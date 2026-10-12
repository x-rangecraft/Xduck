"""Demonstration collection and closed-loop gates for the direct XDuck policy.

Reference samples label training data only. Evaluation sends network outputs
straight to the direct task, with no teacher correction.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from unilab.training import create_env, ensure_registries
from unilab.training.backend_adapter import BackendAdapter

DR_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stage_config(config, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("DR fraction must be in [0,1]")
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    dr = cfg.env.domain_rand
    for name in (
        "mass_inertia_scale_range",
        "foot_friction_range",
        "dof_armature_multiplier_range",
    ):
        dr[name] = [1 + fraction * (v - 1) for v in dr[name]]
    dr.imu_max_angle_deg *= fraction
    dr.randomize_dof_armature = fraction > 0
    cfg.env.ankle_target_offset_max_deg *= fraction
    cfg.env.noise_config.level *= fraction
    for name in ("com_range_stages", "head_com_range_stages"):
        for item in cfg.env.curriculum[name]:
            item.value *= fraction
    # Keep measured transport and latency throughout: the accepted teacher used them.
    assert not dr.push_robots and not dr.velocity_pushes
    return cfg


def make_env(config, root, count):
    ensure_registries()
    override = BackendAdapter(config, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    return create_env(config, num_envs=count, env_cfg_override=override)


def set_posture_command(env, sitting):
    """Set evaluator commands and synchronize the already-materialized observations."""
    command = env.state.info["commands"]
    changed = command[:, 0] != sitting
    command[:, 0] = sitting
    env._command_age[changed] = 0
    # Actor contract: gyro3, gravity3, q14, dq14, action14, command13, clock1.
    env.state.obs["obs"][:, 48:61] = command
    env.state.obs["obs"][:, -1] = np.minimum(env._command_age / 4, 1)
    # Critic has the same command channels with privileged state/contact fields.
    env.state.obs["critic"][:, 51:54] = command[:, :3]
    env.state.obs["critic"][:, 66:76] = command[:, 3:13]
    env.state.obs["critic"][:, -1] = np.minimum(env._command_age / 4, 1)


def teacher_actions(env):
    target = env._ref_target.copy()
    desired = (env.state.info["commands"][:, 0] > 0.5).astype(np.int8)
    direction = env._ref_direction.copy()
    start = (direction == 0) & (desired != env._ref_posture)
    direction[start] = np.where(desired[start], 1, -1)
    elapsed = env._ref_elapsed.copy()
    elapsed[start] = 0
    for sign, name in ((-1, "rise"), (1, "descent")):
        ids = np.flatnonzero(direction == sign)
        if len(ids):
            target[ids] = env._refs[name].sample(elapsed[ids] + env.cfg.ctrl_dt)[0]
    return (target - env.default_angles).astype(np.float32)


def actor_batch(obs):
    array = torch.as_tensor(obs, dtype=torch.float32)
    return TensorDict({"actor": array}, batch_size=[len(array)])


def rollout(
    config,
    root,
    *,
    actor=None,
    count=16,
    segments=8,
    seed=1,
    collect=False,
    teacher_mix=0.0,
    independent=False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.env.nominal_env_fraction = 0.0  # evaluation must not dilute requested DR
    cfg.env.sit_reset_probability = 0
    cfg.env.sit_command_probability = 1
    cfg.env.posture_dwell_seconds = [99, 100]
    cfg.env.max_episode_seconds = 1000
    env = make_env(cfg, root, count)
    rows, actions, critics, groups = [], [], [], []
    metrics = {"rise": [], "descent": []}
    failures = {"rise": [], "descent": []}
    initial_domains = []
    if actor is not None:
        actor.eval()
    try:
        env.init_state()
        for segment in range(segments):
            sitting = segment % 2 == 0
            name = "descent" if sitting else "rise"
            if independent:
                # A full fresh environment + per-trial seed prevents failures
                # in earlier trials changing the next trial's DR/transport RNG.
                env.close()
                trial_seed = seed + 1009 * segment
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                cfg.env.sit_reset_probability = float(not sitting)
                cfg.env.sit_command_probability = float(sitting)
                env = make_env(cfg, root, count)
                env.init_state()
                digest = hashlib.sha256()
                for value in (
                    env._mass_inertia_scale,
                    env._foot_friction,
                    env._imu_rotation,
                    env._ankle_target_offset,
                ):
                    digest.update(value.tobytes())
                initial_domains.append(
                    {"seed": trial_seed, "sampled_dr_sha256": digest.hexdigest()}
                )
            initial_height = env._backend.get_base_pos()[:, 2]
            alive = abs(initial_height - (0.3129 if sitting else 0.2182)) < 0.015
            failed_at = np.full(count, -1)
            failed_at[~alive] = 0
            held = np.ones(count, bool)
            set_posture_command(env, sitting)
            for step in range(200):
                set_posture_command(env, sitting)
                obs = env.state.obs["obs"].copy()
                expert = teacher_actions(env)
                if collect:
                    rows.append(obs[alive])
                    actions.append(expert[alive])
                    critics.append(env.state.obs["critic"][alive].copy())
                    groups.append(np.flatnonzero(alive))
                if actor is None:
                    action = expert
                else:
                    with torch.no_grad():
                        action = actor(actor_batch(obs)).cpu().numpy()
                    if collect:
                        action = teacher_mix * expert + (1 - teacher_mix) * action
                state = env.step(action)
                failed_at[alive & (state.terminated | state.truncated)] = step + 1
                alive &= ~(state.terminated | state.truncated)
                if step >= 175:  # require last 0.5 s continuously stable
                    height = env._backend.get_base_pos()[:, 2]
                    quat = env._backend.get_base_quat()
                    up_z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
                    speed = np.linalg.norm(env._backend.get_base_lin_vel(), axis=1)
                    target = env._refs[name].nodes[-1]
                    pose_rms = np.sqrt(np.mean((env.get_dof_pos() - target) ** 2, axis=1))
                    held &= abs(height - (0.2182 if sitting else 0.3129)) < 0.01
                    held &= (up_z > np.cos(np.deg2rad(5))) & (speed < 0.03) & (pose_rms < 0.08)
            metrics[name].extend((alive & held).tolist())
            failures[name].extend(failed_at.tolist())
        result = {
            k: {
                "successes": int(sum(v)),
                "trials": len(v),
                "rate": float(np.mean(v)),
                "first_failure_step": failures[k],
                "invalid_starts": sum(x == 0 for x in failures[k]),
                "motion_failures": sum(x > 0 for x in failures[k]),
                "hold_failures": sum(x == -1 and not ok for x, ok in zip(failures[k], v)),
                "valid_start_success_rate": sum(v) / max(1, sum(x != 0 for x in failures[k])),
            }
            for k, v in metrics.items()
        }
        result["protocol"] = "independent_fresh_env_v2" if independent else "continuous_cycles"
        if independent:
            result["initial_domains"] = initial_domains
        if not independent:
            down = np.asarray(metrics["descent"]).reshape(-1, count)
            up = np.asarray(metrics["rise"]).reshape(-1, count)
            pairs = min(len(down), len(up))
            cycles = down[:pairs] & up[:pairs]
            result["cycles"] = {
                "successes": int(cycles.sum()),
                "trials": int(cycles.size),
                "rate": float(cycles.mean()) if cycles.size else 0.0,
            }
        data = None
        if collect:
            data = {
                "obs": np.concatenate(rows),
                "actions": np.concatenate(actions),
                "critic": np.concatenate(critics),
                "group": np.concatenate(groups),
            }
        return result, data
    finally:
        env.close()


@dataclass
class SuccessCurriculum:
    """Promote only after two independent passing evaluation windows."""

    stage: int = 0
    streak: int = 0
    threshold: float = 0.9

    def observe(self, metrics):
        passed = all(
            metrics[k]["trials"] >= 32 and metrics[k]["rate"] >= self.threshold
            for k in ("rise", "descent")
        )
        self.streak = self.streak + 1 if passed else 0
        if self.streak >= 2 and self.stage < len(DR_LEVELS) - 1:
            self.stage += 1
            self.streak = 0
            return True
        return False


def no_regression(candidate, baseline):
    """Paired finite sample gate; never claim this proves global robustness."""
    if (
        "initial_domains" in baseline
        and candidate.get("initial_domains") != baseline["initial_domains"]
    ):
        return False
    return all(
        candidate[k]["trials"] == baseline[k]["trials"]
        and candidate[k]["successes"] >= baseline[k]["successes"]
        for k in ("rise", "descent", "cycles")
        if k in baseline
    )
