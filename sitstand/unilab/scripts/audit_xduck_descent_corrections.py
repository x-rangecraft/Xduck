"""Paired physical reward probes after learned descent; no PPO training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.sitstand import LEG_IDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--resume-policy-seconds", type=float, default=0.)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    ensure_registries()
    torch.set_num_threads(2)
    config = OmegaConf.create(json.loads((args.checkpoint.parent / "run_config.json").read_text())["config"])
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    progress = saved["infos"]["unilab_progress"]["env_steps_per_env"]
    actor_cfg = OmegaConf.to_container(config.algo.actor, resolve=True)
    actor_cfg.pop("class_name")
    actor = MLPModel(TensorDict({"actor": torch.zeros(16, 61)}, batch_size=[16]),
                     {"actor": ["actor"]}, "actor", 14, **actor_cfg)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    shared_initial = None
    shared_anchor = None
    modes = {"policy": [], "freeze": [], "correct_yaw": [0], "correct_neck": [5],
             "correct_pair": [0, 5], "correct_all": list(range(14))}
    for mode, joints in modes.items():
        if args.modes is not None and mode not in args.modes:
            continue
        np.random.seed(6701)
        torch.manual_seed(6701)
        cfg = copy.deepcopy(config)
        cfg.env.evaluation_case = 3
        cfg.env.max_episode_seconds = 12.1 + args.resume_policy_seconds
        cfg.env.posture_dwell_seconds = [100., 100.]
        cfg.env.curriculum.step_cap = progress
        for item in cfg.env.curriculum.head_range_stages:
            item.ranges = [[0., 0.]] * 4
        env = create_env(cfg, num_envs=16, env_cfg_override=BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override())
        try:
            env.step_counter = progress
            env._update_curriculum()
            env.init_state()
            env.set_autoreset(False)
            goal = env.default_angles.copy()
            goal[LEG_IDS] = env._sit_legs
            fallen = np.zeros(16, dtype=bool)
            anchor = None
            history = {key: [] for key in ("reward", "qpos", "target", "raw_action", "pose_error", "height", "success", "fallen", "head_error")}
            component_history = {key: [] for key in env._episode_reward_sums}
            for step in range(600 + round(args.resume_policy_seconds / .02)):
                if step == 300:
                    anchor = env._current_action_target.copy()
                    shared = env._backend.get_physics_state().copy()
                    if shared_initial is None:
                        shared_initial = shared
                        shared_anchor = anchor.copy()
                    else:
                        # GPU contact solves are not bitwise reproducible.
                        # Require tightly matched poses/targets and report the
                        # observed differences rather than claim exact states.
                        assert np.max(np.abs(shared[:, 1:22] - shared_initial[:, 1:22])) < .005
                        assert np.max(np.abs(anchor - shared_anchor)) < .005
                if step < 300 or step >= 600 or mode == "policy":
                    obs = TensorDict({"actor": torch.from_numpy(env.state.obs["obs"].copy())}, batch_size=[16])
                    with torch.inference_mode():
                        action = actor(obs).numpy()
                else:
                    desired = anchor.copy()
                    mix = np.clip((step - 300 + 1) * .02 / 3., 0., 1.)
                    desired[:, joints] += mix * (goal[joints] - anchor[:, joints])
                    action = np.arctanh(np.clip((desired - env._current_action_target) / env._increments, -.999999, .999999))
                previous = {key: value.copy() for key, value in env._episode_reward_sums.items()}
                state = env.step(action)
                fallen |= env._failed
                delta = {key: env._episode_reward_sums[key] - value for key, value in previous.items()}
                np.testing.assert_allclose(sum(delta.values()), state.reward, atol=1e-5, rtol=1e-5)
                for key, value in delta.items():
                    component_history[key].append(value.copy())
                values = dict(reward=state.reward, qpos=env._backend.get_physics_state()[:, 1:22],
                              target=env._current_action_target, raw_action=action,
                              pose_error=env.last_pose_error, height=env._backend.get_base_pos()[:, 2],
                              success=env.last_success, fallen=fallen,
                              head_error=env.get_dof_pos()[:, 5:9] - goal[5:9])
                for key, value in values.items():
                    history[key].append(value.copy())
            data = {key: np.asarray(value) for key, value in history.items()}
            components = {key: np.asarray(value) for key, value in component_history.items()}
            initial_ok = ~data["fallen"][299]
            final_ok = ~data["fallen"][599]
            success = data["success"][550:600].all(axis=0) & (np.abs(data["head_error"][550:600]) < .15).all(axis=(0, 2)) & final_ok
            row = dict(mode=mode, trials=16, successes=int(success.sum()), falls=int(data["fallen"][599].sum()),
                       initial_pose_max_difference=float(np.max(np.abs(shared[:, 1:22] - shared_initial[:, 1:22]))),
                       initial_target_max_difference=float(np.max(np.abs(anchor - shared_anchor))),
                       post_descent_return_6s=float(data["reward"][300:600, initial_ok].sum(axis=0).mean()),
                       correction_return_3s=float(data["reward"][300:450, initial_ok].sum(axis=0).mean()),
                       late_reward_per_second=float(data["reward"][500:600, initial_ok].mean() / .02),
                       late_pose_error=float(data["pose_error"][550:600, initial_ok].mean()),
                       late_height=float(data["height"][550:600, initial_ok].mean()),
                       components_per_second={key: dict(correction=float(value[300:450, initial_ok].mean()/.02),
                                                       late=float(value[500:600, initial_ok].mean()/.02))
                                              for key, value in components.items()})
            rows.append(row)
            if args.resume_policy_seconds > 0:
                row["policy_resume"] = dict(seconds=args.resume_policy_seconds,
                    initial_action_mean=data["raw_action"][600].mean(axis=0).tolist(),
                    final_pose_error=float(data["pose_error"][-50:].mean()),
                    final_target_mean=data["target"][-50:].mean(axis=(0, 1)).tolist(),
                    final_successes=int((data["success"][-50:].all(axis=0) & (np.abs(data["head_error"][-50:]) < .15).all(axis=(0, 2)) & ~fallen).sum()),
                    falls=int(fallen.sum()))
            for key, value in components.items():
                data["term_" + key] = value
            np.savez_compressed(args.output / f"{mode}.npz", **data)
            (args.output / "results.json").write_text(json.dumps(dict(
                checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                progress=progress, conditions="16 seeded paired trials, seed6701; same first6s deterministic policy descent with measured GPU numerical differences in starting poses/targets (<0.005 rad/m); original GF/limits/physical/reset/observation DR; fixed zero head. After6s diagnostics freeze/correct targets with unchanged increment decoder over3s, then hold3s. Not policy training.", rows=rows), indent=2))
            print(json.dumps({key: value for key, value in row.items() if key != "components_per_second"}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
