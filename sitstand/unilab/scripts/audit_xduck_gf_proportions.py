"""Read-only, fixed-command reward/actuator comparison for XL330 and XDuck GF.

The paired probes use the same physical joint-angle commands, not two learned
policies. They test units, feasibility and relative reward mass; they do not
prove that either policy can learn the probed open-loop motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base import registry

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "official_xl330": "microduck_xl330_official_velocity_flat/mujoco",
    "xduck_gf43x40": "xduck_gf43x40_velocity_flat/mujoco",
}
CASES = {
    "idle": (0.0, 0.0, 0.0, 0.0),
    "forward_hold": (0.2, 0.0, 0.0, 0.0),
    "forward_single_swing": (0.2, 0.6, 0.6, 0.0),
    "forward_single_swing_08_slow": (0.2, 0.8, 0.8, 0.0),
    "forward_single_swing_10_slow": (0.2, 1.0, 1.0, 0.0),
    "forward_alternating": (0.2, 0.6, 0.6, 1.2),
    "forward_alternating_08_slow": (0.2, 0.8, 0.8, 1.6),
    "forward_alternating_10_slow": (0.2, 1.0, 1.0, 2.0),
}
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def matched_state_scale_checks(samples: int = 512) -> dict:
    """Check momentum/clearance units without confounding learned behavior."""
    paths = (
        ROOT.parent / "mjlab/src/mjlab_microduck/robot/microduck/scene_walk.xml",
        ROOT.parent / "Model/Duck_V1.1.0/scene.xml",
    )
    models = [mujoco.MjModel.from_xml_path(str(path)) for path in paths]
    datas = [mujoco.MjData(model) for model in models]
    homes = []
    for model, name in zip(models, ("STAND", "STAND")):
        homes.append(model.key(name).qpos.copy())
    rng = np.random.default_rng(20260916)
    costs = [[], []]
    for _ in range(samples):
        relative_joint_pos = rng.normal(0.0, 0.12, 14)
        joint_vel = rng.normal(0.0, 1.0, 14)
        root_angular_vel = rng.normal(0.0, 1.0, 3)
        for index, (model, data, home) in enumerate(zip(models, datas, homes)):
            data.qpos[:] = home
            data.qvel[:] = 0.0
            for joint_name, delta, velocity in zip(JOINT_NAMES, relative_joint_pos, joint_vel):
                joint = model.joint(joint_name)
                data.qpos[joint.qposadr[0]] += delta
                data.qvel[joint.dofadr[0]] = velocity
            data.qvel[3:6] = root_angular_vel
            mujoco.mj_forward(model, data)
            matrix = np.zeros((3, model.nv))
            mujoco.mj_angmomMat(model, data, matrix, model.body("trunk_base").id)
            angular_momentum = matrix @ data.qvel
            coefficient = (0.02, 8.0e-6)[index]
            costs[index].append(coefficient * float(np.dot(angular_momentum, angular_momentum)))
    official, xduck = map(np.asarray, costs)
    relative_height_error = 0.2
    absolute_foot_speed = 0.5
    official_clearance = 2.0 * (0.02 * relative_height_error) * absolute_foot_speed
    xduck_clearance = (2.0 * 0.02 / 0.035) * (0.035 * relative_height_error) * absolute_foot_speed
    return {
        "samples": samples,
        "official_mass_kg": float(mujoco.mj_getTotalmass(models[0])),
        "xduck_mass_kg": float(mujoco.mj_getTotalmass(models[1])),
        "momentum_cost_official_mean": float(np.mean(official)),
        "momentum_cost_xduck_mean": float(np.mean(xduck)),
        "momentum_cost_xduck_over_official": float(np.mean(xduck) / np.mean(official)),
        "momentum_cost_sample_ratio_p05_p50_p95": np.quantile(
            xduck / official, [0.05, 0.50, 0.95]
        ).tolist(),
        "clearance_cost_equal_relative_error_ratio": xduck_clearance / official_clearance,
        "clearance_comparison_basis": (
            "same relative target-height error and same absolute foot speed"
        ),
    }


def physical_action(step: int, amplitude: float, swing_s: float, period_s: float) -> np.ndarray:
    action = np.zeros(14, dtype=np.float32)
    if amplitude == 0.0 or step < 50:
        return action
    t = (step - 50) * 0.02
    if period_s == 0.0:
        if t >= swing_s:
            return action
        left, phase = True, t / swing_s
    else:
        local = t % period_s
        left = local < period_s / 2.0
        phase = (local % (period_s / 2.0)) / swing_s
    if phase >= 1.0:
        return action
    pulse = float(np.sin(np.pi * phase) ** 2)
    if left:
        action[[2, 3, 4]] = amplitude * pulse * np.asarray([0.5, 1.0, 0.5])
    else:
        action[[11, 12, 13]] = -amplitude * pulse * np.asarray([0.5, 1.0, 0.5])
    return action


def evaluate(task: str, case: str, *, num_envs: int, steps: int) -> dict:
    import train_rsl_rl as entry

    np.random.seed(42)
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={TASKS[task]}",
                "training.device=cpu",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    env = entry.create_env(
        cfg, num_envs=num_envs, env_cfg_override=entry.build_ppo_env_cfg_override(cfg)
    )
    try:
        env._autoreset = False
        state = env.init_state()
        command_vx, amplitude, swing_s, period_s = CASES[case]
        command = np.zeros((num_envs, 13), dtype=np.float32)
        command[:, 0] = command_vx
        env._maybe_resample_commands = lambda current: current.info["commands"].__setitem__(
            slice(None), command
        )
        state.info["commands"][:] = command
        state.obs["obs"][:, -13:] = command
        scale = float(getattr(env.cfg, "policy_action_scale_rad", 1.0))
        terms = {key: [] for key in env._episode_reward_sums}
        rewards, speeds, yaw_rates, heights, single, airborne, air_window = ([] for _ in range(7))
        torques, joint_speeds, clipped, contact_changes = ([] for _ in range(4))
        foot_clearance_samples = []
        previous_contact = None
        falls = 0
        for step in range(steps):
            before = {key: value.copy() for key, value in env._episode_reward_sums.items()}
            physical = physical_action(step, amplitude, swing_s, period_s)
            action = np.broadcast_to(physical / scale, (num_envs, 14)).copy()
            state = env.step(action)
            if step < 50:
                continue
            for key, values in env._episode_reward_sums.items():
                terms[key].append(float(np.mean((values - before[key]) / env.cfg.ctrl_dt)))
            rewards.append(float(np.mean(state.reward) / env.cfg.ctrl_dt))
            speeds.append(float(np.mean(env.get_local_linvel()[:, 0])))
            yaw_rates.append(float(np.mean(env.get_gyro()[:, 2])))
            heights.append(float(np.mean(env._backend.get_base_pos()[:, 2])))
            count = env._contact.sum(axis=1)
            single.append(float(np.mean(count == 1)))
            airborne.append(float(np.mean(count == 0)))
            air_window.append(float(np.mean((env._air_time > 0.125) & (env._air_time < 0.3))))
            foot_clearance_samples.extend(env._foot_height[~env._contact].tolist())
            if previous_contact is not None:
                contact_changes.append(
                    float(np.mean(np.any(env._contact != previous_contact, axis=1)))
                )
            previous_contact = env._contact.copy()
            torques.append(np.abs(env._motor.torque).copy())
            joint_speeds.append(np.abs(env.get_dof_vel()).copy())
            clipped.append(
                float(
                    np.mean(
                        (physical < np.asarray(env.cfg.action_offset_low))
                        | (physical > np.asarray(env.cfg.action_offset_high))
                    )
                )
            )
            falls += int(np.sum(state.terminated))
            if falls:
                break
        torque = np.asarray(torques)
        joint_speed = np.asarray(joint_speeds)
        foot_clearance = np.asarray(foot_clearance_samples)
        max_envelope_speed = (
            env._motor.parameters.envelope_speed_rad_s[-1] if task == "xduck_gf43x40" else None
        )
        return {
            "case": case,
            "task": task,
            "command_vx_m_s": command_vx,
            "requested_probe_s": steps * env.cfg.ctrl_dt,
            "measured_after_warmup_s": len(rewards) * env.cfg.ctrl_dt,
            "falls": falls,
            "valid_for_reward_comparison": falls == 0 and len(rewards) >= 50,
            "mean_reward_rate": float(np.mean(rewards)) if rewards else None,
            "mean_forward_speed_m_s": float(np.mean(speeds)) if speeds else None,
            "mean_yaw_rate_rad_s": float(np.mean(yaw_rates)) if yaw_rates else None,
            "mean_trunk_height_m": float(np.mean(heights)) if heights else None,
            "single_support_fraction": float(np.mean(single)) if single else None,
            "both_airborne_fraction": float(np.mean(airborne)) if airborne else None,
            "air_window_fraction": float(np.mean(air_window)) if air_window else None,
            "contact_changes_per_s": (
                float(np.mean(contact_changes) / env.cfg.ctrl_dt) if contact_changes else None
            ),
            "action_target_clip_fraction": float(np.mean(clipped)) if clipped else None,
            "torque_p99_nm": float(np.quantile(torque, 0.99)) if torque.size else None,
            "torque_peak_nm": float(np.max(torque)) if torque.size else None,
            "joint_speed_p99_rad_s": (
                float(np.quantile(joint_speed, 0.99)) if joint_speed.size else None
            ),
            "fraction_over_gf_reference_speed": (
                float(np.mean(joint_speed > max_envelope_speed))
                if joint_speed.size and max_envelope_speed is not None
                else None
            ),
            "airborne_foot_clearance_p50_p95_m": (
                np.quantile(foot_clearance, [0.5, 0.95]).tolist() if foot_clearance.size else None
            ),
            "airborne_foot_clearance_max_m": (
                float(np.max(foot_clearance)) if foot_clearance.size else None
            ),
            "airborne_foot_at_target_fraction": (
                float(np.mean(foot_clearance >= env.cfg.reward_config.foot_target_height))
                if foot_clearance.size
                else None
            ),
            "reward_terms_rate": {
                key: float(np.mean(values)) if values else None for key, values in terms.items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    registry.ensure_registries()
    results = {
        "matched_state_scale_checks": matched_state_scale_checks(),
        "open_loop_probes": {
            task: {
                case: evaluate(task, case, num_envs=args.num_envs, steps=args.steps)
                for case in CASES
            }
            for task in TASKS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
