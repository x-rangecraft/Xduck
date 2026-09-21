"""Deterministic fixed-command gate for MicroDuck velocity checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from unilab.utils.rotation import np_quat_apply_inverse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_rsl_rl as entry

COMMANDS = {
    "idle": [0.0, 0.0, 0.0],
    "forward_02": [0.2, 0.0, 0.0],
    "backward_02": [-0.2, 0.0, 0.0],
    "lateral_left_02": [0.0, 0.2, 0.0],
    "lateral_right_02": [0.0, -0.2, 0.0],
    "turn_left_10": [0.0, 0.0, 1.0],
    "turn_right_10": [0.0, 0.0, -1.0],
}

THRESHOLDS = {
    "falls_per_10s_mean": 0.05,
    "idle_planar_speed_m_s": 0.05,
    "idle_abs_yaw_rate_rad_s": 0.20,
    "command_axis_error_m_s": 0.18,
    "command_yaw_error_rad_s": 0.60,
    "minimum_response_gain": 0.15,
    "rated_torque_exceed_fraction": 0.05,
    "no_load_speed_exceed_fraction": 0.01,
    "target_outside_hard_limit_fraction": 0.0,
    "airborne_fraction": 0.10,
    "contact_changes_per_second": 8.0,
    "active_min_single_foot_contact_fraction": 0.75,
    "active_min_contact_changes_per_second": 4.0,
    "action_delta_l2_mean": 0.15,
    "leg_pose_rms_mean_rad": 0.20,
    "head_pose_rms_mean_rad": 0.20,
    "minimum_upright_mean": 0.95,
    "stance_width_min_m": 0.08,
    "stance_width_max_m": 0.22,
    "idle_stance_width_min_m": 0.15,
}


def _legacy_policy_schema(cfg, checkpoint: dict) -> None:
    if "distribution.std_param" not in checkpoint["actor_state_dict"]:
        return
    with open_dict(cfg.algo.policy):
        for key in (
            "distribution_class_name",
            "min_std",
            "max_std",
            "action_low",
            "action_high",
        ):
            if key in cfg.algo.policy:
                del cfg.algo.policy[key]
        cfg.algo.policy.std_type = "scalar"


def evaluate(run_dir: Path, checkpoint_path: Path, *, per_bucket: int, steps: int) -> dict:
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=microduck_dm4310_velocity_flat/mujoco",
                "training.device=cpu",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
            ],
        )
    _legacy_policy_schema(cfg, checkpoint)
    env = entry.create_env(cfg, num_envs=per_bucket * len(COMMANDS))
    try:
        env._mass_inertia_scale[:] = 1.0
        rl_cfg = entry._algo_config_dict(cfg)
        wrapped = entry._resolve_ppo_wrapper_cls(rl_cfg)(env, device="cpu")
        train_cfg = entry.normalize_ppo_train_cfg(rl_cfg)
        entry.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        runner = entry.OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
        runner.load(str(checkpoint_path), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        command = np.zeros((env.num_envs, 13), dtype=np.float32)
        labels = []
        for index, (name, value) in enumerate(COMMANDS.items()):
            section = slice(index * per_bucket, (index + 1) * per_bucket)
            command[section, :3] = value
            labels.extend([name] * per_bucket)
        labels = np.asarray(labels)

        # Fixed commands are applied by the normal per-step command lifecycle.
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), command
        )
        wrapped.reset()
        count = np.zeros(env.num_envs)
        falls = np.zeros(env.num_envs)
        velocity_sum = np.zeros((env.num_envs, 3))
        both = np.zeros(env.num_envs)
        single = np.zeros(env.num_envs)
        airborne = np.zeros(env.num_envs)
        contact_changes = np.zeros(env.num_envs)
        rated = np.zeros(env.num_envs)
        overspeed = np.zeros(env.num_envs)
        outside = np.zeros(env.num_envs)
        reward_rate = np.zeros(env.num_envs)
        upright_sum = np.zeros(env.num_envs)
        leg_pose_rms_sum = np.zeros(env.num_envs)
        head_pose_rms_sum = np.zeros(env.num_envs)
        stance_width_sum = np.zeros(env.num_envs)
        action_delta_sum = np.zeros(env.num_envs)
        previous_action = None
        previous_contact = None
        with torch.inference_mode():
            for _ in range(steps):
                env.state.info["commands"][:] = command
                env._velocity_timer[:] = 1_000_000
                env._head_timer[:] = 1_000_000
                env._body_timer[:] = 1_000_000
                # Patch command slots only: update_state also advances latency/history.
                env.state.obs["obs"][:, -13:] = command
                obs = wrapped.get_observations()
                action = policy(obs)
                _, reward, done, _ = wrapped.step(action)
                valid = ~done.cpu().numpy()
                falls += env.state.terminated
                linvel = env.get_local_linvel()
                gyro = env.get_gyro()
                velocity_sum += np.c_[linvel[:, :2], gyro[:, 2]] * valid[:, None]
                contact = env._contact.copy()
                contact_count = contact.sum(axis=1)
                both += (contact_count == 2) * valid
                single += (contact_count == 1) * valid
                airborne += (contact_count == 0) * valid
                if previous_contact is not None:
                    contact_changes += np.any(contact != previous_contact, axis=1) * valid
                previous_contact = contact
                torque = np.abs(env._motor.torque)
                speed = np.abs(env.get_dof_vel())
                rated += np.mean(torque > env._motor._rated_torque, axis=1) * valid
                overspeed += np.mean(speed > env._motor._maximum_speed, axis=1) * valid
                target = env.default_angles[None, :] + action.cpu().numpy()
                action_np = action.cpu().numpy()
                outside += (
                    np.mean(
                        (target < env._joint_limits[:, 0]) | (target > env._joint_limits[:, 1]),
                        axis=1,
                    )
                    * valid
                )
                reward_rate += reward.cpu().numpy() / env._cfg.ctrl_dt * valid
                quat = env._backend.get_base_quat()
                upright_sum += (1.0 - 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))) * valid
                joint_error = env.get_dof_pos() - env.default_angles
                leg_ids = np.asarray([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
                leg_pose_rms_sum += (
                    np.sqrt(np.mean(np.square(joint_error[:, leg_ids]), axis=1)) * valid
                )
                head_pose_rms_sum += (
                    np.sqrt(np.mean(np.square(joint_error[:, 5:9]), axis=1)) * valid
                )
                foot_delta = env._foot_pos[:, 0] - env._foot_pos[:, 1]
                stance_width_sum += np.abs(np_quat_apply_inverse(quat, foot_delta)[:, 1]) * valid
                if previous_action is not None:
                    action_delta_sum += (
                        np.sum(np.square(action_np - previous_action), axis=1) * valid
                    )
                previous_action = action_np
                count += valid

        buckets = {}
        for name, requested in COMMANDS.items():
            mask = labels == name
            denominator = np.maximum(count[mask], 1.0)
            actual = np.mean(velocity_sum[mask] / denominator[:, None], axis=0)
            buckets[name] = {
                "requested_vx_vy_yaw": requested,
                "actual_vx_vy_yaw_mean": actual.tolist(),
                "falls_per_10s_mean": float(np.mean(falls[mask]) * 500.0 / steps),
                "reward_rate_mean": float(np.mean(reward_rate[mask] / denominator)),
                "both_feet_contact_fraction": float(np.mean(both[mask] / denominator)),
                "single_foot_contact_fraction": float(np.mean(single[mask] / denominator)),
                "airborne_fraction": float(np.mean(airborne[mask] / denominator)),
                "contact_changes_per_second": float(
                    np.mean(contact_changes[mask] / denominator) / env._cfg.ctrl_dt
                ),
                "rated_torque_exceed_fraction": float(np.mean(rated[mask] / denominator)),
                "no_load_speed_exceed_fraction": float(np.mean(overspeed[mask] / denominator)),
                "target_outside_hard_limit_fraction": float(np.mean(outside[mask] / denominator)),
                "upright_mean": float(np.mean(upright_sum[mask] / denominator)),
                "leg_pose_rms_mean_rad": float(np.mean(leg_pose_rms_sum[mask] / denominator)),
                "head_pose_rms_mean_rad": float(np.mean(head_pose_rms_sum[mask] / denominator)),
                "stance_width_mean_m": float(np.mean(stance_width_sum[mask] / denominator)),
                "action_delta_l2_mean": float(np.mean(action_delta_sum[mask] / denominator)),
            }

        checks = {}
        idle = buckets["idle"]
        checks["idle_fall"] = bool(idle["falls_per_10s_mean"] <= THRESHOLDS["falls_per_10s_mean"])
        checks["idle_planar"] = bool(
            np.linalg.norm(idle["actual_vx_vy_yaw_mean"][:2]) <= THRESHOLDS["idle_planar_speed_m_s"]
        )
        checks["idle_yaw"] = bool(
            abs(idle["actual_vx_vy_yaw_mean"][2]) <= THRESHOLDS["idle_abs_yaw_rate_rad_s"]
        )
        for name, bucket in buckets.items():
            checks[f"{name}/falls"] = bool(
                bucket["falls_per_10s_mean"] <= THRESHOLDS["falls_per_10s_mean"]
            )
            requested = np.asarray(bucket["requested_vx_vy_yaw"])
            actual = np.asarray(bucket["actual_vx_vy_yaw_mean"])
            tolerance = np.asarray(
                [
                    THRESHOLDS["command_axis_error_m_s"],
                    THRESHOLDS["command_axis_error_m_s"],
                    THRESHOLDS["command_yaw_error_rad_s"],
                ]
            )
            checks[f"{name}/tracking"] = bool(np.all(np.abs(actual - requested) <= tolerance))
            active = np.abs(requested) > 1.0e-9
            if np.any(active):
                checks[f"{name}/response_gain"] = bool(
                    np.all(
                        actual[active] * np.sign(requested[active])
                        >= THRESHOLDS["minimum_response_gain"] * np.abs(requested[active])
                    )
                )
                checks[f"{name}/single_support"] = bool(
                    bucket["single_foot_contact_fraction"]
                    >= THRESHOLDS["active_min_single_foot_contact_fraction"]
                )
                checks[f"{name}/minimum_contact_cadence"] = bool(
                    bucket["contact_changes_per_second"]
                    >= THRESHOLDS["active_min_contact_changes_per_second"]
                )
            for metric in (
                "rated_torque_exceed_fraction",
                "no_load_speed_exceed_fraction",
                "target_outside_hard_limit_fraction",
                "airborne_fraction",
                "contact_changes_per_second",
                "action_delta_l2_mean",
                "leg_pose_rms_mean_rad",
                "head_pose_rms_mean_rad",
            ):
                checks[f"{name}/{metric}"] = bool(bucket[metric] <= THRESHOLDS[metric])
            checks[f"{name}/upright"] = bool(
                bucket["upright_mean"] >= THRESHOLDS["minimum_upright_mean"]
            )
            checks[f"{name}/stance_width"] = bool(
                THRESHOLDS["stance_width_min_m"]
                <= bucket["stance_width_mean_m"]
                <= THRESHOLDS["stance_width_max_m"]
            )
        checks["idle/stance_width"] = bool(
            THRESHOLDS["idle_stance_width_min_m"]
            <= idle["stance_width_mean_m"]
            <= THRESHOLDS["stance_width_max_m"]
        )
        checks["opposite_commands/forward_separation"] = bool(
            buckets["forward_02"]["actual_vx_vy_yaw_mean"][0]
            - buckets["backward_02"]["actual_vx_vy_yaw_mean"][0]
            >= 0.08
        )
        checks["opposite_commands/lateral_separation"] = bool(
            buckets["lateral_left_02"]["actual_vx_vy_yaw_mean"][1]
            - buckets["lateral_right_02"]["actual_vx_vy_yaw_mean"][1]
            >= 0.06
        )
        checks["opposite_commands/yaw_separation"] = bool(
            buckets["turn_left_10"]["actual_vx_vy_yaw_mean"][2]
            - buckets["turn_right_10"]["actual_vx_vy_yaw_mean"][2]
            >= 0.80
        )
        return {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_iteration": int(checkpoint["iter"]),
            "per_bucket": per_bucket,
            "steps": steps,
            "thresholds": THRESHOLDS,
            "buckets": buckets,
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="model_999.pt")
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.run_dir.resolve(),
        (args.run_dir / args.checkpoint).resolve(),
        per_bucket=args.per_bucket,
        steps=args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
