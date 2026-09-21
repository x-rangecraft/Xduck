"""Deterministic fixed-command check for one XDuck GF velocity checkpoint.

All reward components are recorded as weighted per-second rates, not the
episode-normalized training log. This evaluator does not change a checkpoint
or the training process.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "idle": [0.0, 0.0, 0.0],
    "forward_02": [0.2, 0.0, 0.0],
    "forward_scaled": [0.2 * math.sqrt(2.09), 0.0, 0.0],
    "backward_02": [-0.2, 0.0, 0.0],
    "lateral_left_02": [0.0, 0.2, 0.0],
    "lateral_right_02": [0.0, -0.2, 0.0],
    "turn_left_10": [0.0, 0.0, 1.0],
    "turn_right_10": [0.0, 0.0, -1.0],
}


def evaluate(
    checkpoint: Path,
    *,
    per_bucket: int = 8,
    steps: int = 400,
    warmup: int = 100,
    stochastic: bool = False,
    zero_last_action_obs: bool = False,
    zero_observation_latency: bool = False,
    action_rate_weight: float | None = None,
    progress_weight: float = 0.0,
    motor_kp: float | None = None,
    motor_kd: float | None = None,
    joint_kp: list[float] | None = None,
    joint_kd: list[float] | None = None,
    seed: int = 20260917,
    focus_forward: bool = False,
) -> dict:
    if per_bucket <= 0 or steps <= warmup or warmup < 0:
        raise ValueError("per_bucket > 0 and steps > warmup >= 0 are required")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    entry.ensure_registries()
    overrides = [
        "task=xduck_gf43x40_velocity_flat/mujoco",
        "training.device=cpu",
        "env.noise_config.level=0",
        "env.domain_rand.velocity_pushes=false",
        "env.domain_rand.randomize_foot_friction=false",
        "env.domain_rand.randomize_joint_friction=false",
        "env.domain_rand.randomize_dof_armature=false",
    ]
    if zero_observation_latency:
        overrides.extend(
            [
                "env.latency.imu_delay_ms=[0.0,0.0]",
                "env.latency.joint_velocity_delay_ms=0.0",
            ]
        )
    if action_rate_weight is not None:
        overrides.append(
            f"+env.curriculum.action_rate_stages=[{{step:0,value:{action_rate_weight}}}]"
        )
    if progress_weight:
        overrides.append(f"reward.scales.command_progress={progress_weight}")
    if motor_kp is not None:
        overrides.append(f"env.control_config.Kp={motor_kp}")
    if motor_kd is not None:
        overrides.append(f"env.control_config.Kd={motor_kd}")
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )
    selected_commands = (
        {name: COMMANDS[name] for name in ("idle", "forward_scaled")} if focus_forward else COMMANDS
    )
    num_envs = per_bucket * len(selected_commands)
    env = entry.create_env(
        cfg, num_envs=num_envs, env_cfg_override=entry.build_ppo_env_cfg_override(cfg)
    )
    try:

        def apply_joint_gains(values: list[float] | None, name: str, maximum: float) -> None:
            if values is None:
                return
            gains = np.asarray(values, dtype=np.float64)
            if gains.shape != (14,) or not np.all(np.isfinite(gains)):
                raise ValueError(f"{name} must contain 14 finite gains")
            if np.any(gains < 0.0) or np.any(gains > maximum):
                raise ValueError(f"{name} must be in [0, {maximum}]")
            setattr(env._motor, f"_nominal_{name}", gains.copy())
            getattr(env._motor, name)[:] = gains[None, :]

        apply_joint_gains(joint_kp, "kp", 500.0)
        apply_joint_gains(joint_kd, "kd", 5.0)
        rl_cfg = entry._algo_config_dict(cfg)
        wrapped = entry._resolve_ppo_wrapper_cls(rl_cfg)(env, device="cpu")
        train_cfg = entry.normalize_ppo_train_cfg(rl_cfg)
        entry.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg["logger"] = "none"
        train_cfg.setdefault("runner", {})["logger"] = "none"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = entry.OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
            runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        commands = np.zeros((num_envs, 13), dtype=np.float32)
        masks = {}
        for index, (name, value) in enumerate(selected_commands.items()):
            mask = np.zeros(num_envs, dtype=bool)
            mask[index * per_bucket : (index + 1) * per_bucket] = True
            masks[name] = mask
            commands[mask, :3] = value
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), commands
        )
        wrapped.reset()
        fields = (
            "reward_rate",
            "vx",
            "vy",
            "yaw_rate",
            "upright",
            "single_support",
            "both_airborne",
            "air_window",
            "contact_change",
            "action_target_clip",
            "mean_abs_physical_action_rad",
            "torque_over_8p9",
            "torque_at_8p9",
            "speed_over_reference",
        )
        accum = {
            name: {
                "count": 0,
                "falls": 0,
                "sum": {field: 0.0 for field in fields},
                "terms": {key: 0.0 for key in env._episode_reward_sums},
                "torque_samples": [],
                "speed_samples": [],
                "action_steps": [],
                "joint_steps": [],
                "target_steps": [],
                "torque_steps": [],
            }
            for name in selected_commands
        }
        previous_contact = None
        reference_speed = float(env._motor.parameters.envelope_speed_rad_s[-1])
        physical_scale = float(env.cfg.policy_action_scale_rad)
        with torch.inference_mode():
            for step in range(steps):
                env.state.info["commands"][:] = commands
                env.state.obs["obs"][:, -13:] = commands
                before = {key: value.copy() for key, value in env._episode_reward_sums.items()}
                observations = wrapped.get_observations()
                if zero_last_action_obs:
                    observations = observations.clone()
                    for group in ("actor", "policy"):
                        if group in observations.keys():
                            observations[group][:, 34:48] = 0.0
                action = (
                    runner.alg.actor(observations, stochastic_output=True)
                    if stochastic
                    else policy(observations)
                )
                _, reward, done, _ = wrapped.step(action)
                done_np = done.cpu().numpy().astype(bool)
                contact = env._contact.copy()
                if step < warmup:
                    previous_contact = contact
                    continue
                valid = ~done_np
                local_velocity = env.get_local_linvel()
                gyro = env.get_gyro()
                quat = env._backend.get_base_quat()
                upright = 1.0 - 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
                count = contact.sum(axis=1)
                air_window = np.mean(
                    (env._air_time > env.cfg.reward_config.air_time_threshold_min)
                    & (env._air_time < env.cfg.reward_config.air_time_threshold_max),
                    axis=1,
                )
                change = (
                    np.any(contact != previous_contact, axis=1)
                    if previous_contact is not None
                    else np.zeros(num_envs, dtype=bool)
                )
                previous_contact = contact
                physical_action = action.cpu().numpy() * physical_scale
                target = env.default_angles[None, :] + physical_action
                clipped = np.mean((target < env._target_low) | (target > env._target_high), axis=1)
                torque = np.abs(env._motor.torque)
                speed = np.abs(env.get_dof_vel())
                rates = {
                    key: (value - before[key]) / env.cfg.ctrl_dt
                    for key, value in env._episode_reward_sums.items()
                }
                for name, mask in masks.items():
                    selected = mask & valid
                    bucket = accum[name]
                    bucket["falls"] += int(np.sum(mask & done_np))
                    n = int(np.sum(selected))
                    if n == 0:
                        continue
                    bucket["count"] += n
                    values = {
                        "reward_rate": reward.cpu().numpy() / env.cfg.ctrl_dt,
                        "vx": local_velocity[:, 0],
                        "vy": local_velocity[:, 1],
                        "yaw_rate": gyro[:, 2],
                        "upright": upright,
                        "single_support": count == 1,
                        "both_airborne": count == 0,
                        "air_window": air_window,
                        "contact_change": change / env.cfg.ctrl_dt,
                        "action_target_clip": clipped,
                        "mean_abs_physical_action_rad": np.mean(np.abs(physical_action), axis=1),
                        "torque_over_8p9": np.mean(torque > 8.9, axis=1),
                        "torque_at_8p9": np.mean(np.isclose(torque, 8.9, atol=1.0e-3), axis=1),
                        "speed_over_reference": np.mean(speed > reference_speed, axis=1),
                    }
                    for field, value in values.items():
                        bucket["sum"][field] += float(np.sum(value[selected]))
                    for key, value in rates.items():
                        bucket["terms"][key] += float(np.sum(value[selected]))
                    bucket["torque_samples"].extend(torque[selected].ravel().tolist())
                    bucket["speed_samples"].extend(speed[selected].ravel().tolist())
                    bucket["action_steps"].append(physical_action[mask].copy())
                    bucket["joint_steps"].append(env.get_dof_pos()[mask].copy())
                    bucket["target_steps"].append(
                        env._motor.communication.active.q_des[mask].copy()
                    )
                    bucket["torque_steps"].append(torque[mask].copy())
        results = {}
        for name, bucket in accum.items():
            n = bucket["count"]
            results[name] = {
                "command": selected_commands[name],
                "valid_env_steps": n,
                "falls": bucket["falls"],
                **{field: bucket["sum"][field] / n if n else None for field in fields},
                "torque_p99_nm": (
                    float(np.quantile(bucket["torque_samples"], 0.99)) if n else None
                ),
                "torque_peak_nm": float(max(bucket["torque_samples"])) if n else None,
                "joint_speed_p99_rad_s": (
                    float(np.quantile(bucket["speed_samples"], 0.99)) if n else None
                ),
                "reward_terms_rate": {
                    key: value / n if n else None for key, value in bucket["terms"].items()
                },
            }
            if n and not bucket["falls"]:
                action_steps = np.stack(bucket["action_steps"])
                joint_steps = np.stack(bucket["joint_steps"])
                target_steps = np.stack(bucket["target_steps"])
                torque_steps = np.stack(bucket["torque_steps"])
                leg_ids = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
                results[name]["action_temporal_std_per_joint_rad"] = np.mean(
                    np.std(action_steps, axis=0), axis=0
                ).tolist()
                results[name]["joint_temporal_std_per_joint_rad"] = np.mean(
                    np.std(joint_steps, axis=0), axis=0
                ).tolist()
                results[name]["target_temporal_std_per_joint_rad"] = np.mean(
                    np.std(target_steps, axis=0), axis=0
                ).tolist()
                results[name]["target_tracking_abs_error_per_joint_rad"] = np.mean(
                    np.abs(target_steps - joint_steps), axis=(0, 1)
                ).tolist()
                results[name]["torque_p99_per_joint_nm"] = np.quantile(
                    torque_steps, 0.99, axis=(0, 1)
                ).tolist()
                results[name]["leg_action_step_delta_rms_rad"] = float(
                    np.sqrt(np.mean(np.square(np.diff(action_steps[:, :, leg_ids], axis=0))))
                )
                results[name]["leg_joint_step_delta_rms_rad"] = float(
                    np.sqrt(np.mean(np.square(np.diff(joint_steps[:, :, leg_ids], axis=0))))
                )
                action_centered = action_steps[:, :, leg_ids] - np.mean(
                    action_steps[:, :, leg_ids], axis=0, keepdims=True
                )
                lag1 = np.mean(action_centered[1:] * action_centered[:-1])
                variance = np.mean(np.square(action_centered))
                results[name]["leg_action_lag1_correlation"] = float(
                    lag1 / variance if variance > 0.0 else 0.0
                )
        return {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_index": int(checkpoint.stem.split("_")[-1]),
            "action_mode": "sampled" if stochastic else "deterministic_mean",
            "zero_last_action_obs": zero_last_action_obs,
            "zero_observation_latency": zero_observation_latency,
            "configured_action_rate_weight": float(env.cfg.reward_config.scales["action_rate_l2"]),
            "configured_progress_weight": progress_weight,
            "configured_motor_kp": motor_kp,
            "configured_motor_kd": motor_kd,
            "diagnostic_joint_kp": joint_kp,
            "diagnostic_joint_kd": joint_kd,
            "seed": seed,
            "focus_forward": focus_forward,
            "per_bucket": per_bucket,
            "warmup_s": warmup * env.cfg.ctrl_dt,
            "probe_s": (steps - warmup) * env.cfg.ctrl_dt,
            "gf_reference_speed_rad_s": reference_speed,
            "torque_over_8p9_note": "8.9 Nm is a low-speed prior, not a measured rated limit",
            "buckets": results,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--zero-last-action-obs", action="store_true")
    parser.add_argument("--zero-observation-latency", action="store_true")
    parser.add_argument("--action-rate-weight", type=float)
    parser.add_argument("--progress-weight", type=float, default=0.0)
    parser.add_argument("--motor-kp", type=float)
    parser.add_argument("--motor-kd", type=float)
    parser.add_argument("--joint-kp", type=float, nargs=14)
    parser.add_argument("--joint-kd", type=float, nargs=14)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--focus-forward", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint.resolve(),
        per_bucket=args.per_bucket,
        steps=args.steps,
        warmup=args.warmup,
        stochastic=args.stochastic,
        zero_last_action_obs=args.zero_last_action_obs,
        zero_observation_latency=args.zero_observation_latency,
        action_rate_weight=args.action_rate_weight,
        progress_weight=args.progress_weight,
        motor_kp=args.motor_kp,
        motor_kd=args.motor_kd,
        joint_kp=args.joint_kp,
        joint_kd=args.joint_kd,
        seed=args.seed,
        focus_forward=args.focus_forward,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint_index": result["checkpoint_index"],
                "buckets": {
                    key: {
                        name: value[name]
                        for name in (
                            "falls",
                            "vx",
                            "yaw_rate",
                            "single_support",
                            "air_window",
                            "contact_change",
                            "action_target_clip",
                            "torque_p99_nm",
                        )
                    }
                    for key, value in result["buckets"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
