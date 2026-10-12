"""Read-only closed-loop XL330 teacher transfer into the XDuck GF plant.

This is a physical feasibility probe, not a deployable cross-robot policy.
Every result must be compared with the teacher in its native XL330 env.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "official_xl330": "microduck_xl330_official_velocity_flat/mujoco",
    "xduck_gf43x40": "xduck_gf43x40_velocity_flat/mujoco",
}


def evaluate(
    checkpoint: Path,
    task: str,
    *,
    physical_action_gain: float,
    command_vx_m_s: float = 0.2,
    filter_tau_s: float = 0.0,
    command_yaw_rad_s: float = 0.0,
    teacher_yaw_feedback_gain: float = 0.0,
    hip_yaw_feedback_gain: float = 0.0,
    hip_yaw_feedback_limit_rad: float = 0.08,
    scene_model: Path | None = None,
    motor_kp: float | None = None,
    motor_kd: float | None = None,
    progress_weight: float = 0.0,
    num_envs: int = 8,
    steps: int = 400,
    warmup: int = 100,
    seed: int = 20260917,
) -> dict:
    if not 0.0 <= physical_action_gain <= 1.0:
        raise ValueError("physical_action_gain must be in [0, 1]")
    if filter_tau_s < 0.0:
        raise ValueError("filter_tau_s must be nonnegative")
    if teacher_yaw_feedback_gain < 0.0:
        raise ValueError("teacher_yaw_feedback_gain must be nonnegative")
    if hip_yaw_feedback_gain < 0.0 or hip_yaw_feedback_limit_rad < 0.0:
        raise ValueError("hip-yaw feedback gain and limit must be nonnegative")
    if hip_yaw_feedback_gain and task != "xduck_gf43x40":
        raise ValueError("direct hip-yaw feedback is only defined for XDuck GF")
    if scene_model is not None and task != "xduck_gf43x40":
        raise ValueError("scene override is only defined for the GF diagnostic task")
    if (motor_kp is not None or motor_kd is not None) and task != "xduck_gf43x40":
        raise ValueError("motor gain overrides are only defined for XDuck GF")
    if progress_weight and task != "xduck_gf43x40":
        raise ValueError("command progress reward is only defined for XDuck GF")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    entry.ensure_registries()
    overrides = [
        f"task={TASKS[task]}",
        "training.device=cpu",
        "env.noise_config.level=0",
        "env.domain_rand.velocity_pushes=false",
        "env.domain_rand.randomize_foot_friction=false",
        "env.domain_rand.randomize_joint_friction=false",
        "env.domain_rand.randomize_dof_armature=false",
    ]
    if motor_kp is not None:
        overrides.append(f"env.control_config.Kp={motor_kp}")
    if motor_kd is not None:
        overrides.append(f"env.control_config.Kd={motor_kd}")
    if progress_weight:
        overrides.append(f"reward.scales.command_progress={progress_weight}")
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )
    env_override = entry.build_ppo_env_cfg_override(cfg)
    if scene_model is not None:
        model_path = scene_model.resolve()
        env_override["scene"] = {
            "model_file": str(model_path),
            "visual_model_file": str(model_path),
        }
        env_override["reward_config"]["foot_target_height"] = 0.02
    env = entry.create_env(cfg, num_envs=num_envs, env_cfg_override=env_override)
    try:
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
        command = np.zeros((num_envs, 13), dtype=np.float32)
        command[:, 0] = command_vx_m_s
        command[:, 2] = command_yaw_rad_s
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), command
        )
        wrapped.reset()
        metrics = {
            key: []
            for key in (
                "reward_rate",
                "vx",
                "vy",
                "vz",
                "yaw_rate",
                "roll_pitch_angular_speed_sq",
                "yaw_rate_error_sq",
                "single_support",
                "both_airborne",
                "air_window",
                "air_window_008",
                "air_window_005",
                "contact_changes_per_s",
                "action_target_clip",
                "torque_nm",
                "joint_speed_rad_s",
                "foot_clearance_m",
                "air_time_s",
                "landed_air_time_s",
            )
        }
        falls = 0
        reward_term_sums = {key: 0.0 for key in env._episode_reward_sums}
        valid_reward_samples = 0
        previous_contact = None
        command_action_abs = []
        model_scale = float(getattr(env.cfg, "policy_action_scale_rad", 1.0))
        filtered_action = np.zeros((num_envs, 14), dtype=np.float32)
        teacher_previous_action = np.zeros((num_envs, 14), dtype=np.float32)
        blend = env.cfg.ctrl_dt / (filter_tau_s + env.cfg.ctrl_dt) if filter_tau_s > 0.0 else 1.0
        with torch.inference_mode():
            for step in range(steps):
                env.state.info["commands"][:] = command
                env.state.obs["obs"][:, -13:] = command
                before_terms = {
                    key: value.copy() for key, value in env._episode_reward_sums.items()
                }
                observation = wrapped.get_observations()
                if task == "xduck_gf43x40" or teacher_yaw_feedback_gain:
                    observation = observation.clone()
                if task == "xduck_gf43x40":
                    # The GF learner now sees raw dimensionless actions. This
                    # cross-robot teacher expects its own previous radian offset.
                    for group in ("actor", "policy"):
                        if group in observation.keys():
                            observation[group][:, 34:48] = torch.from_numpy(teacher_previous_action)
                if teacher_yaw_feedback_gain:
                    teacher_yaw = np.clip(
                        command_yaw_rad_s - teacher_yaw_feedback_gain * env.get_gyro()[:, 2],
                        -1.0,
                        1.0,
                    )
                    for group in ("actor", "policy"):
                        if group in observation.keys():
                            observation[group][:, -11] = torch.from_numpy(teacher_yaw)
                raw_action = policy(observation)
                desired_action = raw_action.cpu().numpy() * physical_action_gain
                filtered_action += blend * (desired_action - filtered_action)
                physical_action = filtered_action.copy()
                if hip_yaw_feedback_gain:
                    yaw_error = env.get_gyro()[:, 2] - command_yaw_rad_s
                    hip_yaw_correction = np.clip(
                        -hip_yaw_feedback_gain * yaw_error,
                        -hip_yaw_feedback_limit_rad,
                        hip_yaw_feedback_limit_rad,
                    )
                    # Both positive hip-yaw offsets caused positive body yaw in the
                    # local plant probe, so correct both joints in the same direction.
                    physical_action[:, (0, 9)] += hip_yaw_correction[:, None]
                env_action = torch.from_numpy(physical_action / model_scale)
                _, reward, done, _ = wrapped.step(env_action)
                valid = ~done.cpu().numpy().astype(bool)
                filtered_action[~valid] = 0.0
                teacher_previous_action[:] = physical_action
                teacher_previous_action[~valid] = 0.0
                contact = env._contact.copy()
                if step >= warmup:
                    falls += int(np.sum(~valid))
                    if np.any(valid):
                        valid_reward_samples += int(np.sum(valid))
                        for key, value in env._episode_reward_sums.items():
                            reward_term_sums[key] += float(
                                np.sum((value[valid] - before_terms[key][valid]) / env.cfg.ctrl_dt)
                            )
                        local_velocity = env.get_local_linvel()
                        gyro = env.get_gyro()
                        yaw_rate = gyro[:, 2]
                        count = contact.sum(axis=1)
                        metrics["reward_rate"].extend(
                            (reward.cpu().numpy()[valid] / env.cfg.ctrl_dt).tolist()
                        )
                        metrics["vx"].extend(local_velocity[valid, 0].tolist())
                        metrics["vy"].extend(local_velocity[valid, 1].tolist())
                        metrics["vz"].extend(local_velocity[valid, 2].tolist())
                        metrics["yaw_rate"].extend(yaw_rate[valid].tolist())
                        metrics["roll_pitch_angular_speed_sq"].extend(
                            np.sum(np.square(gyro[valid, :2]), axis=1).tolist()
                        )
                        metrics["yaw_rate_error_sq"].extend(
                            np.square(yaw_rate[valid] - command_yaw_rad_s).tolist()
                        )
                        metrics["single_support"].extend((count[valid] == 1).astype(float).tolist())
                        metrics["both_airborne"].extend((count[valid] == 0).astype(float).tolist())
                        air_window = np.mean(
                            (env._air_time > env.cfg.reward_config.air_time_threshold_min)
                            & (env._air_time < env.cfg.reward_config.air_time_threshold_max),
                            axis=1,
                        )
                        metrics["air_window"].extend(air_window[valid].tolist())
                        metrics["air_window_008"].extend(
                            np.mean(
                                (env._air_time[valid] > 0.08) & (env._air_time[valid] < 0.3),
                                axis=1,
                            ).tolist()
                        )
                        metrics["air_window_005"].extend(
                            np.mean(
                                (env._air_time[valid] > 0.05) & (env._air_time[valid] < 0.3),
                                axis=1,
                            ).tolist()
                        )
                        if previous_contact is not None:
                            changes = np.any(contact != previous_contact, axis=1)
                            metrics["contact_changes_per_s"].extend(
                                (changes[valid] / env.cfg.ctrl_dt).tolist()
                            )
                        target = env.default_angles[None, :] + physical_action
                        clipped = np.mean(
                            (target < env._target_low) | (target > env._target_high), axis=1
                        )
                        metrics["action_target_clip"].extend(clipped[valid].tolist())
                        metrics["torque_nm"].extend(
                            np.abs(env._motor.torque[valid]).ravel().tolist()
                        )
                        metrics["joint_speed_rad_s"].extend(
                            np.abs(env.get_dof_vel()[valid]).ravel().tolist()
                        )
                        clearance = env._foot_height[valid][~contact[valid]]
                        metrics["foot_clearance_m"].extend(clearance.tolist())
                        metrics["air_time_s"].extend(env._air_time[valid][~contact[valid]].tolist())
                        metrics["landed_air_time_s"].extend(
                            env._landed_air_time[valid][env._first_contact[valid]].tolist()
                        )
                        command_action_abs.extend(np.abs(physical_action[valid]).ravel().tolist())
                previous_contact = contact
        return {
            "checkpoint": str(checkpoint.resolve()),
            "task": task,
            "physical_action_gain": physical_action_gain,
            "command_vx_m_s": command_vx_m_s,
            "filter_tau_s": filter_tau_s,
            "command_yaw_rad_s": command_yaw_rad_s,
            "teacher_yaw_feedback_gain": teacher_yaw_feedback_gain,
            "hip_yaw_feedback_gain": hip_yaw_feedback_gain,
            "hip_yaw_feedback_limit_rad": hip_yaw_feedback_limit_rad,
            "scene_model": str(scene_model.resolve()) if scene_model is not None else None,
            "motor_kp": motor_kp,
            "motor_kd": motor_kd,
            "progress_weight": progress_weight,
            "num_envs": num_envs,
            "seed": seed,
            "warmup_s": warmup * env.cfg.ctrl_dt,
            "probe_s": (steps - warmup) * env.cfg.ctrl_dt,
            "valid_samples": len(metrics["vx"]),
            "falls": falls,
            "mean_reward_rate": float(np.mean(metrics["reward_rate"]))
            if metrics["reward_rate"]
            else None,
            "mean_vx_m_s": float(np.mean(metrics["vx"])) if metrics["vx"] else None,
            "vy_rms_m_s": float(np.sqrt(np.mean(np.square(metrics["vy"]))))
            if metrics["vy"]
            else None,
            "vz_rms_m_s": float(np.sqrt(np.mean(np.square(metrics["vz"]))))
            if metrics["vz"]
            else None,
            "mean_yaw_rate_rad_s": float(np.mean(metrics["yaw_rate"]))
            if metrics["yaw_rate"]
            else None,
            "roll_pitch_angular_speed_rms_rad_s": float(
                np.sqrt(np.mean(metrics["roll_pitch_angular_speed_sq"]))
            )
            if metrics["roll_pitch_angular_speed_sq"]
            else None,
            "yaw_error_rms_rad_s": float(np.sqrt(np.mean(metrics["yaw_rate_error_sq"])))
            if metrics["yaw_rate_error_sq"]
            else None,
            "single_support_fraction": float(np.mean(metrics["single_support"]))
            if metrics["single_support"]
            else None,
            "both_airborne_fraction": float(np.mean(metrics["both_airborne"]))
            if metrics["both_airborne"]
            else None,
            "air_window_fraction": float(np.mean(metrics["air_window"]))
            if metrics["air_window"]
            else None,
            "air_window_008_030_fraction": float(np.mean(metrics["air_window_008"]))
            if metrics["air_window_008"]
            else None,
            "air_window_005_030_fraction": float(np.mean(metrics["air_window_005"]))
            if metrics["air_window_005"]
            else None,
            "contact_changes_per_s": float(np.mean(metrics["contact_changes_per_s"]))
            if metrics["contact_changes_per_s"]
            else None,
            "action_target_clip_fraction": float(np.mean(metrics["action_target_clip"]))
            if metrics["action_target_clip"]
            else None,
            "mean_abs_physical_action_rad": float(np.mean(command_action_abs))
            if command_action_abs
            else None,
            "torque_p99_nm": float(np.quantile(metrics["torque_nm"], 0.99))
            if metrics["torque_nm"]
            else None,
            "torque_peak_nm": float(np.max(metrics["torque_nm"])) if metrics["torque_nm"] else None,
            "joint_speed_p99_rad_s": float(np.quantile(metrics["joint_speed_rad_s"], 0.99))
            if metrics["joint_speed_rad_s"]
            else None,
            "airborne_foot_clearance_p95_m": float(np.quantile(metrics["foot_clearance_m"], 0.95))
            if metrics["foot_clearance_m"]
            else None,
            "airborne_air_time_p50_p95_s": np.quantile(metrics["air_time_s"], [0.50, 0.95]).tolist()
            if metrics["air_time_s"]
            else None,
            "landing_air_time_p50_p95_s": np.quantile(
                metrics["landed_air_time_s"], [0.50, 0.95]
            ).tolist()
            if metrics["landed_air_time_s"]
            else None,
            "reward_terms_rate": {
                key: value / valid_reward_samples if valid_reward_samples else None
                for key, value in reward_term_sums.items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--command-vx", type=float, default=0.2)
    parser.add_argument("--filter-tau", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--teacher-yaw-feedback-gain", type=float, default=0.0)
    parser.add_argument("--hip-yaw-feedback-gain", type=float, default=0.0)
    parser.add_argument("--hip-yaw-feedback-limit", type=float, default=0.08)
    parser.add_argument("--scene-model", type=Path)
    parser.add_argument("--motor-kp", type=float)
    parser.add_argument("--motor-kd", type=float)
    parser.add_argument("--progress-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint.resolve(),
        args.task,
        physical_action_gain=args.gain,
        command_vx_m_s=args.command_vx,
        filter_tau_s=args.filter_tau,
        command_yaw_rad_s=args.command_yaw,
        teacher_yaw_feedback_gain=args.teacher_yaw_feedback_gain,
        hip_yaw_feedback_gain=args.hip_yaw_feedback_gain,
        hip_yaw_feedback_limit_rad=args.hip_yaw_feedback_limit,
        scene_model=args.scene_model,
        motor_kp=args.motor_kp,
        motor_kd=args.motor_kd,
        progress_weight=args.progress_weight,
        seed=args.seed,
        num_envs=args.num_envs,
        steps=args.steps,
        warmup=args.warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
