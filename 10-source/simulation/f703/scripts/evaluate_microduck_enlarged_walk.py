#!/usr/bin/env python3

"""Evaluate the enlarged Micro Duck policy with fixed velocity commands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT_DIR / "conf" / "ppo"
sys.path.append(str(ROOT_DIR))

from unilab.base import registry  # noqa: E402


@dataclass(frozen=True)
class Command:
    name: str
    twist: tuple[float, float, float]


@dataclass(frozen=True)
class EvaluationProfile:
    task_config: str
    task_name: str


PROFILES = {
    "enlarged_112": EvaluationProfile(
        task_config="microduck_enlarged_112_walk_flat/mujoco",
        task_name="MicroDuckEnlarged112WalkFlat",
    ),
    "enlarged_112_stand": EvaluationProfile(
        task_config="microduck_enlarged_112_stand_flat/mujoco",
        task_name="MicroDuckEnlarged112StandFlat",
    ),
    # Keep the legacy enlarged owner available for comparisons, but make the
    # current V1.0.4 model the default so a versionless evaluator cannot
    # silently validate a policy against the wrong asset.
    "enlarged": EvaluationProfile(
        task_config="microduck_enlarged_walk_flat/mujoco",
        task_name="MicroDuckEnlargedWalkFlat",
    ),
    "enlarged_104": EvaluationProfile(
        task_config="microduck_enlarged_104_walk_flat/mujoco",
        task_name="MicroDuckEnlarged104WalkFlat",
    ),
    "enlarged_104_stand": EvaluationProfile(
        task_config="microduck_enlarged_104_stand_flat/mujoco",
        task_name="MicroDuckEnlarged104StandFlat",
    ),
    "enlarged_105": EvaluationProfile(
        task_config="microduck_enlarged_105_walk_flat/mujoco",
        task_name="MicroDuckEnlarged105WalkFlat",
    ),
    "enlarged_106": EvaluationProfile(
        task_config="microduck_enlarged_106_walk_flat/mujoco",
        task_name="MicroDuckEnlarged106WalkFlat",
    ),
    "enlarged_106_stand": EvaluationProfile(
        task_config="microduck_enlarged_106_stand_flat/mujoco",
        task_name="MicroDuckEnlarged106StandFlat",
    ),
    "enlarged_107": EvaluationProfile(
        task_config="microduck_enlarged_107_walk_flat/mujoco",
        task_name="MicroDuckEnlarged107WalkFlat",
    ),
    "enlarged_107_stand": EvaluationProfile(
        task_config="microduck_enlarged_107_stand_flat/mujoco",
        task_name="MicroDuckEnlarged107StandFlat",
    ),
    "enlarged_110": EvaluationProfile(
        task_config="microduck_enlarged_110_walk_flat/mujoco",
        task_name="MicroDuckEnlarged110WalkFlat",
    ),
    "enlarged_110_stand": EvaluationProfile(
        task_config="microduck_enlarged_110_stand_flat/mujoco",
        task_name="MicroDuckEnlarged110StandFlat",
    ),
    "enlarged_111": EvaluationProfile(
        task_config="microduck_enlarged_111_walk_flat/mujoco",
        task_name="MicroDuckEnlarged111WalkFlat",
    ),
    "enlarged_111_stand": EvaluationProfile(
        task_config="microduck_enlarged_111_stand_flat/mujoco",
        task_name="MicroDuckEnlarged111StandFlat",
    ),
}


DEFAULT_COMMANDS = (
    Command("stand", (0.0, 0.0, 0.0)),
    Command("forward_0.3", (0.3, 0.0, 0.0)),
    Command("yaw_0.6", (0.0, 0.0, 0.6)),
)


def _parse_command(value: str) -> Command:
    try:
        name, raw_twist = value.split("=", maxsplit=1)
        twist = tuple(float(item) for item in raw_twist.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=VX,VY,WZ") from exc
    if not name or len(twist) != 3 or not np.isfinite(twist).all():
        raise argparse.ArgumentTypeError("expected finite NAME=VX,VY,WZ")
    return Command(name, twist)


def _make_env(
    num_envs: int,
    profile: EvaluationProfile,
    *,
    kp: float | None = None,
    kd: float | None = None,
    identified_dynamics: bool | None = None,
    response_time_ms: tuple[float, float] | None = None,
    motor_constraints: bool | None = None,
    noise_level: float | None = None,
    action_latency: bool | None = None,
    action_scale: float | None = None,
):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config_mlx",
            overrides=[f"task={profile.task_config}"],
        )
    env_override = OmegaConf.to_container(cfg.env, resolve=True)
    if not isinstance(env_override, dict):
        raise TypeError("resolved environment config must be a mapping")
    if kp is not None:
        env_override["control_config"]["Kp"] = kp
    if kd is not None:
        env_override["control_config"]["Kd"] = kd
    if identified_dynamics is not None:
        env_override["identified_dynamics"]["enabled"] = identified_dynamics
    if response_time_ms is not None:
        env_override["identified_dynamics"]["enabled"] = True
        env_override["identified_dynamics"]["response_time_constant_range_s"] = [
            response_time_ms[0] / 1000.0,
            response_time_ms[1] / 1000.0,
        ]
    if motor_constraints is not None:
        env_override["motor_constraints"]["enabled"] = motor_constraints
    if noise_level is not None:
        env_override["noise_config"]["level"] = noise_level
    if action_latency is not None:
        env_override["control_config"]["simulate_action_latency"] = action_latency
    if action_scale is not None:
        env_override["control_config"]["action_scale"] = action_scale
    env_override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    registry.ensure_registries()
    return registry.make(
        profile.task_name,
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=env_override,
    )


def _fixed_observation(state: Any, command: Command) -> np.ndarray:
    twist = np.broadcast_to(
        np.asarray(command.twist, dtype=np.float32),
        (state.obs["obs"].shape[0], 3),
    )
    state.info["commands"] = twist.copy()
    state.info["head_commands"] = np.zeros((twist.shape[0], 4), dtype=np.float32)
    state.info["body_commands"] = np.zeros((twist.shape[0], 6), dtype=np.float32)
    obs = np.asarray(state.obs["obs"], dtype=np.float32).copy()
    obs[:, 48:51] = twist
    obs[:, 51:61] = 0.0
    return obs


def _infer(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    expected_batch = input_meta.shape[0]
    if not isinstance(expected_batch, int) or expected_batch == obs.shape[0]:
        return np.asarray(
            session.run(None, {input_meta.name: obs.astype(np.float32, copy=False)})[0],
            dtype=np.float32,
        )
    if expected_batch != 1:
        raise ValueError(
            f"policy expects batch {expected_batch}, evaluator has batch {obs.shape[0]}"
        )
    return np.concatenate(
        [
            np.asarray(
                session.run(
                    None,
                    {input_meta.name: row[None].astype(np.float32, copy=False)},
                )[0],
                dtype=np.float32,
            )
            for row in obs
        ],
        axis=0,
    )


def _air_durations(contact: np.ndarray, dt: float) -> list[float]:
    durations: list[float] = []
    start: int | None = None
    for index, in_contact in enumerate(contact):
        if not in_contact and start is None:
            start = index
        elif in_contact and start is not None:
            durations.append((index - start) * dt)
            start = None
    return [duration for duration in durations if duration >= 2.0 * dt]


def _landing_periods(contact: np.ndarray, dt: float) -> list[float]:
    landing = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    return (np.diff(landing) * dt).astype(float).tolist() if landing.size >= 2 else []


def _evaluate_seed(
    session: ort.InferenceSession,
    command: Command,
    *,
    seed: int,
    num_envs: int,
    seconds: float,
    profile: EvaluationProfile,
    kp: float | None = None,
    kd: float | None = None,
    identified_dynamics: bool | None = None,
    response_time_ms: tuple[float, float] | None = None,
    motor_constraints: bool | None = None,
    noise_level: float | None = None,
    action_latency: bool | None = None,
    action_scale: float | None = None,
    push_force_n: tuple[float, float, float] | None = None,
    push_start_s: float = 2.5,
    push_duration_s: float = 0.10,
) -> dict[str, Any]:
    np.random.seed(seed)
    env = _make_env(
        num_envs,
        profile,
        kp=kp,
        kd=kd,
        identified_dynamics=identified_dynamics,
        response_time_ms=response_time_ms,
        motor_constraints=motor_constraints,
        noise_level=noise_level,
        action_latency=action_latency,
        action_scale=action_scale,
    )
    env.set_autoreset(False)
    try:
        # ``init_state`` already performs a complete reset and merges its
        # reset-time DR info into the retained state.  Calling ``reset`` again
        # here without merging its info updates would pair the second physical
        # model sample with stale first-reset motor/gain samples.
        state = env.init_state()
        if env.state is None:  # pragma: no cover - owner integration guard
            raise RuntimeError("enlarged Micro Duck environment did not retain reset state")
        obs = _fixed_observation(state, command)
        steps = int(round(seconds / float(env.cfg.ctrl_dt)))
        alive = np.ones(num_envs, dtype=bool)
        previous_action = np.zeros((num_envs, 14), dtype=np.float32)
        ctrl_dt = float(env.cfg.ctrl_dt)
        action_scale_value = float(env.cfg.control_config.action_scale)
        startup_steps = max(1, int(round(1.0 / ctrl_dt)))
        pose_2s_step = max(0, int(round(2.0 / ctrl_dt)) - 1)
        first_step_target_residual_max = 0.0
        startup_target_rate_max = 0.0
        startup_joint_speed_max = 0.0
        startup_requested_torque_max = 0.0
        pose_2s_action_abs_max: float | None = None
        pose_2s_left_hip_yaw_mean: float | None = None
        pose_2s_right_hip_yaw_mean: float | None = None
        velocity_samples: list[np.ndarray] = []
        yaw_samples: list[np.ndarray] = []
        tilt_samples: list[np.ndarray] = []
        action_delta_samples: list[np.ndarray] = []
        base_height_samples: list[np.ndarray] = []
        leg_home_error_samples: list[np.ndarray] = []
        hip_yaw_home_error_samples: list[np.ndarray] = []
        contact_steps: list[np.ndarray] = []
        valid_steps: list[np.ndarray] = []
        linvel_steps: list[np.ndarray] = []
        tilt_steps: list[np.ndarray] = []
        motor_max = {
            "motor_requested_torque_peak_nm": 0.0,
            "motor_effective_torque_peak_nm": 0.0,
            "motor_peak_utilization_max": 0.0,
            "motor_rated_utilization_max": 0.0,
            "motor_rated_speed_utilization_max": 0.0,
            "motor_evidence_speed_utilization_max": 0.0,
            "motor_peak_envelope_excess_max": 0.0,
            "motor_peak_envelope_max_excess_max": 0.0,
            "motor_rated_torque_excess_max": 0.0,
            "motor_rated_power_excess_max": 0.0,
            "motor_evidence_speed_excess_max": 0.0,
            "motor_evidence_speed_max_excess_max": 0.0,
        }
        push_body_ids = None
        if push_force_n is not None:
            push_body_name = env.cfg.domain_rand.push_body_name
            if push_body_name is None:
                raise RuntimeError("push evaluation requires domain_rand.push_body_name")
            push_body_ids = env._backend.get_body_ids([push_body_name])

        for step_index in range(steps):
            action = np.nan_to_num(_infer(session, obs), nan=0.0, posinf=0.0, neginf=0.0)
            physical_target_delta = (action - previous_action) * action_scale_value
            if step_index == 0:
                first_step_target_residual_max = float(
                    np.max(np.abs(action * action_scale_value))
                )
            if step_index < startup_steps:
                startup_target_rate_max = max(
                    startup_target_rate_max,
                    float(np.max(np.abs(physical_target_delta)) / ctrl_dt),
                )
            time_s = step_index * float(env.cfg.ctrl_dt)
            if (
                push_force_n is not None
                and push_body_ids is not None
                and push_start_s <= time_s < push_start_s + push_duration_s
            ):
                force = np.broadcast_to(
                    np.asarray(push_force_n, dtype=np.float64),
                    (num_envs, 1, 3),
                ).copy()
                env._backend.apply_body_force(push_body_ids, force)
            state = env.step(action)
            if step_index < startup_steps and np.any(alive):
                startup_joint_speed_max = max(
                    startup_joint_speed_max,
                    float(np.max(np.abs(np.asarray(env.get_dof_vel())[alive]))),
                )
                startup_requested_torque_max = max(
                    startup_requested_torque_max,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(state.info["motor_requested_torque_nm"])[alive]
                            )
                        )
                    ),
                )
            if step_index == pose_2s_step:
                pose_2s_action_abs_max = float(
                    np.max(np.abs(action * action_scale_value))
                )
                dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float32)
                pose_2s_left_hip_yaw_mean = float(np.mean(dof_pos[:, 0]))
                pose_2s_right_hip_yaw_mean = float(np.mean(dof_pos[:, 9]))
            for metric, info_key in (
                ("motor_requested_torque_peak_nm", "motor_requested_torque_nm"),
                ("motor_effective_torque_peak_nm", "motor_effective_torque_nm"),
                ("motor_peak_utilization_max", "motor_torque_utilization_peak"),
                ("motor_rated_utilization_max", "motor_torque_utilization_rated"),
                (
                    "motor_rated_speed_utilization_max",
                    "motor_speed_utilization_rated",
                ),
                (
                    "motor_evidence_speed_utilization_max",
                    "motor_speed_utilization_evidence",
                ),
                ("motor_peak_envelope_excess_max", "motor_peak_envelope_excess"),
                (
                    "motor_peak_envelope_max_excess_max",
                    "motor_peak_envelope_max_excess",
                ),
                ("motor_rated_torque_excess_max", "motor_rated_torque_excess"),
                ("motor_rated_power_excess_max", "motor_rated_power_excess"),
                ("motor_evidence_speed_excess_max", "motor_evidence_speed_excess"),
                (
                    "motor_evidence_speed_max_excess_max",
                    "motor_evidence_speed_max_excess",
                ),
            ):
                values = np.asarray(state.info[info_key])[alive]
                if values.size:
                    motor_max[metric] = max(motor_max[metric], float(np.max(np.abs(values))))
            linvel = np.asarray(env.get_local_linvel(), dtype=np.float32)
            gyro = np.asarray(env.get_gyro(), dtype=np.float32)
            gravity = np.asarray(env._projected_gravity(), dtype=np.float32)
            tilt = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
            contact = np.asarray(state.info["foot_contact"], dtype=bool)
            base_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
            contact_steps.append(contact.copy())
            valid_steps.append(alive.copy())
            linvel_steps.append(linvel.copy())
            tilt_steps.append(tilt.copy())

            if np.any(alive):
                joint_error = np.asarray(env.get_dof_pos())[alive] - env.default_angles
                leg_home_error_samples.append(joint_error[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]])
                hip_yaw_home_error_samples.append(np.abs(joint_error[:, [0, 9]]))
                velocity_samples.append(linvel[alive])
                yaw_samples.append(gyro[alive, 2])
                tilt_samples.append(tilt[alive])
                action_delta_samples.append(
                    np.mean(np.abs(action[alive] - previous_action[alive]), axis=1)
                )
                base_height_samples.append(base_height[alive])
            previous_action = action
            # Reaching the configured episode horizon is a successful timeout,
            # not a fall.  Only task termination removes an environment from
            # the deterministic survival set.
            alive &= ~np.asarray(state.terminated)
            obs = _fixed_observation(state, command)

        velocity = np.concatenate(velocity_samples, axis=0)
        yaw = np.concatenate(yaw_samples, axis=0)
        tilt = np.concatenate(tilt_samples, axis=0)
        action_delta = np.concatenate(action_delta_samples, axis=0)
        base_height = np.concatenate(base_height_samples, axis=0)
        leg_home_error = np.concatenate(leg_home_error_samples, axis=0)
        hip_yaw_home_error = np.concatenate(hip_yaw_home_error_samples, axis=0)
        contacts = np.stack(contact_steps, axis=0)
        valid = np.stack(valid_steps, axis=0)
        valid_contacts = contacts[valid]
        single_support = np.logical_xor(valid_contacts[:, 0], valid_contacts[:, 1])
        double_support = np.all(valid_contacts, axis=1)
        flight = ~np.any(valid_contacts, axis=1)
        left_air: list[float] = []
        right_air: list[float] = []
        left_periods: list[float] = []
        right_periods: list[float] = []
        for env_index in range(num_envs):
            count = int(np.sum(valid[:, env_index]))
            if count <= 0:
                continue
            env_contact = contacts[:count, env_index]
            left_air.extend(_air_durations(env_contact[:, 0], float(env.cfg.ctrl_dt)))
            right_air.extend(_air_durations(env_contact[:, 1], float(env.cfg.ctrl_dt)))
            left_periods.extend(_landing_periods(env_contact[:, 0], float(env.cfg.ctrl_dt)))
            right_periods.extend(_landing_periods(env_contact[:, 1], float(env.cfg.ctrl_dt)))

        def _median(values: list[float]) -> float | None:
            return float(np.median(values)) if values else None

        target = np.asarray(command.twist, dtype=np.float32)
        tail_steps = max(1, int(round(1.0 / float(env.cfg.ctrl_dt))))
        tail_velocity = np.stack(linvel_steps[-tail_steps:], axis=0)
        tail_tilt = np.stack(tilt_steps[-tail_steps:], axis=0)
        return {
            "seed": seed,
            "survival_rate": float(np.mean(alive)),
            "actual_vx": float(np.mean(velocity[:, 0])),
            "actual_vy": float(np.mean(velocity[:, 1])),
            "actual_wz": float(np.mean(yaw)),
            "vx_mae": float(np.mean(np.abs(velocity[:, 0] - target[0]))),
            "vy_mae": float(np.mean(np.abs(velocity[:, 1] - target[1]))),
            "wz_mae": float(np.mean(np.abs(yaw - target[2]))),
            "tilt_p95_deg": float(np.percentile(tilt, 95)),
            "tilt_max_deg": float(np.max(tilt)),
            "base_height_mean_m": float(np.mean(base_height)),
            "base_height_p95_minus_p05_m": float(
                np.percentile(base_height, 95) - np.percentile(base_height, 5)
            ),
            "action_delta_mean": float(np.mean(action_delta)),
            "leg_home_rms_rad": float(np.sqrt(np.mean(np.square(leg_home_error)))),
            "leg_home_abs_p95_rad": float(np.percentile(np.abs(leg_home_error), 95)),
            "hip_yaw_home_abs_p95_rad": float(np.percentile(hip_yaw_home_error, 95)),
            "hip_yaw_home_abs_max_rad": float(np.max(hip_yaw_home_error)),
            "first_step_target_residual_max_rad": first_step_target_residual_max,
            "first_step_target_rate_max_rad_s": first_step_target_residual_max / ctrl_dt,
            "startup_1s_target_rate_max_rad_s": startup_target_rate_max,
            "startup_1s_joint_speed_max_rad_s": startup_joint_speed_max,
            "startup_1s_requested_torque_max_nm": startup_requested_torque_max,
            "pose_2s_action_abs_max_rad": pose_2s_action_abs_max,
            "pose_2s_left_hip_yaw_mean_rad": pose_2s_left_hip_yaw_mean,
            "pose_2s_right_hip_yaw_mean_rad": pose_2s_right_hip_yaw_mean,
            "single_support_fraction": float(np.mean(single_support)),
            "double_support_fraction": float(np.mean(double_support)),
            "flight_fraction": float(np.mean(flight)),
            "left_air_time_median_s": _median(left_air),
            "right_air_time_median_s": _median(right_air),
            "left_stride_period_median_s": _median(left_periods),
            "right_stride_period_median_s": _median(right_periods),
            "tail_1s_actual_vx": float(np.mean(tail_velocity[:, :, 0])),
            "tail_1s_tilt_max_deg": float(np.max(tail_tilt)),
            **motor_max,
        }
    finally:
        env.close()


def _aggregate(seed_results: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = (
        "survival_rate",
        "actual_vx",
        "actual_vy",
        "actual_wz",
        "vx_mae",
        "vy_mae",
        "wz_mae",
        "tilt_p95_deg",
        "tilt_max_deg",
        "base_height_mean_m",
        "base_height_p95_minus_p05_m",
        "action_delta_mean",
        "leg_home_rms_rad",
        "leg_home_abs_p95_rad",
        "hip_yaw_home_abs_p95_rad",
        "single_support_fraction",
        "double_support_fraction",
        "flight_fraction",
        "tail_1s_actual_vx",
        "tail_1s_tilt_max_deg",
    )
    result: dict[str, float | None] = {
        key: float(np.mean([seed_result[key] for seed_result in seed_results])) for key in keys
    }
    for key in (
        "motor_requested_torque_peak_nm",
        "hip_yaw_home_abs_max_rad",
        "motor_effective_torque_peak_nm",
        "motor_peak_utilization_max",
        "motor_rated_utilization_max",
        "motor_rated_speed_utilization_max",
        "motor_evidence_speed_utilization_max",
        "motor_peak_envelope_excess_max",
        "motor_peak_envelope_max_excess_max",
        "motor_rated_torque_excess_max",
        "motor_rated_power_excess_max",
        "motor_evidence_speed_excess_max",
        "motor_evidence_speed_max_excess_max",
        "first_step_target_residual_max_rad",
        "first_step_target_rate_max_rad_s",
        "startup_1s_target_rate_max_rad_s",
        "startup_1s_joint_speed_max_rad_s",
        "startup_1s_requested_torque_max_nm",
    ):
        result[key] = float(max(seed_result[key] for seed_result in seed_results))
    for key in (
        "left_air_time_median_s",
        "right_air_time_median_s",
        "left_stride_period_median_s",
        "right_stride_period_median_s",
        "pose_2s_action_abs_max_rad",
        "pose_2s_left_hip_yaw_mean_rad",
        "pose_2s_right_hip_yaw_mean_rad",
    ):
        values = [seed_result[key] for seed_result in seed_results if seed_result[key] is not None]
        result[key] = float(np.median(values)) if values else None
    return result


def _print_table(results: dict[str, dict[str, float | None]]) -> None:
    header = (
        f"{'command':<12} {'survive':>8} {'vx':>7} {'vy':>7} {'wz':>7} "
        f"{'vx_mae':>8} {'wz_mae':>8} {'tilt95':>8} {'tiltmax':>8} "
        f"{'height':>7} {'dh95':>7} {'d_action':>9} "
        f"{'step1':>7} {'rate1':>7} {'qd1s':>7} {'tau1s':>7} "
        f"{'single':>7} {'double':>7} {'flight':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<12} {metrics['survival_rate']:>8.3f} "
            f"{metrics['actual_vx']:>7.3f} {metrics['actual_vy']:>7.3f} "
            f"{metrics['actual_wz']:>7.3f} {metrics['vx_mae']:>8.3f} "
            f"{metrics['wz_mae']:>8.3f} {metrics['tilt_p95_deg']:>8.3f} "
            f"{metrics['tilt_max_deg']:>8.3f} {metrics['base_height_mean_m']:>7.3f} "
            f"{metrics['base_height_p95_minus_p05_m']:>7.4f} "
            f"{metrics['action_delta_mean']:>9.5f} "
            f"{metrics['first_step_target_residual_max_rad']:>7.3f} "
            f"{metrics['startup_1s_target_rate_max_rad_s']:>7.3f} "
            f"{metrics['startup_1s_joint_speed_max_rad_s']:>7.3f} "
            f"{metrics['startup_1s_requested_torque_max_nm']:>7.3f} "
            f"{metrics['single_support_fraction']:>7.3f} "
            f"{metrics['double_support_fraction']:>7.3f} "
            f"{metrics['flight_fraction']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="enlarged_104",
        help="model owner used for evaluation (default: enlarged_104)",
    )
    parser.add_argument("--command", type=_parse_command, action="append", dest="commands")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--disable-identified-dynamics", action="store_true")
    parser.add_argument("--response-time-ms", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--disable-motor-constraints", action="store_true")
    parser.add_argument(
        "--noise-level",
        type=float,
        help="Override observation-noise level (1.0 uses the task's declared scales).",
    )
    parser.add_argument(
        "--action-latency",
        action="store_true",
        help="Apply the task's one-control-step action latency model.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Override the policy-to-joint-target scale for legacy checkpoints.",
    )
    parser.add_argument(
        "--push-force-n",
        type=float,
        nargs=3,
        metavar=("FX", "FY", "FZ"),
        help="Apply a deterministic world-frame force pulse to the configured push body.",
    )
    parser.add_argument("--push-start-s", type=float, default=2.5)
    parser.add_argument("--push-duration-s", type=float, default=0.10)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()

    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")
    if args.num_envs <= 0 or args.seconds <= 0.0:
        parser.error("--num-envs and --seconds must be positive")
    if args.action_scale is not None and (
        not np.isfinite(args.action_scale) or args.action_scale <= 0.0
    ):
        parser.error("--action-scale must be finite and positive")
    if args.push_start_s < 0.0 or args.push_duration_s <= 0.0:
        parser.error("--push-start-s must be non-negative and --push-duration-s positive")
    profile = PROFILES[args.profile]
    commands = tuple(args.commands) if args.commands else DEFAULT_COMMANDS
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    results: dict[str, dict[str, float | None]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for command in commands:
        seed_results = [
            _evaluate_seed(
                session,
                command,
                seed=seed,
                num_envs=args.num_envs,
                seconds=args.seconds,
                profile=profile,
                kp=args.kp,
                kd=args.kd,
                identified_dynamics=(False if args.disable_identified_dynamics else None),
                response_time_ms=(tuple(args.response_time_ms) if args.response_time_ms else None),
                motor_constraints=(False if args.disable_motor_constraints else None),
                noise_level=args.noise_level,
                action_latency=(True if args.action_latency else None),
                action_scale=args.action_scale,
                push_force_n=(tuple(args.push_force_n) if args.push_force_n is not None else None),
                push_start_s=args.push_start_s,
                push_duration_s=args.push_duration_s,
            )
            for seed in args.seeds
        ]
        results[command.name] = _aggregate(seed_results)
        details[command.name] = seed_results

    _print_table(results)
    if args.json_path is not None:
        payload = {
            "policy": str(args.policy.resolve()),
            "profile": args.profile,
            "task_name": profile.task_name,
            "seeds": args.seeds,
            "num_envs": args.num_envs,
            "seconds": args.seconds,
            "control_overrides": {
                "kp": args.kp,
                "kd": args.kd,
                "identified_dynamics": (False if args.disable_identified_dynamics else None),
                "response_time_ms": args.response_time_ms,
                "motor_constraints": (False if args.disable_motor_constraints else None),
                "noise_level": args.noise_level,
                "action_latency": True if args.action_latency else None,
                "action_scale": args.action_scale,
            },
            "push": {
                "force_n": args.push_force_n,
                "start_s": args.push_start_s,
                "duration_s": args.push_duration_s,
            },
            "aggregate": results,
            "per_seed": details,
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON: {args.json_path}")


if __name__ == "__main__":
    main()
