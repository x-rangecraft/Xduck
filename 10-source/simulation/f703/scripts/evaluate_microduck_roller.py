#!/usr/bin/env python3

"""Evaluate original and DUCK_V1.01 roller ONNX policies."""

from __future__ import annotations

import argparse
import json
import sys
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
from unilab.envs.locomotion import microduck as _microduck_registry  # noqa: E402, F401
from unilab.envs.locomotion import (  # noqa: E402, F401
    microduck_enlarged_mod1 as _microduck_enlarged_mod1_registry,
)

TASKS = {
    "roller": ("microduck_roller_flat/mujoco", "MicroDuckRollerFlat"),
    "crouch": (
        "microduck_roller_crouch_flat/mujoco",
        "MicroDuckRollerCrouchFlat",
    ),
    "mod1-roller": (
        "microduck_enlarged_mod1_roller_flat/mujoco",
        "MicroDuckEnlargedMod1RollerFlat",
    ),
    "mod1-crouch": (
        "microduck_enlarged_mod1_roller_crouch_flat/mujoco",
        "MicroDuckEnlargedMod1RollerCrouchFlat",
    ),
}

ROLLER_TASKS = {"roller", "mod1-roller"}
CROUCH_TASKS = {"crouch", "mod1-crouch"}


def _make_env(
    task: str,
    num_envs: int,
    *,
    kp: float | None = None,
    kd: float | None = None,
):
    owner, registry_name = TASKS[task]
    overrides = [f"task={owner}"]
    if kp is not None:
        overrides.append(f"env.control_config.Kp={kp}")
    if kd is not None:
        overrides.append(f"env.control_config.Kd={kd}")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config_mlx", overrides=overrides)
    env_override = OmegaConf.to_container(cfg.env, resolve=True)
    if not isinstance(env_override, dict):
        raise TypeError("resolved environment config must be a mapping")
    noise = env_override.setdefault("noise_config", {})
    if not isinstance(noise, dict):
        raise TypeError("noise_config must be a mapping")
    noise["level"] = 0.0
    env_override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    return registry.make(
        registry_name,
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=env_override,
    )


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


def _roller_obs(state: Any, command_x: float) -> np.ndarray:
    command = np.zeros((state.obs["obs"].shape[0], 3), dtype=np.float32)
    command[:, 0] = command_x
    state.info["commands"][:] = command
    for key in ("head_commands", "body_commands"):
        if key in state.info:
            state.info[key][:] = 0.0
    obs = np.asarray(state.obs["obs"], dtype=np.float32).copy()
    obs[:, -13:-10] = command
    obs[:, -10:] = 0.0
    return obs


def _evaluate(
    session: ort.InferenceSession,
    *,
    task: str,
    seed: int,
    num_envs: int,
    seconds: float,
    command_x: float,
    kp: float | None,
    kd: float | None,
    action_multiplier: float,
) -> dict[str, Any]:
    np.random.seed(seed)
    env = _make_env(task, num_envs, kp=kp, kd=kd)
    env.set_autoreset(False)
    try:
        state = env.init_state()
        obs = np.asarray(state.obs["obs"], dtype=np.float32)
        if task in ROLLER_TASKS:
            obs = _roller_obs(state, command_x)
        steps = int(round(seconds / float(env.cfg.ctrl_dt)))
        alive = np.ones(num_envs, dtype=bool)
        survival_steps = np.zeros(num_envs, dtype=np.int32)
        vx_samples: list[np.ndarray] = []
        tilt_samples: list[np.ndarray] = []
        base_height_samples: list[np.ndarray] = []
        one_blade_samples: list[np.ndarray] = []
        double_support_samples: list[np.ndarray] = []
        flight_samples: list[np.ndarray] = []
        pose_error_samples: list[np.ndarray] = []
        joint_pos_samples: list[np.ndarray] = []
        wheel_speed_samples: list[np.ndarray] = []
        action_samples: list[np.ndarray] = []
        final_pose_error: np.ndarray | None = None

        for _ in range(steps):
            action = _infer(session, obs) * action_multiplier
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            state = env.step(action)
            gravity = np.asarray(env._projected_gravity(), dtype=np.float32)
            tilt = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
            contact = np.asarray(state.info["foot_contact"], dtype=bool)
            if np.any(alive):
                vx_samples.append(np.asarray(env.get_local_linvel())[alive, 0])
                tilt_samples.append(tilt[alive])
                base_height_samples.append(env._backend.get_base_pos()[alive, 2])
                contact_count = np.sum(contact[alive], axis=1)
                one_blade_samples.append(contact_count == 1)
                double_support_samples.append(contact_count == 2)
                flight_samples.append(contact_count == 0)
                joint_pos = np.asarray(env.get_dof_pos(), dtype=np.float32).copy()
                joint_pos[~alive] = np.nan
                joint_pos_samples.append(joint_pos)
                wheel_speed = np.asarray(env.get_wheel_vel(), dtype=np.float32).copy()
                wheel_speed[~alive] = np.nan
                wheel_speed_samples.append(wheel_speed)
                sampled_action = np.asarray(action, dtype=np.float32).copy()
                sampled_action[~alive] = np.nan
                action_samples.append(sampled_action)
                if task in CROUCH_TASKS:
                    ctx = env._reward_context(
                        state.info,
                        env.get_local_linvel(),
                        env.get_gyro(),
                        gravity,
                        env.get_dof_pos(),
                        env.get_dof_vel(),
                    )
                    pose_error = np.mean(np.abs(env._pose_error(ctx)), axis=1)
                    pose_error_samples.append(pose_error[alive])
                    final_pose_error = pose_error
            survival_steps[alive] += 1
            alive &= ~np.asarray(state.terminated)
            obs = np.asarray(state.obs["obs"], dtype=np.float32)
            if task in ROLLER_TASKS:
                obs = _roller_obs(state, command_x)

        vx = np.concatenate(vx_samples)
        tilt = np.concatenate(tilt_samples)
        contacts = np.concatenate(one_blade_samples)
        joint_pos = np.stack(joint_pos_samples, axis=0)
        wheel_speed = np.stack(wheel_speed_samples, axis=0)
        sampled_actions = np.stack(action_samples, axis=0)
        left_leg = joint_pos[:, :, 0:5]
        right_leg = joint_pos[:, :, 9:14]
        symmetry_error = np.nanmean(np.abs(left_leg + right_leg), axis=2)
        joint_p2p = np.nanmax(joint_pos, axis=0) - np.nanmin(joint_pos, axis=0)

        def _mean_pair_correlation(left_id: int, right_id: int) -> float:
            correlations = []
            for env_id in range(num_envs):
                left = joint_pos[:, env_id, left_id]
                right = joint_pos[:, env_id, right_id]
                finite = np.isfinite(left) & np.isfinite(right)
                if np.sum(finite) < 3 or np.std(left[finite]) < 1.0e-6 or np.std(right[finite]) < 1.0e-6:
                    continue
                correlations.append(np.corrcoef(left[finite], right[finite])[0, 1])
            return float(np.mean(correlations)) if correlations else float("nan")

        result = {
            "seed": seed,
            "survival_rate": float(np.mean(alive)),
            "mean_survival_seconds": float(np.mean(survival_steps) * env.cfg.ctrl_dt),
            "mean_vx": float(np.mean(vx)),
            "tilt_p95_deg": float(np.percentile(tilt, 95)),
            "tilt_max_deg": float(np.max(tilt)),
            "base_height_min": float(np.min(np.concatenate(base_height_samples))),
            "single_support_fraction": float(np.mean(contacts)),
            "leg_symmetry_l1_rad": float(np.nanmean(symmetry_error)),
            "mean_leg_joint_p2p_rad": float(
                np.nanmean(joint_p2p[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]])
            ),
            "hip_roll_p2p_rad": float(np.nanmean(joint_p2p[:, [1, 10]])),
            "hip_pitch_p2p_rad": float(np.nanmean(joint_p2p[:, [2, 11]])),
            "hip_roll_lr_correlation": _mean_pair_correlation(1, 10),
            "hip_pitch_lr_correlation": _mean_pair_correlation(2, 11),
            "mean_wheel_speed_rad_s": float(np.nanmean(wheel_speed)),
            "action_rms": float(np.sqrt(np.nanmean(np.square(sampled_actions)))),
            "action_abs_p95": float(np.nanpercentile(np.abs(sampled_actions), 95)),
        }
        if task in ROLLER_TASKS:
            result["vx_mae"] = float(np.mean(np.abs(vx - command_x)))

        result["double_support_fraction"] = float(
            np.mean(np.concatenate(double_support_samples))
        )
        result["flight_fraction"] = float(np.mean(np.concatenate(flight_samples)))
        if pose_error_samples:
            result["mean_pose_l1_rad"] = float(np.mean(np.concatenate(pose_error_samples)))
        if final_pose_error is not None:
            result["final_pose_l1_rad"] = float(np.mean(final_pose_error[alive])) if np.any(alive) else float("nan")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--command-x", type=float, default=0.6)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--action-multiplier", type=float, default=1.0)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")
    if args.action_multiplier <= 0.0:
        parser.error("--action-multiplier must be positive")
    seconds = (
        args.seconds
        if args.seconds is not None
        else (10.0 if args.task in ROLLER_TASKS else 5.0)
    )
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    details = [
        _evaluate(
            session,
            task=args.task,
            seed=seed,
            num_envs=args.num_envs,
            seconds=seconds,
            command_x=args.command_x,
            kp=args.kp,
            kd=args.kd,
            action_multiplier=args.action_multiplier,
        )
        for seed in args.seeds
    ]
    numeric_keys = tuple(key for key in details[0] if key != "seed")
    aggregate = {key: float(np.mean([result[key] for result in details])) for key in numeric_keys}
    payload = {
        "policy": str(args.policy.resolve()),
        "task": args.task,
        "seconds": seconds,
        "num_envs": args.num_envs,
        "kp": args.kp,
        "kd": args.kd,
        "action_multiplier": args.action_multiplier,
        "aggregate": aggregate,
        "per_seed": details,
    }
    print(json.dumps(payload, indent=2))
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
