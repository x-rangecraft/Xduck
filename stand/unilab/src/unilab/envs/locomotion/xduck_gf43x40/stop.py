"""Stop from upright walking states; no sitting or fall-recovery starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.envs.locomotion.microduck_dm4310.velocity import NUM_ACTIONS
from unilab.utils.rotation import np_quat_apply, np_quat_mul, np_yaw_to_quat

from .handoff import WalkHandoffBank
from .velocity import (
    XDuckDRProvider,
    XDuckGF43X40VelocityCfg,
    XDuckGF43X40VelocityEnv,
    XDuckGFControlConfig,
    XDuckRewardConfig,
)


def stop_scales():
    scales = XDuckRewardConfig().scales.copy()
    scales.update(
        track_linear_velocity=4.0,
        track_angular_velocity=2.0,
        action_rate_l2=-0.05,
        air_time=0.0,
        foot_clearance=0.0,
        foot_swing_height=0.0,
        command_progress=0.0,
        head_pose_bias=0.0,
    )
    return scales


@dataclass
class XDuckStopRewardConfig(XDuckRewardConfig):
    scales: dict[str, float] = field(default_factory=stop_scales)
    linear_velocity_std: float = 0.10
    angular_velocity_std: float = 0.35


@registry.envcfg("XDuckGF43X40WalkStopFlat")
@dataclass
class XDuckWalkStopCfg(XDuckGF43X40VelocityCfg):
    control_config: XDuckGFControlConfig = field(
        default_factory=lambda: XDuckGFControlConfig(
            action_scale=1.0,
            Kp=60.0,
            Kd=2.0,
            joint_kp=[90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90],
            joint_kd=[2.5, 2.5, 2.5, 3, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3, 3],
        )
    )
    handoff_bank: str = str(
        Path(__file__).resolve().parents[5] / "data/xduck_stop/walk_handoffs.npz"
    )
    max_episode_seconds: float = 6.0
    failure_tilt_deg: float = 55.0
    minimum_root_height_ratio: float = 0.6
    settled_linear_speed: float = 0.03
    settled_angular_speed: float = 0.2
    settled_joint_speed_rms: float = 0.15
    settled_hold_seconds: float = 1.0
    reward_config: XDuckStopRewardConfig = field(default_factory=XDuckStopRewardConfig)


class WalkStopDRProvider(XDuckDRProvider):
    def _sample_commands(self, env, num_reset):
        return np.zeros((num_reset, 13), dtype=env.default_angles.dtype)

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        ids = np.asarray(env_ids, dtype=np.int32)
        indices = np.random.randint(0, len(env._handoff_bank.qpos), len(ids))
        bank = env._handoff_bank
        q, v = bank.qpos[indices].copy(), bank.qvel[indices].copy()
        # Bank states have zero planar origin and heading. Reset yaw rotates
        # world linear velocity, but MuJoCo free-joint angular qvel is local.
        yaw = np.random.uniform(-np.pi, np.pi, len(ids))
        rotation = np_yaw_to_quat(yaw)
        q[:, :2] += plan.qpos[:, :2]
        q[:, 3:7] = np_quat_mul(rotation, q[:, 3:7])
        v[:, :3] = np_quat_apply(rotation, v[:, :3])
        plan.qpos[:] = q
        plan.qvel[:] = v
        previous = bank.actions[indices].astype(env.default_angles.dtype)
        plan.info_updates["last_actions"] = previous.copy()
        plan.info_updates["current_actions"] = previous.copy()
        env._handoff_indices[ids] = indices
        target = np.clip(
            env.default_angles + previous * env.cfg.policy_action_scale_rad,
            env._target_low,
            env._target_high,
        )
        env._current_action_target[ids] = target
        env._previous_action_target[ids] = target
        # A walking reset retains its sampled height/velocity/contact phase;
        # do not lift it to a fresh airborne pose after this assignment.
        return plan


@registry.env("XDuckGF43X40WalkStopFlat", sim_backend="mujoco")
class XDuckWalkStopEnv(XDuckGF43X40VelocityEnv):
    _cfg: XDuckWalkStopCfg

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self._handoff_bank = WalkHandoffBank.load(cfg.handoff_bank)
        self._handoff_indices = np.zeros(num_envs, dtype=np.int64)
        self._settled_duration = np.zeros(num_envs)
        for value in (
            cfg.settled_linear_speed,
            cfg.settled_angular_speed,
            cfg.settled_joint_speed_rms,
            cfg.settled_hold_seconds,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Stop thresholds must be finite and positive")
        if not 0 < cfg.failure_tilt_deg < 90 or not 0 < cfg.minimum_root_height_ratio < 1:
            raise ValueError("Invalid stop failure thresholds")
        super().__init__(cfg, num_envs, backend_type)
        try:
            self._handoff_bank.validate_contract(cfg, self.default_angles)
            low, high = self._reset_joint_limits[:, 0], self._reset_joint_limits[:, 1]
            if np.any(self._handoff_bank.qpos[:, 7:] < low) or np.any(
                self._handoff_bank.qpos[:, 7:] > high
            ):
                raise ValueError("Handoff bank exceeds current joint limits")
        except BaseException:
            self.close()
            raise

    def _make_dr_provider(self):
        return WalkStopDRProvider()

    def _update_curriculum(self):
        # This dedicated task has fixed zero commands and fixed stop weights.
        # Locomotion curricula must not overwrite its tighter tracking widths.
        self._standing_fraction = 1.0

    def _maybe_resample_commands(self, state):
        state.info["commands"][:] = 0.0

    def _reset_episode_history(self, env_ids):
        super()._reset_episode_history(env_ids)
        self._settled_duration[env_ids] = 0.0

    def _terminated(self, upvector):
        return (upvector[:, 2] < np.cos(np.deg2rad(self.cfg.failure_tilt_deg))) | (
            self._backend.get_base_pos()[:, 2]
            < self._init_qpos[2] * self.cfg.minimum_root_height_ratio
        )

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        reward = super()._compute_reward(state, linvel, gyro, upvector, dof_pos)
        settled = (
            (np.linalg.norm(linvel, axis=1) < self.cfg.settled_linear_speed)
            & (np.linalg.norm(gyro, axis=1) < self.cfg.settled_angular_speed)
            & (np.sqrt(np.mean(self.get_dof_vel() ** 2, axis=1)) < self.cfg.settled_joint_speed_rms)
            & self._contact.all(axis=1)
            & (upvector[:, 2] > np.cos(np.deg2rad(15)))
        )
        self._settled_duration[:] = np.where(
            settled, self._settled_duration + self.cfg.ctrl_dt, 0.0
        )
        log = state.info.setdefault("log", {})
        log["stop/settled_fraction"] = float(
            np.mean(self._settled_duration >= self.cfg.settled_hold_seconds)
        )
        log["stop/planar_speed"] = float(np.mean(np.linalg.norm(linvel[:, :2], axis=1)))
        return reward
