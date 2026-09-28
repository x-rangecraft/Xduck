"""Static standing task for the enlarged DM-J4310 Micro Duck."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.walk import (
    _HEAD_JOINT_INDICES,
    _LEG_JOINT_INDICES,
    MicroDuckEnlargedCommands,
    MicroDuckEnlargedRewardConfig,
    MicroDuckEnlargedWalkFlatCfg,
    MicroDuckEnlargedWalkFlatEnv,
)


@dataclass
class MicroDuckEnlargedStandRewardConfig(MicroDuckEnlargedRewardConfig):
    """Pose-tracking parameters measured for the enlarged model."""

    leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.3, 0.3, 0.6, 0.6, 0.4, 0.3, 0.3, 0.6, 0.6, 0.4]
    )
    body_height_std: float = 0.024
    body_angle_std: float = float(np.deg2rad(15.0))
    height_target_min: float = 0.15
    height_target_max: float = 0.22


@dataclass
class MicroDuckEnlargedStandCommands(MicroDuckEnlargedCommands):
    """Retain the 13D command block but hold every slot at zero."""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    rel_standing_envs: float = 1.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    head_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 4, [0.0] * 4]
    )
    body_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 6, [0.0] * 6]
    )


@registry.envcfg("MicroDuckEnlargedStandFlat")
@dataclass
class MicroDuckEnlargedStandFlatCfg(MicroDuckEnlargedWalkFlatCfg):
    commands: MicroDuckEnlargedStandCommands = field(
        default_factory=MicroDuckEnlargedStandCommands
    )
    reward_config: MicroDuckEnlargedStandRewardConfig | None = None
    reset_base_qvel_limit: float = 0.02


@registry.env("MicroDuckEnlargedStandFlat", sim_backend="mujoco")
class MicroDuckEnlargedStandFlatEnv(MicroDuckEnlargedWalkFlatEnv):
    """HOME-pose balance before introducing locomotion commands."""

    _cfg: MicroDuckEnlargedStandFlatCfg

    @property
    def _stand_reward_cfg(self) -> MicroDuckEnlargedStandRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlargedStandRewardConfig):
            raise TypeError("MicroDuckEnlargedStandFlat requires its stand reward config")
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "upright": self._upright_tracking,
            "body_pose": self._body_pose_tracking,
            "leg_pose": self._leg_pose_tracking,
            "head_pose": self._head_pose_tracking,
            "height_target": self._height_target,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "motor_peak_envelope": self._motor_peak_envelope,
            "motor_rated_torque": self._motor_rated_torque_penalty,
            "motor_rated_power": self._motor_rated_power_penalty,
            "motor_speed": self._motor_speed_penalty,
        }

    def _upright_tracking(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        tilt_error = np.sum(np.square(ctx.gravity[:, :2]), axis=1)
        return np.asarray(
            np.exp(-tilt_error / self._stand_reward_cfg.upright_std**2),
            dtype=get_global_dtype(),
        )

    def _body_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        commands = np.asarray(ctx.info["body_commands"], dtype=get_global_dtype())
        z_error = ctx.base_height - (ctx.base_height_target + commands[:, 2])
        z_reward = np.exp(-np.square(z_error / self._stand_reward_cfg.body_height_std))
        roll_error = ctx.gravity[:, 1] - np.sin(commands[:, 3])
        pitch_error = ctx.gravity[:, 0] - np.sin(commands[:, 4])
        angle_scale = np.sin(self._stand_reward_cfg.body_angle_std)
        roll_reward = np.exp(-np.square(roll_error / angle_scale))
        pitch_reward = np.exp(-np.square(pitch_error / angle_scale))
        return np.asarray(
            (z_reward + roll_reward + pitch_reward) / 3.0,
            dtype=get_global_dtype(),
        )

    def _leg_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        error = (
            ctx.dof_pos[:, _LEG_JOINT_INDICES]
            - ctx.default_angles[_LEG_JOINT_INDICES]
        )
        std = np.asarray(self._stand_reward_cfg.leg_pose_stds, dtype=get_global_dtype())
        if std.shape != (_LEG_JOINT_INDICES.size,) or np.any(std <= 0.0):
            raise ValueError("leg_pose_stds must contain ten positive values")
        return np.asarray(
            np.exp(-np.mean(np.square(error / std), axis=1)),
            dtype=get_global_dtype(),
        )

    def _head_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        actual = (
            ctx.dof_pos[:, _HEAD_JOINT_INDICES]
            - ctx.default_angles[_HEAD_JOINT_INDICES]
        )
        command = np.asarray(ctx.info["head_commands"], dtype=get_global_dtype())
        error = actual - command
        return np.asarray(
            np.mean(
                np.exp(-np.square(error / self._stand_reward_cfg.head_pose_std)),
                axis=1,
            ),
            dtype=get_global_dtype(),
        )

    def _height_target(self, ctx: RewardContext) -> np.ndarray:
        low = self._stand_reward_cfg.height_target_min
        high = self._stand_reward_cfg.height_target_max
        below = np.clip(low - ctx.base_height, 0.0, None)
        above = np.clip(ctx.base_height - high, 0.0, None)
        in_range = (ctx.base_height >= low) & (ctx.base_height <= high)
        return np.asarray(
            in_range.astype(get_global_dtype()) - below**2 - above**2,
            dtype=get_global_dtype(),
        )
