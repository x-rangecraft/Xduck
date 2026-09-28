"""Official-Pose-derived standing task for Micro Duck on flat ground."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.walk import (
    MicroDuckCommands,
    MicroDuckRewardConfig,
    MicroDuckWalkFlatCfg,
    MicroDuckWalkFlatEnv,
)

_LEG_JOINT_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
_HEAD_JOINT_INDICES = np.asarray((5, 6, 7, 8), dtype=np.intp)


@dataclass
class MicroDuckStandRewardConfig(MicroDuckRewardConfig):
    """Reward parameters traced to the upstream ``Microduck Pose`` task.

    Phase A keeps all pose commands at zero and trains the nominal standing
    behavior.  The 61D command slots remain present so later phases can widen
    head/body ranges without changing policy I/O.
    """

    leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.3, 0.3, 0.6, 0.6, 0.4, 0.3, 0.3, 0.6, 0.6, 0.4]
    )
    body_height_std: float = 0.015
    body_angle_std: float = float(np.deg2rad(15.0))
    height_target_min: float = 0.07
    height_target_max: float = 0.17


@dataclass
class MicroDuckStandCommands(MicroDuckCommands):
    """Keep the deployment-compatible command block present but identically zero."""

    vel_limit: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rel_standing_envs: float = 1.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    head_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    body_limit: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@registry.envcfg("MicroDuckStandFlat")
@dataclass
class MicroDuckStandFlatCfg(MicroDuckWalkFlatCfg):
    """HOME-pose standing, phase A of the official Pose-policy reproduction."""

    commands: MicroDuckStandCommands = field(default_factory=MicroDuckStandCommands)
    reward_config: MicroDuckStandRewardConfig | None = None
    reset_base_qvel_limit: float = 0.02


@registry.env("MicroDuckStandFlat", sim_backend="mujoco")
class MicroDuckStandFlatEnv(MicroDuckWalkFlatEnv):
    """Static subset of the official stand/body-pose policy recipe."""

    _cfg: MicroDuckStandFlatCfg

    @property
    def _stand_reward_cfg(self) -> MicroDuckStandRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckStandRewardConfig):
            raise TypeError("MicroDuckStandFlat requires MicroDuckStandRewardConfig")
        return cfg

    def _init_reward_functions(self) -> None:
        # The upstream Pose task disables walking rewards and makes pose
        # tracking primary.  These are its zero-command Phase-A terms.
        self._reward_fns = {
            "upright": self._upright_tracking,
            "body_pose": self._body_pose_tracking,
            "leg_pose": self._leg_pose_tracking,
            "head_pose": self._head_pose_tracking,
            "height_target": self._height_target,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
        }

    def _upright_tracking(self, ctx: RewardContext) -> np.ndarray:
        gravity = ctx.gravity
        assert gravity is not None
        tilt_error = np.sum(np.square(gravity[:, :2]), axis=1)
        return np.asarray(
            np.exp(-tilt_error / self._stand_reward_cfg.upright_std**2),
            dtype=get_global_dtype(),
        )

    def _body_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        """Zero-command z/roll/pitch subset of upstream 6D body tracking."""
        gravity = ctx.gravity
        assert gravity is not None
        commands = np.asarray(ctx.info["body_commands"], dtype=get_global_dtype())
        z_error = ctx.base_height - (ctx.base_height_target + commands[:, 2])
        z_reward = np.exp(-np.square(z_error / self._stand_reward_cfg.body_height_std))
        # For projected gravity, x/y are sine-like pitch/roll errors.  This is
        # exact at HOME and monotonic over the standing termination envelope.
        roll_error = gravity[:, 1] - np.sin(commands[:, 3])
        pitch_error = gravity[:, 0] - np.sin(commands[:, 4])
        angle_scale = np.sin(self._stand_reward_cfg.body_angle_std)
        roll_reward = np.exp(-np.square(roll_error / angle_scale))
        pitch_reward = np.exp(-np.square(pitch_error / angle_scale))
        return np.asarray(
            (z_reward + roll_reward + pitch_reward) / 3.0,
            dtype=get_global_dtype(),
        )

    def _leg_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        error = ctx.dof_pos[:, _LEG_JOINT_INDICES] - ctx.default_angles[_LEG_JOINT_INDICES]
        std = np.asarray(self._stand_reward_cfg.leg_pose_stds, dtype=get_global_dtype())
        if std.shape != (_LEG_JOINT_INDICES.size,) or np.any(std <= 0.0):
            raise ValueError("leg_pose_stds must contain ten positive values")
        return np.asarray(
            np.exp(-np.mean(np.square(error / std), axis=1)),
            dtype=get_global_dtype(),
        )

    def _head_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        actual = ctx.dof_pos[:, _HEAD_JOINT_INDICES] - ctx.default_angles[_HEAD_JOINT_INDICES]
        command = np.asarray(ctx.info["head_commands"], dtype=get_global_dtype())
        error = actual - command
        return np.asarray(
            np.mean(np.exp(-np.square(error / self._stand_reward_cfg.head_pose_std)), axis=1),
            dtype=get_global_dtype(),
        )

    def _height_target(self, ctx: RewardContext) -> np.ndarray:
        low = self._stand_reward_cfg.height_target_min
        high = self._stand_reward_cfg.height_target_max
        below = np.clip(low - ctx.base_height, 0.0, None)
        above = np.clip(ctx.base_height - high, 0.0, None)
        in_range = (ctx.base_height >= low) & (ctx.base_height <= high)
        return np.asarray(in_range.astype(get_global_dtype()) - below**2 - above**2)
