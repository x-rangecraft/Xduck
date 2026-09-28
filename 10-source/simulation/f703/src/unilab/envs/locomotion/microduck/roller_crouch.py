"""Phase-driven crouch-and-glide action for the original Micro Duck rollers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.roller import (
    MicroDuckRollerCommands,
    MicroDuckRollerFlatCfg,
    MicroDuckRollerFlatEnv,
    MicroDuckRollerRewardConfig,
)
from unilab.envs.locomotion.microduck.walk import (
    MICRODUCK_JOINT_NAMES,
    MicroDuckWalkDomainRandomizationProvider,
)

OFFICIAL_STAND_POSE = (
    -0.0476,
    -0.0629,
    -0.2869,
    0.9618,
    1.1674,
    0.6029,
    0.5430,
    -0.0690,
    -0.0414,
    -0.0337,
    -0.0061,
    0.1534,
    -0.9725,
    -1.0646,
)

OFFICIAL_CROUCH_POSE = (
    -0.0184,
    0.0307,
    1.4082,
    1.5248,
    -0.0675,
    1.0937,
    1.2149,
    -0.0184,
    -0.0368,
    0.0184,
    -0.0169,
    -1.4757,
    -1.5907,
    0.0568,
)


@dataclass
class MicroDuckRollerCrouchCommands(MicroDuckRollerCommands):
    period: float = 5.0
    entry_velocity_x: tuple[float, float] = (0.2, 0.5)


@dataclass
class MicroDuckRollerCrouchRewardConfig(MicroDuckRollerRewardConfig):
    stand_pose: list[float] = field(default_factory=lambda: list(OFFICIAL_STAND_POSE))
    crouch_pose: list[float] = field(default_factory=lambda: list(OFFICIAL_CROUCH_POSE))
    crouch_pose_std: float = 0.4
    descent_end: float = 0.10
    hold_end: float = 0.50
    rise_end: float = 0.60
    forward_speed_ref: float = 0.2
    crouch_lean_target: float = 0.08
    crouch_lean_std: float = 0.1


@registry.envcfg("MicroDuckRollerCrouchFlat")
@dataclass
class MicroDuckRollerCrouchFlatCfg(MicroDuckRollerFlatCfg):
    commands: MicroDuckRollerCrouchCommands = field(default_factory=MicroDuckRollerCrouchCommands)
    reward_config: MicroDuckRollerCrouchRewardConfig | None = None
    max_episode_seconds: float = 5.0


class MicroDuckRollerCrouchDomainRandomizationProvider(MicroDuckWalkDomainRandomizationProvider):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = np.zeros((num_reset, 3), dtype=get_global_dtype())
        commands[:, 0] = 1.0
        return commands

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        plan.qpos[:, 3:7] = env._init_qpos[3:7]
        low, high = env.cfg.commands.entry_velocity_x
        plan.qvel[:, 0] = np.random.uniform(low, high, size=(len(env_ids),))
        return plan


@registry.env("MicroDuckRollerCrouchFlat", sim_backend="mujoco")
class MicroDuckRollerCrouchFlatEnv(MicroDuckRollerFlatEnv):
    """One 5 s stand→crouch→stand cycle while preserving roller momentum."""

    _cfg: MicroDuckRollerCrouchFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckWalkDomainRandomizationProvider:
        return MicroDuckRollerCrouchDomainRandomizationProvider()

    @property
    def _crouch_reward_cfg(self) -> MicroDuckRollerCrouchRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckRollerCrouchRewardConfig):
            raise TypeError("MicroDuckRollerCrouchFlat requires MicroDuckRollerCrouchRewardConfig")
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "crouch_glide_pose": self._crouch_glide_pose,
            "crouch_glide_pose_l1": self._crouch_glide_pose_l1,
            "forward_speed": self._forward_speed,
            "crouch_forward_lean": self._crouch_forward_lean,
            "upright": self._upright_tracking,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
        }

    def _phase(self, info: dict[str, Any]) -> np.ndarray:
        steps = np.asarray(info["steps"], dtype=get_global_dtype())
        return np.mod(steps * self._cfg.ctrl_dt / self._cfg.commands.period, 1.0)

    def _update_phase_command(self, info: dict[str, Any]) -> None:
        phase = self._phase(info)
        angle = 2.0 * np.pi * phase
        info["commands"][:, 0] = np.cos(angle)
        info["commands"][:, 1] = np.sin(angle)
        info["commands"][:, 2] = 0.0

    def _crouch_blend(self, info: dict[str, Any]) -> np.ndarray:
        phase = self._phase(info)
        cfg = self._crouch_reward_cfg
        blend = np.zeros_like(phase)
        descend = phase < cfg.descent_end
        blend[descend] = phase[descend] / cfg.descent_end
        low = (phase >= cfg.descent_end) & (phase < cfg.hold_end)
        blend[low] = 1.0
        rise = (phase >= cfg.hold_end) & (phase < cfg.rise_end)
        blend[rise] = 1.0 - (phase[rise] - cfg.hold_end) / (cfg.rise_end - cfg.hold_end)
        return blend

    def _pose_error(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._crouch_reward_cfg
        stand = np.asarray(cfg.stand_pose, dtype=get_global_dtype())
        crouch = np.asarray(cfg.crouch_pose, dtype=get_global_dtype())
        expected_shape = (len(MICRODUCK_JOINT_NAMES),)
        if stand.shape != expected_shape or crouch.shape != expected_shape:
            raise ValueError("stand_pose and crouch_pose must contain 14 joint values")
        blend = self._crouch_blend(ctx.info)[:, None]
        target = stand[None, :] + blend * (crouch - stand)[None, :]
        return np.asarray(ctx.dof_pos - target, dtype=get_global_dtype())

    def _crouch_glide_pose(self, ctx: RewardContext) -> np.ndarray:
        error = self._pose_error(ctx) / self._crouch_reward_cfg.crouch_pose_std
        return np.asarray(np.exp(-np.mean(np.square(error), axis=1)), dtype=get_global_dtype())

    def _crouch_glide_pose_l1(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(-np.mean(np.abs(self._pose_error(ctx)), axis=1), dtype=get_global_dtype())

    def _forward_speed(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            np.tanh(
                np.clip(ctx.linvel[:, 0], 0.0, None) / self._crouch_reward_cfg.forward_speed_ref
            ),
            dtype=get_global_dtype(),
        )

    def _crouch_forward_lean(self, ctx: RewardContext) -> np.ndarray:
        gravity = ctx.gravity
        assert gravity is not None
        cfg = self._crouch_reward_cfg
        gate = self._crouch_blend(ctx.info)
        return np.asarray(
            gate
            * np.exp(-np.square((gravity[:, 0] - cfg.crouch_lean_target) / cfg.crouch_lean_std)),
            dtype=get_global_dtype(),
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_phase_command(state.info)
        return super().update_state(state)
