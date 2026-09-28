"""Phase-driven roller crouch-and-glide task for DUCK_V1.01."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.roller_crouch import OFFICIAL_CROUCH_POSE
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MICRODUCK_ENLARGED_JOINT_NAMES,
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_mod1.roller import (
    MicroDuckEnlargedMod1RollerCommands,
    MicroDuckEnlargedMod1RollerFlatCfg,
    MicroDuckEnlargedMod1RollerFlatEnv,
    MicroDuckEnlargedMod1RollerRewardConfig,
)

MOD1_ROLLER_STAND_POSE = (
    0.0,
    -0.08726646259971647,
    -0.457924,
    -0.004940,
    0.452984,
    0.3490658503988659,
    0.3490658503988659,
    0.0,
    0.0,
    0.0,
    0.08726646259971647,
    0.457924,
    0.004940,
    -0.452984,
)
MOD1_CROUCH_DEPTH = 0.4
MOD1_ROLLER_CROUCH_POSE = tuple(
    stand + MOD1_CROUCH_DEPTH * (crouch - stand)
    for stand, crouch in zip(MOD1_ROLLER_STAND_POSE, OFFICIAL_CROUCH_POSE, strict=True)
)


@dataclass
class MicroDuckEnlargedMod1RollerCrouchCommands(
    MicroDuckEnlargedMod1RollerCommands
):
    period: float = 5.0
    entry_velocity_x: tuple[float, float] = (0.2, 0.5)


@dataclass
class MicroDuckEnlargedMod1RollerCrouchRewardConfig(
    MicroDuckEnlargedMod1RollerRewardConfig
):
    stand_pose: list[float] = field(default_factory=lambda: list(MOD1_ROLLER_STAND_POSE))
    crouch_pose: list[float] = field(
        default_factory=lambda: list(MOD1_ROLLER_CROUCH_POSE)
    )
    crouch_pose_std: float = 0.4
    descent_end: float = 0.20
    hold_end: float = 0.50
    rise_end: float = 0.70
    forward_speed_ref: float = 0.2
    crouch_lean_target: float = 0.08
    crouch_lean_std: float = 0.1


@registry.envcfg("MicroDuckEnlargedMod1RollerCrouchFlat")
@dataclass
class MicroDuckEnlargedMod1RollerCrouchFlatCfg(
    MicroDuckEnlargedMod1RollerFlatCfg
):
    commands: MicroDuckEnlargedMod1RollerCrouchCommands = field(
        default_factory=MicroDuckEnlargedMod1RollerCrouchCommands
    )
    reward_config: MicroDuckEnlargedMod1RollerCrouchRewardConfig | None = None
    max_episode_seconds: float = 5.0
    # The 0.25 V1.01 action scale needs about +/-7.8 to reach the official
    # crouch pose.  A finite envelope prevents exploratory Gaussian actions
    # from overflowing the squared action-rate term without clipping the pose.
    action_clip: float = 5.0


class MicroDuckEnlargedMod1RollerCrouchDomainRandomizationProvider(
    MicroDuckEnlargedWalkDomainRandomizationProvider
):
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


@registry.env("MicroDuckEnlargedMod1RollerCrouchFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1RollerCrouchFlatEnv(
    MicroDuckEnlargedMod1RollerFlatEnv
):
    """One stand-to-crouch-to-stand cycle while preserving roller momentum."""

    _cfg: MicroDuckEnlargedMod1RollerCrouchFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlargedMod1RollerCrouchDomainRandomizationProvider:
        return MicroDuckEnlargedMod1RollerCrouchDomainRandomizationProvider()

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clip = float(self._cfg.action_clip)
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError("action_clip must be finite and positive")
        finite = np.clip(
            np.nan_to_num(actions, nan=0.0, posinf=clip, neginf=-clip),
            -2.0 * clip,
            2.0 * clip,
        )
        excess = np.maximum(np.abs(finite) - clip, 0.0)
        state.info["action_excess"] = np.mean(np.square(excess), axis=1)
        bounded = np.clip(
            finite,
            -clip,
            clip,
        )
        return super().apply_action(bounded, state)

    @property
    def _crouch_reward_cfg(self) -> MicroDuckEnlargedMod1RollerCrouchRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlargedMod1RollerCrouchRewardConfig):
            raise TypeError(
                "MicroDuckEnlargedMod1RollerCrouchFlat requires its crouch reward config"
            )
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "crouch_glide_pose": self._crouch_glide_pose,
            "crouch_glide_pose_l1": self._crouch_glide_pose_l1,
            "forward_speed": self._forward_speed,
            "crouch_forward_lean": self._crouch_forward_lean,
            "leg_symmetry": self._leg_symmetry,
            "grounded": self._crouch_grounded,
            "upright": self._upright_tracking,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "action_excess": self._action_excess,
        }

    def _crouch_grounded(self, ctx: RewardContext) -> np.ndarray:
        """Keep both roller blades down throughout the complete crouch cycle."""
        contact_count = np.sum(np.asarray(ctx.info["foot_contact"], dtype=bool), axis=1)
        return np.asarray(contact_count >= 2, dtype=get_global_dtype())

    def _action_excess(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["action_excess"], dtype=get_global_dtype())

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
        blend[rise] = 1.0 - (phase[rise] - cfg.hold_end) / (
            cfg.rise_end - cfg.hold_end
        )
        return blend

    def _pose_error(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._crouch_reward_cfg
        stand = np.asarray(cfg.stand_pose, dtype=get_global_dtype())
        crouch = np.asarray(cfg.crouch_pose, dtype=get_global_dtype())
        expected_shape = (len(MICRODUCK_ENLARGED_JOINT_NAMES),)
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
                np.clip(ctx.linvel[:, 0], 0.0, None)
                / self._crouch_reward_cfg.forward_speed_ref
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
            * np.exp(
                -np.square((gravity[:, 0] - cfg.crouch_lean_target) / cfg.crouch_lean_std)
            ),
            dtype=get_global_dtype(),
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_phase_command(state.info)
        return super().update_state(state)
