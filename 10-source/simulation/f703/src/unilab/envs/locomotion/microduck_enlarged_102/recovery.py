"""Dedicated ground-recovery task for enlarged Micro Duck V1.0.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_102.tasks import (
    MicroDuckEnlarged102StandFlatCfg,
    MicroDuckEnlarged102StandFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_mod1.tasks import (
    MicroDuckEnlargedMod1StandRewardConfig,
)

_RESET_PRONE_FACE_DOWN = 1
_RESET_PRONE_FACE_UP = 2


def _upright_reward_gate(
    gravity: np.ndarray, activation_tilt_deg: float = 40.0
) -> np.ndarray:
    """Fade pose rewards in only after the trunk is mostly upright."""
    cos_tilt = np.clip(-np.asarray(gravity)[:, 2], -1.0, 1.0)
    activation = float(np.cos(np.deg2rad(activation_tilt_deg)))
    return np.asarray(
        np.clip((cos_tilt - activation) / (1.0 - activation), 0.0, 1.0),
        dtype=get_global_dtype(),
    )


def _horizontal_velocity_penalty(linvel: np.ndarray) -> np.ndarray:
    """Return squared local horizontal speed in ``(m/s)^2``."""
    velocity = np.asarray(linvel)
    return np.asarray(
        np.sum(np.square(velocity[:, :2]), axis=1), dtype=get_global_dtype()
    )


@dataclass
class MicroDuckEnlarged102RecoveryRewardConfig(
    MicroDuckEnlargedMod1StandRewardConfig
):
    """V1.0.2 stand rewards plus recovery progress and completion signals."""

    height_progress_ceiling: float = 0.235
    recovery_success_min_fallen_s: float = 0.5
    recovery_fallen_tilt_deg: float = 40.0
    recovered_tilt_deg: float = 25.0
    recovered_height: float = 0.20
    com_upward_max_height: float = 0.235
    com_upward_gate_tilt_deg: float = 40.0


@registry.envcfg("MicroDuckEnlarged102StandRecoveryFlat")
@dataclass
class MicroDuckEnlarged102StandRecoveryFlatCfg(MicroDuckEnlarged102StandFlatCfg):
    """Zero-command prone recovery without changing the 61D/14D contract."""

    reward_config: MicroDuckEnlarged102RecoveryRewardConfig | None = None
    fallen_height_threshold: float = 0.15
    fallen_tilt_deg: float = 40.0
    fallen_timeout_s: float = 10.0
    max_episode_seconds: float = 10.0
    reset_curriculum: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"step": 0, "prone_prob": 1.0, "face_down_prob": 0.5}
        ]
    )
    prone_z_min: float = 0.08
    prone_z_max: float = 0.13


class MicroDuckEnlarged102RecoveryDomainRandomizationProvider(
    MicroDuckEnlargedWalkDomainRandomizationProvider
):
    """V1.0.2 reset provider that samples both ground orientations."""

    @staticmethod
    def _reset_curriculum_params(env: Any) -> tuple[float, float]:
        selected: dict[str, Any] | None = None
        previous_step = -1
        for stage in env.cfg.reset_curriculum:
            step = int(stage["step"])
            if step < previous_step:
                raise ValueError("recovery reset_curriculum steps must be ordered")
            previous_step = step
            if env.step_counter >= step:
                selected = stage
        if selected is None:
            return 0.0, 0.5
        prone_prob = float(selected["prone_prob"])
        face_down_prob = float(selected["face_down_prob"])
        if not 0.0 <= prone_prob <= 1.0 or not 0.0 <= face_down_prob <= 1.0:
            raise ValueError("recovery reset probabilities must be within [0, 1]")
        return prone_prob, face_down_prob

    @staticmethod
    def _set_prone_resets(
        env: Any,
        qpos: np.ndarray,
        qvel: np.ndarray,
        indices: np.ndarray,
        face_down_prob: float,
        reset_mode: np.ndarray,
    ) -> None:
        if indices.size == 0:
            return
        num = indices.size
        yaw = np.random.uniform(-np.pi, np.pi, size=num)
        cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        side = 2.0**-0.5
        face_down = np.stack(
            (side * cy, -side * sy, side * cy, side * sy), axis=1
        )
        face_up = np.stack(
            (side * cy, side * sy, -side * cy, side * sy), axis=1
        )
        is_face_down = np.random.uniform(size=num) < face_down_prob
        qpos[indices, 2] = np.random.uniform(
            env.cfg.prone_z_min, env.cfg.prone_z_max, size=num
        )
        qpos[indices, 3:7] = np.where(is_face_down[:, None], face_down, face_up)
        qvel[indices] = 0.0
        reset_mode[indices] = np.where(
            is_face_down, _RESET_PRONE_FACE_DOWN, _RESET_PRONE_FACE_UP
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        prone_prob, face_down_prob = self._reset_curriculum_params(env)
        reset_mode = np.zeros(len(env_ids), dtype=np.int8)
        prone_indices = np.flatnonzero(
            np.random.uniform(size=len(env_ids)) < prone_prob
        )
        self._set_prone_resets(
            env,
            plan.qpos,
            plan.qvel,
            prone_indices,
            face_down_prob,
            reset_mode,
        )
        plan.info_updates["recovery_reset_mode"] = reset_mode
        return plan

    def _build_extra_info_updates(
        self, env: Any, num_reset: int
    ) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        dtype = get_global_dtype()
        updates.update(
            {
                "fallen_time_s": np.zeros(num_reset, dtype=dtype),
                "upright_potential_prev": np.zeros(num_reset, dtype=dtype),
                "height_potential_prev": np.zeros(num_reset, dtype=dtype),
                "recovery_fallen_s": np.zeros(num_reset, dtype=dtype),
                "recovery_armed": np.zeros(num_reset, dtype=bool),
                "fallen_tax_armed": np.zeros(num_reset, dtype=bool),
            }
        )
        return updates

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del linvel, gravity
        projected_gravity = env._projected_gravity()[env_ids]
        height = np.asarray(
            env._backend.get_base_pos()[env_ids, 2], dtype=get_global_dtype()
        )
        info_updates["upright_potential_prev"] = np.asarray(
            -projected_gravity[:, 2], dtype=get_global_dtype()
        )
        info_updates["height_potential_prev"] = np.minimum(
            height, env._reward_cfg.height_progress_ceiling
        )
        return env._compute_obs(
            info_updates, gyro, projected_gravity, dof_pos, dof_vel
        )


@registry.env("MicroDuckEnlarged102StandRecoveryFlat", sim_backend="mujoco")
class MicroDuckEnlarged102StandRecoveryFlatEnv(
    MicroDuckEnlarged102StandFlatEnv
):
    """Prone/face-up recovery owner for the exact V1.0.2 model."""

    _cfg: MicroDuckEnlarged102StandRecoveryFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged102RecoveryDomainRandomizationProvider:
        return MicroDuckEnlarged102RecoveryDomainRandomizationProvider()

    @property
    def _recovery_reward_cfg(self) -> MicroDuckEnlarged102RecoveryRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlarged102RecoveryRewardConfig):
            raise TypeError(
                "MicroDuckEnlarged102StandRecoveryFlat requires its recovery reward config"
            )
        return cfg

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "upright_progress": self._upright_progress,
                "height_progress": self._height_progress,
                "recovery_success": self._recovery_success,
                "com_upward_velocity": self._com_upward_velocity,
                "fallen_tax": self._fallen_tax,
                "horizontal_velocity": self._horizontal_velocity,
                "joint_target_limit": self._joint_target_limit,
            }
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Clip normalized targets to the exact V1.0.2 joint envelope.

        The PPO actor is Gaussian and therefore may emit offsets outside the
        physical joint range.  Keep the action history in the same effective
        units that reach the position actuators, while exposing the clipped
        amount to the reward so training learns not to rely on saturation.
        """
        action = np.asarray(actions, dtype=get_global_dtype())
        if action.shape != (self._num_envs, self._num_action):
            raise ValueError(
                f"recovery action must have shape {(self._num_envs, self._num_action)}, "
                f"got {action.shape}"
            )
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            raise RuntimeError("recovery action clipping requires compiled joint limits")
        scale = float(self._cfg.control_config.action_scale)
        target = self.default_angles[None, :] + action * scale
        clipped_target = np.clip(target, joint_range[:, 0], joint_range[:, 1])
        excess = np.sum(np.abs(target - clipped_target), axis=1)
        state.info["joint_target_limit_excess"] = np.asarray(
            excess, dtype=get_global_dtype()
        )
        effective_action = (clipped_target - self.default_angles[None, :]) / scale
        return super().apply_action(effective_action, state)

    def _horizontal_velocity(self, ctx: RewardContext) -> np.ndarray:
        return _horizontal_velocity_penalty(ctx.linvel)

    def _joint_target_limit(self, ctx: RewardContext) -> np.ndarray:
        """Penalty signal for a policy target clipped by the MJCF limits."""
        return np.asarray(
            ctx.info.get(
                "joint_target_limit_excess",
                np.zeros(ctx.num_envs, dtype=get_global_dtype()),
            ),
            dtype=get_global_dtype(),
        )

    def _upright_progress(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        current = np.asarray(-ctx.gravity[:, 2], dtype=get_global_dtype())
        previous = np.asarray(
            ctx.info.get("upright_potential_prev", current), dtype=get_global_dtype()
        )
        ctx.info["upright_potential_prev"] = current.copy()
        return np.asarray(current - previous, dtype=get_global_dtype())

    def _body_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        """Prevent a sideways trunk from farming partial pose reward forever."""
        assert ctx.gravity is not None
        base_reward = super()._body_pose_tracking(ctx)
        return np.asarray(
            base_reward * _upright_reward_gate(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _height_progress(self, ctx: RewardContext) -> np.ndarray:
        current = np.minimum(
            np.asarray(ctx.base_height, dtype=get_global_dtype()),
            self._recovery_reward_cfg.height_progress_ceiling,
        )
        previous = np.asarray(
            ctx.info.get("height_potential_prev", current), dtype=get_global_dtype()
        )
        ctx.info["height_potential_prev"] = current.copy()
        return np.asarray(current - previous, dtype=get_global_dtype())

    def _recovery_success(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        cfg = self._recovery_reward_cfg
        cos_tilt = np.clip(-ctx.gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(cfg.recovery_fallen_tilt_deg))
        zeros = np.zeros(ctx.num_envs, dtype=get_global_dtype())
        fallen_s = np.asarray(
            ctx.info.get("recovery_fallen_s", zeros), dtype=get_global_dtype()
        )
        fallen_s = np.where(fallen, fallen_s + self._cfg.ctrl_dt, 0.0)
        armed = np.asarray(
            ctx.info.get("recovery_armed", np.zeros(ctx.num_envs, dtype=bool)),
            dtype=bool,
        )
        armed |= fallen_s >= cfg.recovery_success_min_fallen_s
        recovered = (cos_tilt > np.cos(np.deg2rad(cfg.recovered_tilt_deg))) & (
            ctx.base_height > cfg.recovered_height
        )
        success = armed & recovered
        ctx.info["recovery_fallen_s"] = np.asarray(
            fallen_s, dtype=get_global_dtype()
        )
        ctx.info["recovery_armed"] = armed & ~success
        return np.asarray(success, dtype=get_global_dtype())

    def _com_upward_velocity(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        cfg = self._recovery_reward_cfg
        cos_tilt = np.clip(-ctx.gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(cfg.com_upward_gate_tilt_deg))
        below = ctx.base_height < cfg.com_upward_max_height
        world_linvel = np.asarray(
            ctx.info["base_world_linvel"], dtype=get_global_dtype()
        )
        return np.asarray(
            np.maximum(world_linvel[:, 2], 0.0) * fallen * below,
            dtype=get_global_dtype(),
        )

    def _fallen_tax(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        cfg = self._recovery_reward_cfg
        cos_tilt = np.clip(-ctx.gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(cfg.recovery_fallen_tilt_deg))
        recovered = (cos_tilt > np.cos(np.deg2rad(cfg.recovered_tilt_deg))) & (
            ctx.base_height > cfg.recovered_height
        )
        armed = np.asarray(
            ctx.info.get("fallen_tax_armed", np.zeros(ctx.num_envs, dtype=bool)),
            dtype=bool,
        )
        armed = (armed | fallen) & ~recovered
        ctx.info["fallen_tax_armed"] = armed
        return np.asarray(armed, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        gravity = self._projected_gravity()
        base_height = np.asarray(
            self._backend.get_base_pos()[:, 2], dtype=get_global_dtype()
        )
        state.info["base_world_linvel"] = np.asarray(
            self._backend.get_base_lin_vel(), dtype=get_global_dtype()
        )
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        fallen = (cos_tilt < np.cos(np.deg2rad(self._cfg.fallen_tilt_deg))) | (
            base_height < self._cfg.fallen_height_threshold
        )
        previous = np.asarray(
            state.info.get(
                "fallen_time_s",
                np.zeros(self._num_envs, dtype=get_global_dtype()),
            ),
            dtype=get_global_dtype(),
        )
        fallen_time = np.where(fallen, previous + self._cfg.ctrl_dt, 0.0)
        state.info["fallen"] = fallen
        state.info["fallen_time_s"] = np.asarray(
            fallen_time, dtype=get_global_dtype()
        )
        updated = super().update_state(state)
        return updated.replace(terminated=fallen_time >= self._cfg.fallen_timeout_s)
