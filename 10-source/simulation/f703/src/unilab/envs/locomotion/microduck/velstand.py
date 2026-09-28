"""Original Micro Duck walking plus fall-recovery training on flat ground.

This is the first UniLab port slice of the official ``Mjlab-VelStand`` task.
It preserves ``MicroDuckWalkFlat`` verbatim, swaps in the original robot's
full-collision MJCF, replaces permanent fall termination with an 8 s recovery
window after the walking bootstrap, adds the official crouch/prone reset
curriculum, and ports the unfarmable orientation and height potential rewards.

The later recovery-economics rewards remain deliberately deferred until each
potential term has passed isolated behavior validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.walk import (
    MICRODUCK_JOINT_NAMES,
    MicroDuckRewardConfig,
    MicroDuckWalkDomainRandomizationProvider,
    MicroDuckWalkFlatCfg,
    MicroDuckWalkFlatEnv,
)

_RESET_UPRIGHT = 0
_RESET_PRONE_FACE_DOWN = 1
_RESET_PRONE_FACE_UP = 2
_RESET_CROUCH = 3

_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


@dataclass
class MicroDuckVelStandRewardConfig(MicroDuckRewardConfig):
    """Walking reward contract plus official recovery potential terms."""

    height_progress_ceiling: float = 0.115
    recovery_success_enable_after_steps: int = 1200 * 24
    recovery_success_min_fallen_s: float = 0.5
    recovery_fallen_tilt_deg: float = 40.0
    recovered_tilt_deg: float = 25.0
    recovered_height: float = 0.09
    com_upward_enable_after_steps: int = 1200 * 24
    com_upward_max_height: float = 0.125
    com_upward_gate_tilt_deg: float = 40.0
    fallen_tax_enable_after_steps: int = 1200 * 24


@registry.envcfg("MicroDuckVelStandFlat")
@dataclass
class MicroDuckVelStandFlatCfg(MicroDuckWalkFlatCfg):
    """Flat VelStand task for the original, unscaled Micro Duck."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck" / "scene_velstand_flat.xml")
        )
    )
    reward_config: MicroDuckVelStandRewardConfig | None = None
    # Official curriculum: 500 PPO iterations * 24 environment steps.
    fell_over_disable_after_steps: int = 12_000
    fallen_height_threshold: float = 0.08
    fallen_tilt_deg: float = 40.0
    fallen_timeout_s: float = 8.0
    # Official run-7 reset curriculum. Steps are vector-environment steps,
    # matching 24 steps per PPO iteration rather than num_envs * 24 samples.
    reset_curriculum: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "step": 0,
                "prone_prob": 0.0,
                "face_down_prob": 1.0,
                "crouch_prob": 0.0,
            },
            {
                "step": 800 * 24,
                "prone_prob": 0.0,
                "face_down_prob": 1.0,
                "crouch_prob": 0.15,
            },
            {
                "step": 1500 * 24,
                "prone_prob": 0.15,
                "face_down_prob": 0.80,
                "crouch_prob": 0.15,
            },
            {
                "step": 2000 * 24,
                "prone_prob": 0.30,
                "face_down_prob": 0.65,
                "crouch_prob": 0.15,
            },
            {
                "step": 2500 * 24,
                "prone_prob": 0.45,
                "face_down_prob": 0.50,
                "crouch_prob": 0.15,
            },
        ]
    )
    prone_z_min: float = 0.05
    prone_z_max: float = 0.09
    crouch_depth_min: float = 0.35
    crouch_depth_max: float = 1.0
    crouch_pitch_max_deg: float = 55.0
    crouch_roll_noise_deg: float = 8.0
    crouch_pitch_noise_deg: float = 10.0
    crouch_joint_noise: float = 0.12
    crouch_z_stand: float = 0.115
    crouch_z_deep: float = 0.06


class MicroDuckVelStandDomainRandomizationProvider(MicroDuckWalkDomainRandomizationProvider):
    """Walking resets plus state for recovery potential and timeout."""

    @staticmethod
    def _reset_curriculum_params(env: Any) -> tuple[float, float, float]:
        stages = env.cfg.reset_curriculum
        if not stages:
            return 0.0, 1.0, 0.0
        previous_step = -1
        selected: dict[str, Any] | None = None
        for stage in stages:
            step = int(stage["step"])
            if step < previous_step:
                raise ValueError("VelStand reset_curriculum steps must be ordered")
            previous_step = step
            if env.step_counter >= step:
                selected = stage
        if selected is None:
            return 0.0, 1.0, 0.0
        prone_prob = float(selected["prone_prob"])
        face_down_prob = float(selected["face_down_prob"])
        crouch_prob = float(selected["crouch_prob"])
        if not 0.0 <= face_down_prob <= 1.0:
            raise ValueError("VelStand face_down_prob must be within [0, 1]")
        if prone_prob < 0.0 or crouch_prob < 0.0 or prone_prob + crouch_prob > 1.0:
            raise ValueError("VelStand prone_prob + crouch_prob must be within [0, 1]")
        return prone_prob, face_down_prob, crouch_prob

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
        cy = np.cos(0.5 * yaw)
        sy = np.sin(0.5 * yaw)
        s = 2.0**-0.5
        face_down = np.stack((s * cy, -s * sy, s * cy, s * sy), axis=1)
        face_up = np.stack((s * cy, s * sy, -s * cy, s * sy), axis=1)
        is_face_down = np.random.uniform(size=num) < face_down_prob
        qpos[indices, 2] = np.random.uniform(
            env.cfg.prone_z_min,
            env.cfg.prone_z_max,
            size=num,
        )
        qpos[indices, 3:7] = np.where(is_face_down[:, None], face_down, face_up)
        qvel[indices, :6] = 0.0
        reset_mode[indices] = np.where(
            is_face_down,
            _RESET_PRONE_FACE_DOWN,
            _RESET_PRONE_FACE_UP,
        )

    @staticmethod
    def _set_crouch_resets(
        env: Any,
        qpos: np.ndarray,
        qvel: np.ndarray,
        indices: np.ndarray,
        reset_mode: np.ndarray,
    ) -> None:
        if indices.size == 0:
            return
        num = indices.size
        depth = np.random.uniform(
            env.cfg.crouch_depth_min,
            env.cfg.crouch_depth_max,
            size=num,
        )
        joint_qpos = qpos[indices, 7:].copy()
        for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
            joint_id = MICRODUCK_JOINT_NAMES.index(name)
            joint_qpos[:, joint_id] += depth * (anchor - joint_qpos[:, joint_id])
        joint_qpos += np.random.uniform(
            -env.cfg.crouch_joint_noise,
            env.cfg.crouch_joint_noise,
            size=joint_qpos.shape,
        )

        pitch = depth * np.deg2rad(env.cfg.crouch_pitch_max_deg)
        pitch += np.random.uniform(
            -np.deg2rad(env.cfg.crouch_pitch_noise_deg),
            np.deg2rad(env.cfg.crouch_pitch_noise_deg),
            size=num,
        )
        pitch = np.maximum(pitch, np.deg2rad(5.0))
        roll = np.random.uniform(
            -np.deg2rad(env.cfg.crouch_roll_noise_deg),
            np.deg2rad(env.cfg.crouch_roll_noise_deg),
            size=num,
        )
        yaw = np.random.uniform(-np.pi, np.pi, size=num)
        cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
        cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
        qpos[indices, 2] = (
            env.cfg.crouch_z_stand
            + depth * (env.cfg.crouch_z_deep - env.cfg.crouch_z_stand)
            + np.random.uniform(0.0, 0.01, size=num)
        )
        qpos[indices, 3:7] = np.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            axis=1,
        )
        qpos[indices, 7:] = joint_qpos
        qvel[indices] = 0.0
        reset_mode[indices] = _RESET_CROUCH

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        prone_prob, face_down_prob, crouch_prob = self._reset_curriculum_params(env)
        reset_mode = np.full(len(env_ids), _RESET_UPRIGHT, dtype=np.int8)
        selection = np.random.uniform(size=len(env_ids))
        prone_indices = np.flatnonzero(selection < prone_prob)
        crouch_indices = np.flatnonzero(
            (selection >= prone_prob) & (selection < prone_prob + crouch_prob)
        )
        self._set_prone_resets(
            env,
            plan.qpos,
            plan.qvel,
            prone_indices,
            face_down_prob,
            reset_mode,
        )
        self._set_crouch_resets(
            env,
            plan.qpos,
            plan.qvel,
            crouch_indices,
            reset_mode,
        )
        plan.info_updates["velstand_reset_mode"] = reset_mode
        return plan

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        updates.update(
            {
                "fallen_time_s": np.zeros(num_reset, dtype=get_global_dtype()),
                "upright_potential_prev": np.zeros(num_reset, dtype=get_global_dtype()),
                "height_potential_prev": np.zeros(num_reset, dtype=get_global_dtype()),
                "recovery_fallen_s": np.zeros(num_reset, dtype=get_global_dtype()),
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
        del gravity
        projected_gravity = env._projected_gravity()[env_ids]
        # projected gravity is [0, 0, -1] when upright, so -g_z = cos(tilt).
        info_updates["upright_potential_prev"] = np.asarray(
            -projected_gravity[:, 2], dtype=get_global_dtype()
        )
        base_height = np.asarray(
            env._backend.get_base_pos()[env_ids, 2],
            dtype=get_global_dtype(),
        )
        info_updates["base_height"] = base_height
        info_updates["height_potential_prev"] = np.minimum(
            base_height,
            env._reward_cfg.height_progress_ceiling,
        )
        return env._compute_obs(
            info_updates,
            gyro,
            projected_gravity,
            dof_pos,
            dof_vel,
        )


@registry.env("MicroDuckVelStandFlat", sim_backend="mujoco")
class MicroDuckVelStandFlatEnv(MicroDuckWalkFlatEnv):
    """Velocity tracking that keeps fallen episodes alive for recovery."""

    _cfg: MicroDuckVelStandFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckVelStandDomainRandomizationProvider:
        return MicroDuckVelStandDomainRandomizationProvider()

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["upright_progress"] = self._upright_progress
        self._reward_fns["height_progress"] = self._height_progress
        self._reward_fns["recovery_success"] = self._recovery_success
        self._reward_fns["com_upward_velocity"] = self._com_upward_velocity
        self._reward_fns["fallen_tax"] = self._fallen_tax

    def _feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        """Keep the walking reward, but make it unavailable while fallen."""
        reward = super()._feet_air_time(ctx)
        gravity = ctx.gravity
        assert gravity is not None
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        upright = cos_tilt >= np.cos(np.deg2rad(self._reward_cfg.recovery_fallen_tilt_deg))
        return np.asarray(reward * upright, dtype=get_global_dtype())

    def _upright_progress(self, ctx: RewardContext) -> np.ndarray:
        """Official potential shaping: current cos(tilt) minus previous."""
        gravity = ctx.gravity
        assert gravity is not None
        current = np.asarray(-gravity[:, 2], dtype=get_global_dtype())
        previous = np.asarray(
            ctx.info.get("upright_potential_prev", current),
            dtype=get_global_dtype(),
        )
        delta = current - previous
        ctx.info["upright_potential_prev"] = current.copy()
        return np.asarray(delta, dtype=get_global_dtype())

    def _height_progress(self, ctx: RewardContext) -> np.ndarray:
        """Official capped height potential: delta min(trunk z, ceiling)."""
        height = np.asarray(ctx.info["base_height"], dtype=get_global_dtype())
        current = np.minimum(height, self._reward_cfg.height_progress_ceiling)
        previous = np.asarray(
            ctx.info.get("height_potential_prev", current),
            dtype=get_global_dtype(),
        )
        delta = current - previous
        ctx.info["height_potential_prev"] = current.copy()
        return np.asarray(delta, dtype=get_global_dtype())

    def _recovery_success(self, ctx: RewardContext) -> np.ndarray:
        """One-shot reward when a genuine fall is followed by a complete stand."""
        zeros = np.zeros(ctx.num_envs, dtype=get_global_dtype())
        if self.step_counter < self._reward_cfg.recovery_success_enable_after_steps:
            ctx.info["recovery_fallen_s"] = zeros
            ctx.info["recovery_armed"] = np.zeros(ctx.num_envs, dtype=bool)
            return zeros

        gravity = ctx.gravity
        assert gravity is not None
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(self._reward_cfg.recovery_fallen_tilt_deg))
        fallen_s = np.asarray(
            ctx.info.get("recovery_fallen_s", zeros),
            dtype=get_global_dtype(),
        )
        fallen_s = np.where(fallen, fallen_s + self._cfg.ctrl_dt, 0.0)
        armed = np.asarray(
            ctx.info.get("recovery_armed", np.zeros(ctx.num_envs, dtype=bool)),
            dtype=bool,
        )
        armed = np.logical_or(
            armed,
            fallen_s >= self._reward_cfg.recovery_success_min_fallen_s,
        )
        height = np.asarray(ctx.info["base_height"], dtype=get_global_dtype())
        recovered = np.logical_and(
            cos_tilt > np.cos(np.deg2rad(self._reward_cfg.recovered_tilt_deg)),
            height > self._reward_cfg.recovered_height,
        )
        success = np.logical_and(armed, recovered)
        ctx.info["recovery_fallen_s"] = np.asarray(fallen_s, dtype=get_global_dtype())
        ctx.info["recovery_armed"] = np.logical_and(armed, np.logical_not(success))
        return np.asarray(success, dtype=get_global_dtype())

    def _com_upward_velocity(self, ctx: RewardContext) -> np.ndarray:
        """Positive world-frame trunk z velocity while genuinely fallen."""
        if self.step_counter < self._reward_cfg.com_upward_enable_after_steps:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        gravity = ctx.gravity
        assert gravity is not None
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(self._reward_cfg.com_upward_gate_tilt_deg))
        height = np.asarray(ctx.info["base_height"], dtype=get_global_dtype())
        below_ceiling = height < self._reward_cfg.com_upward_max_height
        world_linvel = np.asarray(ctx.info["base_world_linvel"], dtype=get_global_dtype())
        upward = np.maximum(world_linvel[:, 2], 0.0)
        return np.asarray(upward * fallen * below_ceiling, dtype=get_global_dtype())

    def _fallen_tax(self, ctx: RewardContext) -> np.ndarray:
        """Hysteretic state cost: arm when fallen, release only fully upright."""
        if self.step_counter < self._reward_cfg.fallen_tax_enable_after_steps:
            ctx.info["fallen_tax_armed"] = np.zeros(ctx.num_envs, dtype=bool)
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        gravity = ctx.gravity
        assert gravity is not None
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        fallen = cos_tilt < np.cos(np.deg2rad(self._reward_cfg.recovery_fallen_tilt_deg))
        height = np.asarray(ctx.info["base_height"], dtype=get_global_dtype())
        recovered = np.logical_and(
            cos_tilt > np.cos(np.deg2rad(self._reward_cfg.recovered_tilt_deg)),
            height > self._reward_cfg.recovered_height,
        )
        armed = np.asarray(
            ctx.info.get("fallen_tax_armed", np.zeros(ctx.num_envs, dtype=bool)),
            dtype=bool,
        )
        armed = np.logical_or(armed, fallen)
        armed = np.logical_and(armed, np.logical_not(recovered))
        ctx.info["fallen_tax_armed"] = armed
        return np.asarray(armed, dtype=get_global_dtype())

    def _fallen_mask(self, gravity: np.ndarray, base_height: np.ndarray) -> np.ndarray:
        cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
        tilt = np.arccos(cos_tilt)
        return np.logical_or(
            tilt > np.deg2rad(self._cfg.fallen_tilt_deg),
            base_height < self._cfg.fallen_height_threshold,
        )

    def _update_fallen_timer(self, info: dict[str, Any], fallen: np.ndarray) -> np.ndarray:
        previous = np.asarray(
            info.get(
                "fallen_time_s",
                np.zeros(self._num_envs, dtype=get_global_dtype()),
            ),
            dtype=get_global_dtype(),
        )
        elapsed = np.where(fallen, previous + self._cfg.ctrl_dt, 0.0)
        info["fallen_time_s"] = np.asarray(elapsed, dtype=get_global_dtype())
        return info["fallen_time_s"]

    def _termination_mask(
        self,
        gravity: np.ndarray,
        base_height: np.ndarray,
        fallen_time_s: np.ndarray,
    ) -> np.ndarray:
        if self.step_counter < self._cfg.fell_over_disable_after_steps:
            cos_tilt = np.clip(-gravity[:, 2], -1.0, 1.0)
            tilt = np.arccos(cos_tilt)
            return np.logical_or(
                tilt > np.deg2rad(self._reward_cfg.max_tilt_deg),
                base_height < self._reward_cfg.min_base_height,
            )
        return fallen_time_s >= self._cfg.fallen_timeout_s

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_height = np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())
        state.info["base_height"] = base_height
        state.info["base_world_linvel"] = np.asarray(
            self._backend.get_base_lin_vel(),
            dtype=get_global_dtype(),
        )
        self._update_foot_contact_history(state.info)

        fallen = self._fallen_mask(gravity, base_height)
        fallen_time_s = self._update_fallen_timer(state.info, fallen)
        state.info["fallen"] = fallen
        terminated = self._termination_mask(gravity, base_height, fallen_time_s)

        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_scales_for_step(),
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        self._resample_commands_for_next_step(state.info)
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)
