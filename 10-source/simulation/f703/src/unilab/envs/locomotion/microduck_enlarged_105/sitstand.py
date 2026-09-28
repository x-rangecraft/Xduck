"""Commanded sit/stand owner for the enlarged Micro Duck V1.0.5 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.stand import (
    MicroDuckEnlargedStandCommands,
    MicroDuckEnlargedStandRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged.walk import (
    _HEAD_JOINT_INDICES,
    _LEG_JOINT_INDICES,
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_104.tasks import (
    MicroDuckEnlarged104StandFlatCfg,
    MicroDuckEnlarged104StandFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_105.tasks import (
    MICRODUCK_ENLARGED_105_MOTOR_SPECS,
)

V105_STAND_Z = 0.2446
V105_SIT_Z = 0.1463
V105_POSTURE_RAMP_S = 2.0


@dataclass
class MicroDuckEnlarged105SitStandCommands(MicroDuckEnlargedStandCommands):
    """Binary posture command in the first three command observation slots."""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    sit_prob: float = 0.5
    rel_standing_envs: float = 0.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0


@dataclass
class MicroDuckEnlarged105SitStandRewardConfig(MicroDuckEnlargedStandRewardConfig):
    """Reward parameters for the V1.0.5 keyframe-driven posture task."""

    posture_pose_std: float = 0.45
    posture_height_std: float = 0.035
    posture_height_sharp_std: float = 0.014
    max_descent_speed: float = 0.07
    max_rise_speed: float = 0.10
    rise_bootstrap_max_height: float = 0.20
    rise_bootstrap_max_speed: float = 0.10
    upright_tall_low: float = 0.17
    upright_tall_high: float = 0.22
    stillness_band_full: float = 0.014
    stillness_band_zero: float = 0.035
    stillness_vel_std: float = 0.06
    stillness_tilt_full_deg: float = 25.0
    stillness_tilt_zero_deg: float = 60.0
    composite_height_std: float = 0.035
    composite_upright_std: float = 0.40
    composite_pose_std: float = 0.40
    composite_head_std: float = 0.40


class MicroDuckEnlarged105SitStandDomainRandomizationProvider(
    MicroDuckEnlargedWalkDomainRandomizationProvider
):
    """Reset provider using the cached ``home`` and ``source_sit`` keyframes."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        sit_prob = float(env.cfg.commands.sit_prob)
        if not 0.0 <= sit_prob <= 1.0:
            raise ValueError("V1.0.5 sit_prob must be within [0, 1]")
        commands = np.zeros((num_reset, 3), dtype=get_global_dtype())
        commands[:, 0] = (
            np.random.uniform(size=num_reset) < sit_prob
        ).astype(get_global_dtype())
        return commands

    def _build_extra_info_updates(
        self, env: Any, num_reset: int
    ) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        low, high = (float(value) for value in env.cfg.twist_resample_s)
        if low <= 0.0 or high < low:
            raise ValueError("V1.0.5 twist_resample_s must satisfy 0 < low <= high")
        updates.update(
            {
                "twist_resample_s": np.asarray(
                    np.random.uniform(low, high, size=num_reset),
                    dtype=get_global_dtype(),
                ),
                "posture_blend": np.zeros(num_reset, dtype=get_global_dtype()),
                "sitstand_target_blend": np.zeros(
                    num_reset, dtype=get_global_dtype()
                ),
                "sitstand_reset_mode": np.zeros(num_reset, dtype=np.int8),
                "sitstand_reset_is_sitting": np.zeros(num_reset, dtype=bool),
                "previous_base_vz": np.zeros(num_reset, dtype=get_global_dtype()),
            }
        )
        return updates

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        if num_reset == 0:
            return plan

        sit_prob = float(env.cfg.sitting_reset_prob)
        stand_prob = float(env.cfg.standing_reset_prob)
        if sit_prob < 0.0 or stand_prob < 0.0 or sit_prob + stand_prob <= 0.0:
            raise ValueError(
                "V1.0.5 reset probabilities must be non-negative and have a positive sum"
            )
        sitting = np.random.uniform(size=num_reset) < sit_prob / (sit_prob + stand_prob)

        home_tail = np.asarray(env._sitstand_home_qpos_tail, dtype=get_global_dtype())
        sit_tail = np.asarray(env._sitstand_source_sit_qpos_tail, dtype=get_global_dtype())
        if home_tail.shape != sit_tail.shape or home_tail.shape != (21 - 2,):
            raise RuntimeError("V1.0.5 posture keyframes must provide 19 qpos values after xyz")
        # Keep the provider's randomized x/y spawn, but use source-exact pose,
        # orientation and base height for both posture starts.
        plan.qpos[:, 2:] = np.where(
            sitting[:, None], sit_tail[None, :], home_tail[None, :]
        )
        plan.qvel[:, :] = 0.0
        plan.info_updates["posture_blend"] = sitting.astype(get_global_dtype())
        plan.info_updates["sitstand_target_blend"] = plan.info_updates[
            "posture_blend"
        ].copy()
        plan.info_updates["sitstand_reset_mode"] = sitting.astype(np.int8)
        plan.info_updates["sitstand_reset_is_sitting"] = sitting
        plan.info_updates["previous_base_vz"] = np.zeros(
            num_reset, dtype=get_global_dtype()
        )
        return plan

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
        return env._compute_obs(
            info_updates,
            gyro,
            env._projected_gravity()[env_ids],
            dof_pos,
            dof_vel,
        )


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_105" / "scene_flat.xml"
        )
    )


@registry.envcfg("MicroDuckEnlarged105SitStandFlat")
@dataclass
class MicroDuckEnlarged105SitStandFlatCfg(MicroDuckEnlarged104StandFlatCfg):
    """Two-way posture task bound to the V1.0.5 fitted GF43 model."""

    scene: SceneCfg = field(default_factory=_scene)
    commands: MicroDuckEnlarged105SitStandCommands = field(
        default_factory=MicroDuckEnlarged105SitStandCommands
    )
    reward_config: MicroDuckEnlarged105SitStandRewardConfig | None = None
    max_episode_seconds: float = 16.0
    twist_resample_s: tuple[float, float] = (3.5, 6.5)
    sit_z: float = V105_SIT_Z
    stand_z: float = V105_STAND_Z
    posture_ramp_s: float = V105_POSTURE_RAMP_S
    sitting_reset_prob: float = 0.5
    standing_reset_prob: float = 0.5
    reset_tilt_max_deg: float = 0.0
    action_rate_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": -0.10},
            {"step": 24000, "weight": -0.20},
            {"step": 48000, "weight": -0.40},
        ]
    )
    descent_speed_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [{"step": 0, "weight": 8.0}]
    )
    rise_speed_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [{"step": 0, "weight": 0.0}]
    )


@registry.env("MicroDuckEnlarged105SitStandFlat", sim_backend="mujoco")
class MicroDuckEnlarged105SitStandFlatEnv(MicroDuckEnlarged104StandFlatEnv):
    """V1.0.5 sit/stand environment with a stable 61D/14D contract."""

    _cfg: MicroDuckEnlarged105SitStandFlatCfg
    MOTOR_SPECS = MICRODUCK_ENLARGED_105_MOTOR_SPECS

    def __init__(
        self,
        cfg: MicroDuckEnlarged105SitStandFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        home_qpos = np.asarray(self._init_qpos, dtype=get_global_dtype())
        source_sit_qpos = np.asarray(
            self._backend.get_keyframe_qpos("source_sit"), dtype=get_global_dtype()
        )
        if home_qpos.shape != source_sit_qpos.shape or home_qpos.shape != (21,):
            raise RuntimeError(
                "V1.0.5 sit/stand requires matching 21D home and source_sit keyframes"
            )
        self._sitstand_home_qpos_tail = home_qpos[2:].copy()
        self._sitstand_source_sit_qpos_tail = source_sit_qpos[2:].copy()
        self._sitstand_sitting_angles = source_sit_qpos[-self._num_action :].copy()

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged105SitStandDomainRandomizationProvider:
        return MicroDuckEnlarged105SitStandDomainRandomizationProvider()

    @property
    def _sitstand_reward_cfg(self) -> MicroDuckEnlarged105SitStandRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlarged105SitStandRewardConfig):
            raise TypeError(
                "MicroDuckEnlarged105SitStandFlat requires its sit/stand reward config"
            )
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "posture_pose_legs": self._posture_pose_legs,
            "head_pose_tracking": self._head_pose_tracking,
            "posture_pose_l1": self._posture_pose_l1,
            "posture_height": self._posture_height_gaussian,
            "posture_height_sharp": self._posture_height_sharp,
            "posture_height_l1": self._posture_height_l1,
            "rise_bootstrap": self._rise_bootstrap,
            "descent_speed": self._descent_speed,
            "rise_speed": self._rise_speed,
            "gentle_motion": self._gentle_motion,
            "upright_linear": self._upright_linear,
            "upright_while_tall": self._upright_while_tall,
            "posture_stillness": self._posture_stillness,
            "posture_composite": self._posture_composite,
            "action_rate_l2": rewards.action_rate,
            "action_rate": rewards.action_rate,
            "body_ang_vel": rewards.ang_vel_xy,
            "motor_peak_envelope": self._motor_peak_envelope,
            "motor_rated_torque": self._motor_rated_torque_penalty,
            "motor_rated_power": self._motor_rated_power_penalty,
            "motor_speed": self._motor_speed_penalty,
        }

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _world_linvel_from_info(info: dict[str, Any], num_envs: int) -> np.ndarray:
        value = info.get("base_world_linvel")
        if value is None:
            return np.zeros((num_envs, 3), dtype=get_global_dtype())
        return np.asarray(
            np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )

    def _posture_blend(self, info: dict[str, Any]) -> np.ndarray:
        blend = info.get("posture_blend", info.get("sitstand_target_blend"))
        if blend is None:
            blend = np.asarray(info["commands"])[:, 0]
        return np.asarray(
            np.clip(np.nan_to_num(blend, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
            dtype=get_global_dtype(),
        )

    def _advance_posture_blend(self, info: dict[str, Any]) -> np.ndarray:
        blend = self._posture_blend(info)
        command = np.clip(
            np.nan_to_num(
                np.asarray(info["commands"], dtype=get_global_dtype())[:, 0],
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        ramp_s = float(self._cfg.posture_ramp_s)
        if ramp_s <= 0.0:
            blend = command
        else:
            blend += np.clip(
                command - blend,
                -float(self._cfg.ctrl_dt) / ramp_s,
                float(self._cfg.ctrl_dt) / ramp_s,
            )
        blend = np.asarray(np.clip(blend, 0.0, 1.0), dtype=get_global_dtype())
        info["posture_blend"] = blend
        info["sitstand_target_blend"] = blend.copy()
        return blend

    def _posture_target(self, info: dict[str, Any]) -> np.ndarray:
        blend = self._posture_blend(info)
        stand = np.broadcast_to(
            self.default_angles, (self._num_envs, self._num_action)
        ).copy()
        sit = np.broadcast_to(
            self._sitstand_sitting_angles, (self._num_envs, self._num_action)
        )
        return np.asarray(stand + blend[:, None] * (sit - stand), dtype=get_global_dtype())

    def _posture_height_target(self, info: dict[str, Any]) -> np.ndarray:
        blend = self._posture_blend(info)
        return np.asarray(
            float(self._cfg.stand_z)
            + blend * (float(self._cfg.sit_z) - float(self._cfg.stand_z)),
            dtype=get_global_dtype(),
        )

    def _height_error(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            np.nan_to_num(ctx.base_height, nan=0.0, posinf=0.0, neginf=0.0)
            - self._posture_height_target(ctx.info),
            dtype=get_global_dtype(),
        )

    def _posture_pose_legs(self, ctx: RewardContext) -> np.ndarray:
        target = self._posture_target(ctx.info)
        error = np.nan_to_num(
            ctx.dof_pos[:, _LEG_JOINT_INDICES] - target[:, _LEG_JOINT_INDICES],
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
        )
        std = float(self._sitstand_reward_cfg.posture_pose_std)
        if std <= 0.0:
            raise ValueError("posture_pose_std must be positive")
        return np.asarray(np.exp(-np.mean(np.square(error / std), axis=1)), dtype=get_global_dtype())

    def _posture_pose_l1(self, ctx: RewardContext) -> np.ndarray:
        target = self._posture_target(ctx.info)
        return np.asarray(
            -np.mean(
                np.abs(ctx.dof_pos[:, _LEG_JOINT_INDICES] - target[:, _LEG_JOINT_INDICES]),
                axis=1,
            ),
            dtype=get_global_dtype(),
        )

    def _posture_height_gaussian(self, ctx: RewardContext) -> np.ndarray:
        std = float(self._sitstand_reward_cfg.posture_height_std)
        if std <= 0.0:
            raise ValueError("posture_height_std must be positive")
        return np.asarray(np.exp(-np.square(self._height_error(ctx) / std)), dtype=get_global_dtype())

    def _posture_height_sharp(self, ctx: RewardContext) -> np.ndarray:
        std = float(self._sitstand_reward_cfg.posture_height_sharp_std)
        if std <= 0.0:
            raise ValueError("posture_height_sharp_std must be positive")
        return np.asarray(np.exp(-np.square(self._height_error(ctx) / std)), dtype=get_global_dtype())

    def _posture_height_l1(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(-np.abs(self._height_error(ctx)), dtype=get_global_dtype())

    def _rise_bootstrap(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        vz = np.minimum(np.clip(velocity[:, 2], 0.0, None), cfg.rise_bootstrap_max_speed)
        command = np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 0]
        return np.asarray(
            vz
            * (ctx.base_height < cfg.rise_bootstrap_max_height)
            * (1.0 - np.clip(command, 0.0, 1.0)),
            dtype=get_global_dtype(),
        )

    def _descent_speed(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        return np.asarray(
            -np.clip(-velocity[:, 2] - self._sitstand_reward_cfg.max_descent_speed, 0.0, None),
            dtype=get_global_dtype(),
        )

    def _rise_speed(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        return np.asarray(
            -np.clip(velocity[:, 2] - self._sitstand_reward_cfg.max_rise_speed, 0.0, None),
            dtype=get_global_dtype(),
        )

    def _gentle_motion(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        previous = np.asarray(
            ctx.info.get("previous_base_vz", velocity[:, 2]), dtype=get_global_dtype()
        )
        acceleration = (velocity[:, 2] - previous) / max(float(self._cfg.ctrl_dt), 1.0e-6)
        return np.asarray(-np.abs(acceleration), dtype=get_global_dtype())

    def _upright_cosine(self, ctx: RewardContext) -> np.ndarray:
        if ctx.gravity is None:
            return np.ones(ctx.num_envs, dtype=get_global_dtype())
        return np.asarray(
            np.clip(np.nan_to_num(-ctx.gravity[:, 2], nan=1.0), -1.0, 1.0),
            dtype=get_global_dtype(),
        )

    def _upright_linear(self, ctx: RewardContext) -> np.ndarray:
        return self._upright_cosine(ctx)

    def _upright_while_tall(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        gate = self._smoothstep(
            (ctx.base_height - cfg.upright_tall_low)
            / max(cfg.upright_tall_high - cfg.upright_tall_low, 1.0e-6)
        )
        return np.asarray(self._upright_cosine(ctx) * gate, dtype=get_global_dtype())

    def _posture_stillness(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        velocity_score = np.exp(-np.square(np.linalg.norm(velocity, axis=1) / max(cfg.stillness_vel_std, 1.0e-6)))
        height_error = np.abs(self._height_error(ctx))
        height_gate = self._smoothstep(
            (cfg.stillness_band_zero - height_error)
            / max(cfg.stillness_band_zero - cfg.stillness_band_full, 1.0e-6)
        )
        cos_tilt = self._upright_cosine(ctx)
        tilt_gate = self._smoothstep(
            (cos_tilt - np.cos(np.deg2rad(cfg.stillness_tilt_zero_deg)))
            / max(
                np.cos(np.deg2rad(cfg.stillness_tilt_full_deg))
                - np.cos(np.deg2rad(cfg.stillness_tilt_zero_deg)),
                1.0e-6,
            )
        )
        command = np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 0]
        ramp_done = (np.abs(np.clip(command, 0.0, 1.0) - self._posture_blend(ctx.info)) < 0.02)
        return np.asarray(velocity_score * height_gate * tilt_gate * ramp_done, dtype=get_global_dtype())

    def _posture_composite(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        height_score = np.exp(-np.square(self._height_error(ctx) / max(cfg.composite_height_std, 1.0e-6)))
        tilt_sq = np.clip(1.0 - self._upright_cosine(ctx), 0.0, 2.0)
        upright_score = np.exp(-tilt_sq / max(cfg.composite_upright_std**2, 1.0e-6))
        target = self._posture_target(ctx.info)
        pose_error = ctx.dof_pos[:, _LEG_JOINT_INDICES] - target[:, _LEG_JOINT_INDICES]
        pose_score = np.exp(-np.mean(np.square(pose_error), axis=1) / max(cfg.composite_pose_std**2, 1.0e-6))
        actual_head = ctx.dof_pos[:, _HEAD_JOINT_INDICES] - ctx.default_angles[_HEAD_JOINT_INDICES]
        head = np.asarray(ctx.info.get("head_commands", np.zeros_like(actual_head)), dtype=get_global_dtype())
        head_score = np.exp(-np.mean(np.square(actual_head - head), axis=1) / max(cfg.composite_head_std**2, 1.0e-6))
        return np.asarray(height_score * upright_score * pose_score * head_score, dtype=get_global_dtype())

    def _reward_scales_for_step(self) -> dict[str, float]:
        scales = {name: float(value) for name, value in self._reward_cfg.scales.items()}

        def curriculum(name: str, stages: list[dict[str, float | int]]) -> None:
            if name not in scales:
                return
            value = scales[name]
            for stage in stages:
                if self.step_counter < int(stage["step"]):
                    break
                value = float(stage["weight"])
            scales[name] = value

        curriculum("action_rate_l2", self._cfg.action_rate_curriculum)
        curriculum("action_rate", self._cfg.action_rate_curriculum)
        curriculum("descent_speed", self._cfg.descent_speed_curriculum)
        curriculum("rise_speed", self._cfg.rise_speed_curriculum)
        return scales

    def _resample_commands_for_next_step(self, info: dict[str, Any]) -> None:
        timer = np.asarray(info["twist_resample_s"], dtype=get_global_dtype())
        timer -= float(self._cfg.ctrl_dt)
        due = timer <= 0.0
        if np.any(due):
            sit_prob = float(self._cfg.commands.sit_prob)
            info["commands"][due, 0] = (
                np.random.uniform(size=int(np.sum(due))) < sit_prob
            ).astype(get_global_dtype())
            low, high = (float(value) for value in self._cfg.twist_resample_s)
            timer[due] = np.random.uniform(low, high, size=int(np.sum(due)))
        info["twist_resample_s"] = timer

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        base_world_linvel = np.asarray(
            self._backend.get_base_lin_vel(), dtype=get_global_dtype()
        )
        state.info["base_height"] = np.asarray(
            np.nan_to_num(base_pos[:, 2], nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        state.info["base_world_linvel"] = np.asarray(
            np.nan_to_num(base_world_linvel, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        self._advance_posture_blend(state.info)
        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_scales_for_step(),
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        state.info["previous_base_vz"] = state.info["base_world_linvel"][:, 2].copy()
        self._resample_commands_for_next_step(state.info)
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        invalid = np.zeros(self._num_envs, dtype=bool)
        for value in (linvel, gyro, gravity, dof_pos, dof_vel, base_pos, base_quat, base_world_linvel):
            invalid |= ~np.isfinite(np.asarray(value)).reshape(self._num_envs, -1).all(axis=1)
        np.nan_to_num(reward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return state.replace(obs=obs, reward=reward, terminated=invalid)


__all__ = [
    "V105_POSTURE_RAMP_S",
    "V105_SIT_Z",
    "V105_STAND_Z",
    "MicroDuckEnlarged105SitStandCommands",
    "MicroDuckEnlarged105SitStandDomainRandomizationProvider",
    "MicroDuckEnlarged105SitStandFlatCfg",
    "MicroDuckEnlarged105SitStandFlatEnv",
    "MicroDuckEnlarged105SitStandRewardConfig",
]
