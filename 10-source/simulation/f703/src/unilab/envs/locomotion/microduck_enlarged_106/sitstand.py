"""Commanded sit/stand owner for the enlarged Micro Duck V1.0.6 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from unilab.actuators.gf43x import GF43X40_10, GF43MotorSpec
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedDomainRandConfig,
)
from unilab.envs.locomotion.microduck_enlarged_105.sitstand import (
    MicroDuckEnlarged105SitStandCommands,
    MicroDuckEnlarged105SitStandDomainRandomizationProvider,
    MicroDuckEnlarged105SitStandFlatCfg,
    MicroDuckEnlarged105SitStandFlatEnv,
    MicroDuckEnlarged105SitStandRewardConfig,
)

V106_STAND_Z = 0.2446
V106_SIT_Z = 0.1285
V106_POSTURE_RAMP_S = 2.0
V106_HEAD_GROUND_CONTACT_SENSORS: tuple[str, ...] = (
    "top_head_ground_contact",
    "jaw_ground_contact",
    "bottom_head_ground_contact",
)
V106_TRUNK_GROUND_CONTACT_SENSOR = "trunk_ground_contact"

MICRODUCK_ENLARGED_106_MOTOR_SPECS: tuple[GF43MotorSpec, ...] = (
    GF43X40_10,
) * 14


@dataclass
class MicroDuckEnlarged106SitStandCommands(
    MicroDuckEnlarged105SitStandCommands
):
    """Official-compatible binary posture command for V1.0.6."""


@dataclass
class MicroDuckEnlarged106SitStandRewardConfig(
    MicroDuckEnlarged105SitStandRewardConfig
):
    """V1.0.6 reward parameters preserving the official posture semantics."""

    posture_pose_std: float = 0.50
    posture_height_std: float = 0.040
    posture_height_sharp_std: float = 0.015
    rise_bootstrap_max_height: float = 0.255
    upright_tall_low: float = 0.157
    upright_tall_high: float = 0.209
    stillness_band_full: float = 0.012
    stillness_band_zero: float = 0.030
    stillness_vel_std: float = 0.050
    composite_height_std: float = 0.030
    forbidden_contact_threshold_n: float = 1.0
    foot_support_threshold_n: float = 1.0


@dataclass
class MicroDuckEnlarged106SitStandDomainRandConfig(
    MicroDuckEnlargedDomainRandConfig
):
    """V1.0.6-only DR surface for conservative sit/stand fine-tuning."""

    # Optional lateral/vertical COM offsets are declared here so Hydra struct
    # mode accepts them.  All switches remain disabled in the base profile.
    com_offset_y: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])


class MicroDuckEnlarged106SitStandDomainRandomizationProvider(
    MicroDuckEnlarged105SitStandDomainRandomizationProvider
):
    """Source-keyframe reset provider bound to the V1.0.6 owner."""

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray, int, np.ndarray]:
        """Cache model tables needed by friction and armature DR."""

        cached = getattr(self, "_reset_randomization_baselines", None)
        if cached is None:
            backend = env._backend
            cached = (
                None,
                backend.get_geom_friction(),
                backend.get_geom_id(env.cfg.asset.ground),
                backend.get_dof_armature(),
            )
            self._reset_randomization_baselines = cached
        return cached

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        # Do not perturb the robot on its very first control step.  The first
        # push is due only after a complete configured interval has elapsed.
        if step_counter <= 0:
            return None
        return super().build_interval_randomization_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        match_prob = float(env.cfg.reset_command_match_prob)
        if not 0.0 <= match_prob <= 1.0:
            raise ValueError("V1.0.6 reset_command_match_prob must be within [0, 1]")
        if len(env_ids) == 0 or match_prob == 0.0:
            return plan

        match = np.random.uniform(size=len(env_ids)) < match_prob
        sitting = np.asarray(plan.info_updates["sitstand_reset_is_sitting"])
        commands = np.asarray(plan.info_updates["commands"])
        commands[match, 0] = sitting[match].astype(commands.dtype)
        return plan


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH
            / "robots"
            / "microduck_enlarged_106"
            / "scene_flat.xml"
        )
    )


@registry.envcfg("MicroDuckEnlarged106SitStandFlat")
@dataclass
class MicroDuckEnlarged106SitStandFlatCfg(
    MicroDuckEnlarged105SitStandFlatCfg
):
    """Two-way posture task bound to the supplied V1.0.6 mechanics."""

    scene: SceneCfg = field(default_factory=_scene)
    commands: MicroDuckEnlarged106SitStandCommands = field(
        default_factory=MicroDuckEnlarged106SitStandCommands
    )
    reward_config: MicroDuckEnlarged106SitStandRewardConfig | None = None
    domain_rand: MicroDuckEnlarged106SitStandDomainRandConfig = field(
        default_factory=MicroDuckEnlarged106SitStandDomainRandConfig
    )
    sit_z: float = V106_SIT_Z
    stand_z: float = V106_STAND_Z
    posture_ramp_s: float = V106_POSTURE_RAMP_S
    reset_command_match_prob: float = 0.0
    terminate_forbidden_ground_contact: bool = False
    curriculum_terminate_tilt_deg: float = 0.0


@registry.env("MicroDuckEnlarged106SitStandFlat", sim_backend="mujoco")
class MicroDuckEnlarged106SitStandFlatEnv(
    MicroDuckEnlarged105SitStandFlatEnv
):
    """V1.0.6 sit/stand environment with the stable 61D/14D contract."""

    _cfg: MicroDuckEnlarged106SitStandFlatCfg
    MOTOR_SPECS: ClassVar[tuple[GF43MotorSpec, ...]] = (
        MICRODUCK_ENLARGED_106_MOTOR_SPECS
    )

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged106SitStandDomainRandomizationProvider:
        return MicroDuckEnlarged106SitStandDomainRandomizationProvider()

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["forbidden_ground_contact"] = (
            self._forbidden_ground_contact
        )
        self._reward_fns["both_feet_support"] = self._both_feet_support
        self._reward_fns["endpoint_progress_upright"] = (
            self._endpoint_progress_upright
        )
        self._reward_fns["motor_evidence_speed"] = (
            self._motor_evidence_speed_penalty
        )
        self._reward_fns["motor_evidence_speed_max"] = (
            self._motor_evidence_speed_max_penalty
        )

    def _contact_force_norm(self, sensor_name: str) -> np.ndarray:
        force = np.asarray(
            self._backend.get_sensor_data(sensor_name), dtype=get_global_dtype()
        ).reshape(self._num_envs, -1)
        return np.asarray(np.linalg.norm(force, axis=1), dtype=get_global_dtype())

    def _forbidden_ground_contact(self, ctx: RewardContext) -> np.ndarray:
        """Penalize the head/trunk support exploit while keeping feet valid."""

        threshold = float(self._sitstand_reward_cfg.forbidden_contact_threshold_n)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "forbidden_contact_threshold_n must be finite and non-negative"
            )
        head_force = np.maximum.reduce(
            [
                self._contact_force_norm(name)
                for name in V106_HEAD_GROUND_CONTACT_SENSORS
            ]
        )
        trunk_force = self._contact_force_norm(V106_TRUNK_GROUND_CONTACT_SENSOR)
        return np.asarray(
            (head_force > threshold).astype(get_global_dtype())
            + (trunk_force > threshold).astype(get_global_dtype()),
            dtype=get_global_dtype(),
        )

    def _both_feet_support(self, ctx: RewardContext) -> np.ndarray:
        """Reward the intended two-foot support instead of airborne escapes."""

        del ctx
        threshold = float(self._sitstand_reward_cfg.foot_support_threshold_n)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "foot_support_threshold_n must be finite and non-negative"
            )
        left = self._contact_force_norm("left_foot_contact")
        right = self._contact_force_norm("right_foot_contact")
        return np.asarray(
            (left > threshold) & (right > threshold), dtype=get_global_dtype()
        )

    def _endpoint_progress_upright(self, ctx: RewardContext) -> np.ndarray:
        """Pay posture quality only after making real commanded-height progress."""

        span = float(self._cfg.stand_z) - float(self._cfg.sit_z)
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("stand_z must be greater than sit_z")
        stand_progress = np.clip(
            (ctx.base_height - float(self._cfg.sit_z)) / span, 0.0, 1.0
        )
        sit_progress = np.clip(
            (float(self._cfg.stand_z) - ctx.base_height) / span, 0.0, 1.0
        )
        command = np.clip(
            np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 0],
            0.0,
            1.0,
        )
        progress = (1.0 - command) * stand_progress + command * sit_progress
        upright = np.clip((self._upright_cosine(ctx) - 0.55) / 0.45, 0.0, 1.0)
        return np.asarray(progress * upright, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Optionally terminate failed curriculum samples without changing production."""

        state = super().update_state(state)
        terminated = np.asarray(state.terminated, dtype=bool).copy()
        if self._cfg.terminate_forbidden_ground_contact:
            threshold = float(
                self._sitstand_reward_cfg.forbidden_contact_threshold_n
            )
            head_force = np.maximum.reduce(
                [
                    self._contact_force_norm(name)
                    for name in V106_HEAD_GROUND_CONTACT_SENSORS
                ]
            )
            trunk_force = self._contact_force_norm(
                V106_TRUNK_GROUND_CONTACT_SENSOR
            )
            terminated |= (head_force > threshold) | (trunk_force > threshold)
        tilt_limit_deg = float(self._cfg.curriculum_terminate_tilt_deg)
        if tilt_limit_deg > 0.0:
            gravity = self._projected_gravity()
            tilt_deg = np.rad2deg(
                np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
            )
            terminated |= tilt_deg > tilt_limit_deg
        return state.replace(terminated=terminated)

    @property
    def _sitstand_reward_cfg(
        self,
    ) -> MicroDuckEnlarged106SitStandRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlarged106SitStandRewardConfig):
            raise TypeError(
                "MicroDuckEnlarged106SitStandFlat requires its sit/stand reward config"
            )
        return cfg


__all__ = [
    "MICRODUCK_ENLARGED_106_MOTOR_SPECS",
    "V106_HEAD_GROUND_CONTACT_SENSORS",
    "V106_POSTURE_RAMP_S",
    "V106_SIT_Z",
    "V106_STAND_Z",
    "V106_TRUNK_GROUND_CONTACT_SENSOR",
    "MicroDuckEnlarged106SitStandCommands",
    "MicroDuckEnlarged106SitStandDomainRandConfig",
    "MicroDuckEnlarged106SitStandDomainRandomizationProvider",
    "MicroDuckEnlarged106SitStandFlatCfg",
    "MicroDuckEnlarged106SitStandFlatEnv",
    "MicroDuckEnlarged106SitStandRewardConfig",
]
