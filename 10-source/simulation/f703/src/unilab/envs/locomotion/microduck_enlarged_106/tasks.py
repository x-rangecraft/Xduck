"""Standing and walking owners for the enlarged Micro Duck V1.0.6 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from unilab.actuators.gf43x import GF43MotorSpec
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedDomainRandConfig,
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_104.tasks import (
    MicroDuckEnlarged104StandFlatCfg,
    MicroDuckEnlarged104StandFlatEnv,
    MicroDuckEnlarged104StandRewardConfig,
    MicroDuckEnlarged104WalkFlatCfg,
    MicroDuckEnlarged104WalkFlatEnv,
    MicroDuckEnlarged104WalkRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged_106.sitstand import (
    MICRODUCK_ENLARGED_106_MOTOR_SPECS,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_106" / "scene_flat.xml")
    )


@dataclass
class MicroDuckEnlarged106WalkDomainRandConfig(MicroDuckEnlargedDomainRandConfig):
    """Conservative V1.0.6 walk robustness surface."""

    com_offset_y: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    randomize_joint_position_bias: bool = False
    joint_position_bias_range_rad: list[float] = field(default_factory=lambda: [-0.005, 0.005])
    randomize_joint_velocity_bias: bool = False
    joint_velocity_bias_range_rad_s: list[float] = field(default_factory=lambda: [-0.0025, 0.0025])
    randomize_gyro_bias: bool = False
    gyro_bias_range_rad_s: list[float] = field(default_factory=lambda: [-0.01, 0.01])


class MicroDuckEnlarged106WalkDomainRandomizationProvider(
    MicroDuckEnlargedWalkDomainRandomizationProvider
):
    """Cache the model tables required by V1.0.6 walk reset randomization."""

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray, int, np.ndarray]:
        cached = getattr(self, "_reset_randomization_baselines", None)
        if cached is None:
            backend = env._backend
            cached = (
                backend.get_body_mass(),
                backend.get_geom_friction(),
                backend.get_geom_id(env.cfg.asset.ground),
                backend.get_dof_armature(),
            )
            self._reset_randomization_baselines = cached
        return cached

    @staticmethod
    def _sample_bias(
        enabled: bool,
        value_range: list[float],
        shape: tuple[int, int],
        *,
        name: str,
    ) -> np.ndarray:
        if not enabled:
            return np.zeros(shape, dtype=np.float32)
        lower, upper = (float(value) for value in value_range)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError(f"{name} must be a finite ordered pair")
        return np.asarray(np.random.uniform(lower, upper, size=shape), dtype=np.float32)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        domain_rand = env.cfg.domain_rand
        updates.update(
            {
                "joint_position_bias_rad": self._sample_bias(
                    bool(domain_rand.randomize_joint_position_bias),
                    domain_rand.joint_position_bias_range_rad,
                    (num_reset, env._num_action),
                    name="joint_position_bias_range_rad",
                ),
                "joint_velocity_bias_rad_s": self._sample_bias(
                    bool(domain_rand.randomize_joint_velocity_bias),
                    domain_rand.joint_velocity_bias_range_rad_s,
                    (num_reset, env._num_action),
                    name="joint_velocity_bias_range_rad_s",
                ),
                "gyro_bias_rad_s": self._sample_bias(
                    bool(domain_rand.randomize_gyro_bias),
                    domain_rand.gyro_bias_range_rad_s,
                    (num_reset, 3),
                    name="gyro_bias_range_rad_s",
                ),
            }
        )
        return updates


@registry.envcfg("MicroDuckEnlarged106StandFlat")
@dataclass
class MicroDuckEnlarged106StandFlatCfg(MicroDuckEnlarged104StandFlatCfg):
    """V1.0.6 HOME-pose balance task on the all-X40 mechanics."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104StandRewardConfig | None = None
    domain_rand: MicroDuckEnlarged106WalkDomainRandConfig = field(
        default_factory=MicroDuckEnlarged106WalkDomainRandConfig
    )


@registry.env("MicroDuckEnlarged106StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged106StandFlatEnv(MicroDuckEnlarged104StandFlatEnv):
    """Flat-ground standing with the V1.0.6 actuator and scene contract."""

    _cfg: MicroDuckEnlarged106StandFlatCfg
    MOTOR_SPECS: ClassVar[tuple[GF43MotorSpec, ...]] = MICRODUCK_ENLARGED_106_MOTOR_SPECS

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged106WalkDomainRandomizationProvider:
        return MicroDuckEnlarged106WalkDomainRandomizationProvider()

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "motor_peak_envelope_max": self._motor_peak_envelope_max,
                "motor_joint_target_limit": self._motor_joint_target_limit,
                "motor_evidence_speed": self._motor_evidence_speed_penalty,
                "motor_evidence_speed_max": self._motor_evidence_speed_max_penalty,
            }
        )


@registry.envcfg("MicroDuckEnlarged106WalkFlat")
@dataclass
class MicroDuckEnlarged106WalkFlatCfg(MicroDuckEnlarged104WalkFlatCfg):
    """V1.0.6 forward-walking task with the stable 61D/14D contract."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104WalkRewardConfig | None = None
    domain_rand: MicroDuckEnlarged106WalkDomainRandConfig = field(
        default_factory=MicroDuckEnlarged106WalkDomainRandConfig
    )


@registry.env("MicroDuckEnlarged106WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged106WalkFlatEnv(MicroDuckEnlarged104WalkFlatEnv):
    """Flat-ground locomotion with fourteen GF43X40-10 actuators."""

    _cfg: MicroDuckEnlarged106WalkFlatCfg
    MOTOR_SPECS: ClassVar[tuple[GF43MotorSpec, ...]] = MICRODUCK_ENLARGED_106_MOTOR_SPECS

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged106WalkDomainRandomizationProvider:
        return MicroDuckEnlarged106WalkDomainRandomizationProvider()


__all__ = [
    "MicroDuckEnlarged106StandFlatCfg",
    "MicroDuckEnlarged106StandFlatEnv",
    "MicroDuckEnlarged106WalkFlatCfg",
    "MicroDuckEnlarged106WalkDomainRandConfig",
    "MicroDuckEnlarged106WalkFlatEnv",
]
