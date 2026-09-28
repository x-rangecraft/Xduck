"""Walking owner for the enlarged Micro Duck V1.0.5 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from unilab.actuators.gf43x import GF43X40_10, GF43MotorSpec
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged_104.tasks import (
    MicroDuckEnlarged104WalkFlatCfg,
    MicroDuckEnlarged104WalkFlatEnv,
    MicroDuckEnlarged104WalkRewardConfig,
)

MICRODUCK_ENLARGED_105_MOTOR_SPECS: tuple[GF43MotorSpec, ...] = (GF43X40_10,) * 14


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_105" / "scene_flat.xml"
        )
    )


@registry.envcfg("MicroDuckEnlarged105WalkFlat")
@dataclass
class MicroDuckEnlarged105WalkFlatCfg(MicroDuckEnlarged104WalkFlatCfg):
    """V1.0.5 flat walking config with the shared 61D policy contract."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104WalkRewardConfig | None = None


@registry.env("MicroDuckEnlarged105WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged105WalkFlatEnv(MicroDuckEnlarged104WalkFlatEnv):
    """Flat-ground locomotion with fourteen GF43X40-10 actuators."""

    _cfg: MicroDuckEnlarged105WalkFlatCfg
    MOTOR_SPECS: ClassVar[tuple[GF43MotorSpec, ...]] = MICRODUCK_ENLARGED_105_MOTOR_SPECS
