"""Standing and walking owners for the enlarged Micro Duck V1.0.4 model."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedMotorConstraintConfig,
)
from unilab.envs.locomotion.microduck_enlarged_102.tasks import (
    MicroDuckEnlarged102StandFlatCfg,
    MicroDuckEnlarged102StandFlatEnv,
    MicroDuckEnlarged102WalkFlatCfg,
    MicroDuckEnlarged102WalkFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_mod1.tasks import (
    MicroDuckEnlargedMod1StandRewardConfig,
    MicroDuckEnlargedMod1WalkRewardConfig,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_104" / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlarged104WalkRewardConfig(MicroDuckEnlargedMod1WalkRewardConfig):
    """V1.0.4 walk reward parameters on the existing enlarged 61D contract."""


@dataclass
class MicroDuckEnlarged104StandRewardConfig(MicroDuckEnlargedMod1StandRewardConfig):
    """V1.0.4 standing rewards on the exact enlarged mechanical model."""


@registry.envcfg("MicroDuckEnlarged104StandFlat")
@dataclass
class MicroDuckEnlarged104StandFlatCfg(MicroDuckEnlarged102StandFlatCfg):
    """HOME-pose balance task bound to the supplied V1.0.4 scene."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104StandRewardConfig | None = None
    motor_constraints: MicroDuckEnlargedMotorConstraintConfig = field(
        default_factory=lambda: MicroDuckEnlargedMotorConstraintConfig(enabled=True)
    )


@registry.env("MicroDuckEnlarged104StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged104StandFlatEnv(MicroDuckEnlarged102StandFlatEnv):
    """V1.0.4 flat-ground standing with the source joint ordering and limits."""

    _cfg: MicroDuckEnlarged104StandFlatCfg


@registry.envcfg("MicroDuckEnlarged104WalkFlat")
@dataclass
class MicroDuckEnlarged104WalkFlatCfg(MicroDuckEnlarged102WalkFlatCfg):
    """Flat walking task bound to the supplied V1.0.4 mechanics."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104WalkRewardConfig | None = None
    motor_constraints: MicroDuckEnlargedMotorConstraintConfig = field(
        default_factory=lambda: MicroDuckEnlargedMotorConstraintConfig(enabled=True)
    )


@registry.env("MicroDuckEnlarged104WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged104WalkFlatEnv(MicroDuckEnlarged102WalkFlatEnv):
    """V1.0.4 flat-ground locomotion with source joint ordering and limits."""

    _cfg: MicroDuckEnlarged104WalkFlatCfg
