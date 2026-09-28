"""Standing and walking owners for enlarged Micro Duck V1.0.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged_102.tasks import (
    MicroDuckEnlarged102StandFlatCfg,
    MicroDuckEnlarged102StandFlatEnv,
    MicroDuckEnlarged102WalkFlatCfg,
    MicroDuckEnlarged102WalkFlatEnv,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_103" / "scene_flat.xml"
        )
    )


@registry.envcfg("MicroDuckEnlarged103WalkFlat")
@dataclass
class MicroDuckEnlarged103WalkFlatCfg(MicroDuckEnlarged102WalkFlatCfg):
    """V1.0.3 mechanics with the compatible enlarged locomotion contract."""

    scene: SceneCfg = field(default_factory=_scene)


@registry.env("MicroDuckEnlarged103WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged103WalkFlatEnv(MicroDuckEnlarged102WalkFlatEnv):
    """Flat-ground locomotion using the supplied coarse position actuators."""

    _cfg: MicroDuckEnlarged103WalkFlatCfg


@registry.envcfg("MicroDuckEnlarged103StandFlat")
@dataclass
class MicroDuckEnlarged103StandFlatCfg(MicroDuckEnlarged102StandFlatCfg):
    """V1.0.3 mechanics with the compatible enlarged balance contract."""

    scene: SceneCfg = field(default_factory=_scene)


@registry.env("MicroDuckEnlarged103StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged103StandFlatEnv(MicroDuckEnlarged102StandFlatEnv):
    """HOME-pose balance using the supplied coarse position actuators."""

    _cfg: MicroDuckEnlarged103StandFlatCfg
