"""Standing and walking owners for enlarged Micro Duck V1.0.2."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged_mod1.tasks import (
    MicroDuckEnlargedMod1StandFlatCfg,
    MicroDuckEnlargedMod1StandFlatEnv,
    MicroDuckEnlargedMod1StandRewardConfig,
    MicroDuckEnlargedMod1WalkFlatCfg,
    MicroDuckEnlargedMod1WalkFlatEnv,
    MicroDuckEnlargedMod1WalkRewardConfig,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_102" / "scene_flat.xml")
    )


@registry.envcfg("MicroDuckEnlarged102WalkFlat")
@dataclass
class MicroDuckEnlarged102WalkFlatCfg(MicroDuckEnlargedMod1WalkFlatCfg):
    """V1.0.2 scene with the V1.01-compatible locomotion contract."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlargedMod1WalkRewardConfig | None = None


@registry.env("MicroDuckEnlarged102WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged102WalkFlatEnv(MicroDuckEnlargedMod1WalkFlatEnv):
    """Flat-ground locomotion for the V1.0.2 mass and inertia model."""

    _cfg: MicroDuckEnlarged102WalkFlatCfg


@registry.envcfg("MicroDuckEnlarged102StandFlat")
@dataclass
class MicroDuckEnlarged102StandFlatCfg(MicroDuckEnlargedMod1StandFlatCfg):
    """V1.0.2 scene with the V1.01-compatible balance contract."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlargedMod1StandRewardConfig | None = None


@registry.env("MicroDuckEnlarged102StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged102StandFlatEnv(MicroDuckEnlargedMod1StandFlatEnv):
    """HOME-pose balance for the V1.0.2 mass and inertia model."""

    _cfg: MicroDuckEnlarged102StandFlatCfg
