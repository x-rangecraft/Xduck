"""Standing and walking owners for enlarged Micro Duck modification 1."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.stand import (
    MicroDuckEnlargedStandFlatCfg,
    MicroDuckEnlargedStandFlatEnv,
    MicroDuckEnlargedStandRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedAsset,
    MicroDuckEnlargedControlConfig,
    MicroDuckEnlargedRewardConfig,
    MicroDuckEnlargedWalkFlatCfg,
    MicroDuckEnlargedWalkFlatEnv,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_mod1" / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlargedMod1Asset(MicroDuckEnlargedAsset):
    """Cold-path semantic mapping measured from DUCK_V1.01."""

    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -0.049772514, -0.029438068),
        (0.0, -0.049772514, -0.029438068),
    )


@dataclass
class MicroDuckEnlargedMod1ControlConfig(MicroDuckEnlargedControlConfig):
    """Stable V1.01 hold gains validated for the supplied STAND2 pose."""

    Kp: float = 30.0
    Kd: float = 0.5


@dataclass
class MicroDuckEnlargedMod1WalkRewardConfig(MicroDuckEnlargedRewardConfig):
    base_height_target: float = 0.242
    min_base_height: float = 0.13
    foot_height_target: float = 0.0418
    swing_time_min: float = 0.17
    swing_time_max: float = 0.44
    swing_time_target: float = 0.275
    swing_time_std: float = 0.069
    short_swing_threshold: float = 0.18
    base_height_std: float = 0.016


@registry.envcfg("MicroDuckEnlargedMod1WalkFlat")
@dataclass
class MicroDuckEnlargedMod1WalkFlatCfg(MicroDuckEnlargedWalkFlatCfg):
    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlargedMod1Asset = field(default_factory=MicroDuckEnlargedMod1Asset)
    control_config: MicroDuckEnlargedMod1ControlConfig = field(
        default_factory=MicroDuckEnlargedMod1ControlConfig
    )
    reward_config: MicroDuckEnlargedMod1WalkRewardConfig | None = None


@registry.env("MicroDuckEnlargedMod1WalkFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1WalkFlatEnv(MicroDuckEnlargedWalkFlatEnv):
    """Flat-ground locomotion with the V1.01 mechanical contract."""

    _cfg: MicroDuckEnlargedMod1WalkFlatCfg


@dataclass
class MicroDuckEnlargedMod1StandRewardConfig(MicroDuckEnlargedStandRewardConfig):
    base_height_target: float = 0.242
    min_base_height: float = 0.13
    body_height_std: float = 0.032
    height_target_min: float = 0.20
    height_target_max: float = 0.29


@registry.envcfg("MicroDuckEnlargedMod1StandFlat")
@dataclass
class MicroDuckEnlargedMod1StandFlatCfg(MicroDuckEnlargedStandFlatCfg):
    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlargedMod1Asset = field(default_factory=MicroDuckEnlargedMod1Asset)
    control_config: MicroDuckEnlargedMod1ControlConfig = field(
        default_factory=MicroDuckEnlargedMod1ControlConfig
    )
    reward_config: MicroDuckEnlargedMod1StandRewardConfig | None = None


@registry.env("MicroDuckEnlargedMod1StandFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1StandFlatEnv(MicroDuckEnlargedStandFlatEnv):
    """HOME-pose balance for the V1.01 mechanical contract."""

    _cfg: MicroDuckEnlargedMod1StandFlatCfg
