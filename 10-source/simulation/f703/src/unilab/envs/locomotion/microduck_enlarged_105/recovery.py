"""Ground-recovery owner for the enlarged Micro Duck V1.0.5 model."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged_104.recovery import (
    MicroDuckEnlarged104RecoveryRewardConfig,
    MicroDuckEnlarged104StandRecoveryFlatCfg,
    MicroDuckEnlarged104StandRecoveryFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_105.tasks import (
    MICRODUCK_ENLARGED_105_MOTOR_SPECS,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_105" / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlarged105RecoveryRewardConfig(
    MicroDuckEnlarged104RecoveryRewardConfig
):
    """V1.0.5 recovery rewards with the inherited progress signals."""


@registry.envcfg("MicroDuckEnlarged105StandRecoveryFlat")
@dataclass
class MicroDuckEnlarged105StandRecoveryFlatCfg(
    MicroDuckEnlarged104StandRecoveryFlatCfg
):
    """Zero-command prone/face-up recovery bound to the V1.0.5 scene."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged105RecoveryRewardConfig | None = None


@registry.env("MicroDuckEnlarged105StandRecoveryFlat", sim_backend="mujoco")
class MicroDuckEnlarged105StandRecoveryFlatEnv(
    MicroDuckEnlarged104StandRecoveryFlatEnv
):
    """Prone/face-up recovery with fourteen fitted GF43X40-10 actuators."""

    _cfg: MicroDuckEnlarged105StandRecoveryFlatCfg
    MOTOR_SPECS = MICRODUCK_ENLARGED_105_MOTOR_SPECS


__all__ = [
    "MicroDuckEnlarged105RecoveryRewardConfig",
    "MicroDuckEnlarged105StandRecoveryFlatCfg",
    "MicroDuckEnlarged105StandRecoveryFlatEnv",
]
