"""Registry exports for enlarged Micro Duck V1.0.5."""

from unilab.envs.locomotion.microduck_enlarged_105.recovery import (
    MicroDuckEnlarged105RecoveryRewardConfig,
    MicroDuckEnlarged105StandRecoveryFlatCfg,
    MicroDuckEnlarged105StandRecoveryFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_105.sitstand import (
    MicroDuckEnlarged105SitStandCommands,
    MicroDuckEnlarged105SitStandDomainRandomizationProvider,
    MicroDuckEnlarged105SitStandFlatCfg,
    MicroDuckEnlarged105SitStandFlatEnv,
    MicroDuckEnlarged105SitStandRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged_105.tasks import (
    MICRODUCK_ENLARGED_105_MOTOR_SPECS,
    MicroDuckEnlarged105WalkFlatCfg,
    MicroDuckEnlarged105WalkFlatEnv,
)

__all__ = [
    "MICRODUCK_ENLARGED_105_MOTOR_SPECS",
    "MicroDuckEnlarged105RecoveryRewardConfig",
    "MicroDuckEnlarged105SitStandCommands",
    "MicroDuckEnlarged105SitStandDomainRandomizationProvider",
    "MicroDuckEnlarged105SitStandFlatCfg",
    "MicroDuckEnlarged105SitStandFlatEnv",
    "MicroDuckEnlarged105SitStandRewardConfig",
    "MicroDuckEnlarged105StandRecoveryFlatCfg",
    "MicroDuckEnlarged105StandRecoveryFlatEnv",
    "MicroDuckEnlarged105WalkFlatCfg",
    "MicroDuckEnlarged105WalkFlatEnv",
]
