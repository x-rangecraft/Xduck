"""Registry exports for enlarged Micro Duck V1.0.6."""

from unilab.envs.locomotion.microduck_enlarged_106.roulade import (
    MICRODUCK_ENLARGED_106_ROULADE_CRITIC_OBS_DIM,
    MicroDuckEnlarged106RouladeFlatCfg,
    MicroDuckEnlarged106RouladeFlatEnv,
    MicroDuckEnlarged106RouladeRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged_106.sitstand import (
    MICRODUCK_ENLARGED_106_MOTOR_SPECS,
    MicroDuckEnlarged106SitStandCommands,
    MicroDuckEnlarged106SitStandDomainRandomizationProvider,
    MicroDuckEnlarged106SitStandFlatCfg,
    MicroDuckEnlarged106SitStandFlatEnv,
    MicroDuckEnlarged106SitStandRewardConfig,
)
from unilab.envs.locomotion.microduck_enlarged_106.tasks import (
    MicroDuckEnlarged106StandFlatCfg,
    MicroDuckEnlarged106StandFlatEnv,
    MicroDuckEnlarged106WalkFlatCfg,
    MicroDuckEnlarged106WalkFlatEnv,
)

__all__ = [
    "MICRODUCK_ENLARGED_106_MOTOR_SPECS",
    "MICRODUCK_ENLARGED_106_ROULADE_CRITIC_OBS_DIM",
    "MicroDuckEnlarged106RouladeFlatCfg",
    "MicroDuckEnlarged106RouladeFlatEnv",
    "MicroDuckEnlarged106RouladeRewardConfig",
    "MicroDuckEnlarged106SitStandCommands",
    "MicroDuckEnlarged106SitStandDomainRandomizationProvider",
    "MicroDuckEnlarged106SitStandFlatCfg",
    "MicroDuckEnlarged106SitStandFlatEnv",
    "MicroDuckEnlarged106SitStandRewardConfig",
    "MicroDuckEnlarged106StandFlatCfg",
    "MicroDuckEnlarged106StandFlatEnv",
    "MicroDuckEnlarged106WalkFlatCfg",
    "MicroDuckEnlarged106WalkFlatEnv",
]
