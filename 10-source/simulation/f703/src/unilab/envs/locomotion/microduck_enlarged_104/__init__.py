"""Registry exports for enlarged Micro Duck V1.0.4."""

from unilab.envs.locomotion.microduck_enlarged_104.recovery import (
    MicroDuckEnlarged104RecoveryRewardConfig,
    MicroDuckEnlarged104StandRecoveryFlatCfg,
    MicroDuckEnlarged104StandRecoveryFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_104.tasks import (
    MicroDuckEnlarged104StandFlatCfg,
    MicroDuckEnlarged104StandFlatEnv,
    MicroDuckEnlarged104StandRewardConfig,
    MicroDuckEnlarged104WalkFlatCfg,
    MicroDuckEnlarged104WalkFlatEnv,
    MicroDuckEnlarged104WalkRewardConfig,
)

__all__ = [
    "MicroDuckEnlarged104StandRewardConfig",
    "MicroDuckEnlarged104StandFlatCfg",
    "MicroDuckEnlarged104StandFlatEnv",
    "MicroDuckEnlarged104WalkRewardConfig",
    "MicroDuckEnlarged104WalkFlatCfg",
    "MicroDuckEnlarged104WalkFlatEnv",
    "MicroDuckEnlarged104RecoveryRewardConfig",
    "MicroDuckEnlarged104StandRecoveryFlatCfg",
    "MicroDuckEnlarged104StandRecoveryFlatEnv",
]
