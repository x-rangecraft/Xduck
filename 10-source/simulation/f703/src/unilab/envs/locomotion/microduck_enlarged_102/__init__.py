"""Registry exports for enlarged Micro Duck V1.0.2."""

from unilab.envs.locomotion.microduck_enlarged_102.recovery import (
    MicroDuckEnlarged102RecoveryRewardConfig,
    MicroDuckEnlarged102StandRecoveryFlatCfg,
    MicroDuckEnlarged102StandRecoveryFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_102.tasks import (
    MicroDuckEnlarged102StandFlatCfg,
    MicroDuckEnlarged102StandFlatEnv,
    MicroDuckEnlarged102WalkFlatCfg,
    MicroDuckEnlarged102WalkFlatEnv,
)

__all__ = [
    "MicroDuckEnlarged102StandFlatCfg",
    "MicroDuckEnlarged102StandFlatEnv",
    "MicroDuckEnlarged102WalkFlatCfg",
    "MicroDuckEnlarged102WalkFlatEnv",
    "MicroDuckEnlarged102RecoveryRewardConfig",
    "MicroDuckEnlarged102StandRecoveryFlatCfg",
    "MicroDuckEnlarged102StandRecoveryFlatEnv",
]
