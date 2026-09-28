"""Micro Duck locomotion task registry."""

from unilab.envs.locomotion.microduck.roller import (
    MicroDuckRollerFlatCfg,
    MicroDuckRollerFlatEnv,
)
from unilab.envs.locomotion.microduck.roller_crouch import (
    MicroDuckRollerCrouchFlatCfg,
    MicroDuckRollerCrouchFlatEnv,
)
from unilab.envs.locomotion.microduck.roulade import (
    MicroDuckRouladeFlatCfg,
    MicroDuckRouladeFlatEnv,
)
from unilab.envs.locomotion.microduck.recovery import (
    MicroDuckStandRecoveryFlatCfg,
    MicroDuckStandRecoveryFlatEnv,
)
from unilab.envs.locomotion.microduck.stand import MicroDuckStandFlatCfg, MicroDuckStandFlatEnv
from unilab.envs.locomotion.microduck.sitstand import (
    MicroDuckSitStandFlatCfg,
    MicroDuckSitStandFlatEnv,
    MicroDuckSitStandRewardConfig,
)
from unilab.envs.locomotion.microduck.velstand import (
    MicroDuckVelStandFlatCfg,
    MicroDuckVelStandFlatEnv,
)
from unilab.envs.locomotion.microduck.walk import MicroDuckWalkFlatCfg, MicroDuckWalkFlatEnv

__all__ = [
    "MicroDuckRollerFlatCfg",
    "MicroDuckRollerFlatEnv",
    "MicroDuckRollerCrouchFlatCfg",
    "MicroDuckRollerCrouchFlatEnv",
    "MicroDuckRouladeFlatCfg",
    "MicroDuckRouladeFlatEnv",
    "MicroDuckStandRecoveryFlatCfg",
    "MicroDuckStandRecoveryFlatEnv",
    "MicroDuckStandFlatCfg",
    "MicroDuckStandFlatEnv",
    "MicroDuckSitStandFlatCfg",
    "MicroDuckSitStandFlatEnv",
    "MicroDuckSitStandRewardConfig",
    "MicroDuckVelStandFlatCfg",
    "MicroDuckVelStandFlatEnv",
    "MicroDuckWalkFlatCfg",
    "MicroDuckWalkFlatEnv",
]
