"""Dedicated prone/face-up recovery owner for enlarged Micro Duck V1.0.4."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedMotorConstraintConfig,
)
from unilab.envs.locomotion.microduck_enlarged_102.recovery import (
    MicroDuckEnlarged102RecoveryDomainRandomizationProvider,
    MicroDuckEnlarged102RecoveryRewardConfig,
    MicroDuckEnlarged102StandRecoveryFlatCfg,
    MicroDuckEnlarged102StandRecoveryFlatEnv,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_104" / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlarged104RecoveryRewardConfig(
    MicroDuckEnlarged102RecoveryRewardConfig
):
    """V1.0.4 recovery rewards with the inherited recovery progress signals."""


@registry.envcfg("MicroDuckEnlarged104StandRecoveryFlat")
@dataclass
class MicroDuckEnlarged104StandRecoveryFlatCfg(
    MicroDuckEnlarged102StandRecoveryFlatCfg
):
    """Zero-command recovery task bound to the supplied V1.0.4 model."""

    scene: SceneCfg = field(default_factory=_scene)
    reward_config: MicroDuckEnlarged104RecoveryRewardConfig | None = None
    motor_constraints: MicroDuckEnlargedMotorConstraintConfig = field(
        default_factory=lambda: MicroDuckEnlargedMotorConstraintConfig(enabled=True)
    )


class MicroDuckEnlarged104RecoveryDomainRandomizationProvider(
    MicroDuckEnlarged102RecoveryDomainRandomizationProvider
):
    """V1.0.4 reset provider; reset semantics stay identical to V1.0.2."""


@registry.env("MicroDuckEnlarged104StandRecoveryFlat", sim_backend="mujoco")
class MicroDuckEnlarged104StandRecoveryFlatEnv(
    MicroDuckEnlarged102StandRecoveryFlatEnv
):
    """Prone/face-up recovery environment for the exact V1.0.4 model."""

    _cfg: MicroDuckEnlarged104StandRecoveryFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged104RecoveryDomainRandomizationProvider:
        return MicroDuckEnlarged104RecoveryDomainRandomizationProvider()
