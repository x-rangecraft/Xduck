"""Dedicated ground-recovery policy task for the original Micro Duck."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.base import registry
from unilab.envs.locomotion.microduck.stand import MicroDuckStandCommands
from unilab.envs.locomotion.microduck.velstand import (
    MicroDuckVelStandFlatCfg,
    MicroDuckVelStandFlatEnv,
)


@registry.envcfg("MicroDuckStandRecoveryFlat")
@dataclass
class MicroDuckStandRecoveryFlatCfg(MicroDuckVelStandFlatCfg):
    """Zero-command recovery task kept separate from the walking policy.

    The full-collision model, prone/crouch reset implementation and recovery
    potential rewards are shared with VelStand.  This owner changes the
    contract: every command is zero and falling never uses the walking
    bootstrap termination, so its exported policy is safe to place in the
    runtime's dedicated ``stand`` slot.
    """

    commands: MicroDuckStandCommands = field(default_factory=MicroDuckStandCommands)
    fell_over_disable_after_steps: int = 0
    fallen_timeout_s: float = 8.0
    max_episode_seconds: float = 8.0


@registry.env("MicroDuckStandRecoveryFlat", sim_backend="mujoco")
class MicroDuckStandRecoveryFlatEnv(MicroDuckVelStandFlatEnv):
    """Original Micro Duck prone/face-up recovery environment."""

    _cfg: MicroDuckStandRecoveryFlatCfg
