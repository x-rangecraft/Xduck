"""XDuck locomotion and walk-to-stop tasks with the calibrated GF43X40 motor."""

from .stop import XDuckWalkStopCfg, XDuckWalkStopEnv
from .velocity import XDuckGF43X40VelocityCfg, XDuckGF43X40VelocityEnv

__all__ = [
    "XDuckGF43X40VelocityCfg",
    "XDuckGF43X40VelocityEnv",
    "XDuckWalkStopCfg",
    "XDuckWalkStopEnv",
]
