"""XDuck locomotion and walk-to-stop tasks with the calibrated GF43X40 motor."""

from .stop import XDuckWalkStopCfg, XDuckWalkStopEnv
from .sitstand import XDuckSitStandCfg, XDuckSitStandEnv
from .velocity import XDuckGF43X40VelocityCfg, XDuckGF43X40VelocityEnv

__all__ = [
    "XDuckGF43X40VelocityCfg",
    "XDuckGF43X40VelocityEnv",
    "XDuckWalkStopCfg",
    "XDuckWalkStopEnv",
    "XDuckSitStandCfg",
    "XDuckSitStandEnv",
]

from .reference import XDuckSitStandReferenceCfg, XDuckSitStandReferenceEnv

from .direct import XDuckSitStandDirectCfg, XDuckSitStandDirectEnv

from .pure import XDuckSitStandPureCfg, XDuckSitStandPureEnv
from .microduck_ppo import MicroduckPPOCfg, MicroduckPPOEnv
from .incremental import IncrementalCfg, IncrementalEnv
