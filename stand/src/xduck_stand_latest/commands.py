"""Fixed zero command for the selected walk-to-stand policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xduck_walk_latest.commands import XDuckWalkCommand, XDuckWalkCommandCfg


@dataclass(kw_only=True)
class XDuckStandCommandCfg(XDuckWalkCommandCfg):
    def build(self, env):
        return XDuckStandCommand(self, env)


class XDuckStandCommand(XDuckWalkCommand):
    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids] = 0.0
