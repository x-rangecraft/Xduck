"""XL330 terms that preserve the existing Xduck policy and reward contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers.manager_base import ManagerTermBaseCfg
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


class raw_foot_contact_forces(SensorTermBase):
    """Existing critic stores six raw force channels, without log compression."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_names"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        names = cfg.params.get("sensor_names")
        if not isinstance(names, (tuple, list)) or len(names) != 2:
            raise ValueError("raw_foot_contact_forces requires two sensor names")
        self._view = self._bind(tuple(names))
        if self._view.dimensions != (3, 3):
            raise ValueError(f"expected two 3D force sensors, got {self._view.dimensions}")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = np.asarray(self._read(self._view, self.name), dtype=get_global_dtype())
        if values.shape != (env.num_envs, 6) or not np.isfinite(values).all():
            raise ValueError(f"raw foot forces must be finite ({env.num_envs}, 6)")
        return values


class normalized_angular_momentum(SensorTermBase):
    """Existing reward uses sum((root angular momentum / 0.25)^2)."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"sensor_name", "reference"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        name = cfg.params.get("sensor_name")
        reference = cfg.params.get("reference")
        if not isinstance(name, str) or not name:
            raise ValueError("angular momentum sensor_name must be nonempty")
        if isinstance(reference, bool) or not isinstance(reference, (float, int)):
            raise TypeError("angular momentum reference must be numeric")
        self._reference = float(reference)
        if not np.isfinite(self._reference) or self._reference <= 0:
            raise ValueError("angular momentum reference must be finite and positive")
        self._view = self._bind((name,))
        if self._view.dimensions != (3,):
            raise ValueError(f"angular momentum sensor must be 3D, got {self._view.dimensions}")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = np.asarray(self._read(self._view, self.name), dtype=get_global_dtype())
        if values.shape != (env.num_envs, 3) or not np.isfinite(values).all():
            raise ValueError(f"angular momentum must be finite ({env.num_envs}, 3)")
        return np.sum(np.square(values / self._reference), axis=1)
