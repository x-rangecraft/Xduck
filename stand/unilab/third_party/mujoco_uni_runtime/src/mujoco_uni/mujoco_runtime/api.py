"""MuJoCoUni-owned access point for the official ``mujoco`` module."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from mujoco_uni.mujoco_runtime.version_control import verify_mujoco_runtime

_mujoco: ModuleType = importlib.import_module("mujoco")
MUJOCO_RUNTIME_VERSION = verify_mujoco_runtime(_mujoco)
__version__ = MUJOCO_RUNTIME_VERSION
__file__ = getattr(_mujoco, "__file__", __file__)


def __getattr__(name: str) -> Any:
    return getattr(_mujoco, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_mujoco)))


__all__ = [
    "MUJOCO_RUNTIME_VERSION",
    "__version__",
    "__file__",
]
