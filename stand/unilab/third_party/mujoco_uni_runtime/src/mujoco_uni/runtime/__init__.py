"""Python-facing MuJoCoUni runtime interface.

The runtime layer validates public Python inputs, owns the stable
``BatchEnvPool`` API, and keeps the C++ extension behind
``mujoco_uni.compiled``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_uni.compiled import MUJOCO_BUILD_VERSION, batch_available, batch_import_error

if TYPE_CHECKING:
    from .batch import BatchEnvPool


def available_backends() -> dict[str, bool]:
    return {"batch": bool(batch_available())}


def batch_diagnostics() -> dict[str, object]:
    detail = batch_import_error()
    available = batch_available()
    return {
        "mode": "batch",
        "available": available,
        "batch_available": available,
        "batch_import_error": None if detail is None else str(detail),
        "mujoco_build_version": MUJOCO_BUILD_VERSION,
    }


__all__ = [
    "BatchEnvPool",
    "SUPPORTED_FIELDS",
    "available_backends",
    "batch_available",
    "batch_diagnostics",
    "batch_import_error",
]


def __getattr__(name: str):
    if name in {"BatchEnvPool", "SUPPORTED_FIELDS"}:
        from . import batch

        return getattr(batch, name)
    raise AttributeError(name)
