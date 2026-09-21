"""Native compiled MuJoCoUni extension boundary.

This package stays lower-level than ``mujoco_uni.runtime``. The stable public
entry point is ``mujoco_uni.batch_env``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _preload_mujoco_library() -> object | None:
    from mujoco_uni.mujoco_runtime import api as mujoco

    mujoco_dir = Path(mujoco.__file__).resolve().parent
    candidates = sorted(mujoco_dir.glob("libmujoco*")) + sorted(mujoco_dir.glob("mujoco.dll"))
    if not candidates:
        return None
    if os.name == "nt":
        # Let the extension's static dependency on mujoco.dll resolve from the
        # official mujoco package directory.
        os.add_dll_directory(str(mujoco_dir))
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(str(candidates[0]), mode=mode)


_MUJOCO_LIBRARY_HANDLE = _preload_mujoco_library()

try:
    from . import _batch_env as _batch_env
except ImportError as exc:  # pragma: no cover - optional local extension.
    _BATCH_IMPORT_ERROR = exc
    _batch_env = None  # type: ignore[assignment]
    NativeBatchEnvPool = None  # type: ignore[assignment]
    SUPPORTED_FIELDS: tuple[str, ...] = ()
    MUJOCO_BUILD_VERSION = None
else:
    _BATCH_IMPORT_ERROR = None
    NativeBatchEnvPool = _batch_env.BatchEnvPool
    SUPPORTED_FIELDS = tuple(_batch_env.SUPPORTED_FIELDS)
    MUJOCO_BUILD_VERSION = str(_batch_env.MUJOCO_BUILD_VERSION)


def batch_available() -> bool:
    return _batch_env is not None


def batch_import_error() -> ImportError | None:
    return _BATCH_IMPORT_ERROR


def require_native():
    if _batch_env is None:
        raise ImportError("MuJoCoUni native batch extension has not been built") from (
            _BATCH_IMPORT_ERROR
        )
    return _batch_env


__all__ = [
    "NativeBatchEnvPool",
    "MUJOCO_BUILD_VERSION",
    "SUPPORTED_FIELDS",
    "batch_available",
    "batch_import_error",
    "require_native",
]
