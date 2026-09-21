"""Public MuJoCoUni batch executor API.

The implementation lives in ``mujoco_uni.runtime``. This module preserves the
stable ``mujoco_uni.batch_env`` import path used by UniLab and parity tests.
"""

from __future__ import annotations

from mujoco_uni.compiled import batch_available, batch_import_error
from mujoco_uni.runtime import SUPPORTED_FIELDS, BatchEnvPool

__all__ = [
    "BatchEnvPool",
    "SUPPORTED_FIELDS",
    "batch_available",
    "batch_import_error",
]
