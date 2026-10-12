"""Cold-loaded mechanical handoff states for the walk-to-stop task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scaling import xml_model_fingerprint


@dataclass(frozen=True)
class WalkHandoffBank:
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> WalkHandoffBank:
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[5] / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing walk handoff bank {path}; run scripts/build_xduck_stop_bank.py first"
            )
        with np.load(path, allow_pickle=False) as data:
            bank = cls(
                data["qpos"].copy(),
                data["qvel"].copy(),
                data["actions"].copy(),
                json.loads(str(data["metadata"].item())),
            )
        n = len(bank.qpos)
        if (
            n == 0
            or bank.qpos.shape != (n, 21)
            or bank.qvel.shape != (n, 20)
            or bank.actions.shape != (n, 14)
        ):
            raise ValueError("Handoff bank requires nonempty qpos[N,21], qvel[N,20], actions[N,14]")
        if not all(np.isfinite(a).all() for a in (bank.qpos, bank.qvel, bank.actions)):
            raise ValueError("Handoff bank contains nonfinite state")
        if bank.metadata.get("version") != 1:
            raise ValueError("Unsupported handoff bank version")
        if not np.allclose(np.linalg.norm(bank.qpos[:, 3:7], axis=1), 1, atol=1e-4):
            raise ValueError("Handoff bank quaternions must be normalized")
        upright = 1 - 2 * np.sum(bank.qpos[:, 4:6] ** 2, axis=1)
        if np.any(upright < np.cos(np.deg2rad(30))) or np.any(bank.qpos[:, 2] < 0.20):
            raise ValueError("Only upright walking handoffs are supported, not ground recovery")
        return bank

    def validate_contract(self, cfg, default_angles: np.ndarray) -> None:
        meta = self.metadata
        if meta["model_fingerprint"] != xml_model_fingerprint(cfg.scene.model_file):
            raise ValueError("Handoff bank model changed; regenerate the bank")
        if tuple(meta["joint_names"]) != tuple(cfg.asset.policy_joint_names):
            raise ValueError("Handoff bank joint order mismatch")
        if not np.allclose(meta["default_angles"], default_angles, atol=1e-6):
            raise ValueError("Handoff bank action reference mismatch")
        if not np.isclose(meta["action_scale_rad"], cfg.policy_action_scale_rad):
            raise ValueError("Handoff bank action scale mismatch")
        for key in ["joint_kp", "joint_kd"]:
            value = getattr(cfg.control_config, key)
            if value is None or not np.allclose(meta[key], value):
                raise ValueError(f"Handoff bank {key} mismatch")
        if not np.isclose(meta["ctrl_dt"], cfg.ctrl_dt) or not np.isclose(
            meta["sim_dt"], cfg.sim_dt
        ):
            raise ValueError("Handoff bank time-step mismatch")


def canonicalize_handoffs(qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove planar origin/world yaw without changing the walking posture."""
    from unilab.utils.rotation import np_quat_apply, np_quat_mul, np_yaw_from_quat, np_yaw_to_quat

    q, v = qpos.copy(), qvel.copy()
    yaw_inverse = np_yaw_to_quat(-np_yaw_from_quat(q[:, 3:7]))
    q[:, :2] = 0.0
    q[:, 3:7] = np_quat_mul(yaw_inverse, q[:, 3:7])
    v[:, :3] = np_quat_apply(yaw_inverse, v[:, :3])
    return q, v
