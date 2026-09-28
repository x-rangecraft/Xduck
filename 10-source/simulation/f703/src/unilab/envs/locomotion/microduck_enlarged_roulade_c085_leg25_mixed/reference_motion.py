"""Task-owned short dynamic motion reference; never an actor input or controller."""

import hashlib
from pathlib import Path

import numpy as np


def motion_tracking_score(cfg, *, joint_pos, base_quat, height, target):
    """One [0,1] composite; joint/rotation radians, root height metres."""
    joint_error = np.mean(np.square(joint_pos - target[:, 7:]), axis=1)
    quat = base_quat / np.linalg.norm(base_quat, axis=1, keepdims=True)
    reference = target[:, 3:7] / np.linalg.norm(target[:, 3:7], axis=1, keepdims=True)
    angle_error = 2 * np.arccos(np.clip(np.abs(np.sum(quat * reference, axis=1)), 0, 1))
    height_error = height - target[:, 2]
    exponent = (
        joint_error / cfg.reference_joint_std_rad**2
        + np.square(angle_error / cfg.reference_rotation_std_rad)
        + np.square(height_error / cfg.reference_height_std_m)
    )
    return np.exp(-exponent), np.sqrt(joint_error), angle_error, height_error


class MotionReference:
    """Cold-load the frozen source matched to the demonstrated reset bank."""

    def __init__(self, cfg, bank, env_cfg):
        self.poses = None
        if cfg is None or cfg.scales.get("reference_motion", 0) == 0:
            return
        if cfg.scales["reference_motion"] < 0:
            raise ValueError("reference_motion requires a positive reward scale")
        widths = [
            cfg.reference_joint_std_rad,
            cfg.reference_rotation_std_rad,
            cfg.reference_height_std_m,
        ]
        if not np.isfinite(widths).all() or min(widths) <= 0:
            raise ValueError("reference tracking widths must be finite and positive")
        if bank.data is None or cfg.reference_trace_file is None:
            raise ValueError("reference_motion requires a trace and matching reset bank")
        root = Path(__file__).resolve().parents[5]
        path = Path(cfg.reference_trace_file)
        if not path.is_absolute():
            path = root / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(bank.data["source_sha256"]):
            raise ValueError("reference source hash does not match the reset bank")
        with np.load(path, allow_pickle=False) as z:
            poses = z["qpos"].copy()
        if poses.ndim != 3 or poses.shape[1:] != (32, 21) or not np.isfinite(poses).all():
            raise ValueError("invalid s1193 motion reference")
        if not np.allclose(np.linalg.norm(poses[:, :, 3:7], axis=2), 1, atol=1e-5):
            raise ValueError("invalid reference quaternions")
        steps = int(round(env_cfg.max_episode_seconds / env_cfg.ctrl_dt))
        frame_min = int(cfg.reference_frame_min)
        frame_max = int(cfg.reference_frame_max)
        if frame_min < 0 or frame_max <= frame_min or frame_max > len(poses):
            raise ValueError("invalid reference frame interval")
        frames = bank.data["frame"][bank.eligible]
        episodes = bank.data["episode"][bank.eligible]
        if not np.isin(episodes, [5, 15, 17, 23]).all():
            raise ValueError("reference resets must come from checked source episodes")
        if steps < 1 or not np.isclose(steps * env_cfg.ctrl_dt, env_cfg.max_episode_seconds):
            raise ValueError("reference episode duration must use whole control steps")
        if frames.min() < frame_min or frames.max() + steps > frame_max:
            raise ValueError(
                "reference reset and horizon must remain inside the configured interval"
            )
        self.poses = poses
        self.frame_min = frame_min
        self.frame_max = frame_max

    def target(self, info):
        # NpEnv.step increments steps AFTER update_state/reward: first action
        # after resetting to source frame F is compared with post-step qpos[F].
        index = info["demonstration_frame"] + info["steps"]
        frame_min = getattr(self, "frame_min", 0)
        frame_max = getattr(self, "frame_max", len(self.poses))
        valid = (index >= frame_min) & (index < frame_max)
        pose = self.poses[np.clip(index, frame_min, frame_max - 1), info["demonstration_episode"]]
        # No continuing reward for replaying a frozen final frame when a
        # diagnostic intentionally steps past the configured episode horizon.
        return pose, valid
