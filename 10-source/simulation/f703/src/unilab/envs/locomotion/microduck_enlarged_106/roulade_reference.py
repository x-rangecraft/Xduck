"""Soft, time-independent pose prior owned by the V1.0.6 roulade task."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.walk import MICRODUCK_ENLARGED_JOINT_NAMES
from unilab.utils.rotation import np_quat_apply_inverse


@dataclass
class RouladeReferenceConfig:
    motion_file: str | None = None
    joint_std_rad: float = 0.6
    gravity_std: float = 0.75
    height_std_m: float = 0.1
    progress_std_rad: float = 0.8


class RouladeReference:
    """Match the best nearby reference pose, allowing retries and free timing.

    The prior is exp(-mean(normalized pose errors)) in [0, 1], activated
    after the first 20 degrees of net forward motion. Progress disambiguates
    an upright start from an upright finish. Existing completion rewards
    remain responsible for completing a roll; this bonus alone is not a
    success metric and could otherwise reward holding a partially rolled pose.
    """

    def __init__(self, cfg: RouladeReferenceConfig):
        self.cfg = cfg
        self.enabled = cfg.motion_file is not None
        for value in (cfg.joint_std_rad, cfg.gravity_std, cfg.height_std_m,
                      cfg.progress_std_rad):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reference pose widths must be finite and positive")
        if not self.enabled:
            return
        path = Path(cfg.motion_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[5] / path
        with np.load(path, allow_pickle=False) as motion:
            poses = np.asarray(motion['qpos'], dtype=get_global_dtype())
            if tuple(motion['joint_names']) != MICRODUCK_ENLARGED_JOINT_NAMES:
                raise ValueError('reference joint order does not match V1.0.6')
        if poses.ndim != 2 or poses.shape[1] != 21 or len(poses) < 2:
            raise ValueError('reference qpos must have shape (T, 21), T >= 2')
        if not np.isfinite(poses).all():
            raise ValueError('reference poses must be finite')
        if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-4):
            raise ValueError('reference quaternions must be normalized')
        self.joints = poses[:, 7:].copy()
        self.height = poses[:, 2].copy()
        self.gravity = np_quat_apply_inverse(
            poses[:, 3:7], np.tile([0., 0., -1.], (len(poses), 1))
        ).astype(get_global_dtype())
        self.progress = np.unwrap(np.arctan2(self.gravity[:, 0], -self.gravity[:, 2]))
        self.progress -= self.progress[0]
        if not (self.progress.max() > np.deg2rad(300)):
            raise ValueError('reference must contain a forward roll beyond 300 degrees')

    def score(self, ctx: RewardContext) -> np.ndarray:
        if not self.enabled:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        assert ctx.gravity is not None
        cfg = self.cfg
        progress = np.asarray(ctx.info['roulade_accum'])
        joint_error = np.mean(
            ((ctx.dof_pos[:, None, :] - self.joints[None, :, :]) / cfg.joint_std_rad)**2,
            axis=2,
        )
        gravity_error = np.mean(
            ((ctx.gravity[:, None, :] - self.gravity[None, :, :]) / cfg.gravity_std)**2,
            axis=2,
        )
        height_error = ((ctx.base_height[:, None] - self.height) / cfg.height_std_m)**2
        progress_error = ((progress[:, None] - self.progress) / cfg.progress_std_rad)**2
        error = (joint_error + gravity_error + height_error + progress_error) / 4
        score = np.exp(-np.min(error, axis=1))
        score *= np.clip(progress / np.deg2rad(20), 0, 1)
        ctx.info['roulade_reference_score'] = score.astype(get_global_dtype())
        return ctx.info['roulade_reference_score']
