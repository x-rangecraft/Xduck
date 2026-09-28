"""Stage-one standing objective, preserving the forward-roll policy contract."""

from dataclasses import dataclass

import numpy as np

from unilab.envs.locomotion.microduck_enlarged_106.roulade import (
    MicroDuckEnlarged106RouladeRewardConfig,
)


@dataclass
class MicroDuckEnlargedRouladeC085Leg25MixedRewardConfig(MicroDuckEnlarged106RouladeRewardConfig):
    reference_trace_file: str | None = None
    reference_joint_std_rad: float = 0.35
    reference_rotation_std_rad: float = 0.5235987755982988
    reference_height_std_m: float = 0.06
    reference_frame_min: int = 15
    reference_frame_max: int = 100
    quiet_height_m: float = 0.255
    quiet_height_std_m: float = 0.05
    quiet_tilt_std_rad: float = 0.35
    quiet_base_speed_std_m_s: float = 0.15
    quiet_base_omega_std_rad_s: float = 0.6
    quiet_joint_speed_std_rad_s: float = 1.0
    quiet_foot_speed_std_m_s: float = 0.06
    quiet_foot_omega_std_rad_s: float = 0.6
    # Stage-local failure cost; zero retains the original historical objective.
    quiet_other_support_penalty: float = 0.0


def feet_are_loaded(force, threshold):
    """Net contact sensors provide world xyz, not normal/tangent components.

    Each sensor is restricted to one foot and the floor. The force norm
    therefore detects foot support independently of yaw and geom ordering.
    """
    return np.all(np.linalg.norm(force, axis=2) > threshold, axis=1)


def quiet_stand_score(
    cfg, *, height, tilt, base_vel, gyro, joint_vel, foot_vel, foot_omega, both_feet, other_support
):
    """One bounded composite: posture × support × motion settling.

    SI units; velocities use Euclidean norms and are frame invariant. A
    0.25 floor on settling/support gives exploration a posture gradient.
    Head angles are free. Head/trunk ground support receives no reward.
    Optional head/trunk support failure cost changes the lower bound to
    -quiet_other_support_penalty. Enable only for the post-roll hold stage;
    head support is legitimate during the roll itself.
    This is a training signal, not the strict behavior acceptance test.
    """
    posture = np.exp(
        -(((height - cfg.quiet_height_m) / cfg.quiet_height_std_m) ** 2)
        - (tilt / cfg.quiet_tilt_std_rad) ** 2
    )
    motion = (
        np.sum((base_vel / cfg.quiet_base_speed_std_m_s) ** 2, axis=1)
        + np.sum((gyro / cfg.quiet_base_omega_std_rad_s) ** 2, axis=1)
        + np.mean((joint_vel / cfg.quiet_joint_speed_std_rad_s) ** 2, axis=1)
        + np.mean(np.sum((foot_vel / cfg.quiet_foot_speed_std_m_s) ** 2, axis=2), axis=1)
        + np.mean(np.sum((foot_omega / cfg.quiet_foot_omega_std_rad_s) ** 2, axis=2), axis=1)
    )
    settling = np.exp(-motion)
    score = posture * (0.25 + 0.75 * settling) * (0.25 + 0.75 * both_feet)
    penalty = cfg.quiet_other_support_penalty
    if not np.isfinite(penalty) or not 0 <= penalty <= 1:
        raise ValueError("quiet_other_support_penalty must be in [0, 1]")
    return np.where(np.asarray(other_support, dtype=bool), -penalty, score), settling
