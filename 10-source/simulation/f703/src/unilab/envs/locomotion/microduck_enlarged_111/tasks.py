"""Standing and walking owners for the supplied V1.1.1 mechanics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import MicroDuckEnlargedAsset
from unilab.envs.locomotion.microduck_enlarged_107.tasks import (
    MicroDuckEnlarged107DomainRandConfig,
    MicroDuckEnlarged107DomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_110.tasks import (
    MicroDuckEnlarged110StandFlatCfg,
    MicroDuckEnlarged110StandFlatEnv,
    MicroDuckEnlarged110WalkFlatCfg,
    MicroDuckEnlarged110WalkFlatEnv,
)
from unilab.utils.rotation import np_quat_apply

MICRODUCK_ENLARGED_111_JOINT_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "neck_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "head_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_111" / "scene_flat.xml")
    )


@dataclass
class MicroDuckEnlarged111Asset(MicroDuckEnlargedAsset):
    """V1.1.1 body names and local foot-site coordinates."""

    base_name: str = "base_link"
    foot_body_names: tuple[str, str] = (
        "left_ankle_pitch_joint",
        "right_ankle_pitch",
    )
    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.053, -0.015445, -0.0291),
        (-0.053, -0.015513, -0.02921),
    )


@dataclass
class MicroDuckEnlarged111DomainRandConfig(MicroDuckEnlarged107DomainRandConfig):
    """V1.1.1 root body for inherited mechanical randomization and pushes."""

    push_body_name: str | None = "base_link"


class _V111DomainRandomizationProvider(MicroDuckEnlarged107DomainRandomizationProvider):
    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        return super()._compute_reset_obs(
            env,
            env_ids,
            info_updates,
            linvel,
            env._imu_vector_to_base(gyro),
            gravity,
            dof_pos,
            dof_vel,
        )


class _V111MechanicalContract:
    """Resolve mechanical joints by their supplied V1.1.1 names."""

    JOINT_NAMES = MICRODUCK_ENLARGED_111_JOINT_NAMES

    def __init__(self, *args, **kwargs):
        # The MJCF remains the authority for the physical IMU mounting. Cache
        # the rotation once: reward/actor vectors use the base frame as in V110.
        robot = (
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_111" / "microduck_enlarged_111.xml"
        )
        site = ET.parse(robot).find("./worldbody/body/site[@name='imu']")
        if site is None:
            raise ValueError("V1.1.1 requires an IMU site directly on base_link")
        quat = np.fromstring(site.get("quat", "1 0 0 0"), sep=" ")
        self._imu_to_base_quat = (quat / np.linalg.norm(quat)).astype(np.float32)
        super().__init__(*args, **kwargs)

    def _imu_vector_to_base(self, vector):
        return np_quat_apply(np.broadcast_to(self._imu_to_base_quat, (len(vector), 4)), vector)

    def get_gyro(self):
        return self._imu_vector_to_base(super().get_gyro())

    def get_local_linvel(self):
        return self._imu_vector_to_base(super().get_local_linvel())

    def _make_domain_randomization_provider(self):
        return _V111DomainRandomizationProvider()


@registry.envcfg("MicroDuckEnlarged111StandFlat")
@dataclass
class MicroDuckEnlarged111StandFlatCfg(MicroDuckEnlarged110StandFlatCfg):
    """V1.1.1 mechanics with the established V1.1.0 actuator contract."""

    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlarged111Asset = field(default_factory=MicroDuckEnlarged111Asset)
    domain_rand: MicroDuckEnlarged111DomainRandConfig = field(
        default_factory=MicroDuckEnlarged111DomainRandConfig
    )


@registry.env("MicroDuckEnlarged111StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged111StandFlatEnv(_V111MechanicalContract, MicroDuckEnlarged110StandFlatEnv):
    """Flat-ground standing on the supplied V1.1.1 mechanics."""

    _cfg: MicroDuckEnlarged111StandFlatCfg

    def _leg_pose_tracking(self, ctx) -> np.ndarray:
        """Keep measured legs near HOME while allowing load-bearing PD offsets.

        Normalized L1 retains a restoring signal away from HOME, unlike the
        inherited Gaussian. The action itself is deliberately not penalized:
        nonzero target offsets can be necessary to hold the measured pose.
        """
        indices = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
        error = ctx.dof_pos[:, indices] - ctx.default_angles[list(indices)]
        std = np.asarray(self._stand_reward_cfg.leg_pose_stds)
        if std.shape != (10,) or not np.all(np.isfinite(std) & (std > 0.0)):
            raise ValueError("leg_pose_stds must contain ten finite positive values")
        return np.asarray(1.0 - np.mean(np.abs(error) / std, axis=1), dtype=ctx.dof_pos.dtype)

@registry.envcfg("MicroDuckEnlarged111WalkFlat")
@dataclass
class MicroDuckEnlarged111WalkFlatCfg(MicroDuckEnlarged110WalkFlatCfg):
    """V1.1.1 mechanics with the established V1.1.0 actuator contract."""

    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlarged111Asset = field(default_factory=MicroDuckEnlarged111Asset)
    domain_rand: MicroDuckEnlarged111DomainRandConfig = field(
        default_factory=MicroDuckEnlarged111DomainRandConfig
    )


@registry.env("MicroDuckEnlarged111WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged111WalkFlatEnv(_V111MechanicalContract, MicroDuckEnlarged110WalkFlatEnv):
    """Flat-ground locomotion on the supplied V1.1.1 mechanics."""

    _cfg: MicroDuckEnlarged111WalkFlatCfg


__all__ = [
    "MICRODUCK_ENLARGED_111_JOINT_NAMES",
    "MicroDuckEnlarged111Asset",
    "MicroDuckEnlarged111DomainRandConfig",
    "MicroDuckEnlarged111StandFlatCfg",
    "MicroDuckEnlarged111StandFlatEnv",
    "MicroDuckEnlarged111WalkFlatCfg",
    "MicroDuckEnlarged111WalkFlatEnv",
]
