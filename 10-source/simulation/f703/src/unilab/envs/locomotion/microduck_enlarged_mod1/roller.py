"""Roller-skating owner for the DUCK_V1.01 mechanical variant."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.envs.locomotion.microduck.roller import (
    MICRODUCK_WHEEL_JOINT_NAMES,
    MicroDuckRollerAsset,
    MicroDuckRollerCommands,
    MicroDuckRollerFlatCfg,
    MicroDuckRollerFlatEnv,
    MicroDuckRollerRewardConfig,
    MicroDuckRollerSensor,
)
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MICRODUCK_ENLARGED_JOINT_NAMES,
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_mod1.tasks import (
    MicroDuckEnlargedMod1ControlConfig,
)

MICRODUCK_ENLARGED_MOD1_WHEEL_RADIUS = 0.0175 * 2.09


def _roller_scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH
            / "robots"
            / "microduck_enlarged_mod1_rollers"
            / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlargedMod1RollerAsset(MicroDuckRollerAsset):
    foot_body_names: tuple[str, str] = ("ankle_left", "ankle_right")
    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -0.099275, -0.030723),
        (0.0, -0.099275, -0.030267171),
    )


@dataclass
class MicroDuckEnlargedMod1RollerCommands(MicroDuckRollerCommands):
    """Retain the official brake/coast/push command semantics in m/s."""


@dataclass
class MicroDuckEnlargedMod1RollerRewardConfig(MicroDuckRollerRewardConfig):
    base_height_target: float = 0.2508
    min_base_height: float = 0.135
    foot_contact_force_threshold: float = 1.0
    wheel_radius: float = MICRODUCK_ENLARGED_MOD1_WHEEL_RADIUS


_LEFT_LEG_IDS = np.asarray((0, 1, 2, 3, 4), dtype=np.intp)
_RIGHT_LEG_IDS = np.asarray((9, 10, 11, 12, 13), dtype=np.intp)


@registry.envcfg("MicroDuckEnlargedMod1RollerFlat")
@dataclass
class MicroDuckEnlargedMod1RollerFlatCfg(MicroDuckRollerFlatCfg):
    scene: SceneCfg = field(default_factory=_roller_scene)
    sensor: MicroDuckRollerSensor = field(default_factory=MicroDuckRollerSensor)  # type: ignore[assignment]
    asset: MicroDuckEnlargedMod1RollerAsset = field(
        default_factory=MicroDuckEnlargedMod1RollerAsset
    )
    control_config: MicroDuckEnlargedMod1ControlConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckEnlargedMod1ControlConfig
    )
    commands: MicroDuckEnlargedMod1RollerCommands = field(
        default_factory=MicroDuckEnlargedMod1RollerCommands
    )
    reward_config: MicroDuckEnlargedMod1RollerRewardConfig | None = None
    max_episode_seconds: float = 10.0


@registry.env("MicroDuckEnlargedMod1RollerFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1RollerFlatEnv(MicroDuckRollerFlatEnv):
    """V1.01 body with four 2.09x passive roller wheels.

    The actor retains the 61D observation and 14D action contracts. Passive
    wheel velocities remain task-internal reward signals.
    """

    _cfg: MicroDuckEnlargedMod1RollerFlatCfg

    def __init__(
        self,
        cfg: MicroDuckEnlargedMod1RollerFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains=cfg.control_config.position_gains(),
            **env_backend_kwargs(cfg),
        )
        LocomotionBaseEnv.__init__(self, cfg, backend, num_envs)
        if self._num_action != len(MICRODUCK_ENLARGED_JOINT_NAMES):
            raise ValueError(
                "V1.01 roller requires "
                f"{len(MICRODUCK_ENLARGED_JOINT_NAMES)} actuators, got {self._num_action}"
            )

        self._actuated_pos_ids = backend.get_joint_dof_pos_indices(
            MICRODUCK_ENLARGED_JOINT_NAMES
        )
        self._actuated_vel_ids = backend.get_joint_dof_vel_indices(
            MICRODUCK_ENLARGED_JOINT_NAMES
        )
        self._wheel_vel_ids = backend.get_joint_dof_vel_indices(
            MICRODUCK_WHEEL_JOINT_NAMES
        )
        root_qpos_dim = self._init_qpos.shape[0] - backend.get_dof_pos().shape[1]
        self.default_angles = np.asarray(
            self._init_qpos[root_qpos_dim + self._actuated_pos_ids],
            dtype=get_global_dtype(),
        )

        self._reward_cfg = cfg.reward_config
        self._foot_body_ids = backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(
            cfg.asset.foot_site_offsets, dtype=get_global_dtype()
        )
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(self._make_domain_randomization_provider())

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlargedWalkDomainRandomizationProvider:
        return MicroDuckEnlargedWalkDomainRandomizationProvider()

    def _init_reward_functions(self) -> None:
        """Classic swizzle recipe: mirrored legs with both roller blades grounded.

        The original roller owner intentionally remains the alternating-stride
        reproduction.  V1.01 owns this reward dispatch because its requested
        motion is the official, hardware-friendlier symmetric swizzle.
        """
        super()._init_reward_functions()
        for anti_swizzle in ("skating_air_time", "single_support", "glide"):
            self._reward_fns.pop(anti_swizzle, None)
        self._reward_fns.update(
            {
                "leg_symmetry": self._leg_symmetry,
                "grounded": self._grounded,
            }
        )

    def _leg_symmetry(self, ctx):
        """Penalize deviations from the robot's mirrored L/R joint convention."""
        pair_sum = ctx.dof_pos[:, _LEFT_LEG_IDS] + ctx.dof_pos[:, _RIGHT_LEG_IDS]
        return np.asarray(-np.mean(np.abs(pair_sum), axis=1), dtype=get_global_dtype())

    def _grounded(self, ctx):
        """Reward double support during forward pushing, matching classic swizzle."""
        contact_count = np.sum(np.asarray(ctx.info["foot_contact"], dtype=bool), axis=1)
        push = np.clip(ctx.info["commands"][:, 0], 0.0, None)
        return np.asarray((contact_count >= 2) * push, dtype=get_global_dtype())
