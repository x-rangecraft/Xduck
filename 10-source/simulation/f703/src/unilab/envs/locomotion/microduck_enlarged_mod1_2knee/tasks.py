"""Independent DUCK_V1.01 tasks with two DM-J4340P knee motors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.actuators.dm_j4340p_v11 import clip_dm_j4340p_v11_torque
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.envs.locomotion.microduck_enlarged.walk import (
    NUM_MICRODUCK_ENLARGED_ACTIONS,
    MicroDuckEnlargedWalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.microduck_enlarged_mod1.tasks import (
    MicroDuckEnlargedMod1Asset,
    MicroDuckEnlargedMod1ControlConfig,
    MicroDuckEnlargedMod1StandFlatCfg,
    MicroDuckEnlargedMod1StandFlatEnv,
    MicroDuckEnlargedMod1StandRewardConfig,
    MicroDuckEnlargedMod1WalkFlatCfg,
    MicroDuckEnlargedMod1WalkFlatEnv,
    MicroDuckEnlargedMod1WalkRewardConfig,
)

KNEE_ACTUATOR_INDICES = np.asarray((3, 12), dtype=np.intp)
POSITION_ACTUATOR_INDICES = np.asarray(
    tuple(index for index in range(NUM_MICRODUCK_ENLARGED_ACTIONS) if index not in (3, 12)),
    dtype=np.int32,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_mod1_2knee" / "scene_flat.xml"
        )
    )


@dataclass
class MicroDuckEnlargedMod1TwoKneeControlConfig(MicroDuckEnlargedMod1ControlConfig):
    """Shared position-target gains; knee torque is computed every sim tick."""

    knee_kp: float = 30.0
    knee_kd: float = 0.5


class _DMJ4340PKneeControlMixin:
    """Keep the 12 J4310 position actuators and torque-control only the knees."""

    _cfg: MicroDuckEnlargedMod1WalkFlatCfg

    def __init__(
        self,
        cfg: MicroDuckEnlargedMod1WalkFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        if backend_type != "mujoco":
            raise ValueError("the independent 2-knee motor envelope is currently MuJoCo-only")
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        control = cfg.control_config
        if not isinstance(control, MicroDuckEnlargedMod1TwoKneeControlConfig):
            raise TypeError("2-knee tasks require MicroDuckEnlargedMod1TwoKneeControlConfig")
        position_gains = control.position_gains()
        position_gains["actuator_ids"] = POSITION_ACTUATOR_INDICES
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains=position_gains,
            **env_backend_kwargs(cfg),
        )
        LocomotionBaseEnv.__init__(self, cfg, backend, num_envs)
        if self._num_action != NUM_MICRODUCK_ENLARGED_ACTIONS:
            raise ValueError(
                f"2-knee Micro Duck requires {NUM_MICRODUCK_ENLARGED_ACTIONS} actuators, "
                f"got {self._num_action}"
            )
        self._reward_cfg = cfg.reward_config
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(cfg.asset.foot_site_offsets, dtype=get_global_dtype())
        self._control_step = 0
        self._enable_reward_log = True
        self._last_knee_torque_nm = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._backend.set_pre_step_control(self._pre_step_knee_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(MicroDuckEnlargedWalkDomainRandomizationProvider())

    def _pre_step_knee_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        native_ctrl = np.asarray(policy_ctrl).copy()
        position = backend.get_dof_pos()[:, KNEE_ACTUATOR_INDICES]
        velocity = backend.get_dof_vel()[:, KNEE_ACTUATOR_INDICES]
        control = self._cfg.control_config
        assert isinstance(control, MicroDuckEnlargedMod1TwoKneeControlConfig)
        requested = (
            control.knee_kp * (native_ctrl[:, KNEE_ACTUATOR_INDICES] - position)
            - control.knee_kd * velocity
        )
        torque = clip_dm_j4340p_v11_torque(requested, velocity)
        native_ctrl[:, KNEE_ACTUATOR_INDICES] = torque
        self._last_knee_torque_nm[:] = torque
        return native_ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state.info["knee_motor_torque_nm"] = self._last_knee_torque_nm.copy()
        knee_speed = self._backend.get_dof_vel()[:, KNEE_ACTUATOR_INDICES]
        state.info["knee_motor_speed_rpm"] = np.asarray(
            knee_speed * 60.0 / (2.0 * np.pi), dtype=get_global_dtype()
        )
        return super().update_state(state)


@registry.envcfg("MicroDuckEnlargedMod1TwoKneeWalkFlat")
@dataclass
class MicroDuckEnlargedMod1TwoKneeWalkFlatCfg(MicroDuckEnlargedMod1WalkFlatCfg):
    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlargedMod1Asset = field(default_factory=MicroDuckEnlargedMod1Asset)
    control_config: MicroDuckEnlargedMod1TwoKneeControlConfig = field(
        default_factory=MicroDuckEnlargedMod1TwoKneeControlConfig
    )
    reward_config: MicroDuckEnlargedMod1WalkRewardConfig | None = None


@registry.env("MicroDuckEnlargedMod1TwoKneeWalkFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1TwoKneeWalkFlatEnv(
    _DMJ4340PKneeControlMixin, MicroDuckEnlargedMod1WalkFlatEnv
):
    """Forward walking on the independent J4340P-knee mechanical identity."""

    _cfg: MicroDuckEnlargedMod1TwoKneeWalkFlatCfg


@registry.envcfg("MicroDuckEnlargedMod1TwoKneeStandFlat")
@dataclass
class MicroDuckEnlargedMod1TwoKneeStandFlatCfg(MicroDuckEnlargedMod1StandFlatCfg):
    scene: SceneCfg = field(default_factory=_scene)
    asset: MicroDuckEnlargedMod1Asset = field(default_factory=MicroDuckEnlargedMod1Asset)
    control_config: MicroDuckEnlargedMod1TwoKneeControlConfig = field(
        default_factory=MicroDuckEnlargedMod1TwoKneeControlConfig
    )
    reward_config: MicroDuckEnlargedMod1StandRewardConfig | None = None


@registry.env("MicroDuckEnlargedMod1TwoKneeStandFlat", sim_backend="mujoco")
class MicroDuckEnlargedMod1TwoKneeStandFlatEnv(
    _DMJ4340PKneeControlMixin, MicroDuckEnlargedMod1StandFlatEnv
):
    """HOME-pose standing on the independent J4340P-knee model."""

    _cfg: MicroDuckEnlargedMod1TwoKneeStandFlatCfg
