"""Flat-ground roller-skating baseline for the original Pollen Micro Duck."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.walk import (
    MICRODUCK_JOINT_NAMES,
    MicroDuckAsset,
    MicroDuckCommands,
    MicroDuckRewardConfig,
    MicroDuckSensor,
    MicroDuckWalkDomainRandomizationProvider,
    MicroDuckWalkFlatCfg,
    MicroDuckWalkFlatEnv,
)
from unilab.utils.rotation import np_quat_apply

MICRODUCK_WHEEL_JOINT_NAMES = (
    "passive_LF_wheel",
    "passive_LR_wheel",
    "passive_RF_wheel",
    "passive_RR_wheel",
)


@dataclass
class MicroDuckRollerSensor(MicroDuckSensor):
    wheel_contacts: tuple[str, str, str, str] = (
        "left_front_wheel_contact",
        "left_rear_wheel_contact",
        "right_front_wheel_contact",
        "right_rear_wheel_contact",
    )


@dataclass
class MicroDuckRollerAsset(MicroDuckAsset):
    foot_body_names: tuple[str, str] = ("ankle_l_v1", "ankle_r_v1")
    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -0.0475, -0.0147),
        (0.0, -0.0475, -0.0144819),
    )


@dataclass
class MicroDuckRollerCommands(MicroDuckCommands):
    """Official roller semantics: negative=brake, zero=coast, positive=push."""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]
    )
    rel_standing_envs: float = 0.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    head_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 4, [0.0] * 4])
    body_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 6, [0.0] * 6])


@dataclass
class MicroDuckRollerRewardConfig(MicroDuckRewardConfig):
    wheel_radius: float = 0.0175
    wheel_speed_scale: float = 0.3
    braking_vel_std: float = 0.3
    skating_air_time_min: float = 0.15
    skating_air_time_max: float = 0.45
    progress_vel_ref: float = 0.2
    glide_stillness_std: float = 5.0
    double_support_penalty: float = 0.25
    forward_lean_target: float = 0.262
    forward_lean_std: float = 0.1


@registry.envcfg("MicroDuckRollerFlat")
@dataclass
class MicroDuckRollerFlatCfg(MicroDuckWalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_rollers" / "scene_flat.xml")
        )
    )
    sensor: MicroDuckRollerSensor = field(default_factory=MicroDuckRollerSensor)  # type: ignore[assignment]
    asset: MicroDuckRollerAsset = field(default_factory=MicroDuckRollerAsset)
    commands: MicroDuckRollerCommands = field(default_factory=MicroDuckRollerCommands)
    reward_config: MicroDuckRollerRewardConfig | None = None
    max_episode_seconds: float = 10.0


@registry.env("MicroDuckRollerFlat", sim_backend="mujoco")
class MicroDuckRollerFlatEnv(MicroDuckWalkFlatEnv):
    """Original Micro Duck with four passive roller-skate wheels.

    The four wheel hinges are physical simulation DoFs but are deliberately
    excluded from the 14D action and 61D actor observation contracts.
    """

    _cfg: MicroDuckRollerFlatCfg

    def __init__(
        self,
        cfg: MicroDuckRollerFlatCfg,
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
            **env_backend_kwargs(cfg),
        )

        # MicroDuckWalkFlatEnv would create a second backend, so enter its parent
        # initialization path through the same owned setup sequence.
        from unilab.envs.locomotion.common.base import LocomotionBaseEnv

        LocomotionBaseEnv.__init__(self, cfg, backend, num_envs)
        if self._num_action != len(MICRODUCK_JOINT_NAMES):
            raise ValueError(
                f"Micro Duck roller requires {len(MICRODUCK_JOINT_NAMES)} actuators, "
                f"got {self._num_action}"
            )

        self._actuated_pos_ids = backend.get_joint_dof_pos_indices(MICRODUCK_JOINT_NAMES)
        self._actuated_vel_ids = backend.get_joint_dof_vel_indices(MICRODUCK_JOINT_NAMES)
        self._wheel_vel_ids = backend.get_joint_dof_vel_indices(MICRODUCK_WHEEL_JOINT_NAMES)

        root_qpos_dim = self._init_qpos.shape[0] - backend.get_dof_pos().shape[1]
        self.default_angles = np.asarray(
            self._init_qpos[root_qpos_dim + self._actuated_pos_ids],
            dtype=get_global_dtype(),
        )

        self._reward_cfg = cfg.reward_config
        self._foot_body_ids = backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(cfg.asset.foot_site_offsets, dtype=get_global_dtype())
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(self._make_domain_randomization_provider())

    def _make_domain_randomization_provider(self) -> MicroDuckWalkDomainRandomizationProvider:
        return MicroDuckWalkDomainRandomizationProvider()

    @property
    def _roller_reward_cfg(self) -> MicroDuckRollerRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckRollerRewardConfig):
            raise TypeError("MicroDuckRollerFlat requires MicroDuckRollerRewardConfig")
        return cfg

    def get_dof_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_dof_pos()[:, self._actuated_pos_ids],
            dtype=get_global_dtype(),
        )

    def get_dof_vel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_dof_vel()[:, self._actuated_vel_ids],
            dtype=get_global_dtype(),
        )

    def get_wheel_vel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_dof_vel()[:, self._wheel_vel_ids],
            dtype=get_global_dtype(),
        )

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "wheel_speed": self._wheel_speed,
            "braking": self._braking,
            "skating_air_time": self._skating_air_time,
            "single_support": self._single_support,
            "glide": self._glide,
            "forward_lean": self._forward_lean,
            "upright": self._upright_tracking,
            "leg_pose": self._leg_pose_tracking,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
        }

    def _wheel_speed(self, ctx: RewardContext) -> np.ndarray:
        """Official core reward: positive command times forward wheel rotation."""
        wheel_omega = np.mean(self.get_wheel_vel(), axis=1)
        wheel_omega = np.nan_to_num(wheel_omega, nan=0.0, posinf=0.0, neginf=0.0)
        forward_omega = np.clip(wheel_omega, 0.0, None)
        command = np.clip(ctx.info["commands"][:, 0], 0.0, None)
        omega_scale = (
            self._roller_reward_cfg.wheel_speed_scale / self._roller_reward_cfg.wheel_radius
        )
        return np.asarray(command * np.tanh(forward_omega / omega_scale), dtype=get_global_dtype())

    def _forward_progress_gate(self, ctx: RewardContext) -> np.ndarray:
        ref = self._roller_reward_cfg.progress_vel_ref
        if ref <= 0.0:
            return np.ones((self._num_envs,), dtype=get_global_dtype())
        return np.asarray(
            np.clip(np.clip(ctx.linvel[:, 0], 0.0, None) / ref, 0.0, 1.0),
            dtype=get_global_dtype(),
        )

    def _braking(self, ctx: RewardContext) -> np.ndarray:
        command = np.clip(-ctx.info["commands"][:, 0], 0.0, None)
        velocity = np.clip(ctx.linvel[:, 0], 0.0, None)
        std = self._roller_reward_cfg.braking_vel_std
        return np.asarray(command * np.exp(-np.square(velocity / std)), dtype=get_global_dtype())

    def _skating_air_time(self, ctx: RewardContext) -> np.ndarray:
        air_time = np.asarray(ctx.info["current_air_time"], dtype=get_global_dtype())
        cfg = self._roller_reward_cfg
        in_window = (air_time > cfg.skating_air_time_min) & (air_time < cfg.skating_air_time_max)
        push = np.clip(ctx.info["commands"][:, 0], 0.0, None)
        return np.asarray(
            np.sum(in_window, axis=1) * push * self._forward_progress_gate(ctx),
            dtype=get_global_dtype(),
        )

    def _single_support(self, ctx: RewardContext) -> np.ndarray:
        contact_count = np.sum(np.asarray(ctx.info["foot_contact"], dtype=bool), axis=1)
        push = np.clip(ctx.info["commands"][:, 0], 0.0, None)
        single = contact_count == 1
        double = contact_count >= 2
        cfg = self._roller_reward_cfg
        return np.asarray(
            single * push * self._forward_progress_gate(ctx)
            - cfg.double_support_penalty * double * push,
            dtype=get_global_dtype(),
        )

    def _glide(self, ctx: RewardContext) -> np.ndarray:
        contact_count = np.sum(np.asarray(ctx.info["foot_contact"], dtype=bool), axis=1)
        leg_ids = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
        leg_speed_sq = np.sum(np.square(ctx.dof_vel[:, leg_ids]), axis=1)
        stillness = np.exp(-leg_speed_sq / self._roller_reward_cfg.glide_stillness_std**2)
        active = ctx.info["commands"][:, 0] >= 0.0
        return np.asarray(
            (contact_count == 1) * self._forward_progress_gate(ctx) * stillness * active,
            dtype=get_global_dtype(),
        )

    def _forward_lean(self, ctx: RewardContext) -> np.ndarray:
        gravity = ctx.gravity
        assert gravity is not None
        cfg = self._roller_reward_cfg
        push = np.clip(ctx.info["commands"][:, 0], 0.0, None)
        return np.asarray(
            push
            * np.exp(-np.square((gravity[:, 0] - cfg.forward_lean_target) / cfg.forward_lean_std)),
            dtype=get_global_dtype(),
        )

    def _update_foot_contact_history(self, info: dict[str, Any]) -> None:
        """Aggregate the front/rear wheel contacts into one state per blade."""
        wheel_contact = []
        for sensor_name in self._cfg.sensor.wheel_contacts:
            value = np.asarray(
                self._backend.get_sensor_data(sensor_name), dtype=get_global_dtype()
            ).reshape(self._num_envs, -1)
            if value.shape[1] < 3:
                raise RuntimeError(f"wheel contact sensor {sensor_name!r} must expose force")
            wheel_contact.append(
                np.linalg.norm(value[:, :3], axis=1) > self._reward_cfg.foot_contact_force_threshold
            )
        contact = np.stack(
            (
                wheel_contact[0] | wheel_contact[1],
                wheel_contact[2] | wheel_contact[3],
            ),
            axis=1,
        )

        zeros = np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        air_time = np.asarray(info.get("current_air_time", zeros), dtype=get_global_dtype())
        contact_time = np.asarray(info.get("current_contact_time", zeros), dtype=get_global_dtype())
        foot_height, foot_site_lin_vel = self._foot_kinematics()
        peak_height = np.asarray(
            info.get("current_peak_foot_height", zeros), dtype=get_global_dtype()
        )
        first_contact = contact & (air_time > 0.0)
        peak_height = np.where(~contact, np.maximum(peak_height, foot_height), peak_height)

        info["foot_contact"] = contact
        info["foot_height"] = foot_height
        info["foot_site_lin_vel_w"] = foot_site_lin_vel
        info["first_foot_contact"] = first_contact
        info["peak_foot_height_at_landing"] = np.where(first_contact, peak_height, 0.0).astype(
            get_global_dtype()
        )
        info["current_peak_foot_height"] = np.where(first_contact, 0.0, peak_height).astype(
            get_global_dtype()
        )
        info["current_air_time"] = np.where(contact, 0.0, air_time + self._cfg.ctrl_dt).astype(
            get_global_dtype()
        )
        info["current_contact_time"] = np.where(
            contact, contact_time + self._cfg.ctrl_dt, 0.0
        ).astype(get_global_dtype())
