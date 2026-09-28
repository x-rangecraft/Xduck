"""Flat-ground Micro Duck locomotion with the upstream 61D/14D policy contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    Sensor,
)
from unilab.envs.locomotion.common.commands import Commands, zero_small_xy_commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

MICRODUCK_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
NUM_MICRODUCK_ACTIONS = len(MICRODUCK_JOINT_NAMES)
MICRODUCK_ACTOR_OBS_DIM = 61


@dataclass
class MicroDuckSensor(Sensor):
    local_linvel: str = "imu_lin_vel"
    gyro: str = "imu_ang_vel"
    upvector: str = "imu_upvector"
    feet_contact: tuple[str, str] = ("left_foot_contact", "right_foot_contact")


@dataclass
class MicroDuckAsset:
    base_name: str = "trunk_base"
    ground: str = "floor"
    foot_body_names: tuple[str, str] = ("ankle_left", "ankle_right")
    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -0.0238146, -0.0140852),
        (0.0, -0.0238146, -0.0140852),
    )


@dataclass
class MicroDuckControlConfig(ControlConfigBase):
    # Upstream policies output joint-position offsets from HOME in radians.
    action_scale: float = 1.0


@dataclass
class MicroDuckNoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.001
    scale_joint_vel: float = 0.25
    scale_gyro: float = 0.03
    scale_gravity: float = 0.01


@dataclass
class MicroDuckCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.4, -0.3, -1.0], [0.4, 0.3, 1.0]]
    )
    # Upstream Velocity starts with 2% exact-zero commands, then widens this
    # share only after the gait exists.
    rel_standing_envs: float = 0.02
    rel_forward_envs: float = 0.20
    # Upstream develop dedicates 15% of samples to in-place turning because
    # independent uniform sampling almost never produces lin=0 with large yaw.
    rel_turn_in_place_envs: float = 0.15
    head_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.05, -0.05, -0.07, -0.015],
            [0.05, 0.05, 0.07, 0.015],
        ]
    )
    body_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.005, -0.005, -0.005, -0.05, -0.05, -0.05],
            [0.005, 0.005, 0.005, 0.05, 0.05, 0.05],
        ]
    )


@dataclass
class MicroDuckRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.1
    angular_tracking_sigma: float = 0.5
    base_height_target: float = 0.12
    min_base_height: float = 0.065
    max_tilt_deg: float = 70.0
    foot_contact_force_threshold: float = 0.1
    standing_leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.1, 0.05, 0.15, 0.15, 0.1, 0.1, 0.05, 0.15, 0.15, 0.1]
    )
    walking_leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.3, 0.05, 0.4, 0.4, 0.25, 0.3, 0.05, 0.4, 0.4, 0.25]
    )
    walking_command_threshold: float = 0.01
    upright_std: float = float(np.sqrt(0.05))
    head_pose_std: float = 0.5
    air_time_min: float = 0.125
    air_time_max: float = 0.3
    air_time_command_threshold: float = 0.01
    foot_height_target: float = 0.02
    foot_regularization_command_threshold: float = 0.01


@dataclass
class MicroDuckDomainRandConfig(DomainRandConfig):
    push_body_name: str | None = "trunk_base"


@registry.envcfg("MicroDuckWalkFlat")
@dataclass
class MicroDuckWalkFlatCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck" / "scene_flat.xml")
        )
    )
    sensor: MicroDuckSensor = field(default_factory=MicroDuckSensor)  # type: ignore[assignment]
    asset: MicroDuckAsset = field(default_factory=MicroDuckAsset)
    control_config: MicroDuckControlConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckControlConfig
    )
    noise_config: MicroDuckNoiseConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckNoiseConfig
    )
    commands: MicroDuckCommands = field(default_factory=MicroDuckCommands)
    reward_config: MicroDuckRewardConfig | None = None
    domain_rand: MicroDuckDomainRandConfig = field(default_factory=MicroDuckDomainRandConfig)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 10.0
    reset_base_qvel_limit: float = 0.05
    # Owner-level continuation hook.  UniLab's generic MLX trainer restores
    # optimizer state but creates a fresh environment, so resumed task
    # curricula need their already-consumed vector-step count explicitly.
    curriculum_step_offset: int = 0
    # Match the upstream velocity owner: commands change within an episode,
    # rather than remaining frozen from reset until timeout.
    twist_resample_s: tuple[float, float] = (3.0, 8.0)
    head_pose_resample_s: tuple[float, float] = (2.0, 5.0)
    body_pose_resample_s: tuple[float, float] = (2.0, 5.0)
    # These values are vector-environment steps (24 steps per PPO iteration),
    # exactly like the upstream mjlab curricula.
    standing_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "rel_standing_envs": 0.02},
            {"step": 500 * 24, "rel_standing_envs": 0.05},
            {"step": 750 * 24, "rel_standing_envs": 0.10},
            {"step": 1000 * 24, "rel_standing_envs": 0.15},
            {"step": 1500 * 24, "rel_standing_envs": 0.20},
            {"step": 2000 * 24, "rel_standing_envs": 0.25},
        ]
    )
    action_rate_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": -0.1},
            {"step": 500 * 24, "weight": -0.2},
            {"step": 750 * 24, "weight": -0.4},
            {"step": 1000 * 24, "weight": -0.6},
            {"step": 1250 * 24, "weight": -0.8},
            {"step": 1500 * 24, "weight": -1.0},
        ]
    )


class MicroDuckWalkDomainRandomizationProvider(LocomotionDRProvider):
    @staticmethod
    def _standing_fraction(env: Any) -> float:
        fraction = float(env.cfg.commands.rel_standing_envs)
        for stage in env.cfg.standing_curriculum:
            if env.step_counter < int(stage["step"]):
                break
            fraction = float(stage["rel_standing_envs"])
        return fraction

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        commands = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)),
            dtype=get_global_dtype(),
        )
        zero_small_xy_commands(commands, threshold=0.08)

        forward_fraction = float(env.cfg.commands.rel_forward_envs)
        max_forward = max(abs(float(low[0])), abs(float(high[0])))
        if forward_fraction > 0.0 and max_forward > 0.0:
            forward_mask = np.random.uniform(size=(num_reset,)) < min(forward_fraction, 1.0)
            commands[forward_mask, 0] = np.clip(
                np.abs(commands[forward_mask, 0]),
                min(0.3, max_forward),
                max_forward,
            )
            commands[forward_mask, 1:] = 0.0

        standing_fraction = self._standing_fraction(env)
        if standing_fraction > 0.0:
            standing_mask = np.random.uniform(size=(num_reset,)) < min(standing_fraction, 1.0)
            commands[standing_mask] = 0.0

        turn_fraction = float(env.cfg.commands.rel_turn_in_place_envs)
        if turn_fraction <= 0.0:
            return commands
        turn_mask = np.random.uniform(size=(num_reset,)) < min(turn_fraction, 1.0)
        num_turn = int(np.sum(turn_mask))
        if num_turn == 0:
            return commands
        yaw_limit = max(
            abs(float(env.cfg.commands.vel_limit[0][2])),
            abs(float(env.cfg.commands.vel_limit[1][2])),
        )
        signs = np.where(np.random.uniform(size=(num_turn,)) < 0.5, -1.0, 1.0)
        magnitudes = np.random.uniform(0.4 * yaw_limit, yaw_limit, size=(num_turn,))
        commands[turn_mask, :2] = 0.0
        commands[turn_mask, 2] = signs * magnitudes
        return np.asarray(commands, dtype=get_global_dtype())

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        head_limit = np.asarray(env.cfg.commands.head_limit, dtype=get_global_dtype())
        body_limit = np.asarray(env.cfg.commands.body_limit, dtype=get_global_dtype())
        return {
            "head_commands": np.asarray(
                np.random.uniform(head_limit[0], head_limit[1], size=(num_reset, 4)),
                dtype=get_global_dtype(),
            ),
            "body_commands": np.asarray(
                np.random.uniform(body_limit[0], body_limit[1], size=(num_reset, 6)),
                dtype=get_global_dtype(),
            ),
            "current_air_time": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "current_contact_time": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "current_peak_foot_height": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "peak_foot_height_at_landing": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "first_foot_contact": np.zeros((num_reset, 2), dtype=bool),
            "twist_resample_s": np.asarray(
                np.random.uniform(*env.cfg.twist_resample_s, size=num_reset),
                dtype=get_global_dtype(),
            ),
            "head_pose_resample_s": np.asarray(
                np.random.uniform(*env.cfg.head_pose_resample_s, size=num_reset),
                dtype=get_global_dtype(),
            ),
            "body_pose_resample_s": np.asarray(
                np.random.uniform(*env.cfg.body_pose_resample_s, size=num_reset),
                dtype=get_global_dtype(),
            ),
        }

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del gravity
        projected_gravity = env._projected_gravity()[env_ids]
        return env._compute_obs(
            info_updates,
            gyro,
            projected_gravity,
            dof_pos,
            dof_vel,
        )


@registry.env("MicroDuckWalkFlat", sim_backend="mujoco")
class MicroDuckWalkFlatEnv(LocomotionBaseEnv):
    """Micro Duck flat locomotion baseline using native MJCF position actuators.

    This first slice reproduces the upstream policy I/O and geometry contract.
    BAM voltage/friction dynamics, delays, and backlash are intentionally not
    claimed by this baseline and remain follow-up control work.
    """

    _cfg: MicroDuckWalkFlatCfg

    def __init__(self, cfg: MicroDuckWalkFlatCfg, num_envs: int = 1, backend_type: str = "mujoco"):
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
        super().__init__(cfg, backend, num_envs)
        if cfg.curriculum_step_offset < 0:
            raise ValueError("curriculum_step_offset must be non-negative")
        self.step_counter = int(cfg.curriculum_step_offset)
        if self._num_action != NUM_MICRODUCK_ACTIONS:
            raise ValueError(
                f"Micro Duck requires {NUM_MICRODUCK_ACTIONS} actuators, got {self._num_action}"
            )
        self._reward_cfg = cfg.reward_config
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(
            cfg.asset.foot_site_offsets,
            dtype=get_global_dtype(),
        )
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(self._make_domain_randomization_provider())

    def _make_domain_randomization_provider(self) -> LocomotionDRProvider:
        """Return the task-owned reset provider.

        VelStand reuses the complete walking environment but needs a few
        additional per-episode buffers.  Keeping provider selection here lets
        that task extend reset state without duplicating this constructor.
        """
        return MicroDuckWalkDomainRandomizationProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + projected gravity(3) + q-q_home(14) + dq(14)
        # + previous action(14) + [twist(3), head(4), body(6)] = 61.
        return {"obs": MICRODUCK_ACTOR_OBS_DIM}

    def _projected_gravity(self) -> np.ndarray:
        gravity = np.zeros((self._num_envs, 3), dtype=get_global_dtype())
        gravity[:, 2] = -1.0
        return np.asarray(
            np_quat_apply_inverse(self._backend.get_base_quat(), gravity),
            dtype=get_global_dtype(),
        )

    def _compute_obs(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise = self._cfg.noise_config
        joint_offset = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(joint_offset))
        twist = info["commands"]
        head = info["head_commands"]
        body = info["body_commands"]
        actor = np.concatenate(
            [
                self._obs_noise(gyro, noise.scale_gyro),
                self._obs_noise(projected_gravity, noise.scale_gravity),
                self._obs_noise(joint_offset, noise.scale_joint_angle),
                self._obs_noise(dof_vel, noise.scale_joint_vel),
                last_actions,
                twist,
                head,
                body,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if actor.shape[1] != MICRODUCK_ACTOR_OBS_DIM:
            raise RuntimeError(
                f"Micro Duck actor observation must be {MICRODUCK_ACTOR_OBS_DIM}D, "
                f"got {actor.shape[1]}"
            )
        return {"obs": actor}

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "track_linear_velocity": self._track_linear_velocity,
            "track_angular_velocity": self._track_angular_velocity,
            "upright": self._upright_tracking,
            "leg_pose": self._leg_pose_tracking,
            "head_pose": self._head_pose_tracking,
            "air_time": self._feet_air_time,
            "foot_clearance": self._foot_clearance,
            "foot_swing_height": self._foot_swing_height,
            "foot_slip": self._foot_slip,
            "ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
        }

    def _track_linear_velocity(self, ctx: RewardContext) -> np.ndarray:
        error = np.sum(np.square(ctx.info["commands"][:, :2] - ctx.linvel[:, :2]), axis=1)
        return np.asarray(
            np.exp(-error / self._reward_cfg.tracking_sigma),
            dtype=get_global_dtype(),
        )

    def _track_angular_velocity(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 2] - ctx.gyro[:, 2])
        return np.asarray(
            np.exp(-error / self._reward_cfg.angular_tracking_sigma),
            dtype=get_global_dtype(),
        )

    def _upright_tracking(self, ctx: RewardContext) -> np.ndarray:
        gravity = ctx.gravity
        assert gravity is not None
        error = np.sum(np.square(gravity[:, :2]), axis=1)
        return np.asarray(
            np.exp(-error / self._reward_cfg.upright_std**2),
            dtype=get_global_dtype(),
        )

    def _leg_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        leg_ids = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
        standing_std = np.asarray(
            self._reward_cfg.standing_leg_pose_stds,
            dtype=get_global_dtype(),
        )
        walking_std = np.asarray(
            self._reward_cfg.walking_leg_pose_stds,
            dtype=get_global_dtype(),
        )
        expected_shape = (leg_ids.size,)
        if standing_std.shape != expected_shape or walking_std.shape != expected_shape:
            raise ValueError("Micro Duck leg pose std lists must contain ten values")
        if np.any(standing_std <= 0.0) or np.any(walking_std <= 0.0):
            raise ValueError("Micro Duck leg pose std values must be positive")
        command = ctx.info["commands"]
        speed = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        std = np.where(
            (speed < self._reward_cfg.walking_command_threshold)[:, None],
            standing_std[None, :],
            walking_std[None, :],
        )
        error = ctx.dof_pos[:, leg_ids] - ctx.default_angles[leg_ids]
        return np.asarray(
            np.exp(-np.mean(np.square(error / std), axis=1)),
            dtype=get_global_dtype(),
        )

    def _head_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        head_ids = np.asarray((5, 6, 7, 8), dtype=np.intp)
        actual = ctx.dof_pos[:, head_ids] - ctx.default_angles[head_ids]
        error = actual - ctx.info["head_commands"]
        return np.asarray(
            np.mean(np.exp(-np.square(error / self._reward_cfg.head_pose_std)), axis=1),
            dtype=get_global_dtype(),
        )

    def _feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        air_time = np.asarray(ctx.info["current_air_time"], dtype=get_global_dtype())
        in_window = (air_time > self._reward_cfg.air_time_min) & (
            air_time < self._reward_cfg.air_time_max
        )
        command = ctx.info["commands"]
        command_norm = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        active = command_norm > self._reward_cfg.air_time_command_threshold
        return np.asarray(np.sum(in_window, axis=1) * active, dtype=get_global_dtype())

    def _foot_regularization_active(self, info: dict[str, Any]) -> np.ndarray:
        command = info["commands"]
        command_norm = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        return command_norm > self._reward_cfg.foot_regularization_command_threshold

    def _foot_clearance(self, ctx: RewardContext) -> np.ndarray:
        height = np.asarray(ctx.info["foot_height"], dtype=get_global_dtype())
        velocity = np.asarray(ctx.info["foot_site_lin_vel_w"], dtype=get_global_dtype())
        speed_xy = np.linalg.norm(velocity[:, :, :2], axis=2)
        error = np.abs(height - self._reward_cfg.foot_height_target)
        active = self._foot_regularization_active(ctx.info)
        return np.asarray(np.sum(error * speed_xy, axis=1) * active, dtype=get_global_dtype())

    def _foot_swing_height(self, ctx: RewardContext) -> np.ndarray:
        peak = np.asarray(ctx.info["peak_foot_height_at_landing"], dtype=get_global_dtype())
        first_contact = np.asarray(ctx.info["first_foot_contact"], dtype=bool)
        error = peak / self._reward_cfg.foot_height_target - 1.0
        active = self._foot_regularization_active(ctx.info)
        return np.asarray(
            np.sum(np.square(error) * first_contact, axis=1) * active,
            dtype=get_global_dtype(),
        )

    def _foot_slip(self, ctx: RewardContext) -> np.ndarray:
        contact = np.asarray(ctx.info["foot_contact"], dtype=bool)
        velocity = np.asarray(ctx.info["foot_site_lin_vel_w"], dtype=get_global_dtype())
        slip_sq = np.sum(np.square(velocity[:, :, :2]), axis=2)
        active = self._foot_regularization_active(ctx.info)
        return np.asarray(np.sum(slip_sq * contact, axis=1) * active, dtype=get_global_dtype())

    def _foot_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        body_pos, body_quat = self._backend.get_body_pose_w(self._foot_body_ids)
        body_lin_vel, body_ang_vel = self._backend.get_body_vel_w(self._foot_body_ids)
        offsets = np.broadcast_to(self._foot_site_offsets, (self._num_envs, 2, 3))
        rotated_offsets = np_quat_apply(
            body_quat.reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape(self._num_envs, 2, 3)
        site_pos = body_pos + rotated_offsets
        site_lin_vel = body_lin_vel + np.cross(body_ang_vel, rotated_offsets)
        return (
            np.asarray(site_pos[:, :, 2], dtype=get_global_dtype()),
            np.asarray(site_lin_vel, dtype=get_global_dtype()),
        )

    def _update_foot_contact_history(self, info: dict[str, Any]) -> None:
        """Update per-foot contact/air timers from task-owned contact sensors."""
        sensor_values = []
        for sensor_name in self._cfg.sensor.feet_contact:
            value = np.asarray(
                self._backend.get_sensor_data(sensor_name),
                dtype=get_global_dtype(),
            ).reshape(self._num_envs, -1)
            if value.shape[1] < 3:
                raise RuntimeError(
                    f"Micro Duck contact sensor {sensor_name!r} must expose a force vector"
                )
            sensor_values.append(np.linalg.norm(value[:, :3], axis=1))
        contact = np.stack(sensor_values, axis=1) > self._reward_cfg.foot_contact_force_threshold

        zeros = np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        air_time = np.asarray(info.get("current_air_time", zeros), dtype=get_global_dtype())
        contact_time = np.asarray(info.get("current_contact_time", zeros), dtype=get_global_dtype())
        if air_time.shape != zeros.shape or contact_time.shape != zeros.shape:
            raise RuntimeError("Micro Duck foot contact timers must have shape (num_envs, 2)")

        foot_height, foot_site_lin_vel = self._foot_kinematics()
        peak_height = np.asarray(
            info.get("current_peak_foot_height", zeros),
            dtype=get_global_dtype(),
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
        info["current_air_time"] = np.where(
            contact,
            0.0,
            air_time + self._cfg.ctrl_dt,
        ).astype(get_global_dtype())
        info["current_contact_time"] = np.where(
            contact,
            contact_time + self._cfg.ctrl_dt,
            0.0,
        ).astype(get_global_dtype())

    def _reward_context(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            gravity=gravity,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )

    def _reward_scales_for_step(self) -> dict[str, float]:
        scales = {name: float(value) for name, value in self._reward_cfg.scales.items()}
        action_rate = scales.get("action_rate", 0.0)
        for stage in self._cfg.action_rate_curriculum:
            if self.step_counter < int(stage["step"]):
                break
            action_rate = float(stage["weight"])
        if "action_rate" in scales:
            scales["action_rate"] = action_rate
        return scales

    @staticmethod
    def _decrement_resample_timer(info: dict[str, Any], name: str, dt: float) -> np.ndarray:
        timer = np.asarray(info[name], dtype=get_global_dtype()) - dt
        info[name] = timer
        return timer <= 0.0

    def _resample_commands_for_next_step(self, info: dict[str, Any]) -> None:
        """Advance upstream-style command timers after rewarding this transition."""
        provider = self._make_domain_randomization_provider()

        twist_due = self._decrement_resample_timer(info, "twist_resample_s", self._cfg.ctrl_dt)
        num_twist = int(np.sum(twist_due))
        if num_twist:
            info["commands"][twist_due] = provider._sample_commands(self, num_twist)
            info["twist_resample_s"][twist_due] = np.random.uniform(
                *self._cfg.twist_resample_s, size=num_twist
            )

        for timer_name, command_name, limits, interval in (
            (
                "head_pose_resample_s",
                "head_commands",
                self._cfg.commands.head_limit,
                self._cfg.head_pose_resample_s,
            ),
            (
                "body_pose_resample_s",
                "body_commands",
                self._cfg.commands.body_limit,
                self._cfg.body_pose_resample_s,
            ),
        ):
            due = self._decrement_resample_timer(info, timer_name, self._cfg.ctrl_dt)
            num_due = int(np.sum(due))
            if not num_due:
                continue
            bounds = np.asarray(limits, dtype=get_global_dtype())
            info[command_name][due] = np.random.uniform(
                bounds[0], bounds[1], size=(num_due, bounds.shape[1])
            )
            info[timer_name][due] = np.random.uniform(*interval, size=num_due)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_foot_contact_history(state.info)
        tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
        terminated = np.logical_or(
            tilt > np.deg2rad(self._reward_cfg.max_tilt_deg),
            self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height,
        )
        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_scales_for_step(),
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        self._resample_commands_for_next_step(state.info)
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)
