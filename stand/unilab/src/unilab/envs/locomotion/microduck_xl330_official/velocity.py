"""Official XL330 MicroDuck velocity task on UniLab's parallel MuJoCo backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.microduck_dm4310.velocity import (
    NUM_ACTIONS,
    POLICY_JOINT_NAMES,
    CurriculumConfig,
    MicroDuckDm4310VelocityCfg,
    MicroDuckDm4310VelocityEnv,
    MicroDuckDRProvider,
    VelocityRewardConfig,
    _stage,
)

from .bam import Xl330BamBatchModel

_WORKSPACE = Path(__file__).resolve().parents[6]
_OFFICIAL_SCENE = _WORKSPACE / "mjlab/src/mjlab_microduck/robot/microduck/scene_walk.xml"
_SENSOR_FRAGMENT = Path(__file__).with_name("sensors.xml")


@dataclass
class OfficialCurriculumConfig(CurriculumConfig):
    step_cap: int | None = None
    action_rate_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -0.1},
            {"step": 500 * 24, "value": -0.2},
            {"step": 750 * 24, "value": -0.4},
            {"step": 1000 * 24, "value": -0.6},
            {"step": 1250 * 24, "value": -0.8},
            {"step": 1500 * 24, "value": -1.0},
        ]
    )
    standing_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.02},
            {"step": 500 * 24, "value": 0.05},
            {"step": 750 * 24, "value": 0.10},
            {"step": 1000 * 24, "value": 0.15},
            {"step": 1500 * 24, "value": 0.20},
            {"step": 2000 * 24, "value": 0.25},
        ]
    )
    velocity_scale_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": 1.0}]
    )
    linear_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": float(np.sqrt(0.1))}]
    )
    angular_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": float(np.sqrt(0.5))}]
    )
    head_range_stages: list[dict] = field(
        default_factory=lambda: [
            {"step": 0, "ranges": [[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]]},
            {
                "step": 500 * 24,
                "ranges": [[-0.17, 0.17], [-0.17, 0.17], [-0.21, 0.21], [-0.047, 0.047]],
            },
            {
                "step": 1000 * 24,
                "ranges": [[-0.39, 0.39], [-0.39, 0.39], [-0.49, 0.49], [-0.11, 0.11]],
            },
            {
                "step": 1500 * 24,
                "ranges": [[-0.72, 0.72], [-0.72, 0.72], [-0.91, 0.91], [-0.20, 0.20]],
            },
            {
                "step": 2000 * 24,
                "ranges": [[-1.10, 1.10], [-1.10, 1.10], [-1.40, 1.40], [-0.31, 0.31]],
            },
        ]
    )
    head_bias_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 600 * 24, "value": 1.0},
            {"step": 1000 * 24, "value": 2.0},
            {"step": 1500 * 24, "value": 3.0},
        ]
    )
    com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 500 * 24, "value": 0.005},
            {"step": 1000 * 24, "value": 0.010},
            {"step": 1500 * 24, "value": 0.015},
        ]
    )
    head_com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 500 * 24, "value": 0.005},
            {"step": 1000 * 24, "value": 0.010},
        ]
    )
    push_stages: list[dict[str, float]] = field(default_factory=lambda: [{"step": 0, "value": 0.3}])

    def __post_init__(self) -> None:
        pass


@dataclass
class OfficialRewardConfig(VelocityRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "pose": 1.0,
            "upright": 2.0,
            "foot_slip": -0.1,
            "self_collisions": -1.0,
            "air_time": 3.0,
            "body_ang_vel": -0.05,
            "angular_momentum": -0.02,
            "track_linear_velocity": 2.0,
            "track_angular_velocity": 2.0,
            "action_rate_l2": -0.10,
            "foot_clearance": -2.0,
            "foot_swing_height": -0.25,
            "head_pose_tracking": 2.0,
            "body_pose_tracking": 0.0,
            "head_pose_bias": 0.0,
            "dof_pos_limits": -1.0,
        }
    )
    linear_velocity_std: float = float(np.sqrt(0.1))
    angular_velocity_std: float = float(np.sqrt(0.5))
    foot_target_height: float = 0.02
    head_bias_tau_s: float = 1.0


@registry.envcfg("MicroDuckXl330OfficialVelocityFlat")
@dataclass
class MicroDuckXl330OfficialVelocityCfg(MicroDuckDm4310VelocityCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(_OFFICIAL_SCENE),
            fragment_files=[str(_SENSOR_FRAGMENT)],
            visual_model_file=str(_OFFICIAL_SCENE),
        )
    )
    curriculum: OfficialCurriculumConfig = field(default_factory=OfficialCurriculumConfig)
    reward_config: OfficialRewardConfig = field(default_factory=OfficialRewardConfig)
    reset_clearance_m: float = 0.002


class OfficialMicroDuckDRProvider(MicroDuckDRProvider):
    def _finalize_reset_pose(self, env, plan):
        # MJLab's reset_root_state_uniform does not lift a sampled root pose
        # after the draw; adding clearance here changes the reset distribution.
        return plan

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        plan.qpos[:, 2] = np.random.uniform(0.12, 0.13, size=len(env_ids))
        return plan

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        env._refresh_foot_height()
        return super()._compute_reset_obs(
            env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
        )


@registry.env("MicroDuckXl330OfficialVelocityFlat", sim_backend="mujoco")
class MicroDuckXl330OfficialVelocityEnv(MicroDuckDm4310VelocityEnv):
    _cfg: MicroDuckXl330OfficialVelocityCfg
    _keyframe_name = "STAND"

    def _make_dr_provider(self):
        return OfficialMicroDuckDRProvider()

    def _sample_velocity_commands(self, count: int) -> np.ndarray:
        """Match MJLab's forward, standing, then turn-in-place precedence."""
        cfg = self._cfg.commands
        low = np.asarray(cfg.velocity_low, dtype=get_global_dtype())
        high = np.asarray(cfg.velocity_high, dtype=get_global_dtype())
        command = np.asarray(
            np.random.uniform(low, high, size=(count, 3)), dtype=get_global_dtype()
        )
        standing = np.random.uniform(size=count) < self._standing_fraction
        forward = np.random.uniform(size=count) < cfg.rel_forward_envs
        command[forward, 0] = np.maximum(np.abs(command[forward, 0]), cfg.forward_min_speed)
        command[forward, 1:] = 0.0
        turn = np.random.uniform(size=count) < cfg.rel_turn_in_place_envs
        command[turn, :2] = 0.0
        nturn = int(np.count_nonzero(turn))
        sign = np.where(np.random.uniform(size=nturn) < 0.5, -1.0, 1.0)
        max_yaw = max(abs(float(low[2])), abs(float(high[2])))
        command[turn, 2] = sign * np.random.uniform(0.4 * max_yaw, max_yaw, size=nturn)
        # The official turn bucket explicitly clears is_standing_env.
        command[standing & ~turn] = 0.0
        return command

    def _create_backend(self, cfg, num_envs: int, backend_type: str):
        armature = np.r_[np.zeros(6), np.full(NUM_ACTIONS, Xl330BamBatchModel.armature)]
        force_limit = 8.2 * Xl330BamBatchModel.kt / Xl330BamBatchModel.resistance
        return create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains={"kp": 1.0, "kd": 0.0},
            actuator_force_range=(-force_limit, force_limit),
            dof_armature=armature,
            dof_friction_solref=(-5.0e4, -2.0e2),
            dof_friction_solimp=(0.99, 0.9999, 0.001, 0.5, 2.0),
            **env_backend_kwargs(cfg),
        )

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if (
            not np.isfinite(cfg.reward_config.head_bias_tau_s)
            or cfg.reward_config.head_bias_tau_s <= 0
        ):
            raise ValueError("head_bias_tau_s must be positive and finite")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._foot_site_ids = self._backend.get_site_ids(("left_foot", "right_foot"))
        self._foot_height = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._episode_reward_sums = {
            name: np.zeros(num_envs, dtype=np.float64) for name in cfg.reward_config.scales
        }
        self._backend.enable_dof_force_cache()
        self._motor = self._build_official_motor(num_envs)
        self._motor.reset(np.arange(num_envs, dtype=np.int32))
        self._backend.set_pre_step_control(self._physics_step_action_delay)

    def _build_official_motor(self, num_envs: int):
        """Allow a physical-robot variant to retain its own external motor."""
        return Xl330BamBatchModel(
            num_envs,
            self._motor_dof_ids,
            self._backend.model.nv,
        )

    def _refresh_foot_height(self) -> None:
        self._foot_height[:] = self._backend.sample_site_clearance(
            self._foot_site_ids, radius=0.04, max_distance=1.0
        )

    def _critic_foot_height(self, foot_pos: np.ndarray, env_ids) -> np.ndarray:
        del foot_pos
        return self._foot_height[env_ids]

    def _reset_episode_history(self, env_ids) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        if hasattr(self, "_episode_reward_sums") and len(ids):
            log = self.state.info.setdefault("log", {})
            for name, values in self._episode_reward_sums.items():
                log[f"Episode_Reward/{name}"] = float(
                    np.mean(values[ids]) / self._cfg.max_episode_seconds
                )
                values[ids] = 0.0
        super()._reset_episode_history(env_ids)

    def _sample_substep_foot_state(self, *, reset_events: bool) -> None:
        if reset_events:
            self._first_contact.fill(False)
        foot_force = np.stack(
            [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_force], axis=1
        )
        foot_pos = np.stack(
            [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_pos], axis=1
        )
        foot_vel = np.stack(
            [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_vel], axis=1
        )
        contact = (
            np.concatenate(
                [self._backend.get_sensor_data(name) for name in self._cfg.sensor.feet_found],
                axis=1,
            )
            > 0.0
        )
        fresh = self._history_fresh
        transition = contact & (~self._previous_contact) & (~fresh[:, None])
        self._first_contact |= transition
        self._landed_air_time[:] = np.where(transition, self._air_time, self._landed_air_time)
        self._air_time = np.where(contact, 0.0, self._air_time + self._cfg.sim_dt)
        self._foot_force[:] = foot_force
        self._foot_pos[:] = foot_pos
        self._foot_vel[:] = foot_vel
        self._contact[:] = contact
        self._last_foot_pos[:] = foot_pos
        self._previous_contact[:] = contact
        self._history_fresh[:] = False

    def _physics_step_action_delay(self, backend, ctrl: np.ndarray) -> np.ndarray:
        self._sample_substep_foot_state(reset_events=self._delay_substep == 0)
        pos = backend.get_dof_pos()
        vel = backend.get_dof_vel()
        torque = self._motor.compute(ctrl, pos, vel, backend)
        self._delay_substep = (self._delay_substep + 1) % self._cfg.sim_substeps
        # The cold-path actuator transform sets kp=1, kd=0, so q+tau makes the
        # native position actuator apply exactly tau.
        return pos + torque

    def _update_foot_history(self) -> None:
        self._sample_substep_foot_state(reset_events=False)
        self._refresh_foot_height()
        # MJLab's first-contact query requires the foot to still be in contact
        # at the control-step boundary; a contact that began and ended entirely
        # inside the step must not score a landing.
        self._first_contact &= self._contact
        # MJLab's feet_swing_height class samples peak terrain clearance once
        # per control step, even though contact air-time tracks physics substeps.
        height = self._foot_height
        self._peak_foot_height = np.where(
            ~self._contact,
            np.maximum(self._peak_foot_height, height),
            self._peak_foot_height,
        )

    def _action_rate_cost(self, state):
        current = state.info.get("current_actions", np.zeros((self.num_envs, NUM_ACTIONS)))
        last = state.info.get("last_actions", np.zeros_like(current))
        return np.sum(np.square(current - last), axis=1)

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        cfg = self._cfg.reward_config
        cmd = state.info["commands"]
        joint_rel = dof_pos - self.default_angles
        leg_ids = np.asarray([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        # MJLab gates foot terms by ||v_xy_cmd|| + |yaw_cmd|, not a 3-D norm.
        moving = (np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])) > 0.01
        standing_std = np.asarray([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
        walking_std = np.asarray([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
        pose_std = np.where(moving[:, None], walking_std, standing_std)
        terms: dict[str, np.ndarray] = {}
        terms["pose"] = np.exp(-np.mean(np.square(joint_rel[:, leg_ids] / pose_std), axis=1))
        terms["upright"] = np.exp(-np.sum(np.square(upvector[:, :2]), axis=1) / cfg.upright_std**2)
        linear_error = np.c_[linvel[:, :2] - cmd[:, :2], linvel[:, 2]]
        terms["track_linear_velocity"] = np.exp(
            -np.sum(np.square(linear_error), axis=1) / cfg.linear_velocity_std**2
        )
        angular_error = np.c_[gyro[:, :2], gyro[:, 2] - cmd[:, 2]]
        terms["track_angular_velocity"] = np.exp(
            -np.sum(np.square(angular_error), axis=1) / cfg.angular_velocity_std**2
        )
        trunk_id = np.asarray([self._trunk_body_id], dtype=np.int32)
        world_ang_vel = self._backend.get_body_ang_vel_w(trunk_id)[:, 0]
        terms["body_ang_vel"] = np.sum(np.square(world_ang_vel[:, :2]), axis=1)
        angmom = self._backend.get_sensor_data(self._cfg.sensor.angular_momentum)
        terms["angular_momentum"] = np.sum(np.square(angmom), axis=1)
        terms["action_rate_l2"] = self._action_rate_cost(state)
        below = np.maximum(self._soft_joint_limits[:, 0] - dof_pos, 0.0)
        above = np.maximum(dof_pos - self._soft_joint_limits[:, 1], 0.0)
        terms["dof_pos_limits"] = np.sum(below + above, axis=1)

        head_error = joint_rel[:, 5:9] - cmd[:, 3:7]
        terms["head_pose_tracking"] = np.mean(np.exp(-np.square(head_error / 0.5)), axis=1)
        alpha = min(1.0, self._cfg.ctrl_dt / cfg.head_bias_tau_s)
        self._head_bias_ema = (1.0 - alpha) * self._head_bias_ema + alpha * head_error
        terms["head_pose_bias"] = -np.mean(np.abs(self._head_bias_ema), axis=1)
        terms["body_pose_tracking"] = np.zeros(self.num_envs)
        terms["self_collisions"] = self._backend.get_sensor_data("self_collision_found")[:, 0]

        foot_speed_xy_sq = np.sum(np.square(self._foot_vel[:, :, :2]), axis=2)
        terms["foot_slip"] = np.sum(foot_speed_xy_sq * self._contact, axis=1) * moving
        foot_speed_xy = np.sqrt(foot_speed_xy_sq)
        height_error = np.abs(self._foot_height - cfg.foot_target_height)
        terms["foot_clearance"] = np.sum(height_error * foot_speed_xy, axis=1) * moving
        peak_error = self._peak_foot_height / cfg.foot_target_height - 1.0
        terms["foot_swing_height"] = (
            np.sum(np.square(peak_error) * self._first_contact, axis=1) * moving
        )
        self._peak_foot_height[self._first_contact] = 0.0
        in_window = (
            (self._air_time > cfg.air_time_threshold_min)
            & (self._air_time < cfg.air_time_threshold_max)
            & (~self._contact)
        )
        terms["air_time"] = self._air_time_reward(in_window, moving)

        reward = np.zeros(self.num_envs, dtype=get_global_dtype())
        log = state.info.setdefault("log", {})
        for name, value in terms.items():
            weighted = cfg.scales[name] * value
            reward += weighted.astype(reward.dtype, copy=False)
            self._episode_reward_sums[name] += weighted * self._cfg.ctrl_dt
            log.pop(f"Episode_Reward/{name}", None)
        log["curriculum/standing_fraction"] = self._standing_fraction
        log["curriculum/action_rate_weight"] = cfg.scales["action_rate_l2"]
        return reward * self._cfg.ctrl_dt

    def _air_time_reward(self, in_window, moving):
        return np.sum(in_window, axis=1) * moving
