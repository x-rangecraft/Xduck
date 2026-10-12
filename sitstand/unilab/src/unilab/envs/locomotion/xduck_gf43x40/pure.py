"""Command-conditioned PPO discovery; no motion files or imitation branches."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.base.scene import SceneCfg

from .sitstand import (
    ANKLE_IDS,
    LEG_IDS,
    SitStandCurriculumConfig,
    SitStandDRProvider,
    SitStandRewardConfig,
    XDuckSitStandCfg,
    XDuckSitStandEnv,
    _sitstand_scene,
)
from .velocity import XDuckGF43X40VelocityEnv

CASE_NAMES = ("stand_hold", "sit_hold", "rise", "descent")


def pure_scene():
    scene = _sitstand_scene()
    scene.model_file = str(Path(__file__).with_name("pure_scene.xml"))
    scene.visual_model_file = scene.model_file
    scene.fragment_files.append(str(Path(__file__).with_name("pure_sensors.xml")))
    return scene


@dataclass
class PureRewardConfig(SitStandRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            **{name: 0.0 for name in SitStandRewardConfig().scales},
            "ramp_height": 4.0,
            "goal_distance": -0.5,
            "pose_distance": -0.5,
            "task_progress": 2.0,
            "goal_composite": 5.0,
            "goal_stillness": 2.0,
            "transition_tilt": -0.2,
            "target_rate": -0.02,
            "torque_rate": 0.0,
            "descent_speed": -2.0,
            "rise_speed": 0.0,
            "impact": -0.1,
            "limit_proximity": -0.5,
            "forbidden_contact": -2.0,
        }
    )
    height_std: float = 0.035
    pose_std: float = 0.3
    max_descent_speed: float = 0.08
    max_rise_speed: float = 0.16
    target_rate_std_rad_s: float = 5.0
    torque_rate_std_nm_s: float = 100.0
    progress_discount: float = 0.995


def pure_curriculum():
    # All upgrades are explicit, validation-gated stages, never iteration ramps.
    cfg = SitStandCurriculumConfig()
    cfg.head_range_stages = [{"step": 0, "ranges": [[0.0, 0.0]] * 4}]
    for name in ("com_range_stages", "head_com_range_stages", "push_stages"):
        setattr(cfg, name, [{"step": 0, "value": 0.0}])
    return cfg


@registry.envcfg("XDuckGF43X40SitStandPureFlat")
@dataclass
class XDuckSitStandPureCfg(XDuckSitStandCfg):
    pure_policy_revision: int = 2
    warp_compiled_model_file: str | None = None
    scene: SceneCfg = field(default_factory=pure_scene)
    reward_config: PureRewardConfig = field(default_factory=PureRewardConfig)
    curriculum: SitStandCurriculumConfig = field(default_factory=pure_curriculum)
    motor_envelope_mode: str = "continuous_conservative"
    policy_action_scale_rad: float = 1.0
    whole_body_mass_dr: bool = True
    max_episode_seconds: float = 12.0
    posture_dwell_seconds: tuple[float, float] = (4.0, 6.0)
    posture_slew_seconds: float = 2.5
    dr_level: float = 0.0
    nominal_env_fraction: float = 0.25
    # Full-strength endpoints preserve the prior engineering acceptance range.
    trunk_com_max_m: float = 0.03135
    head_com_max_m: float = 0.0209
    reset_velocity_max: float = 0.015
    reset_joint_noise_rad: float = 0.004
    reset_clearance_m: float = 0.0005
    # Most resets require a transition; endpoint resets expose goal values to
    # the critic. Every surviving episode subsequently alternates commands.
    case_weights: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.4, 0.4])
    evaluation_case: int = -1
    completion_hold_seconds: float = 0.5
    completion_angular_speed_rad_s: float = 0.3
    completion_joint_speed_rad_s: float = 0.5
    transition_success_reward: float = 4.0
    terminal_failure_cost: float = 8.0
    fall_tilt_deg: float = 75.0
    fall_grace_seconds: float = 0.2


class PureDRProvider(SitStandDRProvider):
    def _get_qvel_limit(self, env):
        return env.cfg.reset_velocity_max

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        cases = (
            np.full(len(env_ids), env.cfg.evaluation_case, dtype=int)
            if env.cfg.evaluation_case >= 0
            else np.random.choice(4, len(env_ids), p=env._case_probabilities)
        )
        sitting = (cases == 1) | (cases == 2)
        commands = ((cases == 1) | (cases == 3)).astype(float)
        env._last_posture_command[env_ids] = commands
        env._case[env_ids] = cases
        env._posture_alpha[env_ids] = sitting
        env._completion_paid[env_ids] = False
        env._goal_hold_steps[env_ids] = 0
        env._fall_steps[env_ids] = 0
        env._torque_fresh[env_ids] = True
        target = np.broadcast_to(env.default_angles, (len(env_ids), 14)).copy()
        target[np.ix_(sitting, LEG_IDS)] = env._sit_legs
        plan.qpos[:, 7:] = target
        plan.qpos[:, 7:] += np.random.uniform(
            -env.cfg.reset_joint_noise_rad, env.cfg.reset_joint_noise_rad, target.shape
        )
        plan.qpos[:, 7:] = np.clip(plan.qpos[:, 7:], env._target_low, env._target_high)
        plan.qpos[:, 2] = np.where(sitting, env.cfg.sit_reset_height, env.cfg.stand_reset_height)
        env._reset_clearance.lift_to_clearance(plan.qpos, env.cfg.reset_clearance_m)
        env._previous_progress[env_ids] = env._posture_progress(plan.qpos[:, 2], commands)
        plan.info_updates["commands"][:, 0] = commands
        env._velocity_timer[env_ids] = np.asarray(
            np.random.uniform(*env.cfg.posture_dwell_seconds, len(env_ids)) / env.cfg.ctrl_dt,
            dtype=int,
        )
        # History is expressed in the SAME raw action units as the actor.
        initial_action = plan.qpos[:, 7:] - env.default_angles
        plan.info_updates["current_actions"][:] = initial_action
        plan.info_updates["last_actions"][:] = initial_action
        env._current_action_target[env_ids] = plan.qpos[:, 7:]
        env._previous_action_target[env_ids] = plan.qpos[:, 7:]
        mask = env._nominal_envs[env_ids]
        payload = plan.randomization
        if payload.dof_armature is not None:
            payload.dof_armature[:] = env._base_dof_armature + env.cfg.dr_level * (
                payload.dof_armature - env._base_dof_armature
            )
        for name, baseline in (
            ("body_ipos", env._base_body_ipos),
            ("body_mass", env._base_body_mass),
            ("body_inertia", env._base_body_inertia),
            ("dof_armature", env._base_dof_armature),
            ("dof_frictionloss", env._base_dof_frictionloss),
            ("geom_friction", env._base_geom_friction),
        ):
            value = getattr(payload, name)
            if value is not None:
                value[mask] = baseline
        return plan


@registry.env("XDuckGF43X40SitStandPureFlat", sim_backend="mujoco")
@registry.env("XDuckGF43X40SitStandPureFlat", sim_backend="mjwarp")
class XDuckSitStandPureEnv(XDuckSitStandEnv):
    def _create_backend(self, cfg, num_envs, backend_type):
        if backend_type == "mjwarp":
            if not cfg.warp_compiled_model_file:
                raise ValueError("GF Warp requires an exact compiled CPU model snapshot")
            from unilab.base.backend.mjwarp.xduck_gf import XDuckGF43X40WarpBackend

            return XDuckGF43X40WarpBackend(
                cfg.scene,
                num_envs,
                cfg.sim_dt,
                compiled_model_file=cfg.warp_compiled_model_file,
                physics_microsteps=1,
                base_name=cfg.asset.base_name,
                push_body_name=cfg.domain_rand.push_body_name,
            )
        return super()._create_backend(cfg, num_envs, backend_type)

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if not 0 <= cfg.dr_level <= 1 or not 0 <= cfg.nominal_env_fraction <= 1:
            raise ValueError("DR level and nominal fraction must be in [0,1]")
        weights = np.asarray(cfg.case_weights, dtype=float)
        if (
            weights.shape != (4,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("case_weights needs four nonnegative finite weights")
        if cfg.evaluation_case not in (-1, 0, 1, 2, 3):
            raise ValueError("invalid evaluation_case")
        if cfg.control_config.action_scale != 1 or cfg.policy_action_scale_rad != 1:
            raise ValueError("pure policy uses one radian per action unit")
        if cfg.control_config.simulate_action_latency or not cfg.clip_action_targets:
            raise ValueError("pure policy requires immediate clipped target mapping")
        if cfg.reward_config.stand_height <= cfg.reward_config.sit_height:
            raise ValueError("stand height must exceed sit height")
        for value in (
            cfg.completion_hold_seconds,
            cfg.completion_angular_speed_rad_s,
            cfg.completion_joint_speed_rad_s,
            cfg.fall_grace_seconds,
            cfg.reward_config.height_std,
            cfg.reward_config.pose_std,
            cfg.reward_config.target_rate_std_rad_s,
            cfg.reward_config.torque_rate_std_nm_s,
            cfg.reward_config.max_descent_speed,
            cfg.reward_config.max_rise_speed,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reward tolerances and time scales must be positive and finite")
        self._case_probabilities = weights / weights.sum()
        self._case = np.zeros(num_envs, dtype=int)
        self._last_posture_command = np.zeros(num_envs)
        self._previous_progress = np.zeros(num_envs)
        self._completion_paid = np.zeros(num_envs, dtype=bool)
        self._goal_hold_steps = np.zeros(num_envs, dtype=int)
        self._fall_steps = np.zeros(num_envs, dtype=int)
        self._torque_fresh = np.ones(num_envs, dtype=bool)
        self._previous_torque = np.zeros((num_envs, 14))
        self._nominal_envs = np.arange(num_envs) < int(num_envs * cfg.nominal_env_fraction)
        self._noise_ids = np.arange(num_envs)
        super().__init__(cfg, num_envs, backend_type)
        for name in ("completion_event", "failure_event"):
            self._episode_reward_sums[name] = np.zeros(num_envs)

    def _make_dr_provider(self):
        return PureDRProvider()

    def _sample_head_commands(self, count):
        return np.zeros((count, 4), dtype=self.default_angles.dtype)

    def _update_curriculum(self):
        # The stage is owned by the validation-gated training workflow.
        self._com_range = self.cfg.dr_level * self.cfg.trunk_com_max_m
        self._head_com_range = self.cfg.dr_level * self.cfg.head_com_max_m

    def _reset_episode_history(self, env_ids):
        super()._reset_episode_history(env_ids)
        ids = np.asarray(env_ids)
        level = self.cfg.dr_level
        for name, bounds in (
            ("_mass_inertia_scale", self.cfg.domain_rand.mass_inertia_scale_range),
            ("_foot_friction", self.cfg.domain_rand.foot_friction_range),
        ):
            low, high = 1 + level * (np.asarray(bounds) - 1)
            getattr(self, name)[ids] = np.random.uniform(low, high, len(ids))
        axis = np.random.normal(size=(len(ids), 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        angle = np.random.uniform(
            0, np.deg2rad(self.cfg.domain_rand.imu_max_angle_deg) * level, len(ids)
        )
        skew = np.zeros((len(ids), 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
        self._imu_rotation[ids] = (
            np.eye(3)
            + np.sin(angle)[:, None, None] * skew
            + (1 - np.cos(angle))[:, None, None] * (skew @ skew)
        )
        self._ankle_target_offset[ids] *= level
        nominal = ids[self._nominal_envs[ids]]
        self._mass_inertia_scale[nominal] = 1
        self._foot_friction[nominal] = 1
        self._imu_rotation[nominal] = np.eye(3)
        self._ankle_target_offset[nominal] = 0
        self._update_curriculum()

    def _obs_noise(self, data, scale):
        result = super()._obs_noise(data, scale * self.cfg.dr_level)
        mask = self._nominal_envs[self._noise_ids]
        result[mask] = data[mask]
        return result

    @property
    def obs_groups_spec(self):
        return {k: v + 1 for k, v in XDuckGF43X40VelocityEnv.obs_groups_spec.fget(self).items()}

    def _compute_obs(self, *args, env_ids=None, **kwargs):
        self._noise_ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        history = self._current_action_target[self._noise_ids] - self.default_angles
        if len(args) > 4:
            args = (*args[:4], history, *args[5:])
        else:
            kwargs["last_actions"] = history
        obs = super()._compute_obs(*args, env_ids=env_ids, **kwargs)
        progress = self._posture_alpha[self._noise_ids, None]
        # Observable task state, NOT a prerecorded trajectory or action prior.
        return {k: np.concatenate([v, progress.astype(v.dtype)], axis=1) for k, v in obs.items()}

    def _maybe_resample_commands(self, state):
        before = self._last_posture_command.copy()
        # Reset-plan commands already match their explicit case. The inherited
        # owner resamples at steps==0 and would silently replace them on step 1.
        if self.cfg.evaluation_case < 0:
            self._velocity_timer -= 1
            due = np.flatnonzero(self._velocity_timer <= 0)
            state.info["commands"][due, 0] = 1.0 - state.info["commands"][due, 0]
            self._velocity_timer[due] = np.asarray(
                np.random.uniform(*self.cfg.posture_dwell_seconds, len(due)) / self.cfg.ctrl_dt,
                dtype=int,
            )
        delta = self.cfg.ctrl_dt / self.cfg.posture_slew_seconds
        self._posture_alpha += np.clip(
            state.info["commands"][:, 0] - self._posture_alpha, -delta, delta
        )
        changed = before != state.info["commands"][:, 0]
        self._completion_paid[changed] = False
        self._goal_hold_steps[changed] = 0
        self._case[changed] = np.where(state.info["commands"][changed, 0] > 0.5, 3, 2)
        self._previous_progress[changed] = self._posture_progress(
            self._backend.get_base_pos()[changed, 2], state.info["commands"][changed, 0]
        )
        self._last_posture_command[:] = state.info["commands"][:, 0]

    def _posture_progress(self, height, command):
        cfg = self.cfg.reward_config
        standing = np.clip((height - cfg.sit_height) / (cfg.stand_height - cfg.sit_height), 0, 1)
        return np.where(command > 0.5, 1.0 - standing, standing)

    def _ground_contact(self, name):
        return self._backend.get_sensor_data(name).reshape(self.num_envs, -1).any(axis=1)

    def _failure(self, upvector):
        low = self._backend.get_base_pos()[:, 2] < 0.13
        bad = (upvector[:, 2] < np.cos(np.deg2rad(self.cfg.fall_tilt_deg))) | low
        self._fall_steps[:] = np.where(bad, self._fall_steps + 1, 0)
        grace = self._fall_steps * self.cfg.ctrl_dt >= self.cfg.fall_grace_seconds
        return (
            grace
            | self._ground_contact("pure_head_floor")
            | self._ground_contact("pure_trunk_floor")
        )

    def _terminated(self, upvector):
        del upvector
        return self._failed.copy()

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        del linvel
        cfg = self.cfg.reward_config
        command = state.info["commands"][:, 0]
        height = self._backend.get_base_pos()[:, 2]
        velocity = self._backend.get_base_lin_vel()
        speed = np.linalg.norm(velocity, axis=1)
        angular_speed = np.linalg.norm(gyro, axis=1)
        joint_speed = np.max(np.abs(self.get_dof_vel()), axis=1)
        travel = cfg.stand_height - cfg.sit_height
        goal_height = cfg.stand_height - command * travel
        ramp_height = cfg.stand_height - self._posture_alpha * travel
        ready = np.abs(command - self._posture_alpha) < 1e-6
        goal_pose = np.broadcast_to(self.default_angles, dof_pos.shape).copy()
        goal_pose[np.ix_(command > 0.5, LEG_IDS)] = self._sit_legs
        pose_error = np.sqrt(np.mean((dof_pos - goal_pose) ** 2, axis=1))
        tilt = np.arccos(np.clip(upvector[:, 2], -1, 1))
        upright = np.exp(-np.square(tilt / np.deg2rad(10)))
        feet = self._ground_contact("left_foot_found") & self._ground_contact("right_foot_found")
        hips = self._ground_contact("pure_left_hip_floor") | self._ground_contact(
            "pure_right_hip_floor"
        )
        support = feet & np.where(command > 0.5, hips, ~hips)
        composite = (
            np.exp(-np.square((height - goal_height) / 0.02))
            * np.exp(-np.square(pose_error / cfg.pose_std))
            * upright
            * ready
            * support
        )
        self._failed = self._failure(upvector)
        composite *= ~self._failed
        stable = (
            ready
            & support
            & ~self._failed
            & (np.abs(height - goal_height) < 0.015)
            & (tilt < np.deg2rad(10))
            & (speed < 0.04)
            & (angular_speed < self.cfg.completion_angular_speed_rad_s)
            & (joint_speed < self.cfg.completion_joint_speed_rad_s)
            & (pose_error < 0.1)
        )
        self._goal_hold_steps[:] = np.where(stable, self._goal_hold_steps + 1, 0)
        complete = self._goal_hold_steps * self.cfg.ctrl_dt >= self.cfg.completion_hold_seconds
        event = complete & ~self._completion_paid & np.isin(self._case, [2, 3])
        self._completion_paid[complete] = True
        accel = np.abs(velocity[:, 2] - self._prev_trunk_vz) / self.cfg.ctrl_dt
        accel[self._accel_history_fresh] = 0
        self._prev_trunk_vz[:] = velocity[:, 2]
        self._accel_history_fresh[:] = False
        torque = self._motor.torque
        torque_rate = (torque - self._previous_torque) / self.cfg.ctrl_dt
        torque_rate[self._torque_fresh] = 0
        self._previous_torque[:] = torque
        self._torque_fresh[:] = False
        margin = 0.075 * (self._joint_limits[:, 1] - self._joint_limits[:, 0])
        limit_cost = np.maximum((self._joint_limits[:, 0] + margin - dof_pos) / margin, 0)
        limit_cost += np.maximum((dof_pos - self._joint_limits[:, 1] + margin) / margin, 0)
        # A signed discounted potential difference cannot pay for repeatedly
        # bouncing up and down. Absorbing failures have zero potential.
        progress = self._posture_progress(height, command)
        progress[self._failed] = 0.0
        progress_reward = cfg.progress_discount * progress - self._previous_progress
        self._previous_progress[:] = progress
        terms = {
            "ramp_height": np.exp(-np.square((height - ramp_height) / cfg.height_std))
            * np.maximum(upvector[:, 2], 0.0)
            * ~self._failed,
            "goal_distance": np.minimum(np.abs(height - goal_height) / travel, 2),
            "pose_distance": np.mean(np.abs(dof_pos[:, LEG_IDS] - goal_pose[:, LEG_IDS]), axis=1),
            # The accumulator applies dt; the potential difference is already
            # per transition and must not be scaled by dt a second time.
            "task_progress": progress_reward / self.cfg.ctrl_dt,
            "goal_composite": composite,
            "goal_stillness": composite
            * np.exp(-np.square(speed / 0.075))
            * np.exp(-np.square(angular_speed / self.cfg.completion_angular_speed_rad_s))
            * np.exp(-np.square(joint_speed / self.cfg.completion_joint_speed_rad_s)),
            "transition_tilt": np.square(tilt / np.pi),
            "target_rate": np.mean(
                np.square(
                    (self._current_action_target - self._previous_action_target)
                    / (self.cfg.ctrl_dt * cfg.target_rate_std_rad_s)
                ),
                axis=1,
            ),
            "torque_rate": np.mean(np.square(torque_rate / cfg.torque_rate_std_nm_s), axis=1),
            "descent_speed": np.maximum(-velocity[:, 2] - cfg.max_descent_speed, 0)
            / cfg.max_descent_speed,
            "rise_speed": np.maximum(velocity[:, 2] - cfg.max_rise_speed, 0) / cfg.max_rise_speed,
            "impact": np.minimum(accel / 5, 10),
            "limit_proximity": np.mean(np.square(limit_cost), axis=1),
            "forbidden_contact": self._failed.astype(float),
        }
        reward = np.zeros(self.num_envs)
        log = state.info.setdefault("log", {})
        for name, value in terms.items():
            weighted = cfg.scales[name] * value * self.cfg.ctrl_dt
            reward += weighted
            self._episode_reward_sums[name] += weighted
            log[f"pure/term_{name}"] = float(np.mean(weighted))
        success_reward = self.cfg.transition_success_reward * event
        failure_reward = -self.cfg.terminal_failure_cost * self._failed
        reward += success_reward + failure_reward
        self._episode_reward_sums["completion_event"] += success_reward
        self._episode_reward_sums["failure_event"] += failure_reward
        self.last_success = complete & ~self._failed
        self.last_pose_error = pose_error
        self.last_height_error = height - goal_height
        self.last_tilt = tilt
        self.last_speed = speed
        raw_target = self.default_angles + state.info["current_actions"]
        log["pure/target_clip_fraction"] = float(
            np.mean((raw_target < self._target_low) | (raw_target > self._target_high))
        )
        log["pure/descent_speed_max"] = float(np.max(np.maximum(-velocity[:, 2], 0)))
        log["pure/goal_success_fraction"] = float(self.last_success.mean())
        log["pure/dr_level"] = self.cfg.dr_level
        actuator = self._motor.actuator
        drive_limit = actuator.drive_torque_limit(np.abs(self._backend.get_dof_vel()))
        brake_limit = min(
            actuator.parameters.physics["braking_limit"],
            actuator.parameters.physics["mechanical_torque_limit"],
        )
        limit = np.where(torque * self._backend.get_dof_vel() < 0, brake_limit, drive_limit)
        log["pure/control_sample_saturation_fraction"] = float(
            np.mean((np.abs(torque) >= limit - 1e-5) & (np.abs(torque) > 1e-6))
        )
        for case, name in enumerate(CASE_NAMES):
            mask = self._case == case
            log[f"pure/{name}_fraction"] = float(mask.mean())
            log[f"pure/{name}_stable_fraction"] = (
                float(self.last_success[mask].mean()) if mask.any() else 0.0
            )
        return reward.astype(dof_pos.dtype)
