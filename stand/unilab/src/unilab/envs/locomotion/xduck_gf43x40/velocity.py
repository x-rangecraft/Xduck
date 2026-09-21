"""Official MicroDuck gait task with XDuck V1.1.2 and GF43X40-10 motors.

The reward formulas and command sampler come from the UniLab XL330 official
owner. Geometry is loaded directly from Model/1.1.2/mjcf; the motor, physical
lengths, action units and 1-kHz transport are explicitly adapted for XDuck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.actuators.gf43x40 import GF43X40Parameters, GF43X40RobotMotor
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseEnv,
    PdControlConfig,
)
from unilab.envs.locomotion.microduck_dm4310.velocity import (
    NUM_ACTIONS,
    SCALE,
    Asset,
    Commands,
    MicroDuckDm4310VelocityEnv,
    MicroDuckDomainRandConfig,
    MicroDuckDRProvider,
    MicroDuckLatencyConfig,
    command_direction_progress,
)
from unilab.envs.locomotion.microduck_xl330_official.velocity import (
    MicroDuckXl330OfficialVelocityCfg,
    MicroDuckXl330OfficialVelocityEnv,
    OfficialCurriculumConfig,
    OfficialRewardConfig,
)
from unilab.utils.rotation import np_quat_apply_inverse

from .scaling import normalized_target_rate_l2, xml_model_fingerprint

_WORKSPACE = Path(__file__).resolve().parents[6]
_SOURCE_SCENE = _WORKSPACE / "Model" / "1.1.2" / "mjcf" / "scene.xml"
_SENSORS = Path(__file__).with_name("sensors.xml")
_TIME_SCALE = math.sqrt(SCALE)
_MASS_SCALE = 8.8933 / 0.73724318


def _xduck_commands() -> Commands:
    return Commands(
        velocity_low=[-0.4 * _TIME_SCALE, -0.3 * _TIME_SCALE, -1.0 / _TIME_SCALE],
        velocity_high=[0.4 * _TIME_SCALE, 0.3 * _TIME_SCALE, 1.0 / _TIME_SCALE],
        resampling_time_range=[3.0 * _TIME_SCALE, 8.0 * _TIME_SCALE],
        forward_min_speed=0.3 * _TIME_SCALE,
        head_resampling_time_range=[2.0 * _TIME_SCALE, 5.0 * _TIME_SCALE],
        body_resampling_time_range=[2.0 * _TIME_SCALE, 5.0 * _TIME_SCALE],
    )


def _xduck_noise() -> BaseNoiseConfig:
    return BaseNoiseConfig(
        level=1.0,
        scale_joint_angle=0.001,
        scale_joint_vel=0.25 / _TIME_SCALE,
        scale_gyro=0.03 / _TIME_SCALE,
        scale_gravity=0.01,
        scale_linvel=0.1 * _TIME_SCALE,
    )


def _xduck_reward_scales() -> dict[str, float]:
    scales = OfficialRewardConfig().scales.copy()
    # Gravity-similar velocity scales as sqrt(length), angular speed as its
    # inverse, and raw angular momentum as mass * length^1.5.
    scales["foot_slip"] /= SCALE
    scales["body_ang_vel"] *= SCALE
    scales["angular_momentum"] /= _MASS_SCALE**2 * SCALE**3
    scales["foot_clearance"] /= SCALE**1.5
    scales["command_progress"] = 0.0
    scales["both_airborne"] = 0.0
    scales["target_limit_l2"] = 0.0
    scales["target_limit_l1"] = 0.0
    scales["joint_limit_margin_l1"] = 0.0
    return scales


@dataclass
class XDuckRewardConfig(OfficialRewardConfig):
    scales: dict[str, float] = field(default_factory=_xduck_reward_scales)
    linear_velocity_std: float = math.sqrt(0.1) * _TIME_SCALE
    angular_velocity_std: float = math.sqrt(0.5) / _TIME_SCALE
    foot_target_height: float = 0.02 * SCALE
    air_time_threshold_min: float = 0.125 * _TIME_SCALE
    air_time_threshold_max: float = 0.300 * _TIME_SCALE
    progress_command_threshold: float = 0.05
    progress_upright_min: float = math.cos(math.radians(30.0))
    action_rate_units: str = "policy"
    action_rate_time_scale: float = 1.0
    action_rate_reference_dt: float = 0.02
    scale_model_fingerprint: str | None = None
    air_time_requires_support: bool = False
    airborne_cost_sampling: str = "control"
    target_limit_margin_rad: float = 0.0
    joint_limit_margin_rad: float = 0.0


@dataclass
class XDuckCurriculumConfig(OfficialCurriculumConfig):
    """Keep official stage boundaries, scaling only dimensional quantities."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for stage in self.linear_velocity_std_stages:
            stage["value"] *= _TIME_SCALE
        for stage in self.angular_velocity_std_stages:
            stage["value"] /= _TIME_SCALE
        for name in ("com_range_stages", "head_com_range_stages"):
            for stage in getattr(self, name):
                stage["value"] *= SCALE
        for stage in self.push_stages:
            stage["value"] *= _TIME_SCALE


@dataclass
class XDuckGFControlConfig(PdControlConfig):
    """Optional 14-axis MIT gains in policy joint order; scalars remain default."""

    joint_kp: list[float] | None = None
    joint_kd: list[float] | None = None


@dataclass
class GFMotorDomainRandConfig:
    enabled: bool = False
    torque_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    friction_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    response_time_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])


@dataclass
class GFDomainRandConfig(MicroDuckDomainRandConfig):
    motor: GFMotorDomainRandConfig = field(default_factory=GFMotorDomainRandConfig)


def _gf_domain_rand() -> GFDomainRandConfig:
    config = GFDomainRandConfig(
        randomize_base_mass=False,
        random_com=False,
        randomize_ground_friction=False,
        randomize_dof_armature=True,
        dof_armature_multiplier_range=[0.9, 1.1],
        push_robots=False,
        push_body_name="base_link",
    )
    # The GF model already owns friction; XML/DR friction would apply it twice.
    config.randomize_joint_friction = False
    config.motor.enabled = False
    return config


@dataclass
class GFCommunicationConfig:
    # Measured sparse late arrivals; uniform mode remains available for sweeps.
    jitter_mode: str = "measured_20260918"
    command_jitter_ms: list[int] = field(default_factory=lambda: [0, 0])
    feedback_jitter_ms: list[int] = field(default_factory=lambda: [0, 0])
    feedback_delay_ms: float = 0.0
    jitter_seed: int = 0


@registry.envcfg("XDuckGF43X40VelocityFlat")
@dataclass
class XDuckGF43X40VelocityCfg(MicroDuckXl330OfficialVelocityCfg):
    reference_keyframe: str | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(_SOURCE_SCENE),
            fragment_files=[str(_SENSORS)],
            visual_model_file=str(_SOURCE_SCENE),
        )
    )
    asset: Asset = field(
        default_factory=lambda: Asset(
            base_name="base_link",
            head_body_names=("neck_pitch", "head_pitch", "head_yaw", "head_roll"),
            policy_joint_names=tuple(
                name + "_joint"
                for name in (
                    "left_hip_yaw",
                    "left_hip_roll",
                    "left_hip_pitch",
                    "left_knee_pitch",
                    "left_ankle_pitch",
                    "neck_pitch",
                    "head_pitch",
                    "head_yaw",
                    "head_roll",
                    "right_hip_yaw",
                    "right_hip_roll",
                    "right_hip_pitch",
                    "right_knee_pitch",
                    "right_ankle_pitch",
                )
            ),
        )
    )
    # V1.1.2 limits relative to the published HOME, with 0.01 rad margin.
    action_offset_low: list[float] = field(
        default_factory=lambda: [
            -1.19,
            -0.11,
            -3.5848448167312594,
            -1.1949999999999998,
            -0.9001515100636341,
            -0.03999999999999998,
            -0.840934149601134,
            -1.99,
            -0.49,
            -0.49,
            -1.19,
            -2.395155183268741,
            -1.485,
            -1.879848489936366,
        ]
    )
    action_offset_high: list[float] = field(
        default_factory=lambda: [
            0.49,
            1.49,
            2.395155183268741,
            1.385,
            1.8798484899363659,
            1.54,
            1.539065850398866,
            1.99,
            0.49,
            1.19,
            0.19,
            2.5848448167312594,
            1.295,
            0.9001515100636339,
        ]
    )
    commands: Commands = field(default_factory=_xduck_commands)
    curriculum: XDuckCurriculumConfig = field(default_factory=XDuckCurriculumConfig)
    reward_config: XDuckRewardConfig = field(default_factory=XDuckRewardConfig)
    noise_config: BaseNoiseConfig = field(default_factory=_xduck_noise)
    control_config: XDuckGFControlConfig = field(
        default_factory=lambda: XDuckGFControlConfig(
            action_scale=1.0,
            simulate_action_latency=False,
            Kp=60.0,
            Kd=2.0,
        )
    )
    domain_rand: GFDomainRandConfig = field(default_factory=_gf_domain_rand)
    latency: MicroDuckLatencyConfig = field(
        default_factory=lambda: MicroDuckLatencyConfig(actuator_delay_ms=[0.0, 0.0])
    )
    sim_dt: float = 0.001
    communication: GFCommunicationConfig = field(default_factory=GFCommunicationConfig)
    # The official policy still explores with unit-std Gaussian actions.  Map
    # that dimensionless action to a feasible initial GF position error.
    policy_action_scale_rad: float = 0.12


class XDuckDRProvider(MicroDuckDRProvider):
    def _get_base_actuator_gains(self, env):
        cfg = env.cfg.control_config
        gains = cfg.position_gains()
        kp = cfg.joint_kp if cfg.joint_kp is not None else np.full(NUM_ACTIONS, gains["kp"])
        kd = cfg.joint_kd if cfg.joint_kd is not None else np.full(NUM_ACTIONS, gains["kd"])
        return np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)

    def _finalize_reset_pose(self, env, plan):
        # Respect the URDF geometry and joint limits when resetting.
        return super()._finalize_reset_pose(env, plan)

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        env._refresh_foot_height()
        return super()._compute_reset_obs(
            env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
        )


@registry.env("XDuckGF43X40VelocityFlat", sim_backend="mujoco")
class XDuckGF43X40VelocityEnv(MicroDuckXl330OfficialVelocityEnv):
    _cfg: XDuckGF43X40VelocityCfg
    _keyframe_name = "HOME"

    def _make_dr_provider(self):
        return XDuckDRProvider()

    def _sample_head_commands(self, count: int) -> np.ndarray:
        # Curriculum ranges use the previous physical direction; all V1.1.2
        # joint axes are reversed. Intersect before sampling so the narrow
        # neck limit does not accumulate commands at a clipped endpoint.
        ranges = -np.asarray(self._head_ranges)[:, ::-1]
        low = np.maximum(ranges[:, 0], self._cfg.action_offset_low[5:9])
        high = np.minimum(ranges[:, 1], self._cfg.action_offset_high[5:9])
        if np.any(low > high):
            raise ValueError("Head command curriculum lies outside V1.1.2 joint limits")
        return np.random.uniform(low, high, size=(count, 4)).astype(self.default_angles.dtype)

    def get_local_linvel(self) -> np.ndarray:
        # Commands/rewards use root-link velocity in the base frame. The CAD
        # IMU has a rotated frame and an offset, so its velocimeter is not this
        # quantity (even after rotating it, omega x offset must be removed).
        return np_quat_apply_inverse(
            self._backend.get_base_quat(), self._backend.get_base_lin_vel()
        )

    def get_gyro(self) -> np.ndarray:
        # Match projected gravity and the command axes before the existing
        # actor IMU misalignment/noise/delay pipeline is applied.
        body_ids = np.asarray([self._trunk_body_id], dtype=np.int32)
        return np_quat_apply_inverse(
            self._backend.get_base_quat(),
            self._backend.get_body_ang_vel_w(body_ids)[:, 0],
        )

    def _refresh_foot_height(self) -> None:
        self._foot_height[:] = self._backend.sample_site_clearance(
            self._foot_site_ids, radius=0.04 * SCALE, max_distance=1.0
        )

    def _create_backend(self, cfg, num_envs: int, backend_type: str):
        if backend_type != "mujoco":
            raise ValueError("The GF43X40 whole-robot adapter currently requires MuJoCo")
        parameters = GF43X40Parameters.from_bundle()
        mechanical_limit = float(
            parameters.physics.get("mechanical_torque_limit", parameters.registers["tmax"])
        )
        control_limit = (
            mechanical_limit + float(parameters.registers["pmax"])
            if "mechanical_torque_limit" in parameters.physics
            else 40.0
        )
        armature = np.r_[
            np.zeros(6),
            np.full(NUM_ACTIONS, parameters.physics["armature"]),
        ]
        return create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains={"kp": 1.0, "kd": 0.0},
            actuator_ctrl_range=(-control_limit, control_limit),
            actuator_force_range=(-mechanical_limit, mechanical_limit),
            dof_armature=armature,
            dof_damping=0.0,
            dof_frictionloss=0.0,
            **env_backend_kwargs(cfg),
        )

    def _build_motor(self, cfg, num_envs: int):
        gains = cfg.control_config.position_gains()
        kp = cfg.control_config.joint_kp
        kd = cfg.control_config.joint_kd
        delay_lo_ms, delay_hi_ms = cfg.latency.actuator_delay_ms
        if delay_lo_ms != delay_hi_ms:
            raise ValueError("GF transport currently requires a fixed actuator delay")
        return GF43X40RobotMotor(
            num_envs,
            NUM_ACTIONS,
            kp=kp if kp is not None else float(gains["kp"]),
            kd=kd if kd is not None else float(gains["kd"]),
            command_delay_s=float(delay_lo_ms) / 1000.0,
            feedback_delay_s=cfg.communication.feedback_delay_ms / 1000.0,
            command_jitter_ms=cfg.communication.command_jitter_ms,
            feedback_jitter_ms=cfg.communication.feedback_jitter_ms,
            jitter_seed=cfg.communication.jitter_seed,
            jitter_mode=cfg.communication.jitter_mode,
            motor_dr_ranges=(
                {
                    "torque_scale_multiplier": tuple(cfg.domain_rand.motor.torque_scale_range),
                    "friction_scale_multiplier": tuple(cfg.domain_rand.motor.friction_scale_range),
                    "response_time_multiplier": tuple(cfg.domain_rand.motor.response_time_scale_range),
                }
                if cfg.domain_rand.motor.enabled else None
            ),
        )

    def _build_official_motor(self, num_envs: int):
        del num_envs
        return self._motor

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config.airborne_cost_sampling not in ("control", "substep"):
            raise ValueError("airborne_cost_sampling must be control or substep")
        self._airborne_substep_sum = np.zeros(num_envs, dtype=np.float64)
        self._airborne_substep_count = 0
        self._joint_margin_substep_min = np.full((num_envs, NUM_ACTIONS), np.inf)
        self._joint_near_substep_sum = np.zeros((num_envs, NUM_ACTIONS))
        self._target_limit_cost = np.zeros(num_envs, dtype=np.float64)
        self._target_limit_l1_cost = np.zeros(num_envs, dtype=np.float64)
        for margin in (cfg.reward_config.target_limit_margin_rad, cfg.reward_config.joint_limit_margin_rad):
            if not math.isfinite(margin) or margin < 0:
                raise ValueError("Joint/target limit margins must be finite and nonnegative")
        if cfg.reference_keyframe is not None:
            self._keyframe_name = cfg.reference_keyframe
        fingerprint = cfg.reward_config.scale_model_fingerprint
        if fingerprint is not None and xml_model_fingerprint(cfg.scene.model_file) != fingerprint:
            raise ValueError("XDuck model changed; regenerate the scale-adapted profile")
        if cfg.reward_config.action_rate_units not in ("policy", "physical_target_rate"):
            raise ValueError("action_rate_units must be policy or physical_target_rate")
        for value in (
            cfg.reward_config.action_rate_time_scale,
            cfg.reward_config.action_rate_reference_dt,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("action-rate time scales must be positive and finite")
        self._current_action_target = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float64)
        self._previous_action_target = np.zeros_like(self._current_action_target)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        if np.any(2 * cfg.reward_config.target_limit_margin_rad >= self._target_high - self._target_low):
            raise ValueError("Target limit margin leaves no interior target range")
        if np.any(2 * cfg.reward_config.joint_limit_margin_rad >= self._joint_limits[:, 1] - self._joint_limits[:, 0]):
            raise ValueError("Joint limit margin leaves no interior joint range")
        self._current_action_target[:] = self.default_angles
        self._previous_action_target[:] = self.default_angles
        # The GF owner has no XL330/DM encoder-bias randomization.
        self._encoder_bias.fill(0.0)

    def _reset_episode_history(self, env_ids):
        super()._reset_episode_history(env_ids)
        self._target_limit_cost[env_ids] = 0.0
        self._target_limit_l1_cost[env_ids] = 0.0
        self._current_action_target[env_ids] = self.default_angles
        self._previous_action_target[env_ids] = self.default_angles

    def _action_rate_cost(self, state):
        cfg = self._cfg.reward_config
        if cfg.action_rate_units == "policy":
            return super()._action_rate_cost(state)
        return normalized_target_rate_l2(
            self._current_action_target,
            self._previous_action_target,
            ctrl_dt=self._cfg.ctrl_dt,
            reference_dt=cfg.action_rate_reference_dt,
            time_scale=cfg.action_rate_time_scale,
        )

    def apply_action(self, actions, state):
        scale = float(self._cfg.policy_action_scale_rad)
        if not 0.0 < scale <= 1.0:
            raise ValueError("policy_action_scale_rad must be in (0, 1]")
        # Keep the learner's raw action in history, as on the XL330 owner.
        # Only the motor position target uses the robot-specific radian scale.
        raw_target = LocomotionBaseEnv.apply_action(self, actions, state)
        target = self.default_angles + (raw_target - self.default_angles) * scale
        margin = self._cfg.reward_config.target_limit_margin_rad
        # Shaping begins before the existing clipping boundary. The actual
        # actuator target still uses the original bounds below.
        residual = target - np.clip(target, self._target_low + margin, self._target_high - margin)
        self._target_limit_cost[:] = np.sum(np.square(residual / scale), axis=1)
        self._target_limit_l1_cost[:] = np.sum(np.abs(residual / scale), axis=1)
        if self._cfg.clip_action_targets:
            target = np.clip(target, self._target_low, self._target_high)
        self._previous_action_target[:] = self._current_action_target
        self._current_action_target[:] = target
        return target

    def _sample_substep_foot_state(self, *, reset_events):
        super()._sample_substep_foot_state(reset_events=reset_events)
        if reset_events:
            self._airborne_substep_sum.fill(0.0)
            self._airborne_substep_count = 0
            self._joint_margin_substep_min.fill(np.inf)
            self._joint_near_substep_sum.fill(0.0)
        else:
            # Skip t=0; accumulate the results of the 20 physics intervals,
            # including the final sample taken by _update_foot_history.
            self._airborne_substep_sum += ~np.any(self._contact, axis=1)
            self._airborne_substep_count += 1
            q = self.get_dof_pos()
            margin = np.minimum(q - self._joint_limits[:, 0], self._joint_limits[:, 1] - q)
            np.minimum(self._joint_margin_substep_min, margin, out=self._joint_margin_substep_min)
            self._joint_near_substep_sum += margin < 0.01

    def _airborne_fraction(self):
        if self._airborne_substep_count == 0:
            return np.zeros_like(self._airborne_substep_sum)
        return self._airborne_substep_sum / self._airborne_substep_count

    def _air_time_reward(self, in_window, moving):
        reward = super()._air_time_reward(in_window, moving)
        if self._cfg.reward_config.air_time_requires_support:
            reward = reward * (np.sum(self._contact, axis=1) == 1)
        return reward

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        reward = super()._compute_reward(state, linvel, gyro, upvector, dof_pos)
        cfg = self._cfg.reward_config
        target_l1_weight = float(cfg.scales.get("target_limit_l1", 0.0))
        if target_l1_weight:
            weighted = target_l1_weight * self._target_limit_l1_cost
            reward += (weighted * self._cfg.ctrl_dt).astype(reward.dtype, copy=False)
            self._episode_reward_sums["target_limit_l1"] += weighted * self._cfg.ctrl_dt
            state.info.setdefault("log", {}).pop("Episode_Reward/target_limit_l1", None)
        joint_margin_weight = float(cfg.scales.get("joint_limit_margin_l1", 0.0))
        if joint_margin_weight:
            margin = cfg.joint_limit_margin_rad
            violation = np.maximum(self._joint_limits[:, 0] + margin - dof_pos, 0.0)
            violation += np.maximum(dof_pos - self._joint_limits[:, 1] + margin, 0.0)
            cost = np.sum(violation / self._cfg.policy_action_scale_rad, axis=1)
            weighted = joint_margin_weight * cost
            reward += (weighted * self._cfg.ctrl_dt).astype(reward.dtype, copy=False)
            self._episode_reward_sums["joint_limit_margin_l1"] += weighted * self._cfg.ctrl_dt
            state.info.setdefault("log", {}).pop("Episode_Reward/joint_limit_margin_l1", None)
        limit_weight = float(cfg.scales.get("target_limit_l2", 0.0))
        if limit_weight:
            weighted_limit = limit_weight * self._target_limit_cost
            reward += (weighted_limit * self._cfg.ctrl_dt).astype(reward.dtype, copy=False)
            self._episode_reward_sums["target_limit_l2"] += weighted_limit * self._cfg.ctrl_dt
            state.info.setdefault("log", {}).pop("Episode_Reward/target_limit_l2", None)
        weight_air = float(cfg.scales.get("both_airborne", 0.0))
        if weight_air:
            air = (
                self._airborne_fraction()
                if cfg.airborne_cost_sampling == "substep"
                else ~np.any(self._contact, axis=1)
            )
            weighted_air = weight_air * air
            reward += (weighted_air * self._cfg.ctrl_dt).astype(reward.dtype, copy=False)
            self._episode_reward_sums["both_airborne"] += weighted_air * self._cfg.ctrl_dt
            state.info.setdefault("log", {}).pop("Episode_Reward/both_airborne", None)
        weight = float(cfg.scales["command_progress"])
        if weight:
            commands = state.info["commands"]
            moving = (
                np.linalg.norm(commands[:, :2], axis=1) + np.abs(commands[:, 2])
                > cfg.progress_command_threshold
            )
            upright = upvector[:, 2] >= cfg.progress_upright_min
            progress = command_direction_progress(linvel, gyro, commands)
            weighted = weight * np.where(moving & upright, progress, 0.0)
            reward += (weighted * self._cfg.ctrl_dt).astype(reward.dtype, copy=False)
            self._episode_reward_sums["command_progress"] += weighted * self._cfg.ctrl_dt
        state.info.setdefault("log", {}).pop("Episode_Reward/command_progress", None)
        return reward

    def _physics_step_action_delay(self, backend, ctrl: np.ndarray) -> np.ndarray:
        motor = self._motor
        if not isinstance(motor, GF43X40RobotMotor):
            return MicroDuckDm4310VelocityEnv._physics_step_action_delay(self, backend, ctrl)
        pos = backend.get_dof_pos()
        vel = backend.get_dof_vel()
        motor.finish_substep(pos, vel)
        self._sample_substep_foot_state(reset_events=self._delay_substep == 0)
        torque = motor.begin_substep(ctrl, pos, vel)
        self._delay_substep = (self._delay_substep + 1) % self._cfg.sim_substeps
        # The source model's position actuators are made unit-gain/zero-damping
        # on the cold path: target q + tau applies exactly the GF torque.
        return pos + torque

    def update_state(self, state):
        if isinstance(self._motor, GF43X40RobotMotor):
            self._motor.finish_substep(self._backend.get_dof_pos(), self._backend.get_dof_vel())
        result = super().update_state(state)
        log = result.info.setdefault("log", {})
        log["audit/gf_peak_abs_torque_nm"] = float(np.max(np.abs(self._motor.torque)))
        log["audit/gf_mean_abs_torque_nm"] = float(np.mean(np.abs(self._motor.torque)))
        return result
