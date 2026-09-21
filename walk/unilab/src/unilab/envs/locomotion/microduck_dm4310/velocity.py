"""MicroDuck V1.1.0 all-DM4340 velocity task ported from mjlab_microduck.

This is a contract-preserving UniLab owner: actor observation is exactly
``gyro(3), projected_gravity(3), joint_pos_rel(14), joint_vel(14),
last_action(14), command(13)`` and the action is the same 14 joint-position
offsets used by the deployed ONNX family.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import IntervalRandomizationPlan, ResetRandomizationPayload
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

from .motor import (
    Dm4310MitBatchModel,
    Dm4310MitConfig,
    Dm4340DomainRandConfig,
    Dm4340MitConfig,
    quantize_unsigned,
)
from .reset_support import ResetGroundClearance

NUM_ACTIONS = 14
NUM_COMMANDS = 13
OBS_DIM = 61
REFERENCE_TRAIN_ENVS = 4096
OWNER_TRAIN_ENVS = 1024
REFERENCE_ROLLOUT_STEPS = 24
OWNER_ROLLOUT_STEPS = 48
CURRICULUM_SAMPLE_SCALE = REFERENCE_TRAIN_ENVS // OWNER_TRAIN_ENVS
FIRST_GATE_STEP = 500 * REFERENCE_ROLLOUT_STEPS * CURRICULUM_SAMPLE_SCALE
HEAD_COM_BODY_NAMES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
# Geometry scale declared by the duck V1.1.0 asset. Keep task-space lengths
# tied to the imported model rather than to the motor housing dimensions.
SCALE = 2.09
POLICY_JOINT_NAMES = (
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


def _projected_gravity_from_quat(base_quat_w: np.ndarray) -> np.ndarray:
    """Express world gravity in the base frame, matching MJLab's 61D contract."""
    world_gravity = np.zeros((len(base_quat_w), 3), dtype=base_quat_w.dtype)
    world_gravity[:, 2] = -1.0
    return np.asarray(np_quat_apply_inverse(base_quat_w, world_gravity))


def pseudo_huber(value: np.ndarray, delta: float) -> np.ndarray:
    """Dimensionless robust cost with a quadratic core and linear tail."""
    if delta <= 0.0:
        raise ValueError("pseudo-Huber delta must be positive")
    return np.sqrt(1.0 + np.square(value / delta)) - 1.0


def pseudo_huber_norm(value: np.ndarray, delta: float) -> np.ndarray:
    """Vector pseudo-Huber cost that retains gradient far from the target."""
    if delta <= 0.0:
        raise ValueError("pseudo-Huber delta must be positive")
    return np.sqrt(1.0 + np.sum(np.square(value / delta), axis=1)) - 1.0


def velocity_tracking_scores(
    linvel: np.ndarray,
    gyro: np.ndarray,
    command: np.ndarray,
    linear_std: float,
    angular_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Official MJLab root velocity tracking, including uncommanded axes."""
    linear = np.exp(
        -(np.sum(np.square(linvel[:, :2] - command[:, :2]), axis=1) + linvel[:, 2] ** 2)
        / linear_std**2
    )
    angular = np.exp(
        -(np.square(gyro[:, 2] - command[:, 2]) + np.sum(gyro[:, :2] ** 2, axis=1)) / angular_std**2
    )
    return linear, angular


def command_tracking_gate(
    linear_score: np.ndarray,
    angular_score: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Require the commanded degrees of freedom before paying base rewards."""
    linear_active = np.linalg.norm(command[:, :2], axis=1) > 0.01
    angular_active = np.abs(command[:, 2]) > 0.01
    gate = linear_score * angular_score  # exact zero command must be still in all axes
    gate = np.where(linear_active & ~angular_active, linear_score, gate)
    gate = np.where(~linear_active & angular_active, angular_score, gate)
    return gate


def command_direction_progress(
    linvel: np.ndarray,
    gyro: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Bounded signed progress; zero at rest and one at exact commanded speed."""
    linear_norm_sq = np.sum(np.square(command[:, :2]), axis=1)
    linear_active = linear_norm_sq > 1.0e-4
    linear = np.divide(
        np.sum(linvel[:, :2] * command[:, :2], axis=1),
        linear_norm_sq,
        out=np.zeros(len(command)),
        where=linear_active,
    )
    angular_active = np.abs(command[:, 2]) > 0.01
    angular = np.divide(
        gyro[:, 2],
        command[:, 2],
        out=np.zeros(len(command)),
        where=angular_active,
    )
    count = linear_active.astype(np.float64) + angular_active.astype(np.float64)
    progress = np.divide(
        np.clip(linear, -1.0, 1.0) * linear_active + np.clip(angular, -1.0, 1.0) * angular_active,
        count,
        out=np.zeros(len(command)),
        where=count > 0.0,
    )
    return progress


def feet_clearance_cost(
    height: np.ndarray,
    velocity_xy: np.ndarray,
    command: np.ndarray,
    target_height: float,
    reference_height: float,
) -> np.ndarray:
    """Official cost in reference-height units, at unchanged absolute foot speed.

    Equal relative clearance error costs the same on robots with different
    target heights. With target_height == reference_height this is exactly
    the upstream abs(height - target_height) * speed formula.
    """
    if target_height <= 0 or reference_height <= 0:
        raise ValueError("clearance target and reference heights must be positive")
    active = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > 0.01
    error = np.abs(height - target_height) * (reference_height / target_height)
    return np.sum(error * np.linalg.norm(velocity_xy, axis=2), axis=1) * active


def feet_air_time_reward(
    air_time: np.ndarray, command: np.ndarray, threshold_min: float, threshold_max: float
) -> np.ndarray:
    """Official per-foot window reward; no added posture/support/progress gates."""
    active = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > 0.01
    return np.sum((air_time > threshold_min) & (air_time < threshold_max), axis=1) * active


def leg_pose_reward(error: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.exp(-np.mean(np.square(error / std), axis=1))


def feet_slip_cost(velocity: np.ndarray, contact: np.ndarray, command: np.ndarray) -> np.ndarray:
    active = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2]) > 0.01
    return np.sum(np.sum(np.square(velocity[:, :, :2]), axis=2) * contact, axis=1) * active


@dataclass
class Asset:
    base_name: str = "trunk_base"
    ground: str = "floor"
    head_body_names: tuple[str, ...] = HEAD_COM_BODY_NAMES
    policy_joint_names: tuple[str, ...] = POLICY_JOINT_NAMES


@dataclass
class Sensor:
    local_linvel: str = "imu_lin_vel"
    gyro: str = "imu_ang_vel"
    upvector: str = "imu_upvector"
    feet_force: tuple[str, str] = ("left_foot_contact", "right_foot_contact")
    feet_found: tuple[str, str] = ("left_foot_found", "right_foot_found")
    feet_pos: tuple[str, str] = ("left_foot_pos", "right_foot_pos")
    feet_vel: tuple[str, str] = ("left_foot_vel", "right_foot_vel")
    angular_momentum: str = "root_angmom"
    self_contacts: tuple[str, str, str] = (
        "self_contact_left",
        "self_contact_right",
        "self_contact_legs",
    )
    actuator_torque: tuple[str, ...] = tuple(f"torque_{name}" for name in POLICY_JOINT_NAMES)


@dataclass
class Commands:
    velocity_low: list[float] = field(default_factory=lambda: [-0.4, -0.3, -1.0])
    velocity_high: list[float] = field(default_factory=lambda: [0.4, 0.3, 1.0])
    resampling_time_range: list[float] = field(default_factory=lambda: [3.0, 8.0])
    rel_forward_envs: float = 0.2
    rel_turn_in_place_envs: float = 0.15
    forward_min_speed: float = 0.3
    head_resampling_time_range: list[float] = field(default_factory=lambda: [2.0, 5.0])
    body_resampling_time_range: list[float] = field(default_factory=lambda: [2.0, 5.0])


@dataclass
class MicroDuckLatencyConfig:
    """Actor/actuator latency ranges expressed in deployment-facing milliseconds."""

    actuator_delay_ms: list[float] = field(default_factory=lambda: [10.0, 20.0])
    imu_delay_ms: list[float] = field(default_factory=lambda: [0.0, 20.0])
    joint_velocity_delay_ms: float = 20.0
    imu_delay_update_period_steps: int = 64


@dataclass
class CurriculumConfig:
    # Later head/body combinations need active balance, not merely static torque
    # feasibility. Keep their automatic promotion closed until policy evaluation.
    # Raise this env-step cap explicitly after per-task acceptance; None opens all.
    # Defaults below are resolved 1024-env steps (source steps multiplied by 4).
    # Gate 0 is therefore 1000 iterations at 48 rollout steps, ending at 47_999.
    step_cap: int | None = FIRST_GATE_STEP - 1
    action_rate_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -0.10},
            {"step": 48000, "value": -0.20},
            {"step": 96000, "value": -0.40},
            {"step": 144000, "value": -0.60},
            {"step": 192000, "value": -0.80},
            {"step": 240000, "value": -1.00},
        ]
    )
    standing_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.20},
            {"step": 48000, "value": 0.25},
        ]
    )
    velocity_scale_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.25},
            {"step": 48000, "value": 0.40},
            {"step": 96000, "value": 0.60},
            {"step": 144000, "value": 0.80},
            {"step": 192000, "value": 1.00},
        ]
    )
    linear_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.08},
            {"step": 48000, "value": 0.10},
            {"step": 96000, "value": 0.12},
            {"step": 144000, "value": 0.14},
            {"step": 192000, "value": 0.15},
        ]
    )
    angular_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.15},
            {"step": 48000, "value": 0.20},
            {"step": 96000, "value": 0.25},
            {"step": 144000, "value": 0.30},
            {"step": 192000, "value": 0.35},
        ]
    )
    head_range_stages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"step": 0, "ranges": [[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]]},
            {"step": 144000, "ranges": [[-0.1, 0.1], [-0.1, 0.1], [-0.15, 0.15], [-0.025, 0.025]]},
            {"step": 288000, "ranges": [[-0.2, 0.2], [-0.2, 0.2], [-0.35, 0.35], [-0.05, 0.05]]},
            {"step": 432000, "ranges": [[-0.3, 0.3], [-0.35, 0.35], [-0.65, 0.65], [-0.1, 0.1]]},
            {"step": 576000, "ranges": [[-0.4, 0.4], [-0.5, 0.5], [-1.0, 1.0], [-0.15, 0.15]]},
        ]
    )
    head_bias_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 48000, "value": 0.5},
            {"step": 96000, "value": 1.0},
            {"step": 144000, "value": 2.0},
            {"step": 192000, "value": 3.0},
        ]
    )
    com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 192000, "value": 0.005},
            {"step": 384000, "value": 0.007},
            {"step": 576000, "value": 0.01},
        ]
    )
    head_com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 192000, "value": 0.004},
            {"step": 384000, "value": 0.005},
            {"step": 576000, "value": 0.006},
        ]
    )

    push_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 192000, "value": 0.03},
            {"step": 384000, "value": 0.06},
            {"step": 576000, "value": 0.10},
        ]
    )

    def __post_init__(self) -> None:
        # Resolved stages are already in 1024-env physical steps. Copy mutable
        # values so dataclasses.replace() cannot mutate the original config.
        from copy import deepcopy

        for name, value in vars(self).items():
            if isinstance(value, list):
                setattr(self, name, deepcopy(value))


@dataclass
class VelocityCurriculumConfig(CurriculumConfig):
    """4096-env source schedule translated to equal samples at 1024 envs."""

    action_rate_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -0.10},
            {"step": 48000, "value": -0.20},
            {"step": 72000, "value": -0.40},
            {"step": 96000, "value": -0.60},
            {"step": 120000, "value": -0.80},
            {"step": 144000, "value": -1.00},
        ]
    )
    standing_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.02},
            {"step": 48000, "value": 0.05},
            {"step": 72000, "value": 0.10},
            {"step": 96000, "value": 0.15},
            {"step": 144000, "value": 0.20},
            {"step": 192000, "value": 0.25},
        ]
    )
    velocity_scale_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": 1.00}]
    )
    linear_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": float(np.sqrt(0.1))}]
    )
    angular_velocity_std_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": float(np.sqrt(0.5))}]
    )
    head_range_stages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"step": 0, "ranges": [[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]]},
            {
                "step": 48000,
                "ranges": [[-0.1, 0.1], [-0.1, 0.1], [-0.15, 0.15], [-0.025, 0.025]],
            },
            {"step": 96000, "ranges": [[-0.2, 0.2], [-0.2, 0.2], [-0.35, 0.35], [-0.05, 0.05]]},
            {"step": 144000, "ranges": [[-0.3, 0.3], [-0.35, 0.35], [-0.65, 0.65], [-0.1, 0.1]]},
            {"step": 192000, "ranges": [[-0.4, 0.4], [-0.5, 0.5], [-1.0, 1.0], [-0.15, 0.15]]},
        ]
    )
    head_bias_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 57600, "value": 1.0},
            {"step": 96000, "value": 2.0},
            {"step": 144000, "value": 3.0},
        ]
    )
    com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 48000, "value": 0.005},
            {"step": 96000, "value": 0.007},
            {"step": 144000, "value": 0.010},
        ]
    )
    head_com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.003},
            {"step": 48000, "value": 0.004},
            {"step": 96000, "value": 0.005},
            {"step": 144000, "value": 0.006},
        ]
    )
    push_stages: list[dict[str, float]] = field(default_factory=lambda: [{"step": 0, "value": 0.3}])


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "pose": 1.0,
            "upright": 2.0,
            "foot_slip": -0.1,
            "self_collisions": -1.0,
            "air_time": 3.0,
            "body_ang_vel": -0.05,
            "angular_momentum": -5.0e-7,
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
    linear_velocity_std: float = 0.08
    angular_velocity_std: float = 0.15
    upright_std: float = float(np.sqrt(0.05))
    # Absolute task tolerances, not geometric similarity laws. See the audit.
    foot_target_height: float = 0.035
    angular_momentum_reference: float = 0.25  # kg m^2 / s; HOME Ixx=0.256 kg m^2
    height_std: float = 0.050
    height_sharp_std: float = 0.015
    posture_composite_height_std: float = 0.035
    stillness_velocity_std: float = 0.050
    linear_velocity_error_delta: float = 0.15
    yaw_velocity_error_delta: float = 0.30
    vertical_velocity_error_delta: float = 0.15
    air_time_target: float = 0.25
    air_time_std: float = 0.08
    joint_limit_margin_fraction: float = 0.15


@dataclass
class VelocityRewardConfig(RewardConfig):
    """Official velocity tracking widths and per-foot air-time window."""

    linear_velocity_std: float = float(np.sqrt(0.1))
    angular_velocity_std: float = float(np.sqrt(0.5))
    # Keep official cost units when adapting the clearance target to this robot.
    foot_clearance_reference_height: float = 0.02
    air_time_threshold_min: float = 0.125
    air_time_threshold_max: float = 0.300


@dataclass
class MicroDuckDomainRandConfig(DomainRandConfig):
    mass_inertia_scale_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    imu_max_angle_deg: float = 6.0
    randomize_kp: bool = False
    randomize_kd: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    velocity_pushes: bool = True
    velocity_push_interval_s: list[float] = field(default_factory=lambda: [3.0, 6.0])
    velocity_push_xy: list[float] = field(default_factory=lambda: [-0.3, 0.3])
    randomize_joint_friction: bool = True
    joint_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_foot_friction: bool = True
    foot_friction_range: list[float] = field(default_factory=lambda: [0.7, 1.3])
    motor: Dm4340DomainRandConfig = field(default_factory=Dm4340DomainRandConfig)


@dataclass
class MicroDuckDm4310VelocityCfg(LocomotionBaseCfg):
    reset_clearance_m: float = 0.005
    posture_reset_joint_jitter_rad: float = 0.02
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_dm4310" / "scene_flat.xml")
        )
    )
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)  # type: ignore[assignment]
    commands: Commands = field(default_factory=Commands)
    curriculum: VelocityCurriculumConfig = field(default_factory=VelocityCurriculumConfig)
    reward_config: VelocityRewardConfig = field(default_factory=VelocityRewardConfig)
    latency: MicroDuckLatencyConfig = field(default_factory=MicroDuckLatencyConfig)
    clip_action_targets: bool = True
    action_offset_low: list[float] = field(
        default_factory=lambda: [
            -0.406332,
            -0.266706,
            -1.082872,
            -1.535856,
            -1.99378,
            -1.889862,
            -1.889862,
            -2.93706,
            -0.406332,
            -0.493599,
            -0.441239,
            -1.99872,
            -1.545736,
            -1.087812,
        ]
    )
    action_offset_high: list[float] = field(
        default_factory=lambda: [
            0.493599,
            0.441239,
            1.99872,
            1.545736,
            1.087812,
            0.668132,
            1.19173,
            2.93706,
            0.406332,
            0.406332,
            0.266706,
            1.082872,
            1.535856,
            1.99378,
        ]
    )
    motor_model: str = "dm4340"
    # Optional per-axis overrides used by the original mixed-motor robot:
    # neck pitch and both hip-roll axes were DM4340 while the other 11 axes
    # inherited ``motor_model=dm4310``.
    neck_pitch_motor: str = "dm4340"
    hip_roll_motor: str = "dm4340"
    control_config: PdControlConfig = field(  # type: ignore[assignment]
        default_factory=lambda: PdControlConfig(
            action_scale=1.0,
            simulate_action_latency=False,
            Kp=60.0,
            Kd=2.0,
        )
    )
    noise_config: BaseNoiseConfig = field(
        default_factory=lambda: BaseNoiseConfig(
            level=1.0,
            scale_joint_angle=0.001,
            scale_joint_vel=0.25,
            scale_gyro=0.03,
            scale_gravity=0.01,
            scale_linvel=0.1,
        )
    )
    domain_rand: MicroDuckDomainRandConfig = field(
        default_factory=lambda: MicroDuckDomainRandConfig(
            randomize_base_mass=False,
            random_com=False,
            randomize_ground_friction=False,
            ground_friction_multiplier_range=[0.7, 1.3],
            randomize_dof_armature=True,
            dof_armature_multiplier_range=[0.9, 1.1],
            push_robots=False,
            push_body_name="trunk_base",
        )
    )
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 20.0


def _stage(stages: list[dict[str, Any]], step: int, key: str = "value"):
    current = stages[0][key]
    for entry in stages:
        if step < int(entry["step"]):
            break
        current = entry[key]
    return current


class MicroDuckDRProvider(LocomotionDRProvider):
    def _finalize_reset_pose(self, env, plan):
        plan.qpos[:, -NUM_ACTIONS:] = np.clip(
            plan.qpos[:, -NUM_ACTIONS:],
            env._reset_joint_limits[:, 0] + 0.01,
            env._reset_joint_limits[:, 1] - 0.01,
        )
        env._reset_clearance.lift_to_clearance(plan.qpos, env._cfg.reset_clearance_m)
        return plan

    def _get_qvel_limit(self, env: Any) -> float:
        del env
        return 0.0

    def validate(self, env: Any, capabilities) -> None:
        super().validate(env, capabilities)
        if (
            env.cfg.domain_rand.velocity_pushes
            and not capabilities.supports_interval_body_velocity_delta
        ):
            raise NotImplementedError(
                f"{env._backend.backend_type} does not support base-velocity push parity"
            )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        if not env.cfg.domain_rand.velocity_pushes:
            return None
        due = step_counter >= env._next_push_step
        if not np.any(due):
            return None
        lo_s, hi_s = env.cfg.domain_rand.velocity_push_interval_s
        env._next_push_step[due] = step_counter + np.asarray(
            np.random.uniform(lo_s, hi_s, size=np.count_nonzero(due)) / env.cfg.ctrl_dt,
            dtype=np.int64,
        )
        lo, hi = env.cfg.domain_rand.velocity_push_xy
        delta = np.zeros((env.num_envs, 1, 3), dtype=get_global_dtype())
        delta[due, 0, :2] = np.random.uniform(lo, hi, size=(np.count_nonzero(due), 2))
        return IntervalRandomizationPlan(
            body_ids=np.asarray([env._trunk_body_id], dtype=np.int32),
            body_linear_velocity_delta=delta,
        )

    def _get_base_actuator_gains(self, env: Any):
        gains = env.cfg.control_config.position_gains()
        return (
            np.full(env._num_action, gains["kp"], dtype=np.float64),
            np.full(env._num_action, gains["kd"], dtype=np.float64),
        )

    def _get_reset_randomization_baselines(self, env: Any):
        cached = getattr(self, "_reset_baselines", None)
        if cached is None:
            backend = env._backend
            cached = (
                None,
                backend.get_geom_friction(),
                backend.get_geom_id(env.cfg.asset.ground),
                backend.get_dof_armature(),
            )
            self._reset_baselines = cached
        return cached

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        env._reset_episode_history(env_ids)
        plan = super().build_reset_plan(env, env_ids)
        n = len(env_ids)
        plan.qpos[:, 2] = np.random.uniform(0.12 * SCALE, 0.13 * SCALE, size=n)
        body_ipos = np.broadcast_to(env._base_body_ipos, (n, *env._base_body_ipos.shape)).copy()
        body_ipos[:, env._trunk_body_id, :] += np.random.uniform(
            -env._com_range, env._com_range, size=(n, 3)
        )
        body_ipos[:, env._head_body_ids, :] += np.random.uniform(
            -env._head_com_range,
            env._head_com_range,
            size=(n, len(env._head_body_ids), 3),
        )
        if plan.randomization is None:
            plan.randomization = ResetRandomizationPayload()
        plan.randomization.body_ipos = body_ipos
        scale = env._mass_inertia_scale[env_ids]
        body_mass = np.broadcast_to(env._base_body_mass, (n, env._base_body_mass.size)).copy()
        body_inertia = np.broadcast_to(
            env._base_body_inertia, (n, *env._base_body_inertia.shape)
        ).copy()
        body_mass[:, env._trunk_body_id] *= scale
        body_inertia[:, env._trunk_body_id, :] *= scale[:, None]
        plan.randomization.body_mass = body_mass
        plan.randomization.body_inertia = body_inertia
        if env.cfg.domain_rand.randomize_foot_friction:
            geom_friction = np.broadcast_to(
                env._base_geom_friction, (n, *env._base_geom_friction.shape)
            ).copy()
            geom_friction[:, env._foot_geom_ids, 0] = env._foot_friction[env_ids, None]
            plan.randomization.geom_friction = geom_friction
        if env.cfg.domain_rand.randomize_joint_friction:
            lo, hi = env.cfg.domain_rand.joint_friction_multiplier_range
            friction = np.broadcast_to(
                env._base_dof_frictionloss, (n, env._base_dof_frictionloss.size)
            ).copy()
            friction *= np.random.uniform(lo, hi, size=friction.shape)
            plan.randomization.dof_frictionloss = friction
        motor_cfg = env.cfg.domain_rand.motor
        motor_joint_ids = env._motor._j4340_joint_ids
        if motor_cfg.enabled and len(motor_joint_ids):
            dof_ids = env._motor_dof_ids[motor_joint_ids]
            if plan.randomization.dof_armature is None:
                plan.randomization.dof_armature = np.broadcast_to(
                    env._base_dof_armature, (n, env._base_dof_armature.size)
                ).copy()
            plan.randomization.dof_armature[:, dof_ids] = env._motor.armature[
                env_ids[:, None], motor_joint_ids[None, :]
            ]
            if plan.randomization.dof_frictionloss is None:
                plan.randomization.dof_frictionloss = np.broadcast_to(
                    env._base_dof_frictionloss, (n, env._base_dof_frictionloss.size)
                ).copy()
            plan.randomization.dof_frictionloss[:, dof_ids] = env._motor.coulomb_friction[
                env_ids[:, None], motor_joint_ids[None, :]
            ]
        # Common gain DR describes firmware gains, not the native torque plant.
        if not motor_cfg.enabled or not len(motor_joint_ids):
            if plan.randomization.kp is not None:
                env._motor.kp[env_ids] = quantize_unsigned(plan.randomization.kp, 500.0, 12)
            if plan.randomization.kd is not None:
                env._motor.kd[env_ids] = quantize_unsigned(plan.randomization.kd, 5.0, 12)
        plan.randomization.kp = None
        plan.randomization.kd = None
        return self._finalize_reset_pose(env, plan)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        return np.concatenate(
            (
                env._sample_velocity_commands(num_reset),
                env._sample_head_commands(num_reset),
                env._sample_body_commands(num_reset),
            ),
            axis=1,
            dtype=get_global_dtype(),
        )

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        del gravity  # framezaxis is world-frame and is not MJLab projected gravity.
        upvector = -env._projected_gravity(env_ids)
        actor_gyro = env._rotate_imu(gyro, env_ids)
        actor_gravity = env._rotate_imu(upvector, env_ids)
        return env._compute_obs(
            actor_gyro,
            actor_gravity,
            env._motor.quantize_position_feedback(dof_pos, env_ids) + env._encoder_bias[env_ids],
            env._motor.quantize_velocity_feedback(dof_vel, env_ids),
            info_updates["last_actions"],
            info_updates["commands"],
            critic_linvel=linvel,
            critic_gyro=gyro,
            critic_upvector=upvector,
            critic_dof_pos=dof_pos,
            critic_dof_vel=dof_vel,
            env_ids=env_ids,
        )


class MicroDuckDm4310VelocityEnv(LocomotionBaseEnv):
    _cfg: MicroDuckDm4310VelocityCfg
    _critic_has_foot_height = True

    def _make_dr_provider(self):
        return MicroDuckDRProvider()

    def get_local_linvel(self) -> np.ndarray:
        """Root translational velocity in body frame, excluding IMU lever arm."""
        return np.asarray(
            np_quat_apply_inverse(self._backend.get_base_quat(), self._backend.get_base_lin_vel())
        )

    def _create_backend(self, cfg: MicroDuckDm4310VelocityCfg, num_envs: int, backend_type: str):
        return create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains=None,
            **env_backend_kwargs(cfg),
        )

    def __init__(self, cfg: MicroDuckDm4310VelocityCfg, num_envs=1, backend_type="mujoco"):
        mass_range = cfg.domain_rand.mass_inertia_scale_range
        if (
            len(mass_range) != 2
            or not np.all(np.isfinite(mass_range))
            or not 0 < mass_range[0] <= mass_range[1]
        ):
            raise ValueError("mass_inertia_scale_range must be positive ordered finite bounds")
        if not np.isfinite(cfg.domain_rand.imu_max_angle_deg) or not (
            0 <= cfg.domain_rand.imu_max_angle_deg <= 180
        ):
            raise ValueError("imu_max_angle_deg must be finite and in [0, 180]")
        backend = self._create_backend(cfg, num_envs, backend_type)
        super().__init__(cfg, backend, num_envs)
        if self._num_action != NUM_ACTIONS:
            raise ValueError(
                f"MicroDuck action contract is 14D, backend exposed {self._num_action}"
            )
        self._standing_fraction = cfg.curriculum.standing_stages[0]["value"]
        self._velocity_command_scale = cfg.curriculum.velocity_scale_stages[0]["value"]
        self._head_ranges = cfg.curriculum.head_range_stages[0]["ranges"]
        self._head_bias_ema = np.zeros((num_envs, 4), dtype=get_global_dtype())
        self._encoder_bias = np.asarray(
            np.random.uniform(-0.015, 0.015, size=(num_envs, NUM_ACTIONS)),
            dtype=get_global_dtype(),
        )
        axis = np.random.normal(size=(num_envs, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        angle = np.random.uniform(0.0, np.deg2rad(cfg.domain_rand.imu_max_angle_deg), size=num_envs)
        skew = np.zeros((num_envs, 3, 3), dtype=get_global_dtype())
        skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
        eye = np.broadcast_to(np.eye(3), (num_envs, 3, 3))
        self._imu_rotation = (
            eye
            + np.sin(angle)[:, None, None] * skew
            + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
        ).astype(get_global_dtype())
        self._last_foot_pos = np.zeros((num_envs, 2, 3), dtype=get_global_dtype())
        self._air_time = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._contact_time = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._foot_substep_pending = False
        self._landed_air_time = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._peak_foot_height = np.zeros((num_envs, 2), dtype=get_global_dtype())
        self._previous_contact = np.zeros((num_envs, 2), dtype=bool)
        self._history_fresh = np.ones(num_envs, dtype=bool)
        self._foot_force = np.zeros((num_envs, 2, 3), dtype=get_global_dtype())
        self._foot_pos = np.zeros((num_envs, 2, 3), dtype=get_global_dtype())
        self._foot_vel = np.zeros((num_envs, 2, 3), dtype=get_global_dtype())
        self._contact = np.zeros((num_envs, 2), dtype=bool)
        self._first_contact = np.zeros((num_envs, 2), dtype=bool)
        joint_range = self._backend.get_joint_range()
        if joint_range is None or joint_range.shape != (NUM_ACTIONS, 2):
            shape = None if joint_range is None else joint_range.shape
            raise ValueError(f"expected 14 hinge joint ranges, got {shape}")
        center = np.mean(joint_range, axis=1)
        self._target_low = self.default_angles + np.asarray(cfg.action_offset_low)
        self._target_high = self.default_angles + np.asarray(cfg.action_offset_high)
        if (
            self._target_low.shape != (NUM_ACTIONS,)
            or self._target_high.shape != (NUM_ACTIONS,)
            or np.any(~np.isfinite(self._target_low))
            or np.any(~np.isfinite(self._target_high))
            or np.any(self._target_low >= self._target_high)
            or np.any(self._target_low < joint_range[:, 0])
            or np.any(self._target_high > joint_range[:, 1])
        ):
            raise ValueError("action offset bounds must define finite targets inside joint limits")
        self._reset_joint_limits = joint_range.copy()
        self._joint_limits = joint_range.copy()
        self._reset_clearance = ResetGroundClearance(cfg.scene.model_file, cfg.asset.ground)
        half = 0.5 * (joint_range[:, 1] - joint_range[:, 0]) * 0.9
        self._soft_joint_limits = np.stack((center - half, center + half), axis=1)
        self._base_body_ipos = self._backend.get_body_ipos()
        self._base_body_mass = self._backend.get_body_mass()
        self._base_body_inertia = self._backend.get_body_inertia()
        self._base_dof_armature = self._backend.get_dof_armature()
        self._base_dof_frictionloss = self._backend.get_dof_frictionloss()
        self._trunk_body_id = self._backend.get_body_id(cfg.asset.base_name)
        # bearing_roll carries right_hip_yaw; it is not part of the head chain.
        self._head_body_ids = self._backend.get_body_ids(cfg.asset.head_body_names)
        self._com_range = cfg.curriculum.com_range_stages[0]["value"]
        self._head_com_range = cfg.curriculum.head_com_range_stages[0]["value"]
        self._mass_inertia_scale = np.random.uniform(*mass_range, size=num_envs)
        self._base_geom_friction = self._backend.get_geom_friction()
        self._foot_geom_ids = np.asarray(
            [
                self._backend.get_geom_id("left_foot_collision"),
                self._backend.get_geom_id("right_foot_collision"),
            ],
            dtype=np.int32,
        )
        foot_lo, foot_hi = cfg.domain_rand.foot_friction_range
        self._foot_friction = np.random.uniform(foot_lo, foot_hi, size=num_envs)
        push_lo, push_hi = cfg.domain_rand.velocity_push_interval_s
        self._next_push_step = np.asarray(
            np.random.uniform(push_lo, push_hi, size=num_envs) / cfg.ctrl_dt,
            dtype=np.int64,
        )
        self._motor = self._build_motor(cfg, num_envs)
        native_gain, native_damping = self._backend.get_actuator_gains()
        if not np.allclose(native_gain, 1.0) or not np.allclose(native_damping, 0.0):
            raise ValueError("MicroDuck external motor model requires unit-gain torque actuators")
        self._motor_dof_ids = self._backend.get_joint_dof_indices(cfg.asset.policy_joint_names)
        bias_lo, bias_hi = self._motor.position_bias_range
        self._encoder_bias[:, self._motor._j4340_joint_ids] = np.random.uniform(
            bias_lo,
            bias_hi,
            size=(num_envs, len(self._motor._j4340_joint_ids)),
        )
        actuator_lo, actuator_hi = self._latency_step_bounds(
            cfg.latency.actuator_delay_ms, cfg.sim_dt, "actuator_delay_ms"
        )
        self._actuator_delay_bounds = (actuator_lo, actuator_hi)
        self._actuator_delay_lag = np.zeros(num_envs, dtype=np.int32)
        encoder_adjusted_home = self.default_angles[None, :] - self._encoder_bias
        self._actuator_history = np.broadcast_to(
            encoder_adjusted_home,
            (actuator_hi + 1, num_envs, NUM_ACTIONS),
        ).copy()
        # Match MJLab's DelayBuffer reset semantics: the first command received
        # after reset backfills the entire row, so no pre-reset/HOME target is
        # replayed into a newly teleported posture.
        self._actuator_history_fresh = np.ones(num_envs, dtype=bool)
        self._actuator_history_cursor = 0
        imu_lo, imu_hi = self._latency_step_bounds(
            cfg.latency.imu_delay_ms, cfg.ctrl_dt, "imu_delay_ms"
        )
        joint_lag, joint_hi = self._latency_step_bounds(
            [cfg.latency.joint_velocity_delay_ms] * 2,
            cfg.ctrl_dt,
            "joint_velocity_delay_ms",
        )
        self._imu_delay_bounds = (imu_lo, imu_hi)
        self._imu_delay_lag = np.zeros(num_envs, dtype=np.int32)
        self._joint_velocity_delay_lag = joint_lag
        obs_history_len = max(imu_hi, joint_hi) + 1
        self._imu_history = np.zeros((obs_history_len, num_envs, 6), dtype=get_global_dtype())
        self._joint_velocity_history = np.zeros(
            (obs_history_len, num_envs, NUM_ACTIONS), dtype=get_global_dtype()
        )
        self._observation_history_cursor = 0
        self._command_provider = self._make_dr_provider()
        self._velocity_timer = np.ones(num_envs, dtype=np.int32)
        self._head_timer = np.ones(num_envs, dtype=np.int32)
        self._body_timer = np.ones(num_envs, dtype=np.int32)
        self._init_domain_randomization(self._command_provider)
        self._delay_substep = 0
        self._backend.set_pre_step_control(self._physics_step_action_delay)

    def _build_motor(self, cfg: MicroDuckDm4310VelocityCfg, num_envs: int):
        """Construct this owner's motor before reset/provider initialization."""
        motor_names = {
            "motor_model": cfg.motor_model,
            "neck_pitch_motor": cfg.neck_pitch_motor,
            "hip_roll_motor": cfg.hip_roll_motor,
        }
        for field_name, motor_name in motor_names.items():
            if motor_name not in {"dm4310", "dm4340"}:
                raise ValueError(f"{field_name} must be 'dm4310' or 'dm4340', got {motor_name!r}")

        def motor_config(name: str):
            base = Dm4340MitConfig() if name == "dm4340" else Dm4310MitConfig()
            gains = cfg.control_config.position_gains()
            return replace(base, kp=gains["kp"], kd=gains["kd"])

        motor_cfg = motor_config(cfg.motor_model)
        joint_cfgs = {
            POLICY_JOINT_NAMES.index("neck_pitch"): motor_config(cfg.neck_pitch_motor),
            POLICY_JOINT_NAMES.index("left_hip_roll"): motor_config(cfg.hip_roll_motor),
            POLICY_JOINT_NAMES.index("right_hip_roll"): motor_config(cfg.hip_roll_motor),
        }
        return Dm4310MitBatchModel(
            num_envs,
            NUM_ACTIONS,
            cfg.sim_dt,
            cfg=motor_cfg,
            joint_cfgs=joint_cfgs,
            j4340_domain_rand=cfg.domain_rand.motor,
        )

    @staticmethod
    def _latency_step_bounds(values, dt: float, name: str) -> tuple[int, int]:
        lo_ms, hi_ms = (float(value) for value in values)
        if lo_ms < 0.0 or hi_ms < lo_ms:
            raise ValueError(f"{name} must be a non-negative [min, max] range")
        lo = int(round(lo_ms / (1000.0 * dt)))
        hi = int(round(hi_ms / (1000.0 * dt)))
        for requested, steps in ((lo_ms, lo), (hi_ms, hi)):
            represented = steps * dt * 1000.0
            if not np.isclose(requested, represented, atol=1e-9):
                raise ValueError(
                    f"{name}={requested} ms is not representable at dt={dt} s; "
                    f"nearest value is {represented} ms"
                )
        return lo, hi

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": OBS_DIM, "critic": 76 if self._critic_has_foot_height else 74}

    def _curriculum_step(self) -> int:
        cap = self._cfg.curriculum.step_cap
        return self.step_counter if cap is None else min(self.step_counter, cap)

    def _update_curriculum(self) -> None:
        cur = self._cfg.curriculum
        reward = self._cfg.reward_config
        reward.scales["action_rate_l2"] = float(
            _stage(cur.action_rate_stages, self._curriculum_step())
        )
        reward.scales["head_pose_bias"] = float(
            _stage(cur.head_bias_stages, self._curriculum_step())
        )
        self._standing_fraction = float(_stage(cur.standing_stages, self._curriculum_step()))
        self._velocity_command_scale = float(
            _stage(cur.velocity_scale_stages, self._curriculum_step())
        )
        reward.linear_velocity_std = float(
            _stage(cur.linear_velocity_std_stages, self._curriculum_step())
        )
        reward.angular_velocity_std = float(
            _stage(cur.angular_velocity_std_stages, self._curriculum_step())
        )
        self._head_ranges = _stage(cur.head_range_stages, self._curriculum_step(), "ranges")
        self._com_range = float(_stage(cur.com_range_stages, self._curriculum_step()))
        self._head_com_range = float(_stage(cur.head_com_range_stages, self._curriculum_step()))
        push = float(_stage(cur.push_stages, self._curriculum_step()))
        self._cfg.domain_rand.velocity_push_xy = [-push, push]

    def _compute_obs(
        self,
        gyro,
        upvector,
        dof_pos,
        dof_vel,
        last_actions,
        commands,
        *,
        critic_linvel,
        critic_gyro,
        critic_upvector,
        critic_dof_pos,
        critic_dof_vel,
        env_ids=None,
    ):
        noise = self._cfg.noise_config
        gyro, upvector, dof_vel = self._apply_actor_observation_latency(
            gyro, upvector, dof_vel, env_ids=env_ids
        )
        joint_rel = dof_pos - self.default_angles
        actor = np.concatenate(
            (
                self._obs_noise(gyro, noise.scale_gyro),
                self._obs_noise(-upvector, noise.scale_gravity),
                self._obs_noise(joint_rel, noise.scale_joint_angle),
                self._obs_noise(dof_vel, noise.scale_joint_vel),
                last_actions,
                commands,
            ),
            axis=1,
            dtype=get_global_dtype(),
        )
        if actor.shape[1] != OBS_DIM:
            raise ValueError(f"MicroDuck observation contract is 61D, got {actor.shape[1]}")
        ids = slice(None) if env_ids is None else env_ids
        foot_force = np.stack(
            [self._backend.get_sensor_data(name)[ids] for name in self._cfg.sensor.feet_force],
            axis=1,
        )
        foot_pos = np.stack(
            [self._backend.get_sensor_data(name)[ids] for name in self._cfg.sensor.feet_pos],
            axis=1,
        )
        contact = np.concatenate(
            [self._backend.get_sensor_data(name)[ids] for name in self._cfg.sensor.feet_found],
            axis=1,
        ).astype(get_global_dtype())
        critic_parts = [
            critic_linvel,
            critic_gyro,
            -critic_upvector,
            critic_dof_pos - self.default_angles,
            critic_dof_vel,
            last_actions,
            commands[:, :3],
        ]
        if self._critic_has_foot_height:
            critic_parts.append(self._critic_foot_height(foot_pos, ids))
        critic_parts.extend(
            [
                self._air_time[ids],
                contact,
                foot_force.reshape(len(foot_force), -1),
                commands[:, 3:7],
                commands[:, 7:13],
            ]
        )
        critic = np.concatenate(critic_parts, axis=1, dtype=get_global_dtype())
        expected = 76 if self._critic_has_foot_height else 74
        if critic.shape[1] != expected:
            raise ValueError(f"MicroDuck critic contract is {expected}D, got {critic.shape[1]}")
        return {"obs": actor, "critic": critic}

    def _critic_foot_height(self, foot_pos: np.ndarray, env_ids) -> np.ndarray:
        """DM4340's existing world-site-Z critic contract."""
        del env_ids
        return foot_pos[:, :, 2]

    def _reset_episode_history(self, env_ids) -> None:
        ids = np.asarray(env_ids, dtype=np.int32)
        self._head_bias_ema[ids] = 0.0
        self._last_foot_pos[ids] = 0.0
        self._air_time[ids] = 0.0
        self._contact_time[ids] = 0.0
        self._landed_air_time[ids] = 0.0
        self._peak_foot_height[ids] = 0.0
        self._previous_contact[ids] = False
        self._foot_force[ids] = 0.0
        self._foot_pos[ids] = 0.0
        self._foot_vel[ids] = 0.0
        self._contact[ids] = False
        self._first_contact[ids] = False
        self._history_fresh[ids] = True
        if hasattr(self, "_prev_vz"):
            self._prev_vz[ids] = 0.0
        if hasattr(self, "_prev_torque"):
            self._prev_torque[ids] = 0.0
        if hasattr(self, "_torque_history_fresh"):
            self._torque_history_fresh[ids] = True
        if hasattr(self, "_actuator_history"):
            self._actuator_history[:, ids] = 0.0
            self._actuator_history_fresh[ids] = True
            self._actuator_delay_lag[ids] = 0
        if hasattr(self, "_imu_history"):
            self._imu_history[:, ids] = 0.0
            self._joint_velocity_history[:, ids] = 0.0
            self._sample_imu_delay(ids)
        if hasattr(self, "_motor"):
            self._motor.reset(ids)

    def _rotate_imu(self, values: np.ndarray, env_ids=None) -> np.ndarray:
        rotation = self._imu_rotation if env_ids is None else self._imu_rotation[env_ids]
        return np.einsum("nij,nj->ni", rotation, values)

    def _projected_gravity(self, env_ids=None) -> np.ndarray:
        quat = self._backend.get_base_quat()
        if env_ids is not None:
            quat = quat[np.asarray(env_ids, dtype=np.int32)]
        return _projected_gravity_from_quat(quat)

    def _sample_imu_delay(self, env_ids=None) -> None:
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=np.int32)
        lo, hi = self._imu_delay_bounds
        self._imu_delay_lag[ids] = np.random.randint(lo, hi + 1, size=len(ids))

    def _apply_actor_observation_latency(
        self,
        gyro: np.ndarray,
        upvector: np.ndarray,
        dof_vel: np.ndarray,
        *,
        env_ids=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imu = np.concatenate((gyro, upvector), axis=1)
        if env_ids is not None:
            ids = np.asarray(env_ids, dtype=np.int32)
            self._imu_history[:, ids] = imu[None, :, :]
            self._joint_velocity_history[:, ids] = dof_vel[None, :, :]
            self._sample_imu_delay(ids)
            return gyro, upvector, dof_vel

        period = int(self._cfg.latency.imu_delay_update_period_steps)
        if period <= 0:
            raise ValueError("imu_delay_update_period_steps must be positive")
        if self.step_counter % period == 0:
            self._sample_imu_delay()
        cursor = self._observation_history_cursor
        self._imu_history[cursor] = imu
        self._joint_velocity_history[cursor] = dof_vel
        env_index = np.arange(self.num_envs)
        imu_index = (cursor - self._imu_delay_lag) % len(self._imu_history)
        joint_index = (cursor - self._joint_velocity_delay_lag) % len(self._joint_velocity_history)
        delayed_imu = self._imu_history[imu_index, env_index]
        delayed_joint_velocity = self._joint_velocity_history[joint_index, env_index]
        self._observation_history_cursor = (cursor + 1) % len(self._imu_history)
        return delayed_imu[:, :3], delayed_imu[:, 3:], delayed_joint_velocity

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # PPO and last-action observations keep raw Gaussian actions. Only the
        # executed position target is clipped, never the learner-owned sample.
        ctrl = super().apply_action(actions, state)
        if self._cfg.clip_action_targets:
            return np.clip(ctrl, self._target_low, self._target_high)
        return ctrl

    def _physics_step_action_delay(self, backend, ctrl: np.ndarray) -> np.ndarray:
        # Consume the preceding physics result before advancing the next substep.
        self._update_foot_history(sample_peak=False)
        self._foot_substep_pending = True
        # Official delay_update_period=0: resample each 5 ms physics step.
        lo, hi = self._actuator_delay_bounds
        self._actuator_delay_lag[:] = np.random.randint(lo, hi + 1, size=self.num_envs)
        cursor = self._actuator_history_cursor
        # MJLab's JointPositionAction subtracts encoder bias before the target
        # enters actuator delay/control, equivalent to closing the firmware
        # position loop on the biased encoder reading.
        adjusted_target = ctrl - self._encoder_bias
        fresh = self._actuator_history_fresh
        if np.any(fresh):
            self._actuator_history[:, fresh] = adjusted_target[fresh][None, :, :]
            self._actuator_history_fresh[fresh] = False
        self._actuator_history[cursor] = adjusted_target
        env_index = np.arange(self.num_envs)
        delayed_index = (cursor - self._actuator_delay_lag) % len(self._actuator_history)
        delayed_target = self._actuator_history[delayed_index, env_index]
        self._actuator_history_cursor = (cursor + 1) % len(self._actuator_history)
        pos = backend.get_dof_pos()
        vel = backend.get_dof_vel()
        torque = self._motor.compute(delayed_target, pos, vel)
        self._delay_substep += 1
        if self._delay_substep >= self._cfg.sim_substeps:
            self._delay_substep = 0
        return torque

    def _sample_velocity_commands(self, count: int) -> np.ndarray:
        cfg = self._cfg.commands
        scale = self._velocity_command_scale
        low = np.asarray(cfg.velocity_low) * scale
        high = np.asarray(cfg.velocity_high) * scale
        velocity = np.asarray(
            np.random.uniform(low, high, size=(count, 3)),
            dtype=get_global_dtype(),
        )
        forward = np.random.uniform(size=count) < cfg.rel_forward_envs
        standing = np.random.uniform(size=count) < self._standing_fraction
        turn = np.random.uniform(size=count) < cfg.rel_turn_in_place_envs
        velocity[forward, 1:] = 0.0
        forward_floor = min(0.3 * scale, max(0.0, high[0]))
        velocity[forward, 0] = np.clip(
            np.maximum(np.abs(velocity[forward, 0]), forward_floor),
            low[0],
            high[0],
        )
        velocity[turn, :2] = 0.0
        nturn = int(np.count_nonzero(turn))
        sign = np.where(np.random.uniform(size=nturn) < 0.5, -1.0, 1.0)
        turn_limit = min(abs(low[2]), abs(high[2]))
        velocity[turn, 2] = sign * np.random.uniform(
            min(0.4 * scale, turn_limit), turn_limit, size=nturn
        )
        # Exact zero is a deployment command and takes precedence over the
        # specialized buckets, so its actual probability equals the curriculum.
        velocity[standing] = 0.0
        return velocity

    def _sample_head_commands(self, count: int) -> np.ndarray:
        ranges = np.asarray(self._head_ranges, dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(ranges[:, 0], ranges[:, 1], size=(count, 4)),
            dtype=get_global_dtype(),
        )

    def _sample_body_commands(self, count: int) -> np.ndarray:
        ranges = np.asarray(
            [[-0.005, 0.005]] * 3 + [[-0.05, 0.05]] * 3,
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.random.uniform(ranges[:, 0], ranges[:, 1], size=(count, 6)),
            dtype=get_global_dtype(),
        )

    def _resample_due(self, timer, commands, slc, sampler, time_range, reset_mask):
        timer -= 1
        due = np.flatnonzero((timer <= 0) | reset_mask)
        if len(due) == 0:
            return
        commands[due, slc] = sampler(len(due))
        lo, hi = time_range
        timer[due] = np.asarray(
            np.random.uniform(lo, hi, size=len(due)) / self._cfg.ctrl_dt,
            dtype=np.int32,
        )

    def _maybe_resample_commands(self, state: NpEnvState) -> None:
        reset_mask = state.info.get("steps", np.zeros(self.num_envs)) == 0
        commands = state.info["commands"]
        self._resample_due(
            self._velocity_timer,
            commands,
            slice(0, 3),
            self._sample_velocity_commands,
            self._cfg.commands.resampling_time_range,
            reset_mask,
        )
        self._resample_due(
            self._head_timer,
            commands,
            slice(3, 7),
            self._sample_head_commands,
            self._cfg.commands.head_resampling_time_range,
            reset_mask,
        )
        self._resample_due(
            self._body_timer,
            commands,
            slice(7, 13),
            self._sample_body_commands,
            self._cfg.commands.body_resampling_time_range,
            reset_mask,
        )

    def _update_foot_history(self, *, sample_peak=True) -> None:
        dt = self._cfg.sim_dt if self._foot_substep_pending else 0.0
        self._foot_substep_pending = False
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
        self._last_foot_pos[fresh] = foot_pos[fresh]
        first_contact = contact & (~self._previous_contact) & (~fresh[:, None])
        previous_air_time = self._air_time.copy()
        self._landed_air_time[:] = np.where(
            first_contact, previous_air_time + dt, self._landed_air_time
        )
        self._air_time = np.where(contact, 0.0, previous_air_time + dt)
        self._contact_time[:] = np.where(contact, self._contact_time + dt, 0.0)
        if sample_peak:
            self._peak_foot_height = np.where(
                ~contact,
                np.maximum(self._peak_foot_height, foot_pos[:, :, 2]),
                self._peak_foot_height,
            )
        self._foot_force[:] = foot_force
        self._foot_pos[:] = foot_pos
        self._foot_vel[:] = foot_vel
        self._contact[:] = contact
        self._first_contact[:] = (
            contact & (self._contact_time > 0.0) & (self._contact_time < self._cfg.ctrl_dt + 1e-8)
        )
        self._last_foot_pos[:] = foot_pos
        self._previous_contact[:] = contact
        self._history_fresh[:] = False

    def _terminated(self, upvector: np.ndarray) -> np.ndarray:
        return upvector[:, 2] < np.cos(np.deg2rad(70.0))

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_curriculum()
        self._maybe_resample_commands(state)
        self._update_foot_history()
        gyro = self.get_gyro()
        projected_gravity = self._projected_gravity()
        upvector = -projected_gravity
        linvel = self.get_local_linvel()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        commands = state.info["commands"]
        last_actions = state.info.get(
            "current_actions", np.zeros((self.num_envs, NUM_ACTIONS), dtype=get_global_dtype())
        )
        observed_dof_pos = self._motor.quantize_position_feedback(dof_pos) + self._encoder_bias
        observed_dof_vel = self._motor.quantize_velocity_feedback(dof_vel)
        obs = self._compute_obs(
            self._rotate_imu(gyro),
            self._rotate_imu(upvector),
            observed_dof_pos,
            observed_dof_vel,
            last_actions,
            commands,
            critic_linvel=linvel,
            critic_gyro=gyro,
            critic_upvector=upvector,
            critic_dof_pos=dof_pos,
            critic_dof_vel=dof_vel,
        )
        # Rewards use privileged simulator truth, matching MJLab. Encoder
        # quantization and per-environment bias belong only to the actor
        # observation; feeding them into reward would shift each environment's
        # physical pose target by its sampled sensor error.
        reward = self._compute_reward(state, linvel, gyro, upvector, dof_pos)
        state.info.setdefault("log", {})["curriculum/effective_step"] = float(
            self._curriculum_step()
        )
        terminated = self._terminated(upvector)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _physical_cost_terms(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        torque: np.ndarray,
    ) -> dict[str, np.ndarray]:
        cfg = self._cfg.reward_config
        joint_range = self._joint_limits[:, 1] - self._joint_limits[:, 0]
        margin = joint_range * cfg.joint_limit_margin_fraction
        lower = np.maximum(self._joint_limits[:, 0] + margin - dof_pos, 0.0) / margin
        upper = np.maximum(dof_pos - (self._joint_limits[:, 1] - margin), 0.0) / margin
        rated = self._motor._rated_torque
        maximum_speed = self._motor._maximum_speed
        torque_ratio = np.divide(
            np.abs(torque), rated, out=np.zeros_like(torque), where=rated > 0.0
        )
        speed_ratio = np.divide(
            np.abs(dof_vel),
            maximum_speed,
            out=np.zeros_like(dof_vel),
            where=maximum_speed > 0.0,
        )
        rated_power = rated * self._motor._rated_speed
        power_ratio = np.divide(
            np.abs(torque * dof_vel),
            rated_power,
            out=np.zeros_like(torque),
            where=rated_power > 0.0,
        )
        i2t_ratio = np.divide(
            self._motor.i2t,
            self._motor._i2t_budget,
            out=np.zeros_like(self._motor.i2t),
            where=self._motor._i2t_budget > 0.0,
        )
        return {
            "joint_limit_proximity": np.sum(np.square(lower) + np.square(upper), axis=1),
            "rated_torque_excess": np.mean(np.square(np.maximum(torque_ratio - 1.0, 0.0)), axis=1),
            "joint_speed_excess": np.mean(np.square(np.maximum(speed_ratio - 1.0, 0.0)), axis=1),
            "mechanical_power": np.mean(power_ratio, axis=1),
            "i2t_usage": np.mean(i2t_ratio, axis=1),
        }

    def _log_physical_audit(
        self,
        log: dict,
        dof_vel: np.ndarray,
        torque: np.ndarray,
        action: np.ndarray,
    ) -> None:
        rated = self._motor._rated_torque
        maximum_speed = self._motor._maximum_speed
        target = self.default_angles[None, :] + action * self._cfg.control_config.action_scale
        log["audit/action_target_clip_fraction"] = (
            float(np.mean((target < self._target_low) | (target > self._target_high)))
            if self._cfg.clip_action_targets
            else 0.0
        )
        if self._cfg.clip_action_targets:
            target = np.clip(target, self._target_low, self._target_high)
        log["audit/rated_torque_exceed_fraction"] = float(np.mean(np.abs(torque) > rated))
        log["audit/no_load_speed_exceed_fraction"] = float(np.mean(np.abs(dof_vel) > maximum_speed))
        log["audit/action_target_outside_hard_limit_fraction"] = float(
            np.mean((target < self._joint_limits[:, 0]) | (target > self._joint_limits[:, 1]))
        )
        log["audit/i2t_usage_fraction"] = float(
            np.mean(
                np.divide(
                    self._motor.i2t,
                    self._motor._i2t_budget,
                    out=np.zeros_like(self._motor.i2t),
                    where=self._motor._i2t_budget > 0.0,
                )
            )
        )
        log["audit/peak_abs_torque_nm"] = float(np.max(np.abs(torque)))
        log["audit/peak_abs_joint_speed_rad_s"] = float(np.max(np.abs(dof_vel)))

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        cfg = self._cfg.reward_config
        cmd = state.info["commands"]
        current = state.info.get("current_actions", np.zeros((self.num_envs, NUM_ACTIONS)))
        last = state.info.get("last_actions", np.zeros_like(current))
        joint_rel = dof_pos - self.default_angles
        leg_ids = np.asarray([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        moving = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2]) > 0.01
        standing_std = np.asarray([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
        walking_std = np.asarray([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
        pose_std = np.where(moving[:, None], walking_std, standing_std)
        terms: dict[str, np.ndarray] = {}
        pose_score = leg_pose_reward(joint_rel[:, leg_ids], pose_std)
        upright_score = np.exp(-np.sum(np.square(upvector[:, :2]), axis=1) / cfg.upright_std**2)
        linear_score, angular_score = velocity_tracking_scores(
            linvel, gyro, cmd, cfg.linear_velocity_std, cfg.angular_velocity_std
        )
        terms["pose"] = pose_score
        terms["upright"] = upright_score
        terms["track_linear_velocity"] = linear_score
        terms["track_angular_velocity"] = angular_score
        world_gyro = np_quat_apply(self._backend.get_base_quat(), gyro)
        terms["body_ang_vel"] = np.sum(np.square(world_gyro[:, :2]), axis=1)
        angmom = self._backend.get_sensor_data(self._cfg.sensor.angular_momentum)
        terms["angular_momentum"] = np.sum(
            np.square(angmom / self._cfg.reward_config.angular_momentum_reference), axis=1
        )
        terms["action_rate_l2"] = np.sum(np.square(current - last), axis=1)
        below = np.maximum(self._soft_joint_limits[:, 0] - dof_pos, 0.0)
        above = np.maximum(dof_pos - self._soft_joint_limits[:, 1], 0.0)
        terms["dof_pos_limits"] = np.sum(below + above, axis=1)
        dof_vel = self.get_dof_vel()
        torque = np.concatenate(
            [self._backend.get_sensor_data(name) for name in self._cfg.sensor.actuator_torque],
            axis=1,
        )

        head_error = joint_rel[:, 5:9] - cmd[:, 3:7]
        terms["head_pose_tracking"] = np.mean(np.exp(-np.square(head_error / 0.5)), axis=1)
        alpha = min(1.0, self._cfg.ctrl_dt / 1.0)
        self._head_bias_ema = (1.0 - alpha) * self._head_bias_ema + alpha * head_error
        terms["head_pose_bias"] = -np.mean(np.abs(self._head_bias_ema), axis=1)
        terms["body_pose_tracking"] = np.zeros(self.num_envs)
        terms["self_collisions"] = np.sum(
            np.concatenate(
                [self._backend.get_sensor_data(name) for name in self._cfg.sensor.self_contacts],
                axis=1,
            )
            > 0.0,
            axis=1,
        )

        foot_pos = self._foot_pos
        contact = self._contact
        foot_vel = self._foot_vel
        terms["foot_slip"] = feet_slip_cost(foot_vel, contact, cmd)
        foot_height = foot_pos[:, :, 2]
        terms["foot_clearance"] = feet_clearance_cost(
            foot_height,
            foot_vel[:, :, :2],
            cmd,
            cfg.foot_target_height,
            cfg.foot_clearance_reference_height,
        )
        first_contact = self._first_contact
        peak_error = self._peak_foot_height / cfg.foot_target_height - 1.0
        terms["foot_swing_height"] = np.sum(np.square(peak_error) * first_contact, axis=1) * moving
        self._peak_foot_height[first_contact] = 0.0
        terms["air_time"] = feet_air_time_reward(
            self._air_time, cmd, cfg.air_time_threshold_min, cfg.air_time_threshold_max
        )

        reward = np.zeros(self.num_envs, dtype=get_global_dtype())
        log = state.info.setdefault("log", {})
        for name, value in terms.items():
            weighted = cfg.scales[name] * value
            reward += weighted.astype(reward.dtype, copy=False)
            log[f"Episode_Reward/{name}"] = float(np.mean(weighted))
        self._log_physical_audit(log, dof_vel, torque, current)
        log["curriculum/standing_fraction"] = self._standing_fraction
        log["curriculum/velocity_command_scale"] = self._velocity_command_scale
        log["curriculum/action_rate_weight"] = cfg.scales["action_rate_l2"]
        log["curriculum/head_pose_bias_weight"] = cfg.scales["head_pose_bias"]
        return reward * self._cfg.ctrl_dt
