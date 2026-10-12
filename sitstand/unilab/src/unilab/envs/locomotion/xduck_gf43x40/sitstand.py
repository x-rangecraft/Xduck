"""Commanded sit/stand task for XDuck V1.1.2 and the GF43X40 motor.

The actor keeps the 61D MicroDuck observation contract.  The first twist
command is a posture flag (0 stand, 1 sit); the remaining twist slots are zero.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import BaseNoiseConfig
from unilab.envs.locomotion.microduck_dm4310.velocity import MicroDuckDomainRandConfig, _stage
from unilab.envs.locomotion.microduck_xl330_official.velocity import OfficialCurriculumConfig
from unilab.envs.locomotion.xduck_gf43x40.velocity import (
    _SENSORS,
    _SOURCE_SCENE,
    XDuckCurriculumConfig,
    XDuckDRProvider,
    XDuckGF43X40VelocityCfg,
    XDuckGF43X40VelocityEnv,
    XDuckGFControlConfig,
    XDuckRewardConfig,
    _gf_domain_rand,
    _xduck_noise,
)

LEG_IDS = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
ANKLE_IDS = np.asarray((4, 13), dtype=np.intp)
_STAND_FRAGMENT = Path(__file__).with_name("stand_v112.xml")
# V1.1.2 pose checked with the GF motor and the user's neutral-head STAND:
# 30 s at ~0.218 m base height and ~0.41 degrees tilt, with both soles in contact.
# It is a static hold target; policy-controlled sit/stand transitions remain
# a separate training problem.
SIT_LEGS = np.asarray((0.0, 0.0, 0.625, -1.14, -0.35, 0.0, 0.0, -0.625, 1.14, 0.35))
SITSTAND_KP = [90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90]
SITSTAND_KD = [2.5, 2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0]
SAMPLE_BUDGET_RATIO = 4  # Official 4096 envs / XDuck 1024 envs; both roll out 24 steps.


def _sample_aligned_stages(name: str) -> list[dict]:
    # XDuckCurriculumConfig.__post_init__ scales lengths. Use upstream raw
    # stages here so the inherited post-init applies that scaling exactly once.
    stages = deepcopy(getattr(OfficialCurriculumConfig(), name))
    for stage in stages:
        stage["step"] *= SAMPLE_BUDGET_RATIO
    return stages


def _sitstand_scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(_SOURCE_SCENE),
        fragment_files=[str(_SENSORS), str(_STAND_FRAGMENT)],
        visual_model_file=str(_SOURCE_SCENE),
    )


def _sitstand_control() -> XDuckGFControlConfig:
    return XDuckGFControlConfig(
        action_scale=1.0,
        simulate_action_latency=False,
        Kp=90.0,
        Kd=2.5,
        joint_kp=SITSTAND_KP.copy(),
        joint_kd=SITSTAND_KD.copy(),
    )


def _sitstand_domain_rand() -> MicroDuckDomainRandConfig:
    cfg = _gf_domain_rand()
    cfg.mass_inertia_scale_range = [0.9, 1.1]
    cfg.imu_max_angle_deg = 12.0
    cfg.foot_friction_range = [0.4, 1.6]
    cfg.velocity_pushes = False
    return cfg


def _sitstand_noise() -> BaseNoiseConfig:
    cfg = _xduck_noise()
    cfg.level = 2.0
    return cfg


@dataclass
class SitStandRewardConfig(XDuckRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "posture_pose": 4.0,
            "posture_pose_l1": -1.0,
            "posture_height": 6.0,
            "posture_height_sharp": 1.0,
            "posture_height_l1": -3.5,
            "posture_composite": 4.0,
            "rise_bootstrap": 0.75,
            "upright_linear": 2.5,
            "upright_while_tall": 1.5,
            "head_pose_tracking": 0.75,
            "posture_stillness": 2.0,
            "action_rate_l2": -0.02,
            "descent_speed": -10.0,
            "rise_speed": 0.0,
            "gentle_motion": -0.05,
            "body_ang_vel": -0.05,
            "dof_pos_limits": -1.0,
            "self_collisions": -1.0,
        }
    )
    sit_height: float = 0.2182
    stand_height: float = 0.3129
    pose_std: float = 0.45
    height_std: float = 0.035
    sharp_height_std: float = 0.026
    upright_std: float = 0.35
    stillness_speed_std: float = 0.075
    composite_height_std: float | None = None
    composite_pose_std: float | None = None
    pose_match_per_joint: bool = False
    upright_from_tilt: bool = False
    stillness_band_full: float | None = None
    stillness_band_zero: float | None = None
    stillness_tilt_full_deg: float = 25.0
    stillness_tilt_zero_deg: float = 60.0
    max_descent_speed: float = 0.10
    max_rise_speed: float = 0.16
    rise_bootstrap_height_margin: float = 0.015
    upright_height_low_fraction: float = 0.25
    upright_height_high_fraction: float = 0.75


@dataclass
class SitStandCurriculumConfig(XDuckCurriculumConfig):
    action_rate_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -0.02},
            {"step": 4000 * 24, "value": -0.05},
            {"step": 12000 * 24, "value": -0.1},
        ]
    )
    head_range_stages: list[dict] = field(
        default_factory=lambda: _sample_aligned_stages("head_range_stages")
    )
    com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: _sample_aligned_stages("com_range_stages")
    )
    head_com_range_stages: list[dict[str, float]] = field(
        default_factory=lambda: _sample_aligned_stages("head_com_range_stages")
    )
    rise_speed_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 6000 * 24, "value": -5.0},
            {"step": 10000 * 24, "value": -10.0},
        ]
    )
    descent_speed_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": -10.0}]
    )
    standing_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": 0.0}]
    )
    push_stages: list[dict[str, float]] = field(
        default_factory=lambda: [{"step": 0, "value": 0.0}]
    )


@registry.envcfg("XDuckGF43X40SitStandFlat")
@dataclass
class XDuckSitStandCfg(XDuckGF43X40VelocityCfg):
    reference_keyframe: str = "STAND"
    # V1.1.2 joint limits minus the task's STAND keyframe, with 0.01 rad margin.
    # The velocity owner's offsets are relative to its different HOME keyframe.
    action_offset_low: list[float] = field(default_factory=lambda: [
        -1.19, -0.11, -3.374, -1.19, -1.106,
        -0.29, -1.19, -1.99, -0.49,
        -0.49, -1.19, -2.606, -1.49, -1.674,
    ])
    action_offset_high: list[float] = field(default_factory=lambda: [
        0.49, 1.49, 2.606, 1.39, 1.674,
        1.29, 1.19, 1.99, 0.49,
        1.19, 0.19, 2.374, 1.29, 1.106,
    ])
    scene: SceneCfg = field(default_factory=_sitstand_scene)
    control_config: XDuckGFControlConfig = field(default_factory=_sitstand_control)
    domain_rand: MicroDuckDomainRandConfig = field(default_factory=_sitstand_domain_rand)
    noise_config: BaseNoiseConfig = field(default_factory=_sitstand_noise)
    reward_config: SitStandRewardConfig = field(default_factory=SitStandRewardConfig)
    curriculum: SitStandCurriculumConfig = field(default_factory=SitStandCurriculumConfig)
    max_episode_seconds: float = 12.0
    posture_dwell_seconds: tuple[float, float] = (3.5, 6.5)
    sit_reset_probability: float = 0.5
    sit_command_probability: float = 0.5
    posture_slew_seconds: float = 2.0
    sit_reset_height: float = 0.22
    sit_leg_angles: list[float] = field(default_factory=lambda: SIT_LEGS.tolist())
    stand_reset_height: float = 0.313459
    ankle_target_offset_max_deg: float = 3.0
    # A 1.15 rad knee excursion needs about 2.3 policy units, rather than the
    # velocity owner's 9.6 units at its 0.12 rad action scale.
    policy_action_scale_rad: float = 0.5
    whole_body_mass_dr: bool = False


@dataclass
class SitStandAlignedRewardConfig(SitStandRewardConfig):
    """Official reward shape, with lengths and speeds adapted to XDuck."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            **SitStandRewardConfig().scales,
            # Official HOME<->SIT mean leg travel ~0.389 rad; XDuck ~0.283.
            "posture_pose_l1": -0.389 / 0.283,
            "posture_height": 1.0,
            "posture_composite": 3.0,
            "action_rate_l2": -0.025,
            "rise_bootstrap": 0.75 / math.sqrt(2.09),
            "descent_speed": -10.0 / math.sqrt(2.09),
            "body_ang_vel": -0.05 * 2.09,
        }
    )
    # Official 55 mm sit-to-stand travel becomes XDuck's measured 94.7 mm.
    # Scale height widths by travel, preserving reward vs fraction completed.
    height_std: float = 0.04 * (0.3129 - 0.2182) / 0.055
    sharp_height_std: float = 0.015 * (0.3129 - 0.2182) / 0.055
    composite_height_std: float = 0.03 * (0.3129 - 0.2182) / 0.055
    composite_pose_std: float = 0.4
    stillness_band_full: float = 0.012 * (0.3129 - 0.2182) / 0.055
    stillness_band_zero: float = 0.03 * (0.3129 - 0.2182) / 0.055
    # Knee excursion is 1.14 rather than the official 1.35 rad; other joints
    # retain the user's stand zero and the measured, stable XDuck sit pose.
    pose_std: float = 0.5 * 1.14 / 1.35
    pose_match_per_joint: bool = True
    upright_from_tilt: bool = True
    upright_std: float = 0.4
    stillness_speed_std: float = 0.05 * math.sqrt(2.09)
    # Gravity-similar speeds scale as sqrt(length); ramp target is 0.047 m/s.
    max_descent_speed: float = 0.05 * math.sqrt(2.09)
    max_rise_speed: float = 0.08 * math.sqrt(2.09)


@dataclass
class SitStandAlignedCurriculumConfig(SitStandCurriculumConfig):
    # Equal fresh samples: 4096 official / 1024 XDuck = 4 updates.
    # The 0.5-rad policy action scale squares into action-rate L2 weights.
    action_rate_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -0.025},
            {"step": 2000 * 24, "value": -0.05},
            {"step": 3000 * 24, "value": -0.10},
            {"step": 4000 * 24, "value": -0.15},
            {"step": 5000 * 24, "value": -0.20},
            {"step": 6000 * 24, "value": -0.25},
        ]
    )
    rise_speed_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": 0.0},
            {"step": 6000 * 24, "value": -5.0 / math.sqrt(2.09)},
            {"step": 10000 * 24, "value": -10.0 / math.sqrt(2.09)},
        ]
    )
    descent_speed_stages: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"step": 0, "value": -10.0 / math.sqrt(2.09)},
            {"step": 2000 * 24, "value": -20.0 / math.sqrt(2.09)},
        ]
    )


@registry.envcfg("XDuckGF43X40SitStandAlignedFlat")
@dataclass
class XDuckSitStandAlignedCfg(XDuckSitStandCfg):
    reward_config: SitStandAlignedRewardConfig = field(default_factory=SitStandAlignedRewardConfig)
    curriculum: SitStandAlignedCurriculumConfig = field(default_factory=SitStandAlignedCurriculumConfig)


class SitStandDRProvider(XDuckDRProvider):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        if env.cfg.whole_body_mass_dr:
            scale = env._mass_inertia_scale[env_ids]
            plan.randomization.body_mass = env._base_body_mass[None, :] * scale[:, None]
            plan.randomization.body_inertia = env._base_body_inertia[None, :, :] * scale[:, None, None]
        sit = np.random.random(len(env_ids)) < env.cfg.sit_reset_probability
        plan.qpos[sit, 2] = env.cfg.sit_reset_height
        plan.qpos[~sit, 2] = env.cfg.stand_reset_height
        plan.qpos[np.ix_(sit, 7 + LEG_IDS)] = env._sit_legs
        env._reset_clearance.lift_to_clearance(plan.qpos, env.cfg.reset_clearance_m)
        env._posture_alpha[env_ids] = sit.astype(get_global_dtype())
        return plan


@registry.env("XDuckGF43X40SitStandFlat", sim_backend="mujoco")
class XDuckSitStandEnv(XDuckGF43X40VelocityEnv):
    _cfg: XDuckSitStandCfg

    def _make_dr_provider(self):
        return SitStandDRProvider()

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if not (0.0 <= cfg.sit_reset_probability <= 1.0):
            raise ValueError("sit_reset_probability must be in [0, 1]")
        if not (0.0 <= cfg.sit_command_probability <= 1.0):
            raise ValueError("sit_command_probability must be in [0, 1]")
        if cfg.posture_slew_seconds <= 0.0:
            raise ValueError("posture_slew_seconds must be positive")
        reward = cfg.reward_config
        if not 0 <= reward.upright_height_low_fraction < reward.upright_height_high_fraction <= 1:
            raise ValueError("upright height fractions must satisfy 0 <= low < high <= 1")
        if not np.isfinite(reward.rise_bootstrap_height_margin) or reward.rise_bootstrap_height_margin < 0:
            raise ValueError("rise bootstrap height margin must be finite and nonnegative")
        if not np.isfinite(cfg.ankle_target_offset_max_deg) or cfg.ankle_target_offset_max_deg < 0:
            raise ValueError("ankle_target_offset_max_deg must be finite and nonnegative")
        cfg.commands.resampling_time_range = list(cfg.posture_dwell_seconds)
        self._posture_alpha = np.zeros(num_envs, dtype=get_global_dtype())
        self._ankle_target_offset = np.zeros((num_envs, 14), dtype=get_global_dtype())
        self._prev_trunk_vz = np.zeros(num_envs, dtype=get_global_dtype())
        self._accel_history_fresh = np.ones(num_envs, dtype=bool)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._sit_legs = np.asarray(cfg.sit_leg_angles, dtype=self.default_angles.dtype)
        if self._sit_legs.shape != (10,) or not np.isfinite(self._sit_legs).all():
            raise ValueError("sit_leg_angles requires ten finite leg angles")
        if np.any(self._sit_legs < self._target_low[LEG_IDS]) or np.any(
            self._sit_legs > self._target_high[LEG_IDS]
        ):
            raise ValueError("sit_leg_angles outside motor target limits")
        limits = self._backend.get_joint_range()[LEG_IDS]
        if np.any(self._sit_legs <= limits[:, 0]) or np.any(self._sit_legs >= limits[:, 1]):
            raise ValueError("sit posture is outside model joint limits")

    def _reset_episode_history(self, env_ids):
        super()._reset_episode_history(env_ids)
        ids = np.asarray(env_ids, dtype=np.intp)
        self._prev_trunk_vz[ids] = 0.0
        self._accel_history_fresh[ids] = True
        # The inherited velocity owner samples these once at construction.
        # Sit/stand episodes need a fresh, non-accumulating DR draw on reset.
        self._mass_inertia_scale[ids] = np.random.uniform(
            *self._cfg.domain_rand.mass_inertia_scale_range, size=len(ids)
        )
        self._foot_friction[ids] = np.random.uniform(
            *self._cfg.domain_rand.foot_friction_range, size=len(ids)
        )
        axis = np.random.normal(size=(len(ids), 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        angle = np.random.uniform(
            0.0, math.radians(self._cfg.domain_rand.imu_max_angle_deg), size=len(ids)
        )
        skew = np.zeros((len(ids), 3, 3), dtype=self._imu_rotation.dtype)
        skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
        eye = np.eye(3, dtype=self._imu_rotation.dtype)
        self._imu_rotation[ids] = (
            eye
            + np.sin(angle)[:, None, None] * skew
            + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
        )
        limit = np.deg2rad(self._cfg.ankle_target_offset_max_deg)
        self._ankle_target_offset[ids] = 0.0
        self._ankle_target_offset[np.ix_(ids, ANKLE_IDS)] = np.random.uniform(
            -limit, limit, size=(len(ids), len(ANKLE_IDS))
        )

    def _physics_step_action_delay(self, backend, ctrl):
        # Fixed per-episode ankle zero error enters the motor target, after
        # policy action scaling and before the 1-kHz GF torque calculation.
        target = np.clip(
            ctrl + self._ankle_target_offset, self._target_low, self._target_high
        )
        return super()._physics_step_action_delay(backend, target)

    def _sample_velocity_commands(self, count):
        command = np.zeros((count, 3), dtype=get_global_dtype())
        command[:, 0] = np.random.random(count) < self._cfg.sit_command_probability
        return command

    def _sample_body_commands(self, count):
        return np.zeros((count, 6), dtype=get_global_dtype())

    def _maybe_resample_commands(self, state):
        super()._maybe_resample_commands(state)
        target = state.info["commands"][:, 0]
        delta = self._cfg.ctrl_dt / self._cfg.posture_slew_seconds
        self._posture_alpha += np.clip(target - self._posture_alpha, -delta, delta)

    def _terminated(self, upvector):
        del upvector
        return np.zeros(self.num_envs, dtype=bool)

    def _update_curriculum(self) -> None:
        super()._update_curriculum()
        self._cfg.reward_config.scales["rise_speed"] = float(
            _stage(self._cfg.curriculum.rise_speed_stages, self._curriculum_step())
        )
        self._cfg.reward_config.scales["descent_speed"] = float(
            _stage(self._cfg.curriculum.descent_speed_stages, self._curriculum_step())
        )

    def _adapt_posture_terms(self, terms, state, gyro):
        """Task-specific reward adaptation before weighting and accounting."""
        return terms

    def _action_rate_cost(self, state):
        # Preserve this task's original raw-action definition by default while
        # dispatching versioned decoders through the same reward owner hook.
        current = state.info.get("current_actions", np.zeros((self.num_envs, 14)))
        last = state.info.get("last_actions", np.zeros_like(current))
        return np.sum(np.square(current - last), axis=1)

    def _head_reward_scores(self, state, head_error):
        return (np.exp(-np.mean(np.square(head_error / 0.4), axis=1)),
                np.mean(np.exp(-np.square(head_error / 0.5)), axis=1))

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        cfg = self._cfg.reward_config
        alpha = self._posture_alpha
        target_legs = (
            self.default_angles[LEG_IDS][None, :] * (1.0 - alpha[:, None])
            + self._sit_legs[None, :] * alpha[:, None]
        )
        target_height = cfg.stand_height + (cfg.sit_height - cfg.stand_height) * alpha
        height = self._backend.get_base_pos()[:, 2]
        pose_error = dof_pos[:, LEG_IDS] - target_legs
        height_error = height - target_height
        if cfg.pose_match_per_joint:
            pose = np.mean(np.exp(-np.square(pose_error / cfg.pose_std)), axis=1)
        else:
            pose = np.exp(-np.mean(np.square(pose_error / cfg.pose_std), axis=1))
        composite_pose = (
            pose
            if cfg.composite_pose_std is None
            else np.exp(-np.mean(np.square(pose_error / cfg.composite_pose_std), axis=1))
        )
        height_score = np.exp(-np.square(height_error / cfg.height_std))
        composite_height_score = (
            height_score
            if cfg.composite_height_std is None
            else np.exp(-np.square(height_error / cfg.composite_height_std))
        )
        sharp_height = np.exp(-np.square(height_error / cfg.sharp_height_std))
        tilt_sq = (
            np.clip(1.0 - upvector[:, 2], 0.0, 2.0)
            if cfg.upright_from_tilt
            else np.sum(np.square(upvector[:, :2]), axis=1)
        )
        upright = np.exp(-tilt_sq / cfg.upright_std**2)
        current = state.info.get("current_actions", np.zeros((self.num_envs, 14)))
        last = state.info.get("last_actions", np.zeros_like(current))
        base_lin_vel = self._backend.get_base_lin_vel()
        vertical_speed = base_lin_vel[:, 2]
        vertical_accel = np.abs(vertical_speed - self._prev_trunk_vz) / self._cfg.ctrl_dt
        vertical_accel[self._accel_history_fresh] = 0.0
        self._prev_trunk_vz[:] = vertical_speed
        self._accel_history_fresh[:] = False
        head_error = dof_pos[:, 5:9] - self.default_angles[5:9] - state.info["commands"][:, 3:7]
        head_score, head_tracking = self._head_reward_scores(state, head_error)
        height_span = cfg.stand_height - cfg.sit_height
        height_gate = np.clip(
            (height - (cfg.sit_height + cfg.upright_height_low_fraction * height_span))
            / ((cfg.upright_height_high_fraction - cfg.upright_height_low_fraction) * height_span),
            0.0,
            1.0,
        )
        height_gate = height_gate * height_gate * (3.0 - 2.0 * height_gate)
        if cfg.stillness_band_full is None or cfg.stillness_band_zero is None:
            stillness_gate = height_score * upright
        else:
            z_band = np.clip(
                (cfg.stillness_band_zero - np.abs(height_error))
                / (cfg.stillness_band_zero - cfg.stillness_band_full),
                0.0, 1.0,
            )
            z_band = z_band * z_band * (3.0 - 2.0 * z_band)
            cos_tilt = upvector[:, 2]
            cos_full = math.cos(math.radians(cfg.stillness_tilt_full_deg))
            cos_zero = math.cos(math.radians(cfg.stillness_tilt_zero_deg))
            tilt_band = np.clip((cos_tilt - cos_zero) / (cos_full - cos_zero), 0.0, 1.0)
            tilt_band = tilt_band * tilt_band * (3.0 - 2.0 * tilt_band)
            stillness_gate = z_band * tilt_band
        below = np.maximum(self._soft_joint_limits[:, 0] - dof_pos, 0.0)
        above = np.maximum(dof_pos - self._soft_joint_limits[:, 1], 0.0)
        terms = {
            "posture_pose": pose,
            "posture_pose_l1": np.mean(np.abs(pose_error), axis=1),
            "posture_height": height_score,
            "posture_height_sharp": sharp_height,
            "posture_height_l1": np.abs(height_error),
            "posture_composite": composite_pose * composite_height_score * upright * head_score,
            "rise_bootstrap": np.clip(vertical_speed, 0.0, cfg.max_rise_speed)
            * (height < cfg.stand_height + cfg.rise_bootstrap_height_margin)
            * (1.0 - state.info["commands"][:, 0]),
            "upright_linear": upvector[:, 2],
            "upright_while_tall": upvector[:, 2] * height_gate,
            "head_pose_tracking": head_tracking,
            "posture_stillness": stillness_gate
            * np.exp(-np.square(np.linalg.norm(base_lin_vel, axis=1) / cfg.stillness_speed_std))
            * (np.abs(state.info["commands"][:, 0] - alpha) < 0.02),
            "action_rate_l2": self._action_rate_cost(state),
            "descent_speed": np.maximum(-vertical_speed - cfg.max_descent_speed, 0.0),
            "rise_speed": np.maximum(vertical_speed - cfg.max_rise_speed, 0.0),
            "gentle_motion": vertical_accel,
            "body_ang_vel": np.sum(np.square(gyro[:, :2]), axis=1),
            "dof_pos_limits": np.sum(below + above, axis=1),
            "self_collisions": self._backend.get_sensor_data("self_collision_found")[:, 0],
        }
        terms = self._adapt_posture_terms(terms, state, gyro)
        reward = np.zeros(self.num_envs, dtype=get_global_dtype())
        log = state.info.setdefault("log", {})
        for name, value in terms.items():
            weighted = cfg.scales[name] * value
            reward += weighted.astype(reward.dtype, copy=False)
            self._episode_reward_sums[name] += weighted * self._cfg.ctrl_dt
            log.pop(f"Episode_Reward/{name}", None)
            log[f"sitstand/term_{name}"] = float(np.mean(weighted * self._cfg.ctrl_dt))
        log["sitstand/sit_command_fraction"] = float(np.mean(state.info["commands"][:, 0]))
        log["sitstand/posture_alpha_mean"] = float(np.mean(alpha))
        log["sitstand/height_error_mean"] = float(np.mean(np.abs(height_error)))
        return reward * self._cfg.ctrl_dt


@registry.env("XDuckGF43X40SitStandAlignedFlat", sim_backend="mujoco")
class XDuckSitStandAlignedEnv(XDuckSitStandEnv):
    """Versioned aligned task; the original checkpoint keeps its recipe."""
