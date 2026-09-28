"""V1.1.2 mechanics with the baseline GF43X40 plant and optional dry friction."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedAsset,
)
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
from unilab.envs.locomotion.microduck_enlarged_111.tasks import MicroDuckEnlarged111StandFlatEnv
from unilab.envs.locomotion.microduck_enlarged_112.friction import V112DryFriction
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

ASSET_ROOT = ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_112"
JOINT_NAMES = (
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

FIXED_REMOTE_SEQUENCE = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.2, 0.0, 0.4],
        [0.2, 0.0, -0.4],
        [0.0, 0.0, 0.0],
        [-0.15, 0.0, 0.0],
        [-0.12, 0.0, 0.3],
        [-0.12, 0.0, -0.3],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _scene():
    return SceneCfg(model_file=str(ASSET_ROOT / "scene_flat.xml"))


@dataclass
class V112Asset(MicroDuckEnlargedAsset):
    base_name: str = "base_link"
    foot_body_names: tuple[str, str] = ("left_ankle_pitch_joint", "right_ankle_pitch")
    foot_site_offsets: tuple = (
        (0.053, -0.015445, -0.0291),
        (-0.053, -0.015513, -0.02921),
    )


@dataclass
class V112DomainRand(MicroDuckEnlarged107DomainRandConfig):
    push_body_name: str | None = "base_link"


class V112Reset(MicroDuckEnlarged107DomainRandomizationProvider):
    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ):
        return super()._compute_reset_obs(
            env,
            env_ids,
            info_updates,
            linvel,
            env._imu_to_base(gyro),
            gravity,
            dof_pos,
            dof_vel,
        )


class V112WalkReset(V112Reset):
    def _sample_commands(self, env, num_reset):
        if getattr(env.cfg, "fixed_remote_sequence", False):
            return np.broadcast_to(
                env._fixed_remote_sequence_commands[0], (num_reset, 3)
            ).copy()
        remote_p = np.asarray(env.cfg.remote_command_probabilities, dtype=np.float64)
        if remote_p.size:
            modes = np.random.choice(5, size=num_reset, p=remote_p)
            low, high = np.asarray(env.cfg.commands.vel_limit, dtype=np.float32)
            commands = np.zeros((num_reset, 3), dtype=np.float32)

            # Official-simulator-style remote commands: stand, forward,
            # backward, forward arc, backward arc.  Lateral velocity is never
            # requested and yaw is only paired with nonzero fore-aft motion.
            for mode, direction in ((1, 1), (2, -1), (3, 1), (4, -1)):
                mask = modes == mode
                count = int(mask.sum())
                if count == 0:
                    continue
                cap = high[0] if direction > 0 else -low[0]
                commands[mask, 0] = direction * np.random.uniform(0.08, cap, count)
                if mode >= 3:
                    yaw_cap = max(abs(float(low[2])), abs(float(high[2])))
                    signs = np.random.choice([-1.0, 1.0], count)
                    commands[mask, 2] = signs * np.random.uniform(0.2, yaw_cap, count)
            return commands
        if not env.cfg.command_mode_probabilities:
            return super()._sample_commands(env, num_reset)
        p = np.asarray(env.cfg.command_mode_probabilities, dtype=np.float64)
        modes = np.random.choice(6, size=num_reset, p=p)
        low, high = np.asarray(env.cfg.commands.vel_limit, dtype=np.float32)
        commands = np.random.uniform(low, high, size=(num_reset, 3)).astype(np.float32)
        # Mutually exclusive stand / forward / backward / lateral / yaw / mixed.
        commands[modes == 0] = 0
        for mode, axis, sign in ((1, 0, 1), (2, 0, -1), (3, 1, 0), (4, 2, 0)):
            mask = modes == mode
            count = int(mask.sum())
            commands[mask] = 0
            signs = np.random.choice([-1, 1], count) if sign == 0 else np.full(count, sign)
            cap = np.where(signs > 0, high[axis], -low[axis])
            floor = 0.2 if axis == 2 else 0.08
            commands[mask, axis] = signs * np.random.uniform(floor, cap)
        return commands

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        fixed_mask = getattr(env, "_fixed_remote_sequence_mask", None)
        if fixed_mask is not None:
            selected = np.asarray(fixed_mask, dtype=bool)[np.asarray(env_ids, dtype=np.intp)]
            if np.any(selected):
                plan.info_updates["commands"][selected] = env._fixed_remote_sequence_commands[0]
        return plan


class V112Mechanics(V112DryFriction):
    JOINT_NAMES = JOINT_NAMES
    HIP_YAW_INDICES = np.asarray((0, 9), dtype=np.intp)
    LEG_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)

    def __init__(self, cfg, *args, **kwargs):
        root = ET.parse(ASSET_ROOT / "microduck_enlarged_112.xml")
        imu = root.find("./worldbody/body/body[@name='IMU']")
        if imu is None or imu.find("site[@name='imu']") is None:
            raise ValueError("Expected the fixed V1.1.2 IMU mounting body")
        self._imu_quat = np.fromstring(imu.get("quat"), sep=" ").astype(np.float32)
        self._hip_yaw_target_rate_limit = float(cfg.hip_yaw_target_rate_limit_rad_s)
        self._hip_yaw_running_rate_limit = float(cfg.hip_yaw_running_rate_limit_rad_s)
        self._hip_yaw_startup_duration = float(cfg.hip_yaw_startup_duration_s)
        if not np.isfinite(self._hip_yaw_target_rate_limit) or self._hip_yaw_target_rate_limit < 0:
            raise ValueError("hip_yaw_target_rate_limit_rad_s must be finite and non-negative")
        if (
            not np.isfinite(self._hip_yaw_running_rate_limit)
            or self._hip_yaw_running_rate_limit < 0
        ):
            raise ValueError("hip_yaw_running_rate_limit_rad_s must be finite and non-negative")
        if not np.isfinite(self._hip_yaw_startup_duration) or self._hip_yaw_startup_duration < 0:
            raise ValueError("hip_yaw_startup_duration_s must be finite and non-negative")
        self._hip_yaw_startup_steps = int(
            np.ceil(self._hip_yaw_startup_duration / float(cfg.ctrl_dt) - 1e-12)
        )
        super().__init__(cfg, *args, **kwargs)

    def _imu_to_base(self, value):
        return np_quat_apply(np.broadcast_to(self._imu_quat, (len(value), 4)), value)

    def get_gyro(self):
        return self._imu_to_base(super().get_gyro())

    def get_local_linvel(self):
        # Reward tracks base translation, without the IMU lever-arm velocity.
        return np_quat_apply_inverse(
            self._backend.get_base_quat(),
            self._backend.get_base_lin_vel(),
        )

    def _make_domain_randomization_provider(self):
        return V112Reset()

    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["joint_limit_clearance"] = self._joint_limit_clearance_penalty

    def _joint_limit_clearance_penalty(self, ctx):
        limits = np.asarray(self._backend.get_joint_range(), dtype=np.float32)
        margin = np.minimum(ctx.dof_pos - limits[:, 0], limits[:, 1] - ctx.dof_pos)
        threshold = self._v112_joint_clearance_reserve
        deficit = np.maximum(threshold - margin, 0.0) / threshold
        barrier = np.maximum(1.0 - margin / self._v112_barrier_margin, 0.0)
        cost = np.square(deficit) + self._v112_barrier_scale * np.square(barrier)
        return np.max(cost, axis=1).astype(np.float32)

    def _init_motor_constraints(self):
        super()._init_motor_constraints()
        # Keep HOME inside the travel reserve when a revised limit is close.
        # Historical neck=-0.25 HOME had only 0.05 rad clearance. Driver HOME
        # centers the neck, so the existing rule restores the full inset.
        limits = np.asarray(self._backend.get_joint_range())
        clearance = np.minimum(
            self.default_angles - limits[:, 0], limits[:, 1] - self.default_angles
        )
        if np.any(clearance <= 0):
            raise ValueError("V1.1.2 HOME must be strictly inside the mechanical limits")
        threshold = float(self._cfg.joint_limit_clearance_threshold_rad)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("joint_limit_clearance_threshold_rad must be finite and positive")
        # All 14 driven joints need physical travel reserve, including the neck.
        # Reserve at most half the HOME clearance so the rest pose stays unpenalized.
        # Cache at initialization; never reparse the asset on the reward hot path.
        self._v112_joint_clearance_reserve = np.minimum(threshold, clearance * 0.5)
        self._v112_barrier_margin = float(self._cfg.joint_limit_barrier_margin_rad)
        self._v112_barrier_scale = float(self._cfg.joint_limit_barrier_scale)
        if (
            not np.isfinite(self._v112_barrier_margin)
            or self._v112_barrier_margin <= 0
            or not np.isfinite(self._v112_barrier_scale)
            or self._v112_barrier_scale < 0
        ):
            raise ValueError("Joint-limit barrier requires positive margin and non-negative scale")
        if np.any(clearance <= self._v112_barrier_margin):
            raise ValueError("HOME must remain outside the joint-limit barrier region")
        self._v112_termination_margin = float(self._cfg.joint_limit_termination_margin_rad)
        if (
            not np.isfinite(self._v112_termination_margin)
            or self._v112_termination_margin < 0
            or np.any(clearance <= self._v112_termination_margin)
        ):
            raise ValueError("Joint-limit termination margin must be finite and leave HOME valid")
        self._v112_physical_joint_limits = limits.copy()
        nominal_inset = np.deg2rad(self._cfg.control_config.joint_limit_margin_deg)
        inset = np.where(clearance > nominal_inset, nominal_inset, clearance * 0.5)
        self._v110_safe_joint_low = limits[:, 0] + inset
        self._v110_safe_joint_high = limits[:, 1] - inset
        outer_roll_margin = np.deg2rad(float(self._cfg.hip_roll_outer_limit_margin_deg))
        if not np.isfinite(outer_roll_margin) or outer_roll_margin < nominal_inset:
            raise ValueError("hip_roll_outer_limit_margin_deg must be finite and at least nominal")
        # The two physical outward directions are left-roll positive and
        # right-roll negative.  These are the directions that carry the
        # lateral landing impulse and need more overshoot reserve.
        self._v110_safe_joint_high[1] = limits[1, 1] - outer_roll_margin
        self._v110_safe_joint_low[10] = limits[10, 0] + outer_roll_margin
        if np.any(self.default_angles <= self._v110_safe_joint_low) or np.any(
            self.default_angles >= self._v110_safe_joint_high
        ):
            raise ValueError("V1.1.2 HOME must remain inside the request governor")

    def apply_action(self, actions, state):
        if not np.isfinite(actions).all():
            raise ValueError("V1.1.2 actions must be finite")
        governed = np.asarray(actions, dtype=np.float32).copy()
        if self._hip_yaw_target_rate_limit > 0:
            previous = np.asarray(
                state.info.get("v112_governed_actions", np.zeros_like(governed)),
                dtype=governed.dtype,
            )
            # A reset always starts from HOME, even if the vectorized info
            # buffer still contains the preceding episode's final request.
            previous = np.where(np.asarray(state.info["steps"])[:, None] == 0, 0, previous)
            rate_limit = np.where(
                np.asarray(state.info["steps"]) < self._hip_yaw_startup_steps,
                self._hip_yaw_target_rate_limit,
                self._hip_yaw_running_rate_limit,
            )
            max_delta = rate_limit[:, None] * float(self._cfg.ctrl_dt)
            indices = self.HIP_YAW_INDICES
            governed[:, indices] = previous[:, indices] + np.clip(
                governed[:, indices] - previous[:, indices], -max_delta, max_delta
            )
            state.info["v112_policy_actions_raw"] = np.asarray(actions).copy()
            state.info["v112_governed_actions"] = governed.copy()
        # The inherited ctrl encodes torque after the fitted response. Clipping
        # it to a joint range here would bypass the existing actuator plant.
        return super().apply_action(governed, state)


@registry.envcfg("MicroDuckEnlarged112StandFlat")
@dataclass
class MicroDuckEnlarged112StandFlatCfg(MicroDuckEnlarged110StandFlatCfg):
    dry_friction_enabled: bool = False
    scene: SceneCfg = field(default_factory=_scene)
    asset: V112Asset = field(default_factory=V112Asset)
    domain_rand: V112DomainRand = field(default_factory=V112DomainRand)
    hip_yaw_target_rate_limit_rad_s: float = 0.0
    hip_yaw_running_rate_limit_rad_s: float = 0.0
    hip_yaw_startup_duration_s: float = 0.0
    hip_roll_outer_limit_margin_deg: float = 5.0
    joint_limit_clearance_threshold_rad: float = 0.1
    joint_limit_barrier_margin_rad: float = 0.01
    joint_limit_barrier_scale: float = 0.0
    joint_limit_termination_margin_rad: float = 0.0


@registry.env("MicroDuckEnlarged112StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged112StandFlatEnv(V112Mechanics, MicroDuckEnlarged110StandFlatEnv):
    """Balance around the new HOME pose."""

    _leg_pose_tracking = MicroDuckEnlarged111StandFlatEnv._leg_pose_tracking


@registry.envcfg("MicroDuckEnlarged112WalkFlat")
@dataclass
class MicroDuckEnlarged112WalkFlatCfg(MicroDuckEnlarged110WalkFlatCfg):
    dry_friction_enabled: bool = False
    scene: SceneCfg = field(default_factory=_scene)
    asset: V112Asset = field(default_factory=V112Asset)
    domain_rand: V112DomainRand = field(default_factory=V112DomainRand)
    # During the loaded first 0.2 s, toe-out requests are capped at 0.01 rad
    # per 20 ms.  Afterwards the cap is 0.04 rad per frame
    # so gait control retains sufficient bandwidth.  The existing motor plant
    # remains the sole owner of torque, speed, delay and friction behavior.
    hip_yaw_target_rate_limit_rad_s: float = 0.5
    hip_yaw_running_rate_limit_rad_s: float = 2.0
    hip_yaw_startup_duration_s: float = 0.2
    hip_roll_outer_limit_margin_deg: float = 10.0
    joint_limit_clearance_threshold_rad: float = 0.1
    joint_limit_barrier_margin_rad: float = 0.01
    joint_limit_barrier_scale: float = 0.0
    joint_limit_termination_margin_rad: float = 0.0
    command_mode_probabilities: tuple[float, ...] = ()
    # Optional official-simulator-style command mix: stand, forward,
    # backward, forward arc, backward arc.  When populated it takes priority
    # over command_mode_probabilities and never emits strafe or pure yaw.
    remote_command_probabilities: tuple[float, ...] = ()
    command_resample_interval_s: float = 0.0
    fixed_remote_sequence: bool = False
    # Training-only blend. The selected vectorized environments follow the
    # frozen nine-stage sequence; the remainder retain random remote commands.
    fixed_remote_sequence_fraction: float = 0.0
    fixed_remote_sequence_commands: tuple[tuple[float, float, float], ...] = ()
    linear_velocity_tracking_tau_s: float = 0.0
    raw_target_feasibility_scale_rad: float = 0.0
    # The official Micro Duck gait alternates feet on an approximately 0.4 s
    # cycle.  This reference is reward-only: phase is not added to the actor
    # observation and therefore does not change the 61D deployment contract.
    phase_gait_period_s: float = 0.4
    phase_foot_height_std_m: float = 0.015
    # Reward-only idle motion scales; never filter or clamp physical velocity.
    idle_leg_velocity_deadband_rad_s: float = 0.15
    idle_leg_velocity_scale_rad_s: float = 1.0
    idle_home_target_scale_rad: float = 0.25
    # Long-term visual head-shell roll relative to the base.  This deliberately
    # does not ask the four neck/head joints to return to HOME independently.
    head_level_bias_tau_s: float = 1.0
    head_level_moving_multiplier: float = 1.0
    head_level_bias_schedule_steps: tuple[int, ...] = (0,)
    head_level_bias_schedule_scales: tuple[float, ...] = (0.0,)
    # Optional terminal cost used only while training robustness refinements.
    # It changes rewards, never observations, actions, physics or inference.
    termination_penalty: float = 0.0


@registry.env("MicroDuckEnlarged112WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged112WalkFlatEnv(V112Mechanics, MicroDuckEnlarged110WalkFlatEnv):
    """Velocity tracking using the V1.1.2 joint signs and established motor plant."""

    def __init__(self, cfg, *args, **kwargs):
        self._idle_velocity_deadband = float(cfg.idle_leg_velocity_deadband_rad_s)
        self._idle_velocity_scale = float(cfg.idle_leg_velocity_scale_rad_s)
        if not np.isfinite(self._idle_velocity_deadband) or self._idle_velocity_deadband < 0:
            raise ValueError("idle_leg_velocity_deadband_rad_s must be finite and non-negative")
        if not np.isfinite(self._idle_velocity_scale) or self._idle_velocity_scale <= 0:
            raise ValueError("idle_leg_velocity_scale_rad_s must be finite and positive")
        self._idle_home_scale = float(cfg.idle_home_target_scale_rad)
        if not np.isfinite(self._idle_home_scale) or self._idle_home_scale <= 0:
            raise ValueError("idle_home_target_scale_rad must be finite and positive")
        self._raw_target_scale = float(cfg.raw_target_feasibility_scale_rad)
        if not np.isfinite(self._raw_target_scale) or self._raw_target_scale < 0:
            raise ValueError("raw_target_feasibility_scale_rad must be finite and non-negative")
        tau = float(cfg.linear_velocity_tracking_tau_s)
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("linear_velocity_tracking_tau_s must be finite and non-negative")
        self._linear_tracking_alpha = None if tau == 0 else -np.expm1(-float(cfg.ctrl_dt) / tau)
        self._head_level_tau = float(cfg.head_level_bias_tau_s)
        if not np.isfinite(self._head_level_tau) or self._head_level_tau <= 0:
            raise ValueError("head_level_bias_tau_s must be finite and positive")
        self._head_level_moving_multiplier = float(cfg.head_level_moving_multiplier)
        if (
            not np.isfinite(self._head_level_moving_multiplier)
            or self._head_level_moving_multiplier < 1.0
        ):
            raise ValueError("head_level_moving_multiplier must be finite and at least one")
        self._termination_penalty = float(cfg.termination_penalty)
        if not np.isfinite(self._termination_penalty) or self._termination_penalty < 0.0:
            raise ValueError("termination_penalty must be finite and non-negative")
        fixed_fraction = float(cfg.fixed_remote_sequence_fraction)
        if not np.isfinite(fixed_fraction) or not 0.0 <= fixed_fraction <= 1.0:
            raise ValueError("fixed_remote_sequence_fraction must be within [0, 1]")
        if cfg.fixed_remote_sequence and fixed_fraction > 0.0:
            raise ValueError(
                "fixed_remote_sequence and fixed_remote_sequence_fraction are mutually exclusive"
            )
        fixed_commands = np.asarray(
            cfg.fixed_remote_sequence_commands or FIXED_REMOTE_SEQUENCE,
            dtype=np.float32,
        )
        if (
            fixed_commands.ndim != 2
            or fixed_commands.shape[1] != 3
            or fixed_commands.shape[0] == 0
            or not np.isfinite(fixed_commands).all()
        ):
            raise ValueError("fixed_remote_sequence_commands must be finite Nx3 commands")
        self._fixed_remote_sequence_commands = fixed_commands
        p = np.asarray(cfg.command_mode_probabilities, dtype=np.float64)
        if p.size and (
            p.shape != (6,)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("command_mode_probabilities must be six probabilities summing to one")
        remote_p = np.asarray(cfg.remote_command_probabilities, dtype=np.float64)
        if remote_p.size and (
            remote_p.shape != (5,)
            or not np.isfinite(remote_p).all()
            or np.any(remote_p < 0)
            or not np.isclose(remote_p.sum(), 1)
        ):
            raise ValueError(
                "remote_command_probabilities must be five probabilities summing to one"
            )
        if p.size and remote_p.size:
            raise ValueError(
                "command_mode_probabilities and remote_command_probabilities are mutually exclusive"
            )
        interval = float(cfg.command_resample_interval_s)
        if not np.isfinite(interval) or interval < 0:
            raise ValueError("command_resample_interval_s must be finite and non-negative")
        self._phase_gait_period = float(cfg.phase_gait_period_s)
        self._phase_foot_height_std = float(cfg.phase_foot_height_std_m)
        if not np.isfinite(self._phase_gait_period) or self._phase_gait_period <= 0:
            raise ValueError("phase_gait_period_s must be finite and positive")
        if not np.isfinite(self._phase_foot_height_std) or self._phase_foot_height_std <= 0:
            raise ValueError("phase_foot_height_std_m must be finite and positive")
        self._command_resample_steps = round(interval / float(cfg.ctrl_dt))
        if (cfg.fixed_remote_sequence or fixed_fraction > 0.0) and self._command_resample_steps <= 0:
            raise ValueError("fixed remote sequence training requires command resampling")
        if p.size:
            low, high = np.asarray(cfg.commands.vel_limit)
            if np.any(low > [-0.08, -0.08, -0.2]) or np.any(high < [0.08, 0.08, 0.2]):
                raise ValueError("Mixed commands require bidirectional translation and yaw ranges")
        if remote_p.size:
            low, high = np.asarray(cfg.commands.vel_limit)
            if low[0] > -0.08 or high[0] < 0.08 or low[2] > -0.2 or high[2] < 0.2:
                raise ValueError(
                    "Remote commands require bidirectional fore-aft and yaw ranges"
                )
        super().__init__(cfg, *args, **kwargs)
        fixed_count = self._num_envs if cfg.fixed_remote_sequence else round(
            fixed_fraction * self._num_envs
        )
        self._fixed_remote_sequence_mask = np.arange(self._num_envs) < fixed_count
        self._head_roll_body_ids = self._backend.get_body_ids(("head_roll",))

    def _make_domain_randomization_provider(self):
        self._v112_walk_reset = V112WalkReset()
        return self._v112_walk_reset

    def update_state(self, state):
        if self._linear_tracking_alpha is not None:
            velocity = self.get_local_linvel()
            previous = state.info.get("v112_tracking_linvel", velocity)
            previous = np.where(np.asarray(state.info["steps"])[:, None] == 0, velocity, previous)
            # Reward-only temporal measurement. Neither commands, actor observations,
            # actions nor physical qpos/qvel are filtered or rewritten here.
            state.info["v112_tracking_linvel"] = previous + self._linear_tracking_alpha * (
                velocity - previous
            )
        state = super().update_state(state)
        # Treat physical travel-reserve violations as failed episodes. Do not
        # repair qpos/qvel, clamp the state, or hide a failure with autoreset.
        if self._v112_termination_margin > 0:
            limits = self._v112_physical_joint_limits
            qpos = self.get_dof_pos()
            margin = np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos)
            failed = np.any(margin < self._v112_termination_margin, axis=1)
            state.info["v112_joint_limit_termination"] = failed
            state = state.replace(terminated=np.logical_or(state.terminated, failed))
        if self._termination_penalty > 0.0:
            state = state.replace(
                reward=state.reward
                - self._termination_penalty * np.asarray(state.terminated, dtype=np.float32)
            )
        if self._command_resample_steps:
            # Score the completed step against its old command, then expose the
            # next command consistently in info and the next actor observation.
            ids = np.flatnonzero(
                (np.asarray(state.info["steps"]) + 1) % self._command_resample_steps == 0
            )
            if ids.size:
                fixed = self._fixed_remote_sequence_mask[ids]
                random_ids = ids[~fixed]
                if random_ids.size:
                    state.info["commands"][random_ids] = self._v112_walk_reset._sample_commands(
                        self, len(random_ids)
                    )
                fixed_ids = ids[fixed]
                if fixed_ids.size:
                    stage = (
                        (np.asarray(state.info["steps"])[fixed_ids] + 1)
                        // self._command_resample_steps
                    ) % len(self._fixed_remote_sequence_commands)
                    state.info["commands"][fixed_ids] = self._fixed_remote_sequence_commands[stage]
                state.obs["obs"][ids, 48:51] = state.info["commands"][ids]
        return state

    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["action_rate"] = self._raw_action_rate
        self._reward_fns["idle_leg_motion"] = self._idle_leg_motion
        self._reward_fns["idle_home_target"] = self._idle_home_target
        self._reward_fns["phase_foot_height"] = self._phase_foot_height_tracking
        self._reward_fns["lateral_command_progress"] = self._lateral_command_progress
        self._reward_fns["head_level_bias"] = self._head_level_bias_penalty
        if self._raw_target_scale > 0:
            self._reward_fns["motor_joint_target_limit"] = self._raw_target_feasibility_penalty
        if self._linear_tracking_alpha is not None:
            self._reward_fns["track_linear_velocity"] = self._mean_linear_velocity_tracking

    def _head_level_bias_penalty(self, ctx):
        """Penalize sustained head-shell side tilt while preserving natural motion."""
        quat_b = self._backend.get_body_quat_b(self._head_roll_body_ids)[:, 0, :]
        head_up_b = np_quat_apply(
            quat_b,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )
        lateral_tilt = np.arcsin(np.clip(head_up_b[:, 1], -1.0, 1.0))
        previous = np.asarray(
            ctx.info.get("head_level_bias_ema", np.zeros_like(lateral_tilt)),
            dtype=np.float32,
        )
        if previous.shape != lateral_tilt.shape:
            raise RuntimeError("head_level_bias_ema must have shape (num_envs,)")
        previous = np.where(np.asarray(ctx.info["steps"]) == 0, 0.0, previous)
        alpha = min(1.0, float(self._cfg.ctrl_dt) / self._head_level_tau)
        ema = (1.0 - alpha) * previous + alpha * lateral_tilt
        ctx.info["head_level_bias_ema"] = ema.astype(np.float32)
        ctx.info["head_lateral_tilt_rad"] = lateral_tilt.astype(np.float32)
        moving = self._gait_active(ctx.info)
        multiplier = np.where(moving, self._head_level_moving_multiplier, 1.0)
        return (-np.abs(ema) * multiplier).astype(np.float32)

    def _update_head_pose_bias_curriculum(self):
        super()._update_head_pose_bias_curriculum()
        if "head_level_bias" not in self._reward_cfg.scales:
            return
        steps = tuple(self._cfg.head_level_bias_schedule_steps)
        scales = tuple(self._cfg.head_level_bias_schedule_scales)
        if len(steps) != len(scales) or not steps:
            raise ValueError("head-level-bias schedule steps/scales must have equal non-zero length")
        if steps[0] != 0 or any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("head-level-bias schedule must start at zero and increase strictly")
        if not np.isfinite(scales).all() or any(scale < 0 for scale in scales):
            raise ValueError("head-level-bias schedule scales must be finite and non-negative")
        index = int(np.searchsorted(np.asarray(steps), self._control_step, side="right") - 1)
        self._reward_cfg.scales["head_level_bias"] = float(scales[max(index, 0)])

    def _idle_leg_motion(self, ctx):
        """Price sustained leg motion only for an exactly zero twist command.

        Small balance corrections remain unpriced. Head commands are independent
        and head joints are excluded. This is a nonnegative, dimensionless cost
        with a negative owner-YAML weight, not a motor speed limit.
        """
        idle = np.all(np.asarray(ctx.info["commands"]) == 0.0, axis=1)
        speed = np.abs(ctx.dof_vel[:, self.LEG_INDICES])
        excess = np.maximum(speed - self._idle_velocity_deadband, 0.0)
        cost = np.mean(np.square(excess / self._idle_velocity_scale), axis=1)
        return np.where(idle, cost, 0.0).astype(np.float32)

    def _idle_home_target(self, ctx):
        """Reward-only HOME reference, not a clamp on balance corrections.

        The raw leg position offsets are measured in radians before any motor
        target governor. Squared normalized offsets give a nonnegative cost.
        """
        idle = np.all(np.asarray(ctx.info["commands"]) == 0.0, axis=1)
        offset = ctx.info["v112_reward_raw_action"][:, self.LEG_INDICES]
        offset = offset * float(self._cfg.control_config.action_scale)
        cost = np.mean(np.square(offset / self._idle_home_scale), axis=1)
        return np.where(idle, cost, 0.0).astype(np.float32)

    def _mean_linear_velocity_tracking(self, ctx):
        error = np.sum(
            np.square(ctx.info["commands"][:, :2] - ctx.info["v112_tracking_linvel"][:, :2]),
            axis=1,
        )
        return np.exp(-error / self._reward_cfg.tracking_sigma).astype(np.float32)

    def _lateral_command_progress(self, ctx):
        """Signed translation for the lateral acquisition lesson, not foot pose.

        Standing and motion in the wrong direction cannot earn positive progress.
        Saturate at the requested speed; the existing tracking reward still
        penalizes overspeed and unwanted forward motion. Use the existing
        reward-only velocity average when enabled to reject gait sway.
        """
        command = np.asarray(ctx.info["commands"], dtype=np.float32)
        velocity = (
            ctx.info["v112_tracking_linvel"]
            if self._linear_tracking_alpha is not None
            else ctx.linvel
        )
        lateral = (np.abs(command[:, 1]) > 0.01) & (np.abs(command[:, 0]) < 0.01)
        progress = (
            np.sign(command[:, 1]) * velocity[:, 1]
            / np.maximum(np.abs(command[:, 1]), 0.08)
        )
        return np.where(lateral, np.clip(progress, -1.0, 1.0), 0.0).astype(np.float32)

    def _phase_foot_height_targets(self, steps):
        steps = np.asarray(steps)
        if steps.shape != (self._num_envs,):
            raise RuntimeError("V1.1.2 phase reference requires one step counter per environment")
        phase = np.remainder(
            steps.astype(np.float64) * float(self._cfg.ctrl_dt) / self._phase_gait_period,
            1.0,
        )
        wave = np.sin(2.0 * np.pi * phase)
        target = float(self._reward_cfg.foot_height_target) * np.stack(
            (np.maximum(wave, 0.0), np.maximum(-wave, 0.0)), axis=1
        )
        return target.astype(np.float32)

    def _phase_foot_height_tracking(self, ctx):
        height = np.asarray(ctx.info["foot_height"], dtype=np.float32)
        if height.shape != (self._num_envs, 2):
            raise RuntimeError("V1.1.2 phase reward requires two foot heights per environment")
        target = self._phase_foot_height_targets(ctx.info["steps"])
        error = (height - target) / self._phase_foot_height_std
        tracking = np.exp(-np.mean(np.square(error), axis=1))
        return (tracking * self._gait_active(ctx.info)).astype(np.float32)

    def apply_action(self, actions, state):
        # Reward the actor's request before the hip-yaw governor or motor plant.
        # Preserve the existing observation/control history for checkpoint parity.
        raw = np.asarray(actions, dtype=np.float32)
        if raw.shape != (self._num_envs, len(self.JOINT_NAMES)) or not np.isfinite(raw).all():
            raise ValueError("V1.1.2 walking actions must be finite with shape (num_envs, 14)")
        previous = state.info.get("v112_reward_raw_action", np.zeros_like(raw))
        previous = np.where(np.asarray(state.info["steps"])[:, None] == 0, 0, previous)
        state.info["v112_reward_previous_raw_action"] = previous.copy()
        state.info["v112_reward_raw_action"] = raw.copy()
        return super().apply_action(actions, state)

    def _raw_action_rate(self, ctx):
        delta = ctx.info["v112_reward_raw_action"] - ctx.info["v112_reward_previous_raw_action"]
        return np.sum(np.square(delta), axis=1)

    def _raw_target_feasibility_penalty(self, ctx):
        # Teach the actor to use the request interval that the existing guard
        # actually accepts. Scoring only physical q or the already-governed
        # target leaves large, permanently clipped raw requests underpriced.
        target = self.default_angles + (
            ctx.info["v112_reward_raw_action"] * float(self._cfg.control_config.action_scale)
        )
        excess = np.maximum(self._v110_safe_joint_low - target, 0.0) + np.maximum(
            target - self._v110_safe_joint_high, 0.0
        )
        return np.sum(np.square(excess / self._raw_target_scale), axis=1).astype(np.float32)
