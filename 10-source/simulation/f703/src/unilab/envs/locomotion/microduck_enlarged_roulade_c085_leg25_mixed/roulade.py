"""V1.0.7 forward-roll experiment owner; unchanged policy I/O.

X40 uses the existing measured passive fit and hybrid torque envelope. X10
uses its own manufacturer sweep, with explicitly unmeasured passive/response
priors. The latter are sensitivity variables, never an X40 identification
mislabelled as an X10 measurement. No hardware support is claimed.
"""

from dataclasses import dataclass, field

import numpy as np

from unilab.actuators.gf43x import (
    GF43X10_10,
    GF43X40_10,
    GF43X40_10_EMPTY_LOAD_FIT,
    gf43x40_torque_bounds,
    static_torque_magnitude_limit,
)
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import MICRODUCK_ENLARGED_JOINT_NAMES
from unilab.envs.locomotion.microduck_enlarged_106.roulade import (
    MicroDuckEnlarged106RouladeCurriculumConfig,
    MicroDuckEnlarged106RouladeFlatCfg,
    MicroDuckEnlarged106RouladeFlatEnv,
)
from unilab.envs.locomotion.microduck_enlarged_roulade_c085_leg25_mixed.demonstration_reset import (
    MOTOR_HISTORY,
    DemonstrationResetBank,
    DemonstrationResetConfig,
    DemonstrationResetProvider,
)
from unilab.envs.locomotion.microduck_enlarged_roulade_c085_leg25_mixed.quiet_stand import (
    MicroDuckEnlargedRouladeC085Leg25MixedRewardConfig,
    feet_are_loaded,
    quiet_stand_score,
)
from unilab.envs.locomotion.microduck_enlarged_roulade_c085_leg25_mixed.reference_motion import (
    MotionReference,
    motion_tracking_score,
)

MIXED_X10_JOINTS = ("left_hip_yaw", "head_yaw", "head_roll", "right_hip_yaw")
MOTOR_MAPS = {
    "all_x40": (),
    "pitch_distal": ("head_pitch", "left_ankle", "right_ankle"),
    "pitch_legs": (
        "head_pitch",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ),
    "mixed": MIXED_X10_JOINTS,
    "fast_distal": (*MIXED_X10_JOINTS, "head_pitch", "left_ankle", "right_ankle"),
    "fast_legs": (
        *MIXED_X10_JOINTS,
        "head_pitch",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ),
}


def policy_phase_features(info, ctrl_dt, launch_delay_s):
    """Runtime-computable phase features in three unused command slots."""
    if not np.isfinite(launch_delay_s) or launch_delay_s <= 0:
        raise ValueError("phase launch delay must be finite and positive")
    raw_frontier = np.asarray(info["roulade_max"])
    frontier = np.clip(raw_frontier / (2 * np.pi), 0.0, 1.0)
    steps = np.asarray(info.get("steps", np.zeros_like(raw_frontier)))
    clock = np.clip(steps * ctrl_dt / launch_delay_s, 0.0, 1.0)
    # Mid-roll resets carry a restored frontier but a fresh episode clock.
    launch = np.maximum(clock, np.clip(frontier * 20.0, 0.0, 1.0))
    # Keep recovery changes mathematically absent through the first300deg,
    # then switch them on smoothly before the CEM rise begins around330deg.
    recovery_frontier = np.clip(
        np.asarray(info.get("policy_roll_frontier", raw_frontier)) / (2 * np.pi), 0.0, 1.0
    )
    recovery = np.clip((recovery_frontier - 5 / 6) * 12.0, 0.0, 1.0)
    recovery = recovery * recovery * (3.0 - 2.0 * recovery)
    recovery *= np.asarray(info.get("roulade_head_latch", np.ones_like(recovery)))
    return np.c_[frontier, launch, recovery].astype(np.float32)


def x10_bounds(velocity, *, conservative=True):
    """Manufacturer motoring sweep plus labelled low-speed continuation.

    No acceleration beyond the highest measured 217.1 rpm point. The small
    positive torque at that point is dropped to zero conservatively. Below
    the lowest measured speed, hold 5.373 Nm (conservative) or interpolate to
    the 7 Nm nameplate peak. Reverse sweep symmetry and rated braking are
    simulation assumptions, not measured four-quadrant data.
    """
    points = sorted(GF43X10_10.bench_points, key=lambda p: p.output_speed_rpm)
    speeds = np.array([p.output_speed_rpm * np.pi / 30 for p in points])
    torques = np.array([p.output_torque_nm for p in points])
    torques[-1] = 0.0
    w = np.asarray(velocity)
    if not np.isfinite(w).all():
        raise ValueError("X10 output velocity must be finite")
    torque = np.interp(
        np.abs(w),
        np.r_[0.0, speeds],
        np.r_[torques[0] if conservative else 7.0, torques],
        right=0.0,
    )
    return np.where(w > 0, -2.2, -torque), np.where(w < 0, 2.2, torque)


@registry.envcfg("MicroDuckEnlargedRouladeC085Leg25MixedFlat")
@dataclass
class MicroDuckEnlargedRouladeC085Leg25MixedFlatCfg(MicroDuckEnlarged106RouladeFlatCfg):
    reward_config: MicroDuckEnlargedRouladeC085Leg25MixedRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(
                ASSETS_ROOT_PATH
                / "robots/microduck_enlarged_roulade_c085_leg25_mixed/scene_roulade.xml"
            )
        )
    )
    motor_map: str = "pitch_distal"
    sim_dt: float = 0.001
    standing_z_range: tuple[float, float] = (0.2671034703103258, 0.2671034703103258)
    standing_tilt_max: float = 0.0
    curriculum: MicroDuckEnlarged106RouladeCurriculumConfig = field(
        default_factory=lambda: MicroDuckEnlarged106RouladeCurriculumConfig(
            spawn_stages=[{"step": 0, "standing_prob": 1.0, "midroll_prob": 0.0}]
        )
    )
    rolling_guards: bool = False
    motor_response_s: float = 0.0225
    x10_coulomb_nm: float = 0.10
    x10_viscous_nm_s: float = 0.025
    x10_conservative: bool = True
    # Optional conservative per-episode stress-test budget, not a measured
    # motor thermal model. None retains the instantaneous peak envelope.
    episode_peak_budget_s: float | None = None
    demonstration_reset: DemonstrationResetConfig = field(default_factory=DemonstrationResetConfig)
    phase_observation: bool = False
    phase_launch_delay_s: float = 0.30
    # Dedicated post-gate experts historically receive column 50 cleared at
    # runtime.  This training-only switch preserves that exact actor contract
    # while leaving the critic's phase feature intact.
    recovery_training_zero_gate: bool = False


@registry.env("MicroDuckEnlargedRouladeC085Leg25MixedFlat", sim_backend="mujoco")
class MicroDuckEnlargedRouladeC085Leg25MixedFlatEnv(MicroDuckEnlarged106RouladeFlatEnv):
    """Per-joint bounded, lagged position control through the public callback."""

    def _make_domain_randomization_provider(self):
        return DemonstrationResetProvider()

    def _update_roulade_state(self, info, gyro, base_quat, base_lin_vel_w):
        # Unlike the reward-owned progress, policy phase must not freeze in
        # flight or during contact switching.  Integrate the measured body gyro
        # continuously; reset planning seeds both values deterministically.
        accum = np.asarray(info.get("policy_roll_accum", info["roulade_accum"]))
        frontier = np.asarray(info.get("policy_roll_frontier", info["roulade_max"]))
        accum = accum + np.nan_to_num(gyro[:, 1], nan=0.0) * self._cfg.ctrl_dt
        frontier = np.maximum(frontier, accum)
        super()._update_roulade_state(info, gyro, base_quat, base_lin_vel_w)
        info["policy_roll_accum"] = accum.astype(np.float32)
        info["policy_roll_frontier"] = frontier.astype(np.float32)

    def _compute_obs(self, info, gyro, projected_gravity, dof_pos, dof_vel):
        obs = super()._compute_obs(info, gyro, projected_gravity, dof_pos, dof_vel)
        if not self._cfg.phase_observation:
            return obs
        phase = policy_phase_features(info, self._cfg.ctrl_dt, self._cfg.phase_launch_delay_s)
        # Actor command slots48:51 and critic command slots51:54 are identically
        # zero in the roulade task. Reuse them without changing61/74 dimensions.
        obs["obs"][:, 48:51] = phase
        obs["critic"][:, 51:54] = phase
        if self._cfg.recovery_training_zero_gate:
            obs["obs"][:, 50] = 0
        return obs

    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["quiet_stand"] = self._quiet_stand
        self._reward_fns["reference_motion"] = self._reference_motion
        self._reward_fns["policy_rise"] = self._policy_rise

    def _policy_rise(self, ctx):
        """Dense post-roll height/upright curriculum signal in [0,1]."""
        frontier = np.asarray(ctx.info["policy_roll_frontier"])
        gate = np.clip((frontier - np.deg2rad(300.0)) / np.deg2rad(30.0), 0.0, 1.0)
        gate = gate * gate * (3.0 - 2.0 * gate)
        height = np.clip((ctx.base_height - 0.09) / (0.255 - 0.09), 0.0, 1.0)
        upright = np.clip(-ctx.gravity[:, 2], 0.0, 1.0)
        return np.asarray(gate * (0.5 * height + 0.5 * upright), dtype=np.float32)

    def _reference_motion(self, ctx):
        target, valid = self._motion_reference.target(ctx.info)
        score, joint_error, rotation_error, height_error = motion_tracking_score(
            self._reward_cfg,
            joint_pos=ctx.dof_pos,
            base_quat=self._backend.get_base_quat(),
            height=ctx.base_height,
            target=target,
        )
        log = {}
        for key, value in {
            "score": score * valid,
            "joint_rmse_rad": joint_error,
            "rotation_error_rad": rotation_error,
            "height_error_m": np.abs(height_error),
            "reference_valid": valid,
        }.items():
            log[f"reference_motion/{key}"] = float(np.mean(value))
        self._reference_log = log
        return score * valid

    def update_state(self, state):
        state = super().update_state(state)
        if self._motion_reference.poses is not None:
            # The shared reward dispatcher replaces info['log']; publish
            # task-specific raw diagnostics after its scaled reward entries.
            state.info.setdefault("log", {}).update(self._reference_log)
        return state

    def _quiet_stand(self, ctx):
        foot_vel, foot_omega = self._backend.get_body_vel_w(self._foot_body_ids)
        tilt = np.arccos(np.clip(-ctx.gravity[:, 2], -1, 1))
        force = ctx.info["roulade_critic_force"].reshape(self._num_envs, 2, 3)
        both = feet_are_loaded(force, self._reward_cfg.foot_contact_force_threshold)
        other = self._contact_any(self.cfg.sensor.support_ground[2:])
        score, settling = quiet_stand_score(
            self._reward_cfg,
            height=ctx.base_height,
            tilt=tilt,
            base_vel=ctx.linvel,
            gyro=ctx.gyro,
            joint_vel=ctx.dof_vel,
            foot_vel=foot_vel,
            foot_omega=foot_omega,
            both_feet=both,
            other_support=other,
        )
        log = ctx.info.setdefault("log", {})
        for name, value in {
            "score": score,
            "settling": settling,
            "height_m": ctx.base_height,
            "tilt_deg": np.rad2deg(tilt),
            "both_feet": both,
            "base_speed_m_s": np.linalg.norm(ctx.linvel, axis=1),
            "base_omega_rad_s": np.linalg.norm(ctx.gyro, axis=1),
            "foot_speed_m_s": np.linalg.norm(foot_vel, axis=2),
            "foot_omega_rad_s": np.linalg.norm(foot_omega, axis=2),
            "joint_speed_rad_s": np.abs(ctx.dof_vel),
        }.items():
            log[f"quiet_stand/{name}"] = float(np.mean(value))
        return score

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self._demonstration_bank = DemonstrationResetBank(cfg.demonstration_reset, cfg)
        self._motion_reference = MotionReference(cfg.reward_config, self._demonstration_bank, cfg)
        self._reference_log = {}
        if cfg.identified_dynamics.enabled or cfg.bootstrap_actuator.enabled:
            raise ValueError(
                "C085/leg25 mixed roulade owns per-motor dynamics; disable inherited dynamics"
            )
        if not cfg.motor_constraints.enabled:
            raise ValueError("C085/leg25 mixed roulade requires motor constraints")
        if cfg.motor_map not in MOTOR_MAPS:
            raise ValueError(f"motor_map must be one of {tuple(MOTOR_MAPS)}")
        priors = [cfg.motor_response_s, cfg.x10_coulomb_nm, cfg.x10_viscous_nm_s]
        if not np.isfinite(priors).all() or cfg.motor_response_s <= 0 or min(priors[1:]) < 0:
            raise ValueError("invalid C085/leg25 mixed roulade motor response/friction prior")
        if cfg.episode_peak_budget_s is not None and (
            not np.isfinite(cfg.episode_peak_budget_s) or cfg.episode_peak_budget_s < 0
        ):
            raise ValueError("episode_peak_budget_s must be finite and nonnegative, or None")
        if cfg.phase_observation:
            policy_phase_features(
                {"roulade_max": np.zeros(1), "steps": np.zeros(1)},
                cfg.ctrl_dt,
                cfg.phase_launch_delay_s,
            )
        self.MOTOR_SPECS = tuple(
            GF43X10_10 if n in MOTOR_MAPS[cfg.motor_map] else GF43X40_10
            for n in MICRODUCK_ENLARGED_JOINT_NAMES
        )
        self._x10_mask = np.array([s is GF43X10_10 for s in self.MOTOR_SPECS])
        self._roulade_motor_torque = np.zeros((num_envs, 14), dtype=np.float32)
        self._peak_time = np.zeros((num_envs, 14), dtype=np.float64)
        self._peak_run = np.zeros_like(self._peak_time)
        self._peak_longest = np.zeros_like(self._peak_time)
        self._positive_work = np.zeros_like(self._peak_time)
        self._negative_work = np.zeros_like(self._peak_time)
        self._torque_peak = np.zeros_like(self._peak_time)
        if cfg.rolling_guards:
            cfg.sensor.head_ground = (*cfg.sensor.head_ground, "rolling_head_contact")
            cfg.sensor.support_ground = (
                *cfg.sensor.support_ground,
                "rolling_head_contact",
                "rolling_trunk_contact",
            )
        super().__init__(cfg, num_envs, backend_type)
        # Cold-path consistency gate: changing only the motor-map YAML must
        # not silently pair X10 control with an X40 mechanical asset (or vice versa).
        expected_armature = np.r_[
            np.zeros(6),
            np.where(self._x10_mask, 0.0018, GF43X40_10_EMPTY_LOAD_FIT.effective_inertia_kg_m2),
        ]
        actual_armature = self._backend.get_dof_armature()
        if actual_armature.shape != expected_armature.shape or not np.allclose(
            actual_armature, expected_armature, rtol=1e-6, atol=1e-9
        ):
            self.close()
            raise ValueError(
                "C085/leg25 mixed roulade scene armature does not match its configured motor_map"
            )

    def reset(self, env_indices):
        result = super().reset(env_indices)
        if self._demonstration_bank.data is not None:
            selected = result[1]["demonstration_reset_index"]
            for name in MOTOR_HISTORY:
                getattr(self, name)[np.asarray(env_indices, dtype=np.intp)] = (
                    self._demonstration_bank.data[name][selected]
                )
            # Direct reset callers (including the existing training startup)
            # use the returned obs immediately, before the autoreset scatter.
            # Keep restored controller/action history in sync with the physical
            # sample. Autoreset's later identical scatter remains valid.
            if self._state is not None:
                for key, value in result[1].items():
                    if isinstance(value, np.ndarray) and key in self._state.info:
                        self._state.info[key][env_indices] = value
            return result
        self._roulade_motor_torque[np.asarray(env_indices, dtype=np.intp)] = 0
        for value in (
            self._peak_time,
            self._peak_run,
            self._peak_longest,
            self._positive_work,
            self._negative_work,
            self._torque_peak,
        ):
            value[np.asarray(env_indices, dtype=np.intp)] = 0
        return result

    def _motor_torque_bounds_for_velocity(self, dof_vel, *, duty):
        w = np.asarray(dof_vel)
        lower, upper = gf43x40_torque_bounds(w)
        if np.any(self._x10_mask):
            lo, hi = x10_bounds(w[:, self._x10_mask], conservative=self.cfg.x10_conservative)
            lower[:, self._x10_mask], upper[:, self._x10_mask] = lo, hi
        if duty == "rated":
            cap = np.stack(
                [
                    static_torque_magnitude_limit(w[:, i], s, duty="rated")
                    for i, s in enumerate(self.MOTOR_SPECS)
                ],
                axis=1,
            )
            lower, upper = np.maximum(lower, -cap), np.minimum(upper, cap)
        return lower.astype(np.float32), upper.astype(np.float32)

    def _motor_pre_step_control(self, backend, ctrl):
        q = np.asarray(backend.get_dof_pos(), dtype=np.float32)
        dq = np.asarray(backend.get_dof_vel(), dtype=np.float32)
        kp, kd = self._active_actuator_gains(None if self._state is None else self._state.info)
        limits = backend.get_joint_range()
        requested = (np.clip(ctrl, limits[:, 0], limits[:, 1]) - q) * kp - kd * dq
        lo, hi = self._motor_torque_bounds_for_velocity(
            dq, duty=self.cfg.motor_constraints.hard_duty
        )
        lo *= self.cfg.motor_constraints.hard_limit_margin
        hi *= self.cfg.motor_constraints.hard_limit_margin
        rated_lo, rated_hi = self._motor_torque_bounds_for_velocity(dq, duty="rated")
        rated_lo *= self.cfg.motor_constraints.hard_limit_margin
        rated_hi *= self.cfg.motor_constraints.hard_limit_margin
        if self.cfg.episode_peak_budget_s is not None:
            available = self._peak_time + self.cfg.sim_dt <= self.cfg.episode_peak_budget_s + 1e-10
            lo = np.where(available, lo, np.maximum(lo, rated_lo))
            hi = np.where(available, hi, np.minimum(hi, rated_hi))
        alpha = -np.expm1(-self.cfg.sim_dt / self.cfg.motor_response_s)
        self._roulade_motor_torque += alpha * (
            np.clip(requested, lo, hi) - self._roulade_motor_torque
        )
        friction = GF43X40_10_EMPTY_LOAD_FIT.friction_torque(dq)
        friction[:, self._x10_mask] = (
            self.cfg.x10_coulomb_nm * np.tanh(dq[:, self._x10_mask] / 0.01)
            + self.cfg.x10_viscous_nm_s * dq[:, self._x10_mask]
        )
        net = np.clip(self._roulade_motor_torque - friction, lo, hi)
        above_rated = (net < rated_lo - 1e-5) | (net > rated_hi + 1e-5)
        self._peak_time += above_rated * self.cfg.sim_dt
        self._peak_run = np.where(above_rated, self._peak_run + self.cfg.sim_dt, 0)
        self._peak_longest = np.maximum(self._peak_longest, self._peak_run)
        power = net * dq
        self._positive_work += np.maximum(power, 0) * self.cfg.sim_dt
        self._negative_work += np.maximum(-power, 0) * self.cfg.sim_dt
        self._torque_peak = np.maximum(self._torque_peak, np.abs(net))
        # Project the requested angle, not the equivalent actuator target:
        # clipping again here can violate torque bounds at a mechanical stop.
        target = q + (net + kd * dq) / kp
        if self._state is not None:
            self._state.info.update(
                motor_fit_net_torque_nm=net.copy(),
                motor_substep_torque_lower_nm=lo.copy(),
                motor_substep_torque_upper_nm=hi.copy(),
                motor_substep_target_rad=target.copy(),
                motor_episode_peak_time_s=self._peak_time.copy(),
                motor_episode_peak_longest_s=self._peak_longest.copy(),
                motor_episode_positive_work_j=self._positive_work.copy(),
                motor_episode_negative_work_j=self._negative_work.copy(),
                motor_episode_torque_peak_nm=self._torque_peak.copy(),
            )
        return target
