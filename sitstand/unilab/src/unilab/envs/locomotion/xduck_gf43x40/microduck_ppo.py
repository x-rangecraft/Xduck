"""Microduck sitstand learning recipe on the calibrated XDuck plant.

No reference clip, history shortcut, transition bonus, or fall termination.
The original pure task remains versioned for its existing checkpoints.
"""

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry

from .pure import PureDRProvider, PureRewardConfig, XDuckSitStandPureCfg, XDuckSitStandPureEnv
from .sitstand import LEG_IDS, XDuckSitStandEnv
from .velocity import XDuckGF43X40VelocityEnv


def stage(stages, step, key="value"):
    return next(item[key] for item in reversed(stages) if step >= item["step"])


@dataclass
class MicroduckRewardConfig(PureRewardConfig):
    # Disabled by default so saved pre-repair configurations retain semantics.
    repair_enabled: bool = False
    target_overflow_std_rad: float = 0.25
    stationary_joint_speed_std: float = 0.5
    stationary_angular_speed_std: float = 0.5
    discovery_gate_positive_terms: bool = False
    discovery_motion_cost_multiplier: float = 1.0
    endpoint_refinement_enabled: bool = False
    endpoint_refinement_start_step: int = 30000
    endpoint_refinement_ramp_steps: int = 6000
    endpoint_height_std: float = 0.03
    endpoint_leg_std: float = 0.15
    endpoint_head_std: float = 0.20
    endpoint_pose_weight: float = 4.0
    endpoint_head_weight: float = 2.0
    endpoint_velocity_weight: float = -0.5


def stationary_factor(joint_velocity, gyro, joint_std, angular_std):
    # Legs only: commanded head motion must remain possible while seated.
    return np.exp(
        -np.mean(np.square(joint_velocity[:, LEG_IDS] / joint_std), axis=1)
        - np.mean(np.square(gyro / angular_std), axis=1)
    )


@registry.envcfg("XDuckGF43X40SitStandMicroduckPPOFlat")
@dataclass
class MicroduckPPOCfg(XDuckSitStandPureCfg):
    reward_config: MicroduckRewardConfig = field(default_factory=MicroduckRewardConfig)
    pure_policy_revision: int = 3
    encoder_bias_rad: float = 0.015
    reset_tilt_deg: float = 8.0
    reset_height_jitter_m: float = 0.02


@registry.envcfg("XDuckGF43X40SitStandContinuousPPOFlat")
@dataclass
class ContinuousPPOCfg(MicroduckPPOCfg):
    pure_policy_revision: int = 4
    target_rate_limit_rad_s: float = 6.0


@registry.envcfg("XDuckGF43X40SitStandDiscoveryPilotFlat")
@dataclass
class DiscoveryPilotCfg(MicroduckPPOCfg):
    pure_policy_revision: int = 5
    pilot_spawn_enabled: bool = True
    pilot_spawn_alpha: float = 0.25
    pilot_easy_fraction: float = 0.75
    pilot_progress_weight: float = 5.0
    pilot_progress_discount: float = 0.99


class MicroduckReset(PureDRProvider):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        n = len(env_ids)
        # Official noisy seated bootstrap, scaled clearance for this geometry.
        sitting = np.isin(env._case[env_ids], [1, 2])
        angles = np.random.normal(0, env.cfg.reset_joint_noise_rad, (n, 14))
        target = np.broadcast_to(env.default_angles, (n, 14)).copy()
        target[np.ix_(sitting, LEG_IDS)] = env._sit_legs
        plan.qpos[:, 7:] = np.clip(target + angles, env._target_low, env._target_high)
        roll, pitch = np.random.uniform(
            -np.deg2rad(env.cfg.reset_tilt_deg), np.deg2rad(env.cfg.reset_tilt_deg), (2, n)
        )
        # Preserve the sampled yaw while introducing a bounded seated tilt.
        yaw = 2 * np.arctan2(plan.qpos[:, 6], plan.qpos[:, 3])
        roll *= sitting
        pitch *= sitting
        cr, sr, cp, sp, cy, sy = (
            np.cos(roll / 2),
            np.sin(roll / 2),
            np.cos(pitch / 2),
            np.sin(pitch / 2),
            np.cos(yaw / 2),
            np.sin(yaw / 2),
        )
        plan.qpos[:, 3:7] = np.stack(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            axis=1,
        )
        plan.qpos[:, 2] += np.random.uniform(0, env.cfg.reset_height_jitter_m, n)
        env._reset_clearance.lift_to_clearance(plan.qpos, env.cfg.reset_clearance_m)
        # Official reset_action_history is zero, not a pose-preserving skip.
        plan.info_updates["current_actions"][:] = 0
        plan.info_updates["last_actions"][:] = 0
        env._current_action_target[env_ids] = plan.qpos[:, 7:]
        env._previous_action_target[env_ids] = plan.qpos[:, 7:]
        env._encoder_bias[env_ids] = np.random.uniform(
            -env.cfg.encoder_bias_rad, env.cfg.encoder_bias_rad, (n, 14)
        )
        env._motor.actuator.friction_scale[env_ids] = np.random.uniform(0.9, 1.1, (n, 14))
        return plan


class ContinuousReset(MicroduckReset):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        target = np.broadcast_to(env.default_angles, (len(env_ids), 14)).copy()
        sitting = np.isin(env._case[env_ids], (1, 2))
        target[np.ix_(sitting, LEG_IDS)] = env._sit_legs
        # Keep the starting posture, independent of the requested destination.
        action = target - env.default_angles
        plan.info_updates["current_actions"][:] = action
        plan.info_updates["last_actions"][:] = action
        env._current_action_target[env_ids] = target
        env._previous_action_target[env_ids] = target
        return plan


class DiscoveryPilotReset(MicroduckReset):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        if not env.cfg.pilot_spawn_enabled:
            return plan
        easy = (env._case[env_ids] == 2) & (
            np.random.random(len(env_ids)) < env.cfg.pilot_easy_fraction
        )
        alpha = env.cfg.pilot_spawn_alpha
        noise = plan.qpos[:, 7:][:, LEG_IDS] - env._sit_legs
        legs = pilot_support_legs(env.default_angles[LEG_IDS], env._sit_legs, alpha)
        plan.qpos[np.ix_(easy, 7 + LEG_IDS)] = legs + 0.25 * noise[easy]
        # Intermediate foot-supported resets start upright, without the seated
        # tilt perturbation; full seated resets retain the official noise.
        yaw = 2 * np.arctan2(plan.qpos[easy, 6], plan.qpos[easy, 3])
        plan.qpos[easy, 3:7] = np.stack(
            [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)],
            axis=1,
        )
        plan.qpos[easy, 2] = (
            env.cfg.stand_reset_height * (1 - alpha) + env.cfg.sit_reset_height * alpha
        )
        env._reset_clearance.lift_to_clearance(plan.qpos, env.cfg.reset_clearance_m)
        env._posture_alpha[np.asarray(env_ids)[easy]] = alpha
        env._current_action_target[env_ids] = plan.qpos[:, 7:]
        env._previous_action_target[env_ids] = plan.qpos[:, 7:]
        return plan


def pilot_support_legs(standing, sitting, alpha):
    legs = standing + alpha * (sitting - standing)
    # GF knee axes oppose hip axes; ankle body frames reverse again. Bend
    # around the standing support line rather than the ground-supported SIT.
    for hip, knee, ankle in ((2, 3, 4), (7, 8, 9)):
        delta_knee = legs[knee] - standing[knee]
        legs[hip] = standing[hip] + 0.5 * delta_knee
        legs[ankle] = standing[ankle] - (legs[hip] - standing[hip]) + delta_knee
    return legs


@registry.env("XDuckGF43X40SitStandMicroduckPPOFlat", sim_backend="mujoco")
@registry.env("XDuckGF43X40SitStandMicroduckPPOFlat", sim_backend="mjwarp")
class MicroduckPPOEnv(XDuckSitStandPureEnv):
    def __init__(self, cfg, *args, **kwargs):
        reward = cfg.reward_config
        for value in (
            reward.target_overflow_std_rad,
            reward.stationary_joint_speed_std,
            reward.stationary_angular_speed_std,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Microduck reward normalization scales must be finite and positive"
                )
        if reward.repair_enabled and reward.scales.get("target_overflow", 0) >= 0:
            raise ValueError("Repaired recipe requires a negative target_overflow weight")
        if not 0 <= reward.discovery_motion_cost_multiplier <= 1:
            raise ValueError("Discovery motion cost multiplier must be in [0, 1]")
        super().__init__(cfg, *args, **kwargs)

    def _raw_action_target(self, actions):
        return self.default_angles + (
            actions * self.cfg.control_config.action_scale * self.cfg.policy_action_scale_rad
        )

    def _adapt_posture_terms(self, terms, state, gyro):
        cfg = self.cfg.reward_config
        if cfg.discovery_gate_positive_terms:
            command = state.info["commands"][:, 0]
            goal = cfg.stand_height + command * (cfg.sit_height - cfg.stand_height)
            width = 0.5 * (cfg.stand_height - cfg.sit_height)
            gate = np.exp(-np.square((self._backend.get_base_pos()[:, 2] - goal) / width))
            for name in ("posture_pose", "upright_linear", "upright_while_tall"):
                value = terms[name]
                terms[name] = np.minimum(value, 0) + np.maximum(value, 0) * gate
        if cfg.discovery_motion_cost_multiplier != 1.0:
            for name in ("action_rate_l2", "gentle_motion", "body_ang_vel"):
                terms[name] *= cfg.discovery_motion_cost_multiplier
        if not cfg.repair_enabled:
            return terms
        factor = stationary_factor(
            self.get_dof_vel(),
            gyro,
            cfg.stationary_joint_speed_std,
            cfg.stationary_angular_speed_std,
        )
        terms["posture_stillness"] *= factor
        # During a commanded transition preserve the pose/height learning signal.
        # At the endpoint, composite credit requires joints and trunk to settle.
        settled_command = np.abs(state.info["commands"][:, 0] - self._posture_alpha) < 0.02
        terms["posture_composite"] *= np.where(settled_command, factor, 1.0)
        raw = self._raw_action_target(state.info["current_actions"])
        excess = raw - np.clip(raw, self._target_low, self._target_high)
        terms["target_overflow"] = np.mean(np.square(excess / cfg.target_overflow_std_rad), axis=1)
        return terms

    def _make_dr_provider(self):
        return MicroduckReset()

    @property
    def obs_groups_spec(self):
        return XDuckGF43X40VelocityEnv.obs_groups_spec.fget(self)

    def _compute_obs(self, *args, **kwargs):
        # 61-D official actor layout, raw previous action; no ramp phase input.
        env_ids = kwargs.get("env_ids")
        self._noise_ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        return XDuckGF43X40VelocityEnv._compute_obs(self, *args, **kwargs)

    def _sample_head_commands(self, count):
        return XDuckGF43X40VelocityEnv._sample_head_commands(self, count)

    def _update_curriculum(self):
        XDuckSitStandEnv._update_curriculum(self)

    def _reset_episode_history(self, env_ids):
        super()._reset_episode_history(env_ids)
        # Official mass/inertia DR is a startup draw, retained across resets.
        if not hasattr(self, "_startup_mass_scale"):
            self._startup_mass_scale = np.random.uniform(
                *self.cfg.domain_rand.mass_inertia_scale_range, self.num_envs
            )
        self._mass_inertia_scale[env_ids] = self._startup_mass_scale[env_ids]

    def _maybe_resample_commands(self, state):
        before = state.info["commands"][:, 0].copy()
        if self.cfg.evaluation_case < 0:
            self._velocity_timer -= 1
            due = np.flatnonzero(self._velocity_timer <= 0)
            state.info["commands"][due, :3] = self._sample_velocity_commands(len(due))
            self._velocity_timer[due] = (
                np.random.uniform(*self.cfg.posture_dwell_seconds, len(due)) / self.cfg.ctrl_dt
            ).astype(int)
        # Head commands use the same timer and curriculum as official sitstand.
        self._resample_due(
            self._head_timer,
            state.info["commands"],
            slice(3, 7),
            self._sample_head_commands,
            self.cfg.commands.head_resampling_time_range,
            np.zeros(self.num_envs, dtype=bool),
        )
        delta = self.cfg.ctrl_dt / self.cfg.posture_slew_seconds
        self._posture_alpha += np.clip(
            state.info["commands"][:, 0] - self._posture_alpha, -delta, delta
        )
        changed = before != state.info["commands"][:, 0]
        self._case[changed] = np.where(state.info["commands"][changed, 0] > 0.5, 3, 2)
        self._goal_hold_steps[changed] = 0

    def _terminated(self, upvector):
        # Let the learner experience impacts and attempt recovery. Nonfinite
        # state remains a runtime error, never silently fed into PPO.
        return np.zeros(self.num_envs, dtype=bool)

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        reward = XDuckSitStandEnv._compute_reward(self, state, linvel, gyro, upvector, dof_pos)
        cfg = self.cfg.reward_config
        torque = self._motor.torque
        difference = torque - self._previous_torque
        difference[self._torque_fresh] = 0
        self._previous_torque[:] = torque
        self._torque_fresh[:] = False
        # Same delayed regularizers as official, using XDuck torque/momentum units.
        iteration = self._curriculum_step() // 24
        torque_weight = 0 if iteration < 750 else (-0.002 if iteration < 1250 else -0.004)
        momentum = self._backend.get_sensor_data(self.cfg.sensor.angular_momentum)
        extra = {
            "joint_torque_rate_l2": torque_weight * np.sum(np.square(difference / 50.0), axis=1),
            "angular_momentum": cfg.scales["angular_momentum"]
            * np.sum(np.square(momentum / cfg.angular_momentum_reference), axis=1),
        }
        for name, weighted in extra.items():
            weighted *= cfg.discovery_motion_cost_multiplier
            reward += (weighted * self.cfg.ctrl_dt).astype(reward.dtype)
            self._episode_reward_sums[name] += weighted * self.cfg.ctrl_dt
        height = self._backend.get_base_pos()[:, 2]
        command = state.info["commands"][:, 0]
        goal_height = cfg.stand_height + command * (cfg.sit_height - cfg.stand_height)
        target = np.broadcast_to(self.default_angles, dof_pos.shape).copy()
        target[np.ix_(command > 0.5, LEG_IDS)] = self._sit_legs
        target[:, 5:9] += state.info["commands"][:, 3:7]
        self.last_pose_error = np.sqrt(np.mean(np.square(dof_pos - target), axis=1))
        self.last_height_error = height - goal_height
        self.last_tilt = np.arccos(np.clip(upvector[:, 2], -1, 1))
        self.last_speed = np.linalg.norm(self._backend.get_base_lin_vel(), axis=1)
        self._failed = (height < 0.13) | (self.last_tilt > np.deg2rad(75))
        stable = (
            (np.abs(self.last_height_error) < 0.015)
            & (self.last_tilt < np.deg2rad(10))
            & (self.last_speed < 0.04)
            & (self.last_pose_error < 0.1)
            & (np.max(np.abs(self.get_dof_vel()), axis=1) < 0.5)
        )
        stable = self._adapt_goal_stable(stable)
        self._goal_hold_steps[:] = np.where(stable, self._goal_hold_steps + 1, 0)
        self.last_success = self._goal_hold_steps * self.cfg.ctrl_dt >= 0.5
        log = state.info.setdefault("log", {})
        log["pure/goal_success_fraction"] = float(self.last_success.mean())
        log["pure/fallen_fraction"] = float(self._failed.mean())
        raw_target = self._raw_action_target(state.info["current_actions"])
        log["pure/joint_speed_rms"] = float(np.sqrt(np.mean(np.square(self.get_dof_vel()))))
        log["pure/target_overflow_rms_rad"] = float(
            np.sqrt(
                np.mean(
                    np.square(raw_target - np.clip(raw_target, self._target_low, self._target_high))
                )
            )
        )
        log["pure/target_clip_fraction"] = float(
            np.mean((raw_target < self._target_low) | (raw_target > self._target_high))
        )
        for case, name in enumerate(("stand_hold", "sit_hold", "rise", "descent")):
            mask = self._case == case
            log[f"pure/{name}_fraction"] = float(mask.mean())
        return reward


    def _adapt_goal_stable(self, stable):
        return stable


@registry.env("XDuckGF43X40SitStandContinuousPPOFlat", sim_backend="mujoco")
@registry.env("XDuckGF43X40SitStandContinuousPPOFlat", sim_backend="mjwarp")
class ContinuousPPOEnv(MicroduckPPOEnv):
    """Versioned action memory and failure rewards for the GF plant."""

    def __init__(self, cfg, *args, **kwargs):
        if cfg.pure_policy_revision != 4:
            raise ValueError("Continuous target task requires policy revision 4")
        if not np.isfinite(cfg.target_rate_limit_rad_s) or cfg.target_rate_limit_rad_s <= 0:
            raise ValueError("target_rate_limit_rad_s must be finite and positive")
        if cfg.reward_config.scales.get("fallen", 0) >= 0:
            raise ValueError("Continuous task requires a negative fallen reward")
        super().__init__(cfg, *args, **kwargs)

    def _make_dr_provider(self):
        return ContinuousReset()

    def apply_action(self, actions, state):
        target = super().apply_action(actions, state)
        delta = self.cfg.target_rate_limit_rad_s * self.cfg.ctrl_dt
        target = np.clip(
            target, self._previous_action_target - delta, self._previous_action_target + delta
        )
        self._current_action_target[:] = target
        return target

    def _compute_obs(self, gyro, upvector, dof_pos, dof_vel, last_actions, commands, **kwargs):
        ids = kwargs.get("env_ids")
        targets = self._current_action_target if ids is None else self._current_action_target[ids]
        # PPO stores sampled actions separately. Observations expose executed
        # target memory, so saturation cannot accumulate unseen raw offsets.
        history = (targets - self.default_angles).copy()
        return super()._compute_obs(gyro, upvector, dof_pos, dof_vel, history, commands, **kwargs)

    def _adapt_posture_terms(self, terms, state, gyro):
        terms = super()._adapt_posture_terms(terms, state, gyro)
        failed = (self._backend.get_base_pos()[:, 2] < 0.13) | (
            -self._projected_gravity()[:, 2] < np.cos(np.deg2rad(75))
        )
        for name, value in terms.items():
            if self.cfg.reward_config.scales[name] > 0:
                terms[name] = np.where(failed, np.minimum(value, 0), value)
        terms["fallen"] = failed.astype(float)
        return terms


@registry.env("XDuckGF43X40SitStandDiscoveryPilotFlat", sim_backend="mujoco")
@registry.env("XDuckGF43X40SitStandDiscoveryPilotFlat", sim_backend="mjwarp")
class DiscoveryPilotEnv(MicroduckPPOEnv):
    """Bounded experiment: goal progress plus optional intermediate resets."""

    def __init__(self, cfg, *args, **kwargs):
        if not (0 <= cfg.pilot_spawn_alpha <= 1 and 0 <= cfg.pilot_easy_fraction <= 1):
            raise ValueError("Pilot reset fractions must be in [0, 1]")
        if not (0 < cfg.pilot_progress_discount < 1 and cfg.pilot_progress_weight >= 0):
            raise ValueError("Invalid pilot potential shaping parameters")
        super().__init__(cfg, *args, **kwargs)

    def _make_dr_provider(self):
        return DiscoveryPilotReset()

    def _pilot_potential(self, height, upvector, dof_pos, command):
        cfg = self.cfg.reward_config
        goal_height = cfg.stand_height + command * (cfg.sit_height - cfg.stand_height)
        height_score = 1 - np.clip(
            np.abs(height - goal_height) / (cfg.stand_height - cfg.sit_height), 0, 1
        )
        target = self.default_angles[LEG_IDS] + command[:, None] * (
            self._sit_legs - self.default_angles[LEG_IDS]
        )
        pose = np.exp(-np.mean(np.square((dof_pos[:, LEG_IDS] - target) / 0.5), axis=1))
        upright = np.clip(upvector[:, 2], 0, 1) ** 2
        return (0.5 * height_score + 0.5 * pose) * upright

    def apply_action(self, actions, state):
        # Capture the actual pre-step state, including after partial resets.
        self._pilot_before = self._pilot_potential(
            self._backend.get_base_pos()[:, 2], -self._projected_gravity(),
            self.get_dof_pos(), state.info["commands"][:, 0],
        )
        return super().apply_action(actions, state)

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        reward = super()._compute_reward(state, linvel, gyro, upvector, dof_pos)
        potential = self._pilot_potential(
            self._backend.get_base_pos()[:, 2], upvector, dof_pos,
            state.info["commands"][:, 0],
        )
        # This is a per-transition potential difference, not a reward rate.
        shaping = self.cfg.pilot_progress_weight * (
            self.cfg.pilot_progress_discount * potential - self._pilot_before
        )
        reward += shaping.astype(reward.dtype)
        state.info["log"]["pilot/progress_reward"] = float(shaping.mean())
        return reward
