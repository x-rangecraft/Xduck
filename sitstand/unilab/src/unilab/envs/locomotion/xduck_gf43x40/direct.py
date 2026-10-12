"""Direct joint policy: demonstration data is used only by training rewards."""

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry

from .reference import (
    ReferenceDRProvider,
    ReferenceRewardConfig,
    XDuckSitStandReferenceCfg,
    XDuckSitStandReferenceEnv,
)
from .sitstand import LEG_IDS, SIT_LEGS
from .velocity import XDuckGF43X40VelocityEnv


def direct_rewards():
    cfg = ReferenceRewardConfig()
    # The inherited rate term uses 0.5 rad/s normalization. This coefficient
    # equals -0.02 with a 5 rad/s normalization for full joint commands.
    cfg.scales["residual_rate"] = -0.0002
    return cfg


class DirectDRProvider(ReferenceDRProvider):
    """Keep a fixed nominal cohort in every rollout, including partial resets."""

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        mask = env._nominal_envs[env_ids]
        ids = np.asarray(env_ids)[mask]
        if len(ids):
            payload = plan.randomization
            for name, baseline in (
                ("body_ipos", env._base_body_ipos),
                ("body_mass", env._base_body_mass),
                ("body_inertia", env._base_body_inertia),
                ("dof_armature", env._base_dof_armature),
                ("dof_frictionloss", env._base_dof_frictionloss),
            ):
                value = getattr(payload, name)
                if value is not None:
                    value[mask] = baseline
            if payload.geom_friction is not None:
                payload.geom_friction[mask] = env._base_geom_friction
                rows = np.flatnonzero(mask)
                payload.geom_friction[rows[:, None], env._foot_geom_ids, 0] = 1.0
            env._mass_inertia_scale[ids] = 1
            env._foot_friction[ids] = 1
            env._imu_rotation[ids] = np.eye(3)
            env._ankle_target_offset[ids] = 0
        return plan


@registry.envcfg("XDuckGF43X40SitStandDirectFlat")
@dataclass
class XDuckSitStandDirectCfg(XDuckSitStandReferenceCfg):
    policy_action_scale_rad: float = 1.0
    nominal_env_fraction: float = 0.25
    transition_success_reward: float = 4.0
    terminal_failure_cost: float = 8.0
    completion_hold_seconds: float = 0.5
    reward_config: ReferenceRewardConfig = field(default_factory=direct_rewards)


@registry.env("XDuckGF43X40SitStandDirectFlat", sim_backend="mujoco")
class XDuckSitStandDirectEnv(XDuckSitStandReferenceEnv):
    """Actor receives measured state/command/history, never reference samples.

    Inference uses the GF velocity owner's mapping: default_angles + action
    (radians), clipped to model limits. Only the reward evaluator uses the
    demonstration clock and samples. No reference file is needed by inference.
    """

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        if cfg.control_config.action_scale != 1 or cfg.policy_action_scale_rad != 1:
            raise ValueError("direct policy requires one radian per action unit")
        if cfg.control_config.simulate_action_latency or not cfg.clip_action_targets:
            raise ValueError("direct policy requires immediate, limit-clipped target mapping")
        if not 0 <= cfg.nominal_env_fraction <= 1:
            raise ValueError("nominal_env_fraction must lie in [0,1]")
        self._nominal_envs = np.arange(num_envs) < int(num_envs * cfg.nominal_env_fraction)
        self._noise_ids = np.arange(num_envs)
        self._transition_pending = np.zeros(num_envs, dtype=bool)
        self._stable_steps = np.zeros(num_envs, dtype=int)
        self._command_age = np.zeros(num_envs)
        self._clock_fresh = np.ones(num_envs, dtype=bool)
        super().__init__(cfg, num_envs, backend_type)
        self._episode_reward_sums["transition_success_event"] = np.zeros(num_envs)
        self._episode_reward_sums["terminal_failure_event"] = np.zeros(num_envs)

    def _make_dr_provider(self):
        return DirectDRProvider()

    def _obs_noise(self, data, scale):
        result = super()._obs_noise(data, scale)
        mask = self._nominal_envs[self._noise_ids]
        result[mask] = data[mask]
        return result

    def _reset_reference(self, ids, sitting):
        super()._reset_reference(ids, sitting)
        self._transition_pending[ids] = False
        self._stable_steps[ids] = 0
        self._command_age[ids] = 0
        self._clock_fresh[ids] = True
        # A sit reset must not charge a fictitious stand-to-sit target jump.
        for sit, name in [(False, "rise"), (True, "descent")]:
            subset = np.asarray(ids)[sitting == sit]
            self._current_action_target[subset] = self._refs[name].nodes[-1]
            self._previous_action_target[subset] = self._refs[name].nodes[-1]

    @property
    def obs_groups_spec(self):
        return {k: v + 1 for k, v in XDuckGF43X40VelocityEnv.obs_groups_spec.fget(self).items()}

    def _compute_obs(self, *args, env_ids=None, **kwargs):
        self._noise_ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        obs = XDuckGF43X40VelocityEnv._compute_obs(self, *args, env_ids=env_ids, **kwargs)
        ids = slice(None) if env_ids is None else env_ids
        indices = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        fresh = self._clock_fresh[indices]
        if np.any(fresh):
            # At controller reset, already-at-goal means hold, not replay a transition.
            # This uses two static end poses, never a trajectory sample.
            pose = args[2] if len(args) > 2 else kwargs["dof_pos"]
            command = args[5] if len(args) > 5 else kwargs["commands"]
            sit_pose = self.default_angles.copy()
            sit_pose[LEG_IDS] = SIT_LEGS
            sitting = np.mean((pose - sit_pose) ** 2, axis=1) < np.mean(
                (pose - self.default_angles) ** 2, axis=1
            )
            already_there = sitting == (command[:, 0] > 0.5)
            self._command_age[indices[fresh & already_there]] = 4.0
            self._clock_fresh[indices] = False
        # Deployment can compute this clock from command changes alone.
        clock = np.minimum(self._command_age[ids] / 4.0, 1.0)[:, None]
        return {k: np.concatenate([v, clock.astype(v.dtype)], axis=1) for k, v in obs.items()}

    def _maybe_resample_commands(self, state):
        before = state.info["commands"][:, 0].copy()
        super()._maybe_resample_commands(state)
        self._command_age[before != state.info["commands"][:, 0]] = 0

    def _tracking_score(self, joint_error, height_error, orientation_error):
        # A product collapses near zero when any one component is initially bad.
        return (
            0.5 * np.exp(-joint_error - orientation_error)
            + 0.25 * np.exp(-np.square(height_error / 0.02))
            + 0.25 * np.exp(-orientation_error)
        )

    def _vertical_velocity_cost(self, error):
        # Bounded shaping avoids a -200/s velocity cost dwarfing +8/s tracking.
        return 1.0 - np.exp(-np.square(error / 0.1))

    def apply_action(self, actions, state):
        self._command_age += self.cfg.ctrl_dt
        desired = (state.info["commands"][:, 0] > 0.5).astype(np.int8)
        start = (self._ref_direction == 0) & (desired != self._ref_posture)
        self._transition_pending[start] = True
        self._stable_steps[start] = 0
        self._advance_training_reference(state)
        target = XDuckGF43X40VelocityEnv.apply_action(self, actions, state)
        raw = self.default_angles + actions
        self._target_clipped[:] = np.abs(raw - target) > 1e-8
        # Shared reward implementation; this now measures full target velocity.
        self._residual_velocity[:] = (
            self._current_action_target - self._previous_action_target
        ) / self.cfg.ctrl_dt
        return target

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        reward = super()._compute_reward(state, linvel, gyro, upvector, dof_pos)
        height = self._backend.get_base_pos()[:, 2]
        speed = np.linalg.norm(self._backend.get_base_lin_vel(), axis=1)
        quat = self._backend.get_base_quat()
        up_z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        target_height = np.where(state.info["commands"][:, 0] > 0.5, 0.2182, 0.3129)
        pose_error = np.sqrt(np.mean((dof_pos - self._ref_target) ** 2, axis=1))
        # Count settling after motor completion, before the recorded trace ends.
        duration = np.where(
            state.info["commands"][:, 0] > 0.5,
            self._refs["descent"].duration,
            self._refs["rise"].duration,
        )
        stable = (
            (self._command_age >= duration)
            & (abs(height - target_height) < 0.01)
            & (up_z > np.cos(np.deg2rad(5)))
            & (speed < 0.03)
            & (pose_error < 0.08)
        )
        self._stable_steps[:] = np.where(stable, self._stable_steps + 1, 0)
        bad = (up_z < np.cos(np.deg2rad(65))) | (height < 0.13)
        terminal = self._forbidden_contact() | (bad & (self._fall_steps >= 9))
        success = (
            self._transition_pending
            & ~terminal
            & (self._stable_steps * self.cfg.ctrl_dt >= self.cfg.completion_hold_seconds)
        )
        # Events are not scaled by dt: +4 equals 0.5 s of perfect tracking;
        # -8 equals 1 s. Dense shaping remains available throughout motion.
        success_reward = self.cfg.transition_success_reward * success
        failure_reward = -self.cfg.terminal_failure_cost * terminal
        reward += success_reward + failure_reward
        self._episode_reward_sums["transition_success_event"] += success_reward
        self._episode_reward_sums["terminal_failure_event"] += failure_reward
        self._transition_pending[success | terminal] = False
        log = state.info.setdefault("log", {})
        log["sitstand/completion_events"] = float(success.sum())
        log["sitstand/terminal_events"] = float(terminal.sum())
        return reward
