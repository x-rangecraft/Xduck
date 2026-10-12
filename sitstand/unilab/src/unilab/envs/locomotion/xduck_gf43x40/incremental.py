"""Versioned PPO target-increment decoder; GF physics remains unchanged."""
from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.utils.rotation import np_quat_apply
from .microduck_ppo import MicroduckPPOCfg, MicroduckPPOEnv, stationary_factor
from .sitstand import LEG_IDS


def endpoint_factors(height_error, leg_error, head_error, height_std, leg_std, head_std):
    """Precision credit near the goal; settling requires pose as well as height."""
    height_gate = np.exp(-np.square(height_error / height_std))
    legs = np.exp(-np.mean(np.square(leg_error / leg_std), axis=1))
    head = np.exp(-np.mean(np.square(head_error / head_std), axis=1))
    return height_gate, legs, head, height_gate * legs * head


def integrate_target(previous, actions, increments, low, high):
    raw = previous + increments * np.tanh(actions)
    return raw, np.clip(raw, low, high)


def command_style_terms(command, head_error, head_velocity, body_up, head_tilt,
                        gyro, head_width_deg, pitch_width_deg, speed_width):
    """Instantaneous command-specific costs; no reference path or head control."""
    sitting=np.asarray(command)>.5
    head=np.max(np.abs(head_error),axis=1)/np.deg2rad(head_width_deg)
    speed=np.max(np.abs(head_velocity),axis=1)/speed_width
    pitch=np.abs(np.arctan2(body_up[:,0],body_up[:,2]))/np.deg2rad(pitch_width_deg)
    world_head=head_tilt/np.deg2rad(head_width_deg)
    sit_quality=np.exp(-head**2-world_head**2-.1*speed**2)
    rise_quality=np.exp(-pitch**2)
    costs=dict(style_sit_head_displacement=sitting*np.minimum(head**2,36.),
               style_sit_head_speed=sitting*np.minimum(speed**2,36.),
               style_sit_head_tilt=sitting*np.minimum(world_head**2,36.),
               style_rise_pitch=(~sitting)*np.minimum(pitch**2,36.),
               style_rise_pitch_rate=(~sitting)*np.minimum((gyro[:,1]/speed_width)**2,36.))
    return np.where(sitting,sit_quality,rise_quality),costs


@registry.envcfg("XDuckGF43X40SitStandIncrementalFlat")
@dataclass
class IncrementalCfg(MicroduckPPOCfg):
    pure_policy_revision: int = 6
    # Radians per control tick, in training joint order. Not a total pose range.
    target_increment_rad: list[float] = field(default_factory=lambda: [
        .015, .010, .080, .060, .075, .110, .025, .025, .010,
        .015, .010, .080, .060, .075,
    ])
    target_history_scale_rad: float = .12
    # Optional absolute neck policy-target bounds; physical limits are untouched.
    neck_target_bounds_deg: list[float] = field(default_factory=list)
    head_absolute_position: bool = False
    head_absolute_scale_rad: float = .25
    head_official_joint_reward: bool = False
    endpoint_safe_soft_limits: bool = False
    executed_increment_cost_enabled: bool = False
    geometric_head_enabled: bool = False
    head_up_axis: list[float] = field(default_factory=lambda: [0., 0.9999999999932535, 3.673192317532603e-6])
    head_upright_std_deg: float = 15.
    head_upright_height_std_m: float = .03
    head_upright_full_motion: bool = False
    head_only_repair_enabled: bool = False
    head_only_ramp_steps: int = 12000
    head_only_neutral_band_deg: float = 15.
    head_only_excursion_width_deg: float = 30.
    head_only_speed_band_rad_s: float = 1.5
    quiet_transition_enabled: bool = False
    transition_head_std_deg: float = 5.
    transition_body_band_deg: float = 5.
    transition_head_speed_std_rad_s: float = 1.
    quiet_transition_blend: float = 1.
    quiet_transition_start_step: int = 0
    quiet_transition_ramp_steps: int = 0
    target_increment_start_rad: list[float] = field(default_factory=list)
    command_style_enabled: bool = False
    command_style_free_leg_pose: bool = False
    command_style_start_step: int = 0
    command_style_ramp_steps: int = 2400
    command_style_blend_start: float = .15
    command_style_blend_end: float = .8
    command_style_head_width_start_deg: float = 45.
    command_style_head_width_end_deg: float = 15.
    command_style_pitch_width_start_deg: float = 15.
    command_style_pitch_width_end_deg: float = 6.
    command_style_speed_width_start: float = 4.
    command_style_speed_width_end: float = 1.5


@registry.env("XDuckGF43X40SitStandIncrementalFlat", sim_backend="mujoco")
@registry.env("XDuckGF43X40SitStandIncrementalFlat", sim_backend="mjwarp")
class IncrementalEnv(MicroduckPPOEnv):
    def __init__(self, cfg, *args, **kwargs):
        increments = np.asarray(cfg.target_increment_rad)
        if cfg.pure_policy_revision != 6 or increments.shape != (14,):
            raise ValueError("Incremental decoder requires revision 6 and 14 increments")
        if not np.isfinite(increments).all() or (increments <= 0).any():
            raise ValueError("Target increments must be finite and positive")
        if not np.isfinite(cfg.target_history_scale_rad) or cfg.target_history_scale_rad <= 0:
            raise ValueError("Target history scale must be finite and positive")
        if not np.isfinite(cfg.head_absolute_scale_rad) or cfg.head_absolute_scale_rad <= 0:
            raise ValueError('Absolute head action scale must be positive')
        if cfg.head_official_joint_reward and not cfg.geometric_head_enabled:
            raise ValueError('Joint head reward with uprightness requires head geometry')
        if cfg.head_upright_full_motion and not cfg.geometric_head_enabled:
            raise ValueError('Full-motion head reward requires geometric head measurement')
        if cfg.head_only_repair_enabled:
            if not cfg.geometric_head_enabled or cfg.head_only_ramp_steps <= 0:
                raise ValueError('Head-only repair requires geometry and a positive ramp')
            if any(not np.isfinite(v) or v <= 0 for v in (
                cfg.head_only_neutral_band_deg, cfg.head_only_excursion_width_deg,
                cfg.head_only_speed_band_rad_s)):
                raise ValueError('Head-only repair widths must be positive')
        super().__init__(cfg, *args, **kwargs)
        if cfg.neck_target_bounds_deg:
            bounds = np.deg2rad(np.asarray(cfg.neck_target_bounds_deg, dtype=float))
            if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                    or bounds[0] >= bounds[1]
                    or bounds[0] < self._target_low[5]
                    or bounds[1] > self._target_high[5]):
                raise ValueError('Neck target bounds must be ordered inside original policy limits')
            self._target_low[5], self._target_high[5] = bounds
        if cfg.command_style_enabled:
            widths=(cfg.command_style_head_width_start_deg,cfg.command_style_head_width_end_deg,
                    cfg.command_style_pitch_width_start_deg,cfg.command_style_pitch_width_end_deg,
                    cfg.command_style_speed_width_start,cfg.command_style_speed_width_end)
            if not cfg.geometric_head_enabled or any(not np.isfinite(v) or v<=0 for v in widths):
                raise ValueError('Command style requires geometric head and positive widths')
            if cfg.command_style_start_step<0 or cfg.command_style_ramp_steps<=0 or not (0<=cfg.command_style_blend_start<=cfg.command_style_blend_end<=1):
                raise ValueError('Invalid command style curriculum')
        if cfg.quiet_transition_enabled:
            if not cfg.geometric_head_enabled:
                raise ValueError("Quiet transitions require geometric head rewards")
            widths = (cfg.transition_head_std_deg, cfg.transition_body_band_deg, cfg.transition_head_speed_std_rad_s)
            if any(not np.isfinite(v) or v <= 0 for v in widths):
                raise ValueError("Transition reward widths must be positive")
            if not 0 <= cfg.quiet_transition_blend <= 1 or cfg.quiet_transition_start_step < 0 or cfg.quiet_transition_ramp_steps < 0:
                raise ValueError("Invalid quiet transition schedule")
        if cfg.geometric_head_enabled:
            axis = np.asarray(cfg.head_up_axis)
            if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.):
                raise ValueError("Head up axis must be a finite unit vector")
            if any(not np.isfinite(v) or v <= 0 for v in (cfg.head_upright_std_deg, cfg.head_upright_height_std_m)):
                raise ValueError("Geometric head reward widths must be positive")
            self._head_body_ids = self._backend.get_body_ids(["head_roll"])
        if cfg.endpoint_safe_soft_limits:
            # Reward-only bounds include both validated endpoints. The physical
            # model, target decoder limits and actuator gains remain untouched.
            endpoints = np.stack([self.default_angles.copy(), self.default_angles.copy()])
            endpoints[1, LEG_IDS] = self._sit_legs
            self._soft_joint_limits[:, 0] = np.minimum(self._soft_joint_limits[:, 0], endpoints.min(axis=0))
            self._soft_joint_limits[:, 1] = np.maximum(self._soft_joint_limits[:, 1], endpoints.max(axis=0))
        r = cfg.reward_config
        if r.endpoint_refinement_enabled:
            if any(not np.isfinite(v) or v <= 0 for v in (r.endpoint_height_std, r.endpoint_leg_std, r.endpoint_head_std)):
                raise ValueError("Endpoint reward widths must be finite and positive")
            if r.endpoint_refinement_ramp_steps <= 0 or r.endpoint_refinement_start_step < 0:
                raise ValueError("Invalid endpoint refinement schedule")
            if r.endpoint_pose_weight <= 0 or r.endpoint_head_weight <= 0 or r.endpoint_velocity_weight >= 0:
                raise ValueError("Endpoint precision weights must be positive and velocity weight negative")
        self._increments = increments
        self._start_increments = np.asarray(cfg.target_increment_start_rad or cfg.target_increment_rad)
        if self._start_increments.shape != (14,) or not np.isfinite(self._start_increments).all() or (self._start_increments <= 0).any():
            raise ValueError("Initial target increments require 14 positive values")
        self._decoded_raw_target = self._current_action_target.copy()

    def _transition_ramp(self):
        cfg = self.cfg
        if cfg.quiet_transition_ramp_steps == 0:
            return 1.
        return float(np.clip((self._curriculum_step() - cfg.quiet_transition_start_step) / cfg.quiet_transition_ramp_steps, 0., 1.))

    def _transition_blend(self):
        return self.cfg.quiet_transition_blend * self._transition_ramp()

    def apply_action(self, actions, state):
        # Preserve base action ownership and optional one-tick action latency.
        LocomotionBaseEnv.apply_action(self, actions, state)
        executed = state.info["last_actions"] if self.cfg.control_config.simulate_action_latency else state.info["current_actions"]
        # Reset provider initializes this target to the actual sampled reset pose.
        # No command-conditioned posture branch, reference trajectory or actor prior.
        previous = self._current_action_target.copy()
        if self.cfg.target_increment_start_rad:
            ramp = self._transition_ramp()
            self._increments[:] = (1 - ramp) * self._start_increments + ramp * np.asarray(self.cfg.target_increment_rad)
        raw, target = integrate_target(previous, executed, self._increments,
                                       self._target_low, self._target_high)
        if self.cfg.head_absolute_position:
            # Same absolute-position semantics as official; GF action scale is
            # a policy-interface setting, not a change to the calibrated plant.
            raw[:, 5:9] = self.default_angles[5:9] + executed[:, 5:9] * self.cfg.head_absolute_scale_rad
            target[:, 5:9] = np.clip(raw[:, 5:9], self._target_low[5:9], self._target_high[5:9])
        self._decoded_raw_target[:] = raw
        self._previous_action_target[:] = previous
        self._current_action_target[:] = target
        return target

    def _raw_action_target(self, actions):
        # The just-decoded target must be used after physics advances as well.
        return self._decoded_raw_target

    def _compute_obs(self, gyro, upvector, dof_pos, dof_vel, last_actions, commands, **kwargs):
        # Expose integrator state in the existing 14 history slots; remain Markov
        # without adding observation dimensions or privileging physical truth.
        env_ids = kwargs.get("env_ids")
        ids = slice(None) if env_ids is None else env_ids
        history = (self._current_action_target[ids] - self.default_angles) / self.cfg.target_history_scale_rad
        return super()._compute_obs(gyro, upvector, dof_pos, dof_vel, history, commands, **kwargs)

    def _action_rate_cost(self, state):
        if not self.cfg.executed_increment_cost_enabled:
            return super()._action_rate_cost(state)
        # Charge the executed physical increment, not an unbounded latent action.
        delta = (self._current_action_target - self._previous_action_target) / self._increments
        return np.sum(np.square(delta), axis=1)

    def _geometric_head_score(self):
        quat = self._backend.get_body_quat_w(self._head_body_ids)[:, 0]
        up = np_quat_apply(quat, np.broadcast_to(self.cfg.head_up_axis, (self.num_envs, 3)))
        tilt = np.arccos(np.clip(up[:, 2], -1., 1.))
        return tilt, np.exp(-np.square(tilt / np.deg2rad(self.cfg.head_upright_std_deg)))

    def _head_reward_scores(self, state, head_error):
        if not self.cfg.geometric_head_enabled:
            return super()._head_reward_scores(state, head_error)
        tilt, score = self._geometric_head_score()
        if self.cfg.head_official_joint_reward:
            # Official kernels: per-joint mean tracking and exponential mean
            # error composite. Actual world uprightness cannot hide bad joints,
            # and there is no height-dependent release of head feedback.
            composite = np.exp(-np.mean((head_error / .4) ** 2, axis=1)) * score
            tracking = np.mean(np.exp(-(head_error / .5) ** 2), axis=1) * score
            log = state.info.setdefault('log', {})
            log['sitstand/head_tilt_deg'] = float(np.rad2deg(tilt).mean())
            log['sitstand/head_upright_gate'] = 1.
            log['sitstand/head_upright_factor'] = float(score.mean())
            return composite, tracking
        cfg = self.cfg.reward_config
        goal = cfg.stand_height + self._posture_alpha * (cfg.sit_height - cfg.stand_height)
        gate = np.exp(-np.square((self._backend.get_base_pos()[:, 2] - goal) / self.cfg.head_upright_height_std_m))
        if self.cfg.head_upright_full_motion:
            # Bad head orientation must lose credit at every height.
            gate = np.ones(self.num_envs)
        elif self.cfg.quiet_transition_enabled:
            blend = self._transition_blend()
            gate = (1 - blend) * gate + blend
        # Older tasks preserve their endpoint gate; the repaired task uses
        # actual physical orientation throughout the complete transition.
        physical = 1. - gate * (1. - score)
        yaw_tracking = np.exp(-np.square(head_error[:, 2] / .5))
        state.info.setdefault("log", {})["sitstand/head_tilt_deg"] = float(np.rad2deg(tilt).mean())
        state.info['log']['sitstand/head_upright_gate'] = float(gate.mean())
        state.info['log']['sitstand/head_upright_factor'] = float(physical.mean())
        tracking = physical * yaw_tracking
        if self.cfg.head_only_repair_enabled:
            # Repair only standalone head tracking, leaving composite height,
            # body, leg and stillness rewards on the original task semantics.
            ramp = min(1., self._curriculum_step() / self.cfg.head_only_ramp_steps)
            ramp = ramp * (state.info['commands'][:, 0] > .5)
            tracking = ((1 - ramp) * physical + ramp * score) * yaw_tracking
        return physical, tracking

    def _adapt_goal_stable(self, stable):
        if not self.cfg.geometric_head_enabled:
            return super()._adapt_goal_stable(stable)
        tilt, _ = self._geometric_head_score()
        return ((np.abs(self.last_height_error) < .015)
                & (self.last_tilt < np.deg2rad(10))
                & (tilt < np.deg2rad(10))
                & (self.last_speed < .04)
                & (np.max(np.abs(self.get_dof_vel()), axis=1) < .5)
                & ~self._failed)

    def _adapt_posture_terms(self, terms, state, gyro):
        terms = super()._adapt_posture_terms(terms, state, gyro)
        cfg = self.cfg.reward_config
        if self.cfg.geometric_head_enabled:
            # Quiet but tilted heads must not receive full endpoint stillness.
            _, score = self._geometric_head_score()
            terms["posture_stillness"] *= score
        if self.cfg.head_only_repair_enabled:
            # Sitting only; exclude commanded yaw and never charge leg motion.
            # Add bounded costs, not a quality gate on movement rewards.
            c = self.cfg
            ramp = float(np.clip(self._curriculum_step() / c.head_only_ramp_steps, 0., 1.))
            sitting = state.info['commands'][:, 0] > .5
            ids = [5, 6, 8]
            error = np.abs(self.get_dof_pos()[:, ids] - self.default_angles[ids])
            excess = np.maximum(error - np.deg2rad(c.head_only_neutral_band_deg), 0.)
            excursion = np.max((excess / np.deg2rad(c.head_only_excursion_width_deg)) ** 2, axis=1)
            speed = np.maximum(np.abs(self.get_dof_vel()[:, ids]) / c.head_only_speed_band_rad_s - 1., 0.)
            delta = (self._current_action_target[:, ids] - self._previous_action_target[:, ids]) / self._increments[ids]
            terms['head_only_excursion'] = sitting * ramp * np.minimum(excursion, 4.)
            terms['head_only_speed'] = sitting * ramp * np.minimum(np.max(speed ** 2, axis=1), 4.)
            terms['head_only_increment'] = sitting * ramp * np.sum(delta ** 2, axis=1)
            state.info.setdefault('log', {})['sitstand/head_only_ramp'] = ramp
        if self.cfg.command_style_enabled:
            c=self.cfg
            if c.command_style_free_leg_pose:
                # Geometry and quiet motion, without an inherited leg template.
                up=-self._projected_gravity()
                goal=cfg.stand_height+self._posture_alpha*(cfg.sit_height-cfg.stand_height)
                height_error=self._backend.get_base_pos()[:,2]-goal
                width=cfg.composite_height_std or cfg.height_std
                tilt_sq=(np.clip(1-up[:,2],0.,2.) if cfg.upright_from_tilt
                          else np.sum(up[:,:2]**2,axis=1))
                head_score,_=self._head_reward_scores(state,
                    self.get_dof_pos()[:,5:9]-self.default_angles[5:9]-state.info['commands'][:,3:7])
                terms['posture_composite']=np.exp(-(height_error/width)**2-tilt_sq/cfg.upright_std**2)*head_score
                if cfg.repair_enabled:
                    settled=np.abs(state.info['commands'][:,0]-self._posture_alpha)<.02
                    factor=stationary_factor(self.get_dof_vel(),gyro,
                        cfg.stationary_joint_speed_std,cfg.stationary_angular_speed_std)
                    terms['posture_composite']*=np.where(settled,factor,1.)
                terms['posture_pose']=np.zeros(self.num_envs)
                terms['posture_pose_l1']=np.zeros(self.num_envs)
            ramp=float(np.clip((self._curriculum_step()-c.command_style_start_step)/c.command_style_ramp_steps,0.,1.))
            def interpolate(a,b):return a+(b-a)*ramp
            blend=interpolate(c.command_style_blend_start,c.command_style_blend_end)
            head_width=interpolate(c.command_style_head_width_start_deg,c.command_style_head_width_end_deg)
            pitch_width=interpolate(c.command_style_pitch_width_start_deg,c.command_style_pitch_width_end_deg)
            speed_width=interpolate(c.command_style_speed_width_start,c.command_style_speed_width_end)
            tilt,_=self._geometric_head_score()
            quality,costs=command_style_terms(state.info['commands'][:,0],
                self.get_dof_pos()[:,5:9]-self.default_angles[5:9],self.get_dof_vel()[:,5:9],
                -self._projected_gravity(),tilt,gyro,head_width,pitch_width,speed_width)
            gate=1-blend+blend*quality
            for name in ('posture_height','posture_height_sharp','posture_composite',
                         'upright_linear','upright_while_tall','posture_stillness',
                         'head_pose_tracking'):
                if name in terms:terms[name]=np.minimum(terms[name],0.)+np.maximum(terms[name],0.)*gate
            terms.update({name:blend*value for name,value in costs.items()})
            log=state.info.setdefault('log',{})
            log['sitstand/style_blend']=blend;log['sitstand/style_head_width_deg']=head_width
            log['sitstand/style_pitch_width_deg']=pitch_width
            log['sitstand/style_quality']=float(quality.mean())
        if self.cfg.quiet_transition_enabled:
            # Joint neutrality prevents neck/head counter-rotation from hiding
            # large movements behind a level physical head. All head commands
            # are zero in this task; the policy still owns every motor target.
            error = self.get_dof_pos()[:, 5:9] - self.default_angles[5:9]
            head_speed = self.get_dof_vel()[:, 5:9]
            up = -self._projected_gravity()
            tilt = np.arccos(np.clip(up[:, 2], -1., 1.))
            band = np.deg2rad(self.cfg.transition_body_band_deg)
            terms["transition_head_neutral"] = np.exp(-np.mean(np.square(error / np.deg2rad(self.cfg.transition_head_std_deg)), axis=1))
            terms["transition_head_excursion"] = np.mean(np.maximum(np.abs(error) / np.deg2rad(self.cfg.transition_head_std_deg) - 1., 0.), axis=1)
            terms["transition_head_speed"] = np.mean(np.square(head_speed / self.cfg.transition_head_speed_std_rad_s), axis=1)
            terms["transition_body_tilt"] = np.square(np.maximum(tilt / band - 1., 0.))
            for name in ("transition_head_neutral", "transition_head_excursion", "transition_head_speed", "transition_body_tilt"):
                terms[name] *= self._transition_blend()
            state.info.setdefault("log", {})["sitstand/quiet_transition_blend"] = self._transition_blend()
        if not cfg.endpoint_refinement_enabled:
            return terms
        commands = state.info["commands"]
        goal = np.broadcast_to(self.default_angles, (self.num_envs, 14)).copy()
        goal[np.ix_(commands[:, 0] > .5, LEG_IDS)] = self._sit_legs
        goal[:, 5:9] += commands[:, 3:7]
        error = self.get_dof_pos() - goal
        height_goal = cfg.stand_height + commands[:, 0] * (cfg.sit_height - cfg.stand_height)
        height, legs, head, settling = endpoint_factors(
            self._backend.get_base_pos()[:, 2] - height_goal,
            error[:, LEG_IDS], error[:, 5:9], cfg.endpoint_height_std,
            cfg.endpoint_leg_std, cfg.endpoint_head_std,
        )
        ramp = np.clip((self._curriculum_step() - cfg.endpoint_refinement_start_step) / cfg.endpoint_refinement_ramp_steps, 0., 1.)
        # Fade the revised objective in without an abrupt loss of old credit.
        # At full refinement, wrong poses cannot earn existing stillness credit.
        terms["posture_stillness"] *= (1 - ramp) + ramp * legs * head
        # Head motion is commanded: penalize leg speed, rather than freezing the
        # head while it is following a changing command. Head precision remains
        # rewarded independently, including before the legs fully settle.
        speed = np.max(np.abs(self.get_dof_vel()[:, LEG_IDS]), axis=1)
        velocity_cost = np.square(speed / .5) + np.square(np.linalg.norm(self._backend.get_base_lin_vel(), axis=1) / .04)
        additions = {
            "endpoint_pose": cfg.endpoint_pose_weight * ramp * height * legs,
            "endpoint_head": cfg.endpoint_head_weight * ramp * height * head,
            "endpoint_velocity": cfg.endpoint_velocity_weight * ramp * settling * velocity_cost,
        }
        # Parent reward owns weighting/accounting; add explicit weighted terms
        # with unit scales so saved configs expose every contribution.
        for name, value in additions.items():
            terms[name] = value
        state.info.setdefault("log", {})["sitstand/endpoint_refinement_ramp"] = float(ramp)
        return terms
