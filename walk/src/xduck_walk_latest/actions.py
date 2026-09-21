"""GF43X40-10 state-feedback action for the published UniLab manager contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from unilab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .gf43x40 import GF43X40RobotMotor


@dataclass(kw_only=True)
class GF43X40ActionCfg(BaseActionCfg):
    """Raw policy action -> clipped joint target -> 1 kHz MIT motor torque.

    Set `actuator_names` in policy order and supply fourteen per-joint gains.
    The MJCF actuator must have kp=1, kv=0 and a control range wide enough
    for `q + torque`; see the task scene. Position targets are sampled by the
    motor every 20 physics ticks, while torque closes at every physics tick.
    """

    kp: tuple[float, ...]
    kd: tuple[float, ...]
    target_low: tuple[float, ...]
    target_high: tuple[float, ...]
    command_delay_s: float = 0.0
    feedback_delay_s: float = 0.0
    jitter_mode: str = "measured_20260918"
    jitter_seed: int = 0
    kp_multiplier_range: tuple[float, float] = (0.8, 1.2)
    kd_multiplier_range: tuple[float, float] = (0.8, 1.2)
    motor_dr_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "torque_scale_multiplier": (0.8, 1.2),
        "friction_scale_multiplier": (0.8, 1.2),
        "response_time_multiplier": (0.8, 1.2),
    })

    def build(self, env):
        return GF43X40Action(self, env)


class GF43X40Action(BaseAction):
    requires_substep_state_feedback = True
    cfg: GF43X40ActionCfg

    def __init__(self, cfg: GF43X40ActionCfg, env):
        super().__init__(cfg, env)
        if not np.isclose(env.physics_dt, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("GF43X40 requires a 1 ms physics step")
        if not np.isclose(env.step_dt, 0.020, rtol=0.0, atol=1e-12):
            raise ValueError("GF43X40 host control requires a 20 ms policy step")
        if self.action_dim != 14:
            raise ValueError(f"GF43X40 requires 14 policy actuators, got {self.action_dim}")
        self._low = np.asarray(cfg.target_low, dtype=np.float64)
        self._high = np.asarray(cfg.target_high, dtype=np.float64)
        if self._low.shape != (14,) or self._high.shape != (14,) or np.any(self._low >= self._high):
            raise ValueError("GF43X40 target bounds must contain 14 ordered pairs")
        self._offset = self._entity.data.default_joint_pos[:, self.target_ids].copy()
        self._target = np.zeros((self.num_envs, 14), dtype=np.float64)
        self.motor = GF43X40RobotMotor(
            self.num_envs, 14, kp=np.asarray(cfg.kp), kd=np.asarray(cfg.kd),
            command_delay_s=cfg.command_delay_s,
            feedback_delay_s=cfg.feedback_delay_s,
            jitter_mode=cfg.jitter_mode, jitter_seed=cfg.jitter_seed,
            motor_dr_ranges=cfg.motor_dr_ranges,
        )

    def process_actions(self, actions: np.ndarray) -> None:
        super().process_actions(actions)
        np.clip(self._processed_actions, self._low, self._high, out=self._target)

    def apply_actions(self) -> None:
        # Data facade reads public SimBackend joint state, freshly copied out by
        # the MuJoCo pre-step callback. Never inspect mjbatch/private buffers.
        position = self._entity.data.joint_pos[:, self.target_ids]
        velocity = self._entity.data.joint_vel[:, self.target_ids]
        self.motor.finish_substep(position, velocity)
        torque = self.motor.begin_substep(self._target, position, velocity)
        self._entity.set_joint_position_target(position + torque, joint_ids=self.target_ids)

    def finish_control_step(self) -> None:
        """Publish the final motor feedback before post-step observations."""
        position = self._entity.data.joint_pos[:, self.target_ids]
        velocity = self._entity.data.joint_vel[:, self.target_ids]
        self.motor.finish_substep(position, velocity)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            ids = np.arange(self.num_envs, dtype=np.int32)[env_ids]
        else:
            ids = np.asarray(env_ids, dtype=np.int32)
        self.motor.reset(ids)
        if len(ids):
            self.motor.kp[ids] = self.motor._nominal_kp * self._env.rng.uniform(
                *self.cfg.kp_multiplier_range, size=(len(ids), self.action_dim)
            )
            self.motor.kd[ids] = self.motor._nominal_kd * self._env.rng.uniform(
                *self.cfg.kd_multiplier_range, size=(len(ids), self.action_dim)
            )
        self._target[ids] = self._offset[ids]

    def seed_previous_action(self, env_ids: np.ndarray, raw_actions: np.ndarray) -> None:
        """Seed the manager's raw-action history for a walk-to-stand handoff.

        Call after ActionManager.reset(), since reset clears its history.
        """
        ids = np.asarray(env_ids, dtype=np.intp)
        values = np.asarray(raw_actions, dtype=self._raw_actions.dtype)
        if values.shape != (len(ids), self.action_dim) or not np.isfinite(values).all():
            raise ValueError("handoff raw actions must be finite (selected_envs, 14)")
        self._raw_actions[ids] = values
        self._env.action_manager.action[ids] = values
        self._env.action_manager.prev_action[ids] = values
