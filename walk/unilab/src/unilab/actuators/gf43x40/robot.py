"""Whole-robot, simulator-clock adapter for the calibrated GF43X40-10.

The motor parameters are shared by all fourteen axes.  The calibrated
single-axis sticking solver is not valid for coupled robot dynamics, so this
adapter explicitly selects the smooth coupled approximation.  It must not be
described as a whole-robot motor validation.
"""

from __future__ import annotations

import numpy as np

from .actuator import GF43X40Actuator, GF43X40Parameters
from .communication import GF43X40Communication, MITCommand, MotorFeedback, _mapped


class GF43X40RobotMotor:
    """Hold host position targets at 50 Hz and update all motors at 1 kHz."""

    def __init__(
        self,
        num_envs: int,
        num_motors: int,
        *,
        kp: float | list[float] | np.ndarray,
        kd: float | list[float] | np.ndarray,
        command_delay_s: float = 0.0,
        feedback_delay_s: float = 0.0,
        parameters: GF43X40Parameters | None = None,
        command_jitter_ms: tuple[int, int] = (0, 0),
        feedback_jitter_ms: tuple[int, int] = (0, 0),
        jitter_seed: int = 0,
        jitter_mode: str = "uniform",
        motor_dr_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.parameters = parameters or GF43X40Parameters.from_bundle()
        self.motor_dr_ranges = motor_dr_ranges or {}
        allowed_dr = {"torque_scale_multiplier", "friction_scale_multiplier", "response_time_multiplier"}
        if set(self.motor_dr_ranges) - allowed_dr:
            raise ValueError("Unknown GF motor DR parameter")
        for bounds in self.motor_dr_ranges.values():
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("GF motor DR multiplier bounds must be positive, finite and ordered")
        self.shape = (num_envs, num_motors)
        self.actuator = GF43X40Actuator(
            num_envs,
            num_motors,
            parameters=self.parameters,
            friction_mode="smooth_coupled_approx",
        )
        self.communication = GF43X40Communication(
            num_envs,
            num_motors,
            command_delay_s=command_delay_s,
            feedback_delay_s=feedback_delay_s,
            command_jitter_ms=command_jitter_ms,
            feedback_jitter_ms=feedback_jitter_ms,
            jitter_seed=jitter_seed,
            jitter_mode=jitter_mode,
        )
        self._nominal_kp = self._joint_gains(kp, "kp", num_motors, 500.0)
        self._nominal_kd = self._joint_gains(kd, "kd", num_motors, 5.0)
        self.kp = np.broadcast_to(self._nominal_kp, self.shape).copy()
        self.kd = np.broadcast_to(self._nominal_kd, self.shape).copy()
        self.torque = np.zeros(self.shape, dtype=np.float64)
        self.feedback: MotorFeedback | None = None
        self._initial_position = np.zeros(self.shape, dtype=np.float64)
        self._initial_velocity = np.zeros(self.shape, dtype=np.float64)
        self._initial_position_valid = np.zeros(num_envs, dtype=bool)
        self._initial_velocity_valid = np.zeros(num_envs, dtype=bool)
        self._velocity_before = np.zeros(self.shape, dtype=np.float64)
        self._pending = False
        self._tick = 0
        # The legacy reset provider uses this empty selector to skip DM-only DR.
        self._j4340_joint_ids = np.empty(0, dtype=np.intp)
        self.position_bias_range = (0.0, 0.0)

    @staticmethod
    def _joint_gains(
        values: float | list[float] | np.ndarray,
        name: str,
        num_motors: int,
        maximum: float,
    ) -> np.ndarray:
        gains = np.asarray(values, dtype=np.float64)
        if gains.ndim == 0:
            gains = np.full(num_motors, float(gains))
        if gains.shape != (num_motors,) or not np.all(np.isfinite(gains)):
            raise ValueError(f"{name} must be a finite scalar or one value per motor")
        if np.any(gains < 0.0) or np.any(gains > maximum):
            raise ValueError(f"{name} must be in [0, {maximum}]")
        return gains.copy()

    @property
    def dt(self) -> float:
        return self.actuator.dt

    def reset(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self.actuator.reset(ids)
        for name, bounds in self.motor_dr_ranges.items():
            getattr(self.actuator, name)[ids] = np.random.uniform(*bounds, size=(len(ids), self.shape[1]))
        self.communication.reset(ids)
        self.kp[ids] = self._nominal_kp
        self.kd[ids] = self._nominal_kd
        self.torque[ids] = 0.0
        self._velocity_before[ids] = 0.0
        self._initial_position_valid[ids] = False
        self._initial_velocity_valid[ids] = False

    def begin_substep(
        self,
        position_target: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Latch a 20-ms host frame, then return this 1-ms torque command."""
        if self._pending:
            raise RuntimeError("finish_substep must consume the previous physics result")
        at_s = self._tick * self.dt
        if self._tick % 20 == 0:
            zeros = np.zeros(self.shape, dtype=np.float64)
            self.communication.submit(
                MITCommand(position_target, zeros, self.kp, self.kd, zeros), at_s=at_s
            )
        command = self.communication.advance(at_s)
        self.torque[:] = self.actuator.compute_torque(
            position,
            velocity,
            command.q_des,
            command.qd_des,
            command.kp,
            command.kd,
            command.tau_ff,
        )
        self._velocity_before[:] = velocity
        self._tick += 1
        self._pending = True
        return self.torque

    def finish_substep(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Advance the feedback model from the post-physics joint state."""
        if not self._pending:
            return
        acceleration = (velocity - self._velocity_before) / self.dt
        estimate = self.actuator.observe(velocity, acceleration)
        self.feedback = self.communication.sample(
            self._tick * self.dt,
            position,
            self.actuator.velocity_feedback,
            estimate,
        )
        self._pending = False

    def quantize_position_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        if env_ids is None and self.feedback is not None:
            return self.feedback.q.copy()
        limit = float(self.parameters.registers["pmax"])
        return self._initial_feedback(value, env_ids, limit, 16,
                                      self._initial_position, self._initial_position_valid)

    def quantize_velocity_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        if env_ids is None and self.feedback is not None:
            return self.feedback.qd.copy()
        limit = float(self.parameters.registers["vmax"])
        return self._initial_feedback(value, env_ids, limit, 12,
                                      self._initial_velocity, self._initial_velocity_valid)

    @staticmethod
    def _initial_feedback(value, env_ids, limit, bits, held, valid) -> np.ndarray:
        """Hold the initial/reset observation until a transport frame arrives."""
        encoded = _mapped(np.asarray(value), -limit, limit, bits)
        if env_ids is not None:
            held[env_ids] = encoded
            valid[env_ids] = True
            return encoded
        held[~valid] = encoded[~valid]
        valid[:] = True
        return held.copy()
