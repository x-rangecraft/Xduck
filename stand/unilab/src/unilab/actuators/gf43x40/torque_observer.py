"""Causal residual observer of CAN current estimate, never a mechanical force.

Feature and time-constant ordering is a versioned numeric contract with the C
replay and the portable UniLab implementation. No measured feedback is an input.
"""

import numpy as np

TIME_CONSTANTS = np.array([0.002, 0.008, 0.030, 0.150, 0.500])
FEATURE_NAMES = (
    "acceleration_x005",
    "effort",
    "request_minus_effort",
    "velocity_sign",
    "velocity",
    "holding_request",
    "holding_gate",
    "holding_memory",
    "position_effort",
    "velocity_effort",
    "feedforward",
    "gain_acceleration",
    "gain_effort",
    "gain_effort_error",
    "gain_holding_request",
    "gain_holding_memory",
    "gain_velocity_sign",
    "gain_velocity",
)


def features(acceleration, velocity, request, effort, memory, kp, kd, q_error, v_des, ff):
    a, v, r, u, z, kp, kd, qe, vd, ff = np.broadcast_arrays(
        acceleration, velocity, request, effort, memory, kp, kd, q_error, v_des, ff
    )
    hold = np.exp(-np.square(v / 0.05))
    g = kp / (kp + 80.0)
    sign = np.tanh(v / 0.05)
    return np.stack(
        (
            0.05 * a,
            u,
            r - u,
            sign,
            v,
            r * hold,
            hold,
            z * hold,
            np.clip(kp * qe, -28.0, 28.0),
            np.clip(kd * (vd - v), -28.0, 28.0),
            ff,
            g * 0.05 * a,
            g * u,
            g * (r - u),
            g * r * hold,
            g * z * hold,
            g * sign,
            g * v,
        ),
        axis=-1,
    )


class TorqueResidualObserver:
    def __init__(self, observation, shape=()):
        self.enabled = "torque_residual_coefficients" in observation
        self.coefficients = np.asarray(
            observation.get("torque_residual_coefficients", np.zeros((5, 18))), float
        )
        schema = observation.get("torque_residual_schema")
        if self.enabled and schema == "causal_bank_v1" and self.coefficients.shape == (3, 18):
            self.coefficients = np.pad(self.coefficients, ((0, 2), (0, 0)))
        elif self.enabled and schema != "causal_bank_v2":
            raise ValueError("Unknown torque residual feature schema or coefficient shape")
        if self.coefficients.shape != (5, 18) or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("Torque residual coefficients must be finite with shape (5,18)")
        self.state = np.zeros((*shape, 5, 18), float)

    def reset(self, index=None):
        if index is None:
            self.state.fill(0.0)
        else:
            self.state[index] = 0.0

    def step(self, acceleration, velocity, request, effort, memory, kp, kd, q_error, v_des, ff, dt):
        if not self.enabled:
            return np.zeros(self.state.shape[:-2])
        f = features(acceleration, velocity, request, effort, memory, kp, kd, q_error, v_des, ff)
        alpha = -np.expm1(-dt / TIME_CONSTANTS)
        self.state += alpha[:, None] * (f[..., None, :] - self.state)
        return np.sum(self.state * self.coefficients, axis=(-2, -1))
