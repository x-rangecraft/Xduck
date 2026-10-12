"""NumPy BAM XL330-M6 actuator matching the upstream MJLab implementation."""

from __future__ import annotations

import numpy as np


class Xl330BamBatchModel:
    """Vectorized XL330 M6 voltage controller and dynamic friction budget."""

    kt = 0.36601349688984386
    resistance = 2.8113923539223227
    armature = 0.0018077432831600838
    kp_fw = 200.0
    error_gain = 0.0028773775022263564
    max_current = 1.75
    friction_base = 0.004771183165566
    friction_stribeck = 0.004676345799486616
    load_friction_motor = 0.2667860954283698
    load_friction_external = 8.515871897059342e-06
    load_friction_motor_stribeck = 1.0722918395099123e-05
    load_friction_external_stribeck = 0.08077928978935671
    load_friction_motor_quad = 0.009972471242139415
    load_friction_external_quad = 0.004902565732332559
    dtheta_stribeck = 2.890372094130307
    alpha = 8.683259907618984
    friction_viscous = 0.005359668274599504

    def __init__(self, num_envs: int, dof_ids: np.ndarray, nv: int):
        self.num_envs = int(num_envs)
        self.dof_ids = np.asarray(dof_ids, dtype=np.intp)
        self.num_joints = len(self.dof_ids)
        self.nv = int(nv)
        self.vin = np.random.uniform(6.5, 8.2, size=(num_envs, 1))
        self.vin_drop_gain = np.random.uniform(0.0, 0.2, size=(num_envs, 1))
        self.friction_scale = np.ones((num_envs, 1), dtype=np.float64)
        self.kp_scale = np.ones((num_envs, 1), dtype=np.float64)
        self.kd_scale = np.ones((num_envs, 1), dtype=np.float64)
        self._prev_motor_torque = np.zeros((num_envs, self.num_joints), dtype=np.float64)
        self.torque = np.zeros_like(self._prev_motor_torque)
        self._history = np.zeros((7, num_envs, self.num_joints), dtype=np.float64)
        self._history_fresh = np.ones(num_envs, dtype=bool)
        self._history_cursor = 0
        self._frictionloss = np.zeros((num_envs, nv), dtype=np.float64)
        self._damping = np.zeros((num_envs, nv), dtype=np.float64)
        self._j4340_joint_ids = np.empty(0, dtype=np.intp)
        self._rated_torque = np.full_like(self.torque, 0.96)
        self._maximum_speed = np.full_like(self.torque, 7.5 / self.kt)
        self._rated_speed = self._maximum_speed.copy()
        self.i2t = np.zeros_like(self.torque)
        self._i2t_budget = np.ones_like(self.torque)
        self.position_bias_range = (0.0, 0.0)

    def reset(self, env_ids) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self._prev_motor_torque[ids] = 0.0
        self.torque[ids] = 0.0
        self._history[:, ids] = 0.0
        self._history_fresh[ids] = True
        self.friction_scale[ids] = np.random.uniform(0.9, 1.1, size=(len(ids), 1))

    def quantize_position_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        del env_ids
        return value

    def quantize_velocity_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        del env_ids
        return value

    def delayed_target(self, target: np.ndarray) -> np.ndarray:
        fresh = self._history_fresh
        if np.any(fresh):
            self._history[:, fresh] = target[fresh][None, :, :]
            self._history_fresh[fresh] = False
        cursor = self._history_cursor
        self._history[cursor] = target
        lag = np.random.randint(3, 7, size=self.num_envs)
        delayed = self._history[(cursor - lag) % len(self._history), np.arange(self.num_envs)]
        self._history_cursor = (cursor + 1) % len(self._history)
        return delayed

    def compute(
        self,
        position_target: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        backend,
    ) -> np.ndarray:
        target = self.delayed_target(position_target)
        effective_vin = self.vin - self.vin_drop_gain * np.sum(
            np.abs(self._prev_motor_torque), axis=1, keepdims=True
        )
        effective_vin = np.maximum(effective_vin, 6.0)
        scaled_vel = vel * self.kd_scale
        duty = (target - pos) * (self.kp_fw * self.kp_scale) * self.error_gain
        back_emf = self.kt * scaled_vel
        duty_span = self.resistance * self.max_current / effective_vin
        duty_center = back_emf / effective_vin
        duty = np.clip(duty, duty_center - duty_span, duty_center + duty_span)
        duty = np.clip(duty, -1.0, 1.0)
        volts = effective_vin * duty
        motor_torque = self.kt * volts / self.resistance
        motor_torque -= (self.kt**2) * scaled_vel / self.resistance

        bias, constraint, previous_actuator, friction_force = backend.get_dof_force_components()
        external = (
            -bias[:, self.dof_ids]
            + constraint[:, self.dof_ids]
            - friction_force[:, self.dof_ids]
        )
        previous_motor = previous_actuator[:, self.dof_ids]
        stribeck = np.exp(-np.power(np.abs(vel) / self.dtheta_stribeck, self.alpha))
        frictionloss = np.full_like(motor_torque, self.friction_base)
        frictionloss += stribeck * self.friction_stribeck
        frictionloss += np.abs(
            external * self.load_friction_external
            - previous_motor * self.load_friction_motor
        )
        frictionloss += stribeck * np.abs(
            external * self.load_friction_external_stribeck
            - previous_motor * self.load_friction_motor_stribeck
        )
        abs_external = np.abs(external)
        abs_motor = np.abs(previous_motor)
        drive = abs_motor > abs_external
        quadratic = np.where(
            drive,
            self.load_friction_external_quad * np.square(abs_external),
            self.load_friction_motor_quad * np.square(abs_motor),
        )
        frictionloss += stribeck * quadratic
        frictionloss *= self.friction_scale

        self._frictionloss.fill(0.0)
        self._damping.fill(0.0)
        self._frictionloss[:, self.dof_ids] = frictionloss
        self._damping[:, self.dof_ids] = self.friction_viscous
        backend.set_runtime_dof_properties(
            frictionloss=self._frictionloss,
            damping=self._damping,
        )
        self._prev_motor_torque[:] = motor_torque
        self.torque[:] = motor_torque
        return motor_torque
