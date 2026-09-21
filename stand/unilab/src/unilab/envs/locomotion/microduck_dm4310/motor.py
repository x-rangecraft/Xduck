"""24 V DM-J4310/4340 MIT actuator models for MicroDuck.

The active all-DM4340 owner uses a rated/peak envelope rather than the former
dynamometer lookup. Legacy DM4310 support retains its measured lookup path.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

RPM_TO_RAD_S = 2.0 * np.pi / 60.0
LEGACY_ACTUATOR_TIME_CONSTANT_S = 1.0 / (2.0 * np.pi * 200.0)

DM4310_MEASURED_TORQUE_NM = (
    0.160,
    0.306,
    0.436,
    0.563,
    0.685,
    0.801,
    0.928,
    1.048,
    1.174,
    1.292,
    1.415,
    1.535,
    1.651,
    1.776,
    1.887,
    2.016,
    2.134,
    2.248,
    2.368,
    2.500,
    2.617,
    2.730,
    2.850,
    2.966,
    3.091,
    3.200,
    3.319,
    3.436,
    3.556,
    3.668,
    3.794,
    3.911,
    4.017,
    4.136,
    4.243,
    4.374,
    4.481,
    4.593,
    4.695,
    4.814,
    4.923,
    5.041,
    5.149,
    5.266,
    5.373,
)
DM4310_MEASURED_SPEED_RPM = (
    217.1,
    214.5,
    212.6,
    210.3,
    208.7,
    207.3,
    205.3,
    203.4,
    201.4,
    199.4,
    197.3,
    195.2,
    193.1,
    191.1,
    188.8,
    186.7,
    184.6,
    182.4,
    180.1,
    177.9,
    175.4,
    172.9,
    170.6,
    168.1,
    165.6,
    163.3,
    160.7,
    158.1,
    155.3,
    152.7,
    149.7,
    147.0,
    144.3,
    141.2,
    138.1,
    134.8,
    131.3,
    128.4,
    124.7,
    120.8,
    117.2,
    113.6,
    109.2,
    105.2,
    101.1,
)

DM4340_MEASURED_TORQUE_NM = (
    1.14,
    4.35,
    5.08,
    6.23,
    7.36,
    7.79,
    8.17,
    8.89,
    9.61,
    9.98,
    10.36,
    10.74,
    11.53,
    12.30,
    12.64,
    12.99,
    13.39,
    13.80,
    14.14,
    14.59,
    14.91,
    15.26,
    15.67,
    16.09,
    16.44,
    16.71,
    17.17,
    17.54,
    17.86,
)
DM4340_MEASURED_SPEED_RPM = (
    55.0,
    52.0,
    50.7,
    49.5,
    48.3,
    48.3,
    47.5,
    46.7,
    45.8,
    45.5,
    44.9,
    44.4,
    43.6,
    42.3,
    41.5,
    41.0,
    40.6,
    40.0,
    39.2,
    38.5,
    37.3,
    37.2,
    36.3,
    35.2,
    34.6,
    33.8,
    32.7,
    31.7,
    30.6,
)

DM4340_EXTRAPOLATED_TORQUE_NM = (18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.5)
DM4340_EXTRAPOLATED_SPEED_RPM = (
    30.221065680849662,
    27.356102914320413,
    24.198011929532797,
    20.725164526549094,
    16.915932505431563,
    12.748687666242489,
    10.524051627390591,
)
DM4340_EXTRAPOLATED_SPEED_RPM_UPPER = (
    30.29752904879123,
    28.082663715513423,
    25.772432524464648,
    23.366835475644912,
    20.865872569054204,
    18.269543804692546,
    16.9356172258476,
)


@dataclass(frozen=True)
class Dm4310MitConfig:
    bus_voltage: float = 24.0
    min_voltage: float = 20.0
    max_voltage: float = 28.0
    rated_torque: float = 2.2
    peak_torque: float = 5.373
    rated_speed_rad_s: float = 101.1 * RPM_TO_RAD_S
    maximum_speed_rad_s: float = 219.94931506849316 * RPM_TO_RAD_S
    torque_speed_mode: str = "lookup"
    torque_speed_fit_p: float = 1.0
    torque_speed_fit_q: float = 1.0
    peak_i2t_seconds: float = 1.0
    measured_torque_nm: tuple[float, ...] = DM4310_MEASURED_TORQUE_NM
    measured_speed_rpm: tuple[float, ...] = DM4310_MEASURED_SPEED_RPM
    curve_torque_nm: tuple[float, ...] = DM4310_MEASURED_TORQUE_NM
    curve_speed_rpm: tuple[float, ...] = DM4310_MEASURED_SPEED_RPM
    curve_speed_rpm_upper: tuple[float, ...] = DM4310_MEASURED_SPEED_RPM
    zero_torque_speed_rad_s: float | None = None
    position_map_limit: float = 12.5
    velocity_map_limit: float = 220.0 * RPM_TO_RAD_S
    torque_map_limit: float = 5.373
    kp: float = 60.0
    kd: float = 4.0
    actuator_time_constant_s: float = LEGACY_ACTUATOR_TIME_CONSTANT_S
    actuator_time_constant_range_s: tuple[float, float] | None = None
    armature: float = 0.02
    coulomb_friction: float = 0.05
    viscous_friction: float = 0.15
    ambient_temperature_c: float = 25.0
    thermal_time_constant_s: float = 150.0
    rated_steady_temperature_c: float = 110.0
    recommended_temperature_c: float = 100.0
    shutdown_temperature_c: float = 120.0


@dataclass(frozen=True)
class Dm4340MitConfig(Dm4310MitConfig):
    """DM-J4340-2EC V1.1 24 V measured output-side curve.

    Thermal time constants remain explicit commissioning assumptions until a
    measured J4340 thermal curve is available.
    """

    rated_torque: float = 8.9
    peak_torque: float = 23.5
    kd: float = 2.0
    rated_speed_rad_s: float = 43.0 * RPM_TO_RAD_S
    maximum_speed_rad_s: float = 56.0 * RPM_TO_RAD_S
    torque_speed_mode: str = "fitted"
    torque_speed_fit_p: float = 2.2320256
    torque_speed_fit_q: float = 0.843808
    measured_torque_nm: tuple[float, ...] = DM4340_MEASURED_TORQUE_NM
    measured_speed_rpm: tuple[float, ...] = DM4340_MEASURED_SPEED_RPM
    curve_torque_nm: tuple[float, ...] = DM4340_MEASURED_TORQUE_NM + DM4340_EXTRAPOLATED_TORQUE_NM
    curve_speed_rpm: tuple[float, ...] = DM4340_MEASURED_SPEED_RPM + DM4340_EXTRAPOLATED_SPEED_RPM
    curve_speed_rpm_upper: tuple[float, ...] = (
        DM4340_MEASURED_SPEED_RPM + DM4340_EXTRAPOLATED_SPEED_RPM_UPPER
    )
    zero_torque_speed_rad_s: float | None = 5.809523809524
    velocity_map_limit: float = 10.0
    torque_map_limit: float = 28.0
    actuator_time_constant_s: float = 0.020
    armature: float = 0.030471555997140534
    coulomb_friction: float = (0.271549014755883 + 0.3247674828927923) / 2.0
    viscous_friction: float = (0.029647020954078847 + 0.030512363650819753) / 2.0


@dataclass
class Dm4340TorqueSpeedCurveRandConfig:
    torque_axis_scale_range: tuple[float, float] = (0.9, 1.0)
    speed_axis_scale_range: tuple[float, float] = (0.9773, 1.0)
    peak_region_shape_blend_range: tuple[float, float] = (0.0, 1.0)
    peak_power_range: tuple[float, float] = (63.0, 70.0)
    nominal_rated_torque: float = 8.9
    nominal_peak_torque: float = 23.5
    nominal_rated_velocity: float = 4.50294947
    nominal_maximum_velocity: float = 5.80952381


@dataclass
class Dm4340DomainRandConfig:
    enabled: bool = False
    torque_speed_curve: Dm4340TorqueSpeedCurveRandConfig = field(
        default_factory=Dm4340TorqueSpeedCurveRandConfig
    )
    armature_range: tuple[float, float] = (0.02590082, 0.03504229)
    coulomb_positive_range: tuple[float, float] = (0.23081666, 0.31228137)
    coulomb_negative_range: tuple[float, float] = (0.27605236, 0.37348261)
    viscous_positive_range: tuple[float, float] = (0.02519997, 0.03409407)
    viscous_negative_range: tuple[float, float] = (0.02102470, 0.04000003)
    delay_steps_range: tuple[int, int] = (0, 2)
    packet_loss_probability_range: tuple[float, float] = (0.00076873, 0.00095739)
    voltage_range: tuple[float, float] = (23.8, 24.2)
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    velocity_gain_positive: float = 0.98274170
    velocity_gain_negative: float = 0.98190142
    torque_feedback_bias: float = -0.00683761
    velocity_bias: float = -0.00244200
    position_noise_std: float = 0.00019074
    velocity_noise_std: float = 0.00244200
    torque_noise_std: float = 0.00683761
    position_bias_range: tuple[float, float] = (-0.0025, 0.0025)
    initial_temperature_range: tuple[float, float] = (32.0, 38.0)


def quantize_symmetric(value: np.ndarray, limit: float, bits: int) -> np.ndarray:
    """Round-trip a signed MIT mapped field through its unsigned CAN integer."""
    levels = float((1 << bits) - 1)
    clipped = np.clip(value, -limit, limit)
    encoded = np.rint((clipped + limit) * levels / (2.0 * limit))
    return encoded * (2.0 * limit) / levels - limit


def quantize_unsigned(value: float | np.ndarray, limit: float, bits: int) -> float | np.ndarray:
    levels = float((1 << bits) - 1)
    encoded = np.rint(np.clip(value, 0.0, limit) * levels / limit)
    result = encoded * limit / levels
    return float(result) if np.ndim(result) == 0 else result


def _interpolation_curve(cfg: Dm4310MitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending rad/s samples and their measured torque envelope."""
    torque = np.asarray(cfg.curve_torque_nm, dtype=np.float64)
    speed_rpm = np.asarray(cfg.curve_speed_rpm, dtype=np.float64)
    if torque.ndim != 1 or len(torque) < 2 or torque.shape != speed_rpm.shape:
        raise ValueError("measured motor curve must contain matching 1-D torque/speed samples")
    if np.any(np.diff(torque) <= 0.0) or np.any(np.diff(speed_rpm) > 0.0):
        raise ValueError("measured torque must increase while measured speed decreases")

    reversed_speed = speed_rpm[::-1]
    reversed_torque = torque[::-1]
    unique_speed, first_indices = np.unique(reversed_speed, return_index=True)
    unique_torque = reversed_torque[first_indices]
    no_load_rpm = (
        cfg.zero_torque_speed_rad_s / RPM_TO_RAD_S
        if cfg.zero_torque_speed_rad_s is not None
        else speed_rpm[0] + torque[0] * (speed_rpm[0] - speed_rpm[1]) / (torque[1] - torque[0])
    )
    speed = np.concatenate(([0.0], unique_speed, [no_load_rpm])) * RPM_TO_RAD_S
    envelope = np.concatenate(([torque[-1]], unique_torque, [0.0]))
    return speed, envelope


class Dm4310MitBatchModel:
    """Per-environment MIT response, rated/peak budget and thermal state."""

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        dt: float,
        cfg=None,
        joint_cfgs: Mapping[int, Dm4310MitConfig] | None = None,
        j4340_domain_rand: Dm4340DomainRandConfig | None = None,
    ):
        self.cfg = Dm4310MitConfig() if cfg is None else cfg
        self.dt = float(dt)
        configs = [self.cfg] * num_joints
        for joint_id, joint_cfg in (joint_cfgs or {}).items():
            if joint_id < 0 or joint_id >= num_joints:
                raise IndexError(f"joint motor config index {joint_id} outside [0, {num_joints})")
            configs[joint_id] = joint_cfg

        def param(name: str) -> np.ndarray:
            return np.asarray([float(getattr(item, name)) for item in configs], dtype=np.float64)

        shape = (num_envs, num_joints)
        self._bus_voltage = np.broadcast_to(param("bus_voltage"), shape).copy()
        self._min_voltage = param("min_voltage")
        self._max_voltage = param("max_voltage")
        self._rated_torque = np.broadcast_to(param("rated_torque"), shape).copy()
        self._peak_torque = np.broadcast_to(param("peak_torque"), shape).copy()
        self._rated_speed = np.broadcast_to(param("rated_speed_rad_s"), shape).copy()
        self._maximum_speed = np.broadcast_to(param("maximum_speed_rad_s"), shape).copy()
        self._zero_torque_speed = np.broadcast_to(
            np.asarray(
                [
                    float(item.zero_torque_speed_rad_s)
                    if item.zero_torque_speed_rad_s is not None
                    else float(item.maximum_speed_rad_s)
                    for item in configs
                ],
                dtype=np.float64,
            ),
            shape,
        ).copy()
        self._torque_speed_fit_p = param("torque_speed_fit_p")
        self._torque_speed_fit_q = param("torque_speed_fit_q")
        self._torque_map_limit = param("torque_map_limit")
        self._position_map_limit = param("position_map_limit")
        self._velocity_map_limit = param("velocity_map_limit")
        self._ambient_temperature = param("ambient_temperature_c")
        self._thermal_time_constant = param("thermal_time_constant_s")
        self._rated_steady_temperature = param("rated_steady_temperature_c")
        self._recommended_temperature = param("recommended_temperature_c")
        self._shutdown_temperature = param("shutdown_temperature_c")
        self._actuator_time_constant = np.broadcast_to(
            param("actuator_time_constant_s"), shape
        ).copy()
        self.armature = np.broadcast_to(param("armature"), shape).copy()
        self.coulomb_friction = np.broadcast_to(param("coulomb_friction"), shape).copy()
        self._viscous_friction = np.broadcast_to(param("viscous_friction"), shape).copy()
        self._coulomb_positive = self.coulomb_friction.copy()
        self._coulomb_negative = self.coulomb_friction.copy()
        self._viscous_positive = self._viscous_friction.copy()
        self._viscous_negative = self._viscous_friction.copy()
        self._velocity_gain_positive = np.ones(shape, dtype=np.float64)
        self._velocity_gain_negative = np.ones(shape, dtype=np.float64)
        self._velocity_bias = np.zeros(shape, dtype=np.float64)
        self._position_noise_std = np.zeros(shape, dtype=np.float64)
        self._velocity_noise_std = np.zeros(shape, dtype=np.float64)
        self._torque_feedback_bias = np.zeros(shape, dtype=np.float64)
        self._torque_noise_std = np.zeros(shape, dtype=np.float64)
        self.position_bias_range = (0.0, 0.0)
        self._curve_scale = np.ones(shape, dtype=np.float64)
        self._speed_curve_scale = np.ones(shape, dtype=np.float64)
        self._peak_curve_blend = np.zeros(shape, dtype=np.float64)
        self._peak_power = np.full(shape, np.inf, dtype=np.float64)
        self._physical_peak_torque = np.broadcast_to(
            np.asarray([max(item.curve_torque_nm) for item in configs]), shape
        ).copy()
        self._delay_steps = np.zeros(shape, dtype=np.int32)
        self._packet_loss_probability = np.zeros(shape, dtype=np.float64)
        self._initial_temperature = np.broadcast_to(self._ambient_temperature, shape).copy()
        base_kp = np.asarray([quantize_unsigned(item.kp, 500.0, 12) for item in configs])
        base_kd = np.asarray([quantize_unsigned(item.kd, 5.0, 12) for item in configs])
        self.kp = np.broadcast_to(base_kp, shape).copy()
        self.kd = np.broadcast_to(base_kd, shape).copy()
        self._j4340_joint_ids = np.asarray(
            [i for i, item in enumerate(configs) if isinstance(item, Dm4340MitConfig)],
            dtype=np.intp,
        )
        self._rated_peak_joint_ids = np.asarray(
            [i for i, item in enumerate(configs) if item.torque_speed_mode == "rated_peak"],
            dtype=np.intp,
        )
        self._lookup_joint_ids = np.asarray(
            [i for i, item in enumerate(configs) if item.torque_speed_mode == "lookup"],
            dtype=np.intp,
        )
        self._fitted_joint_ids = np.asarray(
            [i for i, item in enumerate(configs) if item.torque_speed_mode == "fitted"],
            dtype=np.intp,
        )
        dr = j4340_domain_rand or Dm4340DomainRandConfig()
        if dr.enabled and len(self._j4340_joint_ids):
            ids = self._j4340_joint_ids
            sample_shape = (num_envs, len(ids))

            def sample(bounds, *, size=sample_shape):
                return np.random.uniform(float(bounds[0]), float(bounds[1]), size=size)

            curve_dr = dr.torque_speed_curve
            torque_scale = sample(curve_dr.torque_axis_scale_range)
            speed_scale = sample(curve_dr.speed_axis_scale_range)
            self._rated_torque[:, ids] = curve_dr.nominal_rated_torque * torque_scale
            self._peak_torque[:, ids] = curve_dr.nominal_peak_torque * torque_scale
            self._physical_peak_torque[:, ids] = self._peak_torque[:, ids]
            self._curve_scale[:, ids] = torque_scale
            self._speed_curve_scale[:, ids] = speed_scale
            self._rated_speed[:, ids] = curve_dr.nominal_rated_velocity * speed_scale
            self._maximum_speed[:, ids] = curve_dr.nominal_maximum_velocity * speed_scale
            self._zero_torque_speed[:, ids] = curve_dr.nominal_maximum_velocity * speed_scale
            self._peak_curve_blend[:, ids] = sample(curve_dr.peak_region_shape_blend_range)
            self._peak_power[:, ids] = sample(curve_dr.peak_power_range)
            self.armature[:, ids] = sample(dr.armature_range)
            positive = sample(dr.coulomb_positive_range)
            negative = sample(dr.coulomb_negative_range)
            viscous_positive = sample(dr.viscous_positive_range)
            viscous_negative = sample(dr.viscous_negative_range)
            self._coulomb_positive[:, ids] = positive
            self._coulomb_negative[:, ids] = negative
            self._viscous_positive[:, ids] = viscous_positive
            self._viscous_negative[:, ids] = viscous_negative
            self.coulomb_friction[:, ids] = 0.5 * (positive + negative)
            lo_delay, hi_delay = dr.delay_steps_range
            self._delay_steps[:, ids] = np.random.randint(
                int(lo_delay), int(hi_delay) + 1, size=sample_shape
            )
            self._packet_loss_probability[:, ids] = sample(dr.packet_loss_probability_range)
            self._bus_voltage[:, ids] = sample(dr.voltage_range)
            self.kp[:, ids] *= sample(dr.kp_scale_range)
            self.kd[:, ids] *= sample(dr.kd_scale_range)
            self._velocity_gain_positive[:, ids] = dr.velocity_gain_positive
            self._velocity_gain_negative[:, ids] = dr.velocity_gain_negative
            self._velocity_bias[:, ids] = dr.velocity_bias
            self._position_noise_std[:, ids] = dr.position_noise_std
            self._velocity_noise_std[:, ids] = dr.velocity_noise_std
            self._torque_feedback_bias[:, ids] = dr.torque_feedback_bias
            self._torque_noise_std[:, ids] = dr.torque_noise_std
            self.position_bias_range = dr.position_bias_range
            self._initial_temperature[:, ids] = sample(dr.initial_temperature_range)
        curve_groups: dict[
            tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], list[int]
        ] = {}
        for joint_id, item in enumerate(configs):
            key = (
                item.curve_torque_nm,
                item.curve_speed_rpm,
                item.curve_speed_rpm_upper,
            )
            curve_groups.setdefault(key, []).append(joint_id)
        self._curve_groups = []
        for joint_ids in curve_groups.values():
            speed, torque = _interpolation_curve(configs[joint_ids[0]])
            upper_cfg = copy.copy(configs[joint_ids[0]])
            object.__setattr__(upper_cfg, "curve_speed_rpm", upper_cfg.curve_speed_rpm_upper)
            upper_speed, upper_torque = _interpolation_curve(upper_cfg)
            self._curve_groups.append(
                (
                    np.asarray(joint_ids, dtype=np.intp),
                    speed,
                    torque,
                    upper_speed,
                    upper_torque,
                )
            )
        self._max_curve_torque = np.asarray(
            [max(item.curve_torque_nm) for item in configs], dtype=np.float64
        )
        self.strength_scale = np.ones(num_joints, dtype=np.float64)
        self.torque = np.zeros(shape, dtype=np.float64)
        self.i2t = np.zeros(shape, dtype=np.float64)
        peak_ratio_sq = np.square(self._peak_torque / self._rated_torque)
        self._i2t_budget = np.broadcast_to(param("peak_i2t_seconds"), shape) * (peak_ratio_sq - 1.0)
        max_delay = int(np.max(self._delay_steps))
        self._command_history = np.zeros((max_delay + 1, *shape), dtype=np.float64)
        self._command_history_cursor = 0
        self._accepted_command = np.zeros(shape, dtype=np.float64)
        self.temperature = self._initial_temperature.copy()
        self.shutdown = np.zeros(shape, dtype=bool)

    def set_strength_scale(self, scale: float | np.ndarray) -> None:
        """Scale the measured torque envelope and thermal rating together."""
        values = np.broadcast_to(np.asarray(scale, dtype=np.float64), self.strength_scale.shape)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("actuator strength scale must contain finite positive values")
        self.strength_scale[:] = values

    def reset(self, env_ids) -> None:
        ids = np.asarray(env_ids, dtype=np.int32)
        self.torque[ids] = 0.0
        self.i2t[ids] = 0.0
        self._command_history[:, ids] = 0.0
        self._accepted_command[ids] = 0.0
        self.temperature[ids] = self._initial_temperature[ids]
        self.shutdown[ids] = False

    def quantize_position_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        quantized = quantize_symmetric(value, self._position_map_limit, 16)
        noise_std = (
            self._position_noise_std
            if env_ids is None
            else self._position_noise_std[np.asarray(env_ids, dtype=np.int32)]
        )
        noise = np.random.normal(size=quantized.shape) * noise_std
        return quantized + noise

    def quantize_velocity_feedback(
        self, value: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        quantized = quantize_symmetric(value, self._velocity_map_limit, 12)
        ids = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.int32)
        gain = np.where(
            quantized >= 0.0,
            self._velocity_gain_positive[ids],
            self._velocity_gain_negative[ids],
        )
        noise = np.random.normal(size=quantized.shape) * self._velocity_noise_std[ids]
        return quantized * gain + self._velocity_bias[ids] + noise

    def quantize_torque_feedback(self, value: np.ndarray) -> np.ndarray:
        quantized = quantize_symmetric(value, self._torque_map_limit, 12)
        noise = np.random.normal(size=quantized.shape) * self._torque_noise_std
        return quantized + self._torque_feedback_bias + noise

    def torque_speed_limit(self, vel: np.ndarray) -> np.ndarray:
        """Return the same-direction accelerating torque limit."""
        velocity = np.asarray(vel, dtype=np.float64)
        if velocity.ndim != 2 or velocity.shape[1] != len(self.strength_scale):
            raise ValueError(
                "velocity must have shape (num_envs, num_joints) matching the motor model"
            )
        curve_limit = np.empty_like(velocity)
        if len(self._rated_peak_joint_ids):
            ids = self._rated_peak_joint_ids
            speed = np.abs(velocity[:, ids])
            rated_speed = self._rated_speed[:, ids]
            maximum_speed = self._maximum_speed[:, ids]
            peak = self._peak_torque[:, ids]
            rated = self._rated_torque[:, ids]
            below_rated = peak + (rated - peak) * np.divide(
                speed,
                rated_speed,
                out=np.zeros_like(speed),
                where=rated_speed > 0.0,
            )
            above_rated = rated * np.divide(
                maximum_speed - speed,
                maximum_speed - rated_speed,
                out=np.zeros_like(speed),
                where=maximum_speed > rated_speed,
            )
            curve_limit[:, ids] = np.where(speed <= rated_speed, below_rated, above_rated)
            curve_limit[:, ids] = np.clip(curve_limit[:, ids], 0.0, peak)
        if len(self._fitted_joint_ids):
            ids = self._fitted_joint_ids
            # The fit was identified at the nominal 24 V bus voltage.  Scaling
            # omega_zero with voltage reproduces the DC-motor back-EMF effect:
            # lower voltage reduces no-load speed while the current/peak-torque
            # ceiling remains independently bounded.
            voltage_scale = self._bus_voltage[:, ids] / self.cfg.bus_voltage
            zero_speed = self._zero_torque_speed[:, ids] * voltage_scale
            normalized_speed = np.divide(
                np.abs(velocity[:, ids]),
                zero_speed,
                out=np.full_like(velocity[:, ids], np.inf),
                where=zero_speed > 0.0,
            )
            remaining = np.maximum(
                0.0,
                1.0
                - np.power(
                    normalized_speed,
                    self._torque_speed_fit_p[ids][None, :],
                ),
            )
            curve_limit[:, ids] = self._physical_peak_torque[:, ids] * np.power(
                remaining,
                self._torque_speed_fit_q[ids][None, :],
            )
        for (
            joint_ids,
            curve_speed,
            curve_torque,
            upper_speed,
            upper_torque,
        ) in self._curve_groups:
            if not np.any(np.isin(joint_ids, self._lookup_joint_ids)):
                continue
            normalized_speed = (
                np.abs(velocity[:, joint_ids]) / self._speed_curve_scale[:, joint_ids]
            )
            nominal_limit = np.interp(
                normalized_speed,
                curve_speed,
                curve_torque,
                left=curve_torque[0],
                right=0.0,
            )
            upper_limit = np.interp(
                normalized_speed,
                upper_speed,
                upper_torque,
                left=upper_torque[0],
                right=0.0,
            )
            blend = self._peak_curve_blend[:, joint_ids]
            curve_limit[:, joint_ids] = (
                ((1.0 - blend) * nominal_limit + blend * upper_limit)
                * self.strength_scale[joint_ids]
                * self._curve_scale[:, joint_ids]
            )
        curve_limit = np.minimum(
            curve_limit,
            self._physical_peak_torque * self.strength_scale[None, :],
        )
        speed = np.abs(velocity)
        power_limit = np.divide(
            self._peak_power,
            speed,
            out=np.full_like(speed, np.inf),
            where=speed > 1e-9,
        )
        curve_limit = np.minimum(curve_limit, power_limit)
        return curve_limit

    def compute(self, position_target: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        self.shutdown |= self.temperature >= self._shutdown_temperature
        q_des = quantize_symmetric(position_target, self._position_map_limit, 16)
        mit_command = self.kp * (q_des - pos) - self.kd * vel
        mit_command = np.clip(
            mit_command,
            -self._torque_map_limit[None, :],
            self._torque_map_limit[None, :],
        )
        cursor = self._command_history_cursor
        self._command_history[cursor] = mit_command
        env_ids = np.arange(len(mit_command))[:, None]
        joint_ids = np.arange(mit_command.shape[1])[None, :]
        delayed_index = (cursor - self._delay_steps) % len(self._command_history)
        delayed_command = self._command_history[delayed_index, env_ids, joint_ids]
        lost = np.random.uniform(size=mit_command.shape) < self._packet_loss_probability
        self._accepted_command[:] = np.where(lost, self._accepted_command, delayed_command)
        self._command_history_cursor = (cursor + 1) % len(self._command_history)
        requested = self._accepted_command

        # Interpolate the supplied 24 V curve at the magnitude of current joint
        # speed. It applies only when torque accelerates in the motion direction;
        # reverse braking retains the physical/thermal/structural limits.
        rated_torque = self._rated_torque * self.strength_scale[None, :]
        curve_limit = self.torque_speed_limit(vel)
        physical_limit = self._physical_peak_torque * self.strength_scale[None, :]
        load_sq = np.square(
            np.divide(
                np.abs(self.torque),
                rated_torque,
                out=np.zeros_like(self.torque),
                where=rated_torque > 0.0,
            )
        )
        self.i2t[:] = np.clip(
            self.i2t + self.dt * (load_sq - 1.0),
            0.0,
            self._i2t_budget,
        )
        peak_available = self.i2t < self._i2t_budget
        overload_limit = np.where(peak_available, physical_limit, rated_torque)
        accelerating = requested * vel > 0.0
        requested_limit = np.where(accelerating, curve_limit, physical_limit)
        requested_limit = np.minimum(requested_limit, overload_limit)
        requested = np.clip(requested, -requested_limit, requested_limit)

        # First-order response uses the fitted per-environment time constant.
        alpha = 1.0 - np.exp(-self.dt / self._actuator_time_constant)
        self.torque[:] += alpha * (requested - self.torque)
        accelerating = self.torque * vel > 0.0
        output_limit = np.where(accelerating, curve_limit, physical_limit)
        output_limit = np.minimum(output_limit, overload_limit)
        self.torque[:] = np.clip(self.torque, -output_limit, output_limit)

        steady_rise = (self._rated_steady_temperature - self._ambient_temperature) * np.square(
            np.abs(self.torque) / rated_torque
        )
        target_temp = self._ambient_temperature + steady_rise
        self.temperature[:] += (
            self.dt * (target_temp - self.temperature) / self._thermal_time_constant
        )
        self.shutdown |= self.temperature >= self._shutdown_temperature
        hot = self.temperature >= self._recommended_temperature
        rated_limits = np.broadcast_to(rated_torque, self.torque.shape)
        self.torque[hot] = np.clip(self.torque[hot], -rated_limits[hot], rated_limits[hot])
        self.torque[self.shutdown] = 0.0

        # MuJoCo carries the randomized mean Coulomb term and nominal viscous
        # term. Apply only the fitted directional residual here.
        positive = vel >= 0.0
        directional_coulomb = np.where(positive, self._coulomb_positive, self._coulomb_negative)
        directional_viscous = np.where(positive, self._viscous_positive, self._viscous_negative)
        friction_residual = np.sign(vel) * (directional_coulomb - self.coulomb_friction) + vel * (
            directional_viscous - self._viscous_friction
        )
        output = self.torque - friction_residual
        output[self.shutdown] = 0.0
        return output
