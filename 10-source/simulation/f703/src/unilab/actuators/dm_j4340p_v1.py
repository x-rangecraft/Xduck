"""DM-J4340P-2EC V1.0 reference motor model.

The per-motor specification is kept separate from platform-owned power-bus
and thermal assumptions. Matching the motor part number alone does not prove
that two robots share those system contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_positive(owner: object, names: tuple[str, ...], prefix: str) -> None:
    for name in names:
        value = float(getattr(owner, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")


@dataclass(frozen=True)
class DMJ4340PV10MotorSpec:
    """Locally recorded V1.0 motor and reducer reference values."""

    reference_voltage_v: float = 24.0
    rated_torque_nm: float = 9.0
    peak_torque_nm: float = 27.0
    rated_speed_rpm: float = 36.0
    no_load_speed_rpm: float = 52.0
    rated_current_a: float = 2.5
    phase_current_limit_a: float = 8.0
    pole_pairs: float = 14.0
    gear_ratio: float = 40.0
    phase_resistance_ohm: float = 0.88
    phase_inductance_h: float = 360.0e-6
    flux_wb: float = 0.004543919467762022
    gear_torque_efficiency: float = 0.8842271900641813
    current_bandwidth_hz: float = 1000.0
    encoder_bits: int = 14

    def validate(self) -> None:
        _require_positive(
            self,
            (
                "reference_voltage_v",
                "rated_torque_nm",
                "peak_torque_nm",
                "rated_speed_rpm",
                "no_load_speed_rpm",
                "rated_current_a",
                "phase_current_limit_a",
                "pole_pairs",
                "gear_ratio",
                "phase_resistance_ohm",
                "phase_inductance_h",
                "flux_wb",
                "gear_torque_efficiency",
                "current_bandwidth_hz",
            ),
            "motor",
        )
        if self.peak_torque_nm < self.rated_torque_nm:
            raise ValueError("motor peak torque must not be below rated torque")
        if self.no_load_speed_rpm < self.rated_speed_rpm:
            raise ValueError("motor no-load speed must not be below rated speed")
        if not 0.0 < self.gear_torque_efficiency <= 1.0:
            raise ValueError("motor.gear_torque_efficiency must lie within (0, 1]")
        if self.encoder_bits <= 0:
            raise ValueError("motor.encoder_bits must be positive")

    @property
    def output_torque_constant_nm_per_a(self) -> float:
        """Output-shaft proxy calibrated by 27 N m at 8 A."""

        return self.peak_torque_nm / self.phase_current_limit_a

    @property
    def encoder_step_rad(self) -> float:
        return float(2.0 * np.pi / (1 << self.encoder_bits))


@dataclass(frozen=True)
class SharedDCBusConfig:
    """Robot-owned source limits; construct from the new robot evidence."""

    voltage_v: float
    continuous_current_a: float
    peak_current_a: float
    peak_budget_s: float

    def validate(self) -> None:
        _require_positive(
            self,
            ("voltage_v", "continuous_current_a", "peak_current_a", "peak_budget_s"),
            "power_bus",
        )
        if self.peak_current_a < self.continuous_current_a:
            raise ValueError("power_bus peak current must not be below continuous current")

    @property
    def peak_budget_a2s(self) -> float:
        return float((self.peak_current_a**2 - self.continuous_current_a**2) * self.peak_budget_s)


@dataclass(frozen=True)
class DMJ4340PV10LossThermalProxy:
    """Rider-derived first-order prior; revalidate for every installation."""

    speed_loss_w_per_rad_s: float = 1.65
    ambient_temperature_c: float = 25.0
    thermal_resistance_c_per_w: float = 2.6
    thermal_capacity_j_per_c: float = 73.0
    thermal_derate_start_c: float = 98.0
    thermal_cutoff_c: float = 100.0

    def validate(self) -> None:
        _require_positive(
            self,
            (
                "thermal_resistance_c_per_w",
                "thermal_capacity_j_per_c",
                "thermal_cutoff_c",
            ),
            "thermal",
        )
        if not np.isfinite(self.speed_loss_w_per_rad_s) or self.speed_loss_w_per_rad_s < 0.0:
            raise ValueError("thermal.speed_loss_w_per_rad_s must be finite and non-negative")
        if not np.isfinite(self.ambient_temperature_c):
            raise ValueError("thermal.ambient_temperature_c must be finite")
        if not np.isfinite(self.thermal_derate_start_c):
            raise ValueError("thermal.thermal_derate_start_c must be finite")
        if self.thermal_derate_start_c >= self.thermal_cutoff_c:
            raise ValueError("thermal derating must start below the cutoff")


def quantize_output_position(position_rad: np.ndarray, encoder_step_rad: float) -> np.ndarray:
    """Quantize output-shaft position to an explicitly selected encoder grid."""

    return np.rint(position_rad / encoder_step_rad) * encoder_step_rad


def dq_current_bounds(
    output_speed_rad_s: np.ndarray,
    motor: DMJ4340PV10MotorSpec,
    *,
    bus_voltage_v: float,
    current_scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return four-quadrant q-axis current bounds from the voltage ellipse."""

    if not np.isfinite(bus_voltage_v) or bus_voltage_v <= 0.0:
        raise ValueError("bus_voltage_v must be finite and positive")
    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    electrical_speed = speed * motor.gear_ratio * motor.pole_pairs
    resistance = motor.phase_resistance_ohm
    inductance = motor.phase_inductance_h
    flux = motor.flux_wb
    voltage_limit = float(bus_voltage_v) / np.sqrt(3.0)

    a = resistance**2 + (electrical_speed * inductance) ** 2
    b = 2.0 * resistance * electrical_speed * flux
    c = (electrical_speed * flux) ** 2 - voltage_limit**2
    discriminant = b**2 - 4.0 * a * c
    feasible = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    lower = (-b - root) / (2.0 * a)
    upper = (-b + root) / (2.0 * a)

    scaled_limit = motor.phase_current_limit_a * np.asarray(current_scale, dtype=np.float64)
    lower = np.maximum(lower, -scaled_limit)
    upper = np.minimum(upper, scaled_limit)
    valid = feasible & (lower <= upper)
    return np.where(valid, lower, 0.0), np.where(valid, upper, 0.0)


def motor_power_and_loss(
    output_torque_nm: np.ndarray,
    output_speed_rad_s: np.ndarray,
    phase_current_a: np.ndarray,
    motor: DMJ4340PV10MotorSpec,
    thermal: DMJ4340PV10LossThermalProxy,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate signed DC-bus power and heat generation for every actuator."""

    torque = np.asarray(output_torque_nm, dtype=np.float64)
    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    current = np.asarray(phase_current_a, dtype=np.float64)
    output_power = torque * speed
    motoring = output_power >= 0.0
    efficiency = motor.gear_torque_efficiency
    gear_loss = np.where(
        motoring,
        output_power * (1.0 / efficiency - 1.0),
        -output_power * (1.0 - efficiency),
    )
    copper_loss = 1.5 * motor.phase_resistance_ohm * np.square(current)
    speed_loss = thermal.speed_loss_w_per_rad_s * np.abs(speed)
    heat_loss = copper_loss + speed_loss + gear_loss
    return output_power + heat_loss, heat_loss


def apply_shared_bus_limit(
    output_torque_nm: np.ndarray,
    phase_current_a: np.ndarray,
    output_speed_rad_s: np.ndarray,
    motor: DMJ4340PV10MotorSpec,
    thermal: DMJ4340PV10LossThermalProxy,
    power_bus: SharedDCBusConfig,
    *,
    bus_current_limit_a: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate shared source power without assuming an actuator count.

    Motoring/stall currents are scaled together only when net requested DC
    power exceeds the selected source limit. Braking joints remain available
    and offset motoring demand on the common bus.
    """

    torque = np.asarray(output_torque_nm, dtype=np.float64)
    current = np.asarray(phase_current_a, dtype=np.float64)
    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    if torque.ndim != 2 or current.shape != torque.shape or speed.shape != torque.shape:
        raise ValueError("motor torque, current, and speed must share shape (N, num_motors)")

    power_limit = power_bus.voltage_v * np.broadcast_to(
        np.asarray(bus_current_limit_a, dtype=np.float64), (torque.shape[0],)
    )
    output_power = torque * speed
    motoring = output_power >= 0.0
    copper = 1.5 * motor.phase_resistance_ohm * np.square(current)
    core = thermal.speed_loss_w_per_rad_s * np.abs(speed)
    efficiency = motor.gear_torque_efficiency

    linear = np.sum(np.where(motoring, output_power / efficiency, 0.0), axis=1)
    quadratic = np.sum(np.where(motoring, copper, 0.0), axis=1)
    fixed = np.sum(core, axis=1) + np.sum(
        np.where(motoring, 0.0, output_power * efficiency + copper), axis=1
    )

    scale = np.ones((torque.shape[0],), dtype=np.float64)
    limited = linear + quadratic + fixed > power_limit
    linear_only = limited & (quadratic <= np.finfo(np.float64).eps)
    scale[linear_only] = np.divide(
        power_limit[linear_only] - fixed[linear_only],
        linear[linear_only],
        out=np.zeros_like(scale[linear_only]),
        where=linear[linear_only] > 0.0,
    )
    quadratic_case = limited & ~linear_only
    discriminant = np.maximum(
        linear[quadratic_case] ** 2
        - 4.0 * quadratic[quadratic_case] * (fixed[quadratic_case] - power_limit[quadratic_case]),
        0.0,
    )
    scale[quadratic_case] = (-linear[quadratic_case] + np.sqrt(discriminant)) / (
        2.0 * quadratic[quadratic_case]
    )
    np.clip(scale, 0.0, 1.0, out=scale)

    scaled_torque = np.where(motoring, torque * scale[:, None], torque)
    scaled_current = np.where(motoring, current * scale[:, None], current)
    dc_power, heat_loss = motor_power_and_loss(scaled_torque, speed, scaled_current, motor, thermal)
    return scaled_torque, scaled_current, np.sum(dc_power, axis=1), heat_loss


def thermal_current_scale(
    temperature_c: np.ndarray,
    thermal: DMJ4340PV10LossThermalProxy,
) -> np.ndarray:
    """Linearly derate inside the configured temperature band."""

    remaining = thermal.thermal_cutoff_c - np.asarray(temperature_c, dtype=np.float64)
    width = thermal.thermal_cutoff_c - thermal.thermal_derate_start_c
    return np.clip(remaining / width, 0.0, 1.0)


def update_motor_temperature(
    temperature_c: np.ndarray,
    heat_loss_w: np.ndarray,
    dt_s: float,
    thermal: DMJ4340PV10LossThermalProxy,
) -> np.ndarray:
    """Advance the first-order winding-temperature proxy by one tick."""

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    temperature = np.asarray(temperature_c, dtype=np.float64)
    cooling = (temperature - thermal.ambient_temperature_c) / thermal.thermal_resistance_c_per_w
    derivative = (
        np.asarray(heat_loss_w, dtype=np.float64) - cooling
    ) / thermal.thermal_capacity_j_per_c
    return temperature + float(dt_s) * derivative
