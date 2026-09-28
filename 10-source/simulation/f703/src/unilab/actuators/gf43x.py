"""Datasheet-bounded GF43X10-10 and GF43X40-10 actuator models.

This module is intentionally an early-validation, robot-independent layer.  It
records nameplate values and the supplied 24 V bench sweep without inventing
electrical, controller, or thermal dynamics that the source manual does not
specify.  The optional empty-load fit is kept separate from the nameplate
specification because it is a CAN-observable, output-side equivalent fit, not
a claim about the drive's internal electrical parameters.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal

import numpy as np

Duty = Literal["rated", "peak"]


@dataclass(frozen=True)
class GF43ActuatorResponseFit:
    """One measured first-order response point for an empty-load actuator.

    The response was fitted from CAN torque feedback at a particular MIT
    ``Kd``.  It is deliberately represented as a lookup point: interpolating
    between the two measured ``Kd`` values is acceptable for the task owner,
    while extrapolating a detailed controller model is not justified by the
    supplied data.
    """

    kd: float
    time_constant_s: float
    command_gain: float
    time_constant_range_s: tuple[float, float]
    command_gain_range: tuple[float, float]
    delay_s: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.kd) or self.kd < 0.0:
            raise ValueError("actuator response Kd must be finite and non-negative")
        for name in ("time_constant_s", "command_gain"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"actuator response {name} must be finite and positive")
        for name in ("time_constant_range_s", "command_gain_range"):
            lower, upper = (float(value) for value in getattr(self, name))
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower <= 0.0
                or upper < lower
            ):
                raise ValueError(f"actuator response {name} must be a positive ordered range")
        if not np.isfinite(self.delay_s) or self.delay_s < 0.0:
            raise ValueError("actuator response delay must be finite and non-negative")


@dataclass(frozen=True)
class GF43EmptyLoadFit:
    """Reliable subset of ``j4340_empty_fit.json`` for simulation use.

    Values are aggregated over the three measured motors (CAN IDs 13, 14 and
    15).  The fit is therefore a nominal model for a fleet of fourteen motors,
    not a per-joint calibration.  The source hash is recorded so that a future
    replacement of the downloaded evidence cannot silently change the model.
    Unknown thermal, absolute-position, encoder-resolution, and loaded-joint
    quantities intentionally do not appear here.  In particular, response
    time constants inferred from empty-load CAN torque feedback are reference
    evidence only: the measurement method does not establish a loaded-joint
    torque response and must not be treated as a deployment truth model.
    """

    source_sha256: str
    measured_motor_ids: tuple[int, ...]
    effective_inertia_kg_m2: float
    effective_inertia_range_kg_m2: tuple[float, float]
    coulomb_positive_nm: float
    coulomb_positive_range_nm: tuple[float, float]
    coulomb_negative_nm: float
    coulomb_negative_range_nm: tuple[float, float]
    viscous_positive_nm_s_rad: float
    viscous_positive_range_nm_s_rad: tuple[float, float]
    viscous_negative_nm_s_rad: float
    viscous_negative_range_nm_s_rad: tuple[float, float]
    torque_offset_nm: float
    velocity_gain_positive: float
    velocity_gain_negative: float
    response_points: tuple[GF43ActuatorResponseFit, ...]
    communication_latency_s: float
    communication_latency_range_s: tuple[float, float]
    packet_loss_probability: float
    packet_loss_probability_range: tuple[float, float]
    velocity_bias_rad_s: float
    velocity_noise_std_upper_bound_rad_s: float
    torque_feedback_bias_nm: float
    torque_noise_std_upper_bound_nm: float

    @property
    def symmetric_coulomb_nm(self) -> float:
        """Return the scalar Coulomb loss used when a backend is symmetric."""

        return 0.5 * (self.coulomb_positive_nm + self.coulomb_negative_nm)

    @property
    def symmetric_viscous_nm_s_rad(self) -> float:
        """Return the scalar viscous loss used when a backend is symmetric."""

        return 0.5 * (self.viscous_positive_nm_s_rad + self.viscous_negative_nm_s_rad)

    def validate(self) -> None:
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256.lower()
        ):
            raise ValueError("empty-load fit source_sha256 must be a 64-character hex digest")
        if not self.measured_motor_ids or any(int(motor_id) <= 0 for motor_id in self.measured_motor_ids):
            raise ValueError("empty-load fit must identify positive measured motor IDs")

        positive_fields = (
            "effective_inertia_kg_m2",
            "coulomb_positive_nm",
            "coulomb_negative_nm",
            "viscous_positive_nm_s_rad",
            "viscous_negative_nm_s_rad",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"empty-load fit {field_name} must be finite and positive")

        range_fields = (
            ("effective_inertia_range_kg_m2", "effective_inertia_kg_m2"),
            ("coulomb_positive_range_nm", "coulomb_positive_nm"),
            ("coulomb_negative_range_nm", "coulomb_negative_nm"),
            ("viscous_positive_range_nm_s_rad", "viscous_positive_nm_s_rad"),
            ("viscous_negative_range_nm_s_rad", "viscous_negative_nm_s_rad"),
            ("communication_latency_range_s", "communication_latency_s"),
            ("packet_loss_probability_range", "packet_loss_probability"),
        )
        for range_name, value_name in range_fields:
            lower, upper = (float(value) for value in getattr(self, range_name))
            value = float(getattr(self, value_name))
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower < 0.0
                or upper < lower
                or value < lower
                or value > upper
            ):
                raise ValueError(f"empty-load fit {range_name} must contain its nominal value")

        for field_name in (
            "velocity_gain_positive",
            "velocity_gain_negative",
            "packet_loss_probability",
            "velocity_noise_std_upper_bound_rad_s",
            "torque_noise_std_upper_bound_nm",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"empty-load fit {field_name} must be finite and non-negative")
        if not np.isfinite(self.communication_latency_s) or self.communication_latency_s < 0.0:
            raise ValueError("empty-load fit communication latency must be finite and non-negative")
        if not np.isfinite(self.torque_offset_nm):
            raise ValueError("empty-load fit torque offset must be finite")
        if not self.response_points:
            raise ValueError("empty-load fit requires at least one response point")
        for point in self.response_points:
            point.validate()
        if any(right.kd <= left.kd for left, right in zip(self.response_points, self.response_points[1:])):
            raise ValueError("empty-load response Kd points must increase strictly")

    def response_for_kd(self, kd: float) -> GF43ActuatorResponseFit:
        """Interpolate the measured response lookup, clamping outside its range."""

        kd_value = float(kd)
        if not np.isfinite(kd_value) or kd_value < 0.0:
            raise ValueError("Kd must be finite and non-negative")
        points = self.response_points
        if len(points) == 1:
            return points[0]
        kd_value = float(np.clip(kd_value, points[0].kd, points[-1].kd))
        kd_points = np.asarray([point.kd for point in points], dtype=np.float64)

        def interpolate(field_name: str) -> float:
            values = np.asarray([getattr(point, field_name) for point in points], dtype=np.float64)
            return float(np.interp(kd_value, kd_points, values))

        def interpolate_range(field_name: str) -> tuple[float, float]:
            lower = np.asarray([getattr(point, field_name)[0] for point in points], dtype=np.float64)
            upper = np.asarray([getattr(point, field_name)[1] for point in points], dtype=np.float64)
            return (
                float(np.interp(kd_value, kd_points, lower)),
                float(np.interp(kd_value, kd_points, upper)),
            )

        return GF43ActuatorResponseFit(
            kd=kd_value,
            time_constant_s=interpolate("time_constant_s"),
            command_gain=interpolate("command_gain"),
            time_constant_range_s=interpolate_range("time_constant_range_s"),
            command_gain_range=interpolate_range("command_gain_range"),
            delay_s=interpolate("delay_s"),
        )

    def friction_torque(
        self,
        qvel: np.ndarray | float,
        *,
        transition_speed_rad_s: float = 0.01,
    ) -> np.ndarray:
        """Return signed direction-dependent Coulomb plus viscous loss.

        A smooth ``tanh`` transition avoids a discontinuous impulse at zero
        speed.  The transition width is a simulator regularization parameter;
        it is not presented as an additional measured motor constant.
        """

        if not np.isfinite(transition_speed_rad_s) or transition_speed_rad_s <= 0.0:
            raise ValueError("friction transition speed must be finite and positive")
        speed = np.asarray(qvel, dtype=np.float64)
        if not np.isfinite(speed).all():
            raise ValueError("qvel must be finite")
        positive = speed >= 0.0
        coulomb = np.where(positive, self.coulomb_positive_nm, self.coulomb_negative_nm)
        viscous = np.where(
            positive,
            self.viscous_positive_nm_s_rad,
            self.viscous_negative_nm_s_rad,
        )
        return coulomb * np.tanh(speed / transition_speed_rad_s) + viscous * speed


@dataclass(frozen=True)
class GF43BenchPoint:
    """One raw output-shaft operating point from the manual's bench sweep.

    The table current is treated as bus current because its product with the
    table voltage reproduces input power.  It must not be used as phase current.
    """

    voltage_v: float
    bus_current_a: float
    input_power_w: float
    output_torque_nm: float
    output_speed_rpm: float
    output_power_w: float
    reported_efficiency_percent: float | None = None

    @property
    def calculated_efficiency_percent(self) -> float | None:
        """Return P_out/P_in, or ``None`` for an invalid zero-input row."""

        if self.input_power_w <= 0.0:
            return None
        return 100.0 * self.output_power_w / self.input_power_w


@dataclass(frozen=True)
class GF43MotorSpec:
    """Robot-independent facts available for one GF43 geared actuator.

    Fields ending in ``None`` are deliberate evidence gaps, not values to be
    guessed from another motor.  Rated and peak output limits are nameplate
    values; the duration for which the peak values are allowed is unknown.
    """

    model: str
    gear_ratio: float
    rated_voltage_v: float
    rated_bus_current_a: float
    rated_phase_current_a: float
    rated_power_w: float
    rated_output_speed_rpm: float
    rated_output_torque_nm: float
    peak_power_w: float
    peak_output_torque_nm: float
    listed_torque_constant_nm_per_a: float
    mass_kg: float
    outer_diameter_m: float
    length_m: float
    gearbox_backlash_arcmin: float
    communication: str
    bench_points: tuple[GF43BenchPoint, ...]
    source_pages: tuple[int, ...]

    integrated_driver: bool = True
    dual_encoders: bool = True
    temperature_protection: bool = True
    overload_protection: bool = True

    phase_resistance_ohm: float | None = None
    phase_inductance_h: float | None = None
    pole_pairs: int | None = None
    rotor_inertia_kg_m2: float | None = None
    gearbox_efficiency: float | None = None
    encoder_resolution_bits: int | None = None
    current_control_bandwidth_hz: float | None = None
    position_control_bandwidth_hz: float | None = None
    torque_response_time_s: float | None = None
    peak_torque_duration_s: float | None = None
    thermal_resistance_c_per_w: float | None = None
    thermal_capacity_j_per_c: float | None = None
    maximum_winding_temperature_c: float | None = None
    can_cycle_time_s: float | None = None
    empty_load_fit: GF43EmptyLoadFit | None = None

    @property
    def rated_output_speed_rad_s(self) -> float:
        return rpm_to_rad_s(self.rated_output_speed_rpm)

    @property
    def gearbox_backlash_rad(self) -> float:
        return float(np.deg2rad(self.gearbox_backlash_arcmin / 60.0))

    @property
    def measured_torque_range_nm(self) -> tuple[float, float]:
        torques = [point.output_torque_nm for point in self.bench_points]
        return min(torques), max(torques)

    @property
    def measured_speed_range_rpm(self) -> tuple[float, float]:
        speeds = [point.output_speed_rpm for point in self.bench_points]
        return min(speeds), max(speeds)

    def validate(self) -> None:
        positive = (
            "gear_ratio",
            "rated_voltage_v",
            "rated_bus_current_a",
            "rated_phase_current_a",
            "rated_power_w",
            "rated_output_speed_rpm",
            "rated_output_torque_nm",
            "peak_power_w",
            "peak_output_torque_nm",
            "listed_torque_constant_nm_per_a",
            "mass_kg",
            "outer_diameter_m",
            "length_m",
            "gearbox_backlash_arcmin",
        )
        for field_name in positive:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{self.model}.{field_name} must be finite and positive")
        if self.peak_power_w < self.rated_power_w:
            raise ValueError(f"{self.model} peak power must not be below rated power")
        if self.peak_output_torque_nm < self.rated_output_torque_nm:
            raise ValueError(f"{self.model} peak torque must not be below rated torque")
        if len(self.bench_points) < 2:
            raise ValueError(f"{self.model} requires at least two bench points")
        torques = [point.output_torque_nm for point in self.bench_points]
        if any(right <= left for left, right in zip(torques, torques[1:])):
            raise ValueError(f"{self.model} bench torque must increase strictly")


@dataclass(frozen=True)
class StaticOperatingPointAssessment:
    """Nameplate-envelope diagnostic for one output-shaft operating point."""

    duty: Duty
    torque_magnitude_nm: float
    speed_magnitude_rad_s: float
    mechanical_power_magnitude_w: float
    torque_limit_nm: float
    torque_utilization: float
    power_utilization: float
    within_envelope: bool


def rpm_to_rad_s(speed_rpm: float) -> float:
    """Convert revolutions per minute to radians per second."""

    return float(speed_rpm) * 2.0 * math.pi / 60.0


def rad_s_to_rpm(speed_rad_s: float) -> float:
    """Convert radians per second to revolutions per minute."""

    return float(speed_rad_s) * 60.0 / (2.0 * math.pi)


def mechanical_power_w(output_torque_nm: float, output_speed_rad_s: float) -> float:
    """Return signed output-shaft mechanical power tau*omega."""

    return float(output_torque_nm) * float(output_speed_rad_s)


def static_torque_magnitude_limit(
    output_speed_rad_s: np.ndarray | float,
    spec: GF43MotorSpec,
    *,
    duty: Duty = "rated",
) -> np.ndarray:
    """Return a first-order constant-torque/constant-power magnitude envelope.

    This is a deterministic framework approximation formed from two manual
    limits, not a measured four-quadrant controller model.  It says nothing
    about reverse/braking symmetry, transients, peak duration, or temperature.
    """

    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    if not np.isfinite(speed).all():
        raise ValueError("output speed must be finite")
    if duty == "rated":
        torque_limit = spec.rated_output_torque_nm
        power_limit = spec.rated_power_w
    elif duty == "peak":
        torque_limit = spec.peak_output_torque_nm
        power_limit = spec.peak_power_w
    else:
        raise ValueError("duty must be 'rated' or 'peak'")

    magnitude = np.abs(speed)
    power_limited = np.divide(
        power_limit,
        magnitude,
        out=np.full_like(magnitude, np.inf),
        where=magnitude > 0.0,
    )
    return np.minimum(torque_limit, power_limited)


# GF43X40-10 output-side motoring capability from the supplied manufacturer
# table, augmented only in the two explicitly labelled regions:
#
# * 0--17.86 N m: the constrained high-torque continuation recorded in
#   ``gf43x40_10_v106_tau15_effective.json``.  This is model extrapolation,
#   not a manufacturer measurement.
# * 55.0--55.4768659 rpm: the measured empty-load endpoint from the same
#   three-motor evidence release.  At and above that endpoint the model has no
#   evidence for additional same-direction accelerating torque.
#
# Values are stored output-side and in ascending absolute speed so callers can
# use a monotone interpolation without depending on a Downloads-path artifact.
GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S = 5.809523809523809
GF43X40_10_MOTORING_SPEED_RAD_S: tuple[float, ...] = (
    0.0,
    1.102076109287,
    1.335039450506,
    1.771432309589,
    2.170334154035,
    2.534009883643,
    2.864724398216,
    3.164742597554,
    3.204424506662,
    3.319616237293,
    3.424335992413,
    3.539527723044,
    3.623303527140,
    3.686135380212,
    3.801327110844,
    3.895574890451,
    3.906046865963,
    4.031710572107,
    4.105014400691,
    4.188790204786,
    4.251622057858,
    4.293509959906,
    4.345869837466,
    4.429645641562,
    4.565781323217,
    4.649557127313,
    4.701917004873,
    4.764748857945,
    4.796164784480,
    4.890412564088,
    4.974188368184,
    5.057964172280,
    5.183627878423,
    5.309291584567,
    5.445427266222,
    5.759586531581,
    GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S,
)
GF43X40_10_MOTORING_TORQUE_NM: tuple[float, ...] = (
    23.5,
    23.5,
    23.0,
    22.0,
    21.0,
    20.0,
    19.0,
    18.0,
    17.86,
    17.54,
    17.17,
    16.71,
    16.44,
    16.09,
    15.67,
    15.26,
    14.91,
    14.59,
    14.14,
    13.80,
    13.39,
    12.99,
    12.64,
    12.30,
    11.53,
    10.74,
    10.36,
    9.98,
    9.61,
    8.89,
    8.17,
    7.79,
    6.23,
    5.08,
    4.35,
    1.14,
    0.0,
)


def gf43x40_motoring_torque_limit(
    output_speed_rad_s: np.ndarray | float,
) -> np.ndarray:
    """Return the evidence-bounded same-direction X40 torque limit.

    The input is the output-shaft speed.  Direction is deliberately discarded
    here because the result is a torque magnitude; four-quadrant direction is
    handled by :func:`gf43x40_torque_bounds`.
    """

    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    if not np.isfinite(speed).all():
        raise ValueError("GF43X40-10 output speed must be finite")
    return np.interp(
        np.abs(speed),
        np.asarray(GF43X40_10_MOTORING_SPEED_RAD_S, dtype=np.float64),
        np.asarray(GF43X40_10_MOTORING_TORQUE_NM, dtype=np.float64),
        left=GF43X40_10_MOTORING_TORQUE_NM[0],
        right=0.0,
    )


def gf43x40_torque_bounds(
    output_speed_rad_s: np.ndarray | float,
    *,
    braking_torque_nm: float = 8.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directional X40 torque bounds for motoring and braking.

    Positive mechanical power is motoring and follows the evidence-bounded
    curve.  Opposite-direction electromagnetic torque is braking and uses the
    separately declared magnitude.  The braking value remains a simulation
    prior: thermal duration and regenerative-bus absorption are not proven by
    the available evidence.
    """

    if not np.isfinite(braking_torque_nm) or braking_torque_nm <= 0.0:
        raise ValueError("GF43X40-10 braking torque must be finite and positive")
    speed = np.asarray(output_speed_rad_s, dtype=np.float64)
    if not np.isfinite(speed).all():
        raise ValueError("GF43X40-10 output speed must be finite")
    motoring = gf43x40_motoring_torque_limit(speed)
    braking = np.full_like(speed, float(braking_torque_nm))
    peak = np.full_like(speed, GF43X40_10_MOTORING_TORQUE_NM[0])
    lower = np.where(speed > 0.0, -braking, np.where(speed < 0.0, -motoring, -peak))
    upper = np.where(speed > 0.0, motoring, np.where(speed < 0.0, braking, peak))
    return lower, upper


def assess_static_operating_point(
    output_torque_nm: float,
    output_speed_rad_s: float,
    spec: GF43MotorSpec,
    *,
    duty: Duty = "rated",
    relative_tolerance: float = 0.01,
) -> StaticOperatingPointAssessment:
    """Assess one point against the selected nameplate approximation.

    The default one-percent tolerance prevents independently rounded nameplate
    torque, speed, and power from contradicting one another at the rated point.
    """

    if not np.isfinite(output_torque_nm) or not np.isfinite(output_speed_rad_s):
        raise ValueError("output torque and speed must be finite")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    torque = abs(float(output_torque_nm))
    speed = abs(float(output_speed_rad_s))
    power = torque * speed
    limit = float(static_torque_magnitude_limit(speed, spec, duty=duty))
    torque_reference = (
        spec.rated_output_torque_nm if duty == "rated" else spec.peak_output_torque_nm
    )
    power_reference = spec.rated_power_w if duty == "rated" else spec.peak_power_w
    return StaticOperatingPointAssessment(
        duty=duty,
        torque_magnitude_nm=torque,
        speed_magnitude_rad_s=speed,
        mechanical_power_magnitude_w=power,
        torque_limit_nm=limit,
        torque_utilization=torque / torque_reference,
        power_utilization=power / power_reference,
        within_envelope=torque <= limit * (1.0 + relative_tolerance),
    )


def interpolate_bench_point_by_torque(
    output_torque_nm: float,
    spec: GF43MotorSpec,
) -> GF43BenchPoint:
    """Linearly interpolate the supplied positive-direction bench sweep.

    Extrapolation is rejected because the manual provides no evidence outside
    its tested torque range.  This function also makes no reverse-direction or
    transient claim.
    """

    torque = float(output_torque_nm)
    if not np.isfinite(torque):
        raise ValueError("output torque must be finite")
    points = spec.bench_points
    torques = [point.output_torque_nm for point in points]
    if torque < torques[0] or torque > torques[-1]:
        lower, upper = spec.measured_torque_range_nm
        raise ValueError(
            f"{spec.model} interpolation is limited to measured torque [{lower}, {upper}] N m"
        )
    index = bisect_left(torques, torque)
    if index < len(points) and torques[index] == torque:
        return points[index]
    left = points[index - 1]
    right = points[index]
    fraction = (torque - left.output_torque_nm) / (right.output_torque_nm - left.output_torque_nm)

    def lerp(left_value: float, right_value: float) -> float:
        return left_value + fraction * (right_value - left_value)

    if left.reported_efficiency_percent is None or right.reported_efficiency_percent is None:
        reported_efficiency = None
    else:
        reported_efficiency = lerp(
            left.reported_efficiency_percent,
            right.reported_efficiency_percent,
        )
    return GF43BenchPoint(
        voltage_v=lerp(left.voltage_v, right.voltage_v),
        bus_current_a=lerp(left.bus_current_a, right.bus_current_a),
        input_power_w=lerp(left.input_power_w, right.input_power_w),
        output_torque_nm=torque,
        output_speed_rpm=lerp(left.output_speed_rpm, right.output_speed_rpm),
        output_power_w=lerp(left.output_power_w, right.output_power_w),
        reported_efficiency_percent=reported_efficiency,
    )


def _p(
    voltage_v: float,
    bus_current_a: float,
    input_power_w: float,
    torque_nm: float,
    speed_rpm: float,
    output_power_w: float,
    efficiency_percent: float | None = None,
) -> GF43BenchPoint:
    return GF43BenchPoint(
        voltage_v,
        bus_current_a,
        input_power_w,
        torque_nm,
        speed_rpm,
        output_power_w,
        efficiency_percent,
    )


GF43X40_10_EMPTY_LOAD_FIT = GF43EmptyLoadFit(
    # SHA256 of /Users/rangemini-003/Downloads/j4340_empty_fit.json at the
    # time this canonical subset was imported.  The runtime never reads the
    # Downloads path; this digest is provenance only.
    source_sha256="f2d042627a2b16143e2e5bd1564f7a5b69430009f356c84d491628d7f07e89eb",
    measured_motor_ids=(13, 14, 15),
    effective_inertia_kg_m2=0.028035586440512583,
    effective_inertia_range_kg_m2=(0.025900822597569456, 0.03504228939671161),
    coulomb_positive_nm=0.2612562552472579,
    coulomb_positive_range_nm=(0.23081666254250055, 0.31228136696926545),
    coulomb_negative_nm=0.3180790053096667,
    coulomb_negative_range_nm=(0.27605236045887344, 0.3734826053267111),
    viscous_positive_nm_s_rad=0.02830071806357562,
    viscous_positive_range_nm_s_rad=(0.02519996781096702, 0.034094074097190674),
    viscous_negative_nm_s_rad=0.030750343797478024,
    viscous_negative_range_nm_s_rad=(0.021024700215881553, 0.04000002708575795),
    torque_offset_nm=0.024372667008714693,
    velocity_gain_positive=0.9820166436395464,
    velocity_gain_negative=0.9813448879419004,
    # Reference only.  These time constants and command gains were inferred
    # together from empty-load CAN feedback and may include drive
    # estimation/filtering and fixture motion;
    # V1.0.6 loaded-joint training uses a separately labelled hypothesis range.
    response_points=(
        GF43ActuatorResponseFit(
            kd=0.2,
            time_constant_s=0.09339178790010678,
            command_gain=0.5241761982988015,
            time_constant_range_s=(0.0793209684435288, 0.10731660436477426),
            command_gain_range=(0.45573436000488804, 0.6165817811830838),
        ),
        GF43ActuatorResponseFit(
            kd=4.0,
            time_constant_s=0.0583960985,
            command_gain=0.4569687277666667,
            time_constant_range_s=(0.051641474, 0.0713419932),
            command_gain_range=(0.4219404714, 0.5204052313),
        ),
    ),
    communication_latency_s=0.000677722,
    communication_latency_range_s=(0.000621458, 0.000726083),
    packet_loss_probability=0.0008878734520259245,
    packet_loss_probability_range=(0.0007687281392935388, 0.0009573922167842344),
    velocity_bias_rad_s=-0.0024420024420024,
    velocity_noise_std_upper_bound_rad_s=0.002442002442002442,
    torque_feedback_bias_nm=-0.0068376068376068,
    torque_noise_std_upper_bound_nm=0.006837606837606838,
)
GF43X40_10_EMPTY_LOAD_FIT.validate()


GF43X40_10 = GF43MotorSpec(
    model="GF43X40-10",
    gear_ratio=40.0,
    rated_voltage_v=24.0,
    rated_bus_current_a=2.3,
    rated_phase_current_a=2.0,
    rated_power_w=40.0,
    rated_output_speed_rpm=43.0,
    rated_output_torque_nm=8.9,
    peak_power_w=70.0,
    peak_output_torque_nm=23.5,
    # The manual does not define the torque constant's reference shaft.  The
    # same value is listed for 10:1 and 40:1 variants, so it is retained as a
    # raw field and is intentionally not used as an output torque constant.
    listed_torque_constant_nm_per_a=0.095,
    mass_kg=0.362,
    outer_diameter_m=0.057,
    length_m=0.0565,
    gearbox_backlash_arcmin=15.0,
    communication="CAN",
    source_pages=(7, 8, 9, 10),
    empty_load_fit=GF43X40_10_EMPTY_LOAD_FIT,
    bench_points=(
        _p(24.019, 0.0000, 0.000, 1.14, 55.0, 6.5501),
        _p(24.020, 1.0833, 25.576, 4.35, 52.0, 23.682),
        _p(23.834, 1.2612, 29.517, 5.08, 50.7, 26.958),
        _p(23.958, 1.5342, 36.155, 6.23, 49.5, 32.243),
        _p(23.916, 1.7728, 41.923, 7.36, 48.3, 37.217),
        _p(23.985, 1.9510, 46.466, 7.79, 48.3, 39.430),
        _p(24.034, 2.0618, 48.908, 8.17, 47.5, 40.592),
        _p(23.959, 2.2320, 53.154, 8.89, 46.7, 43.464),
        _p(23.865, 2.4408, 57.695, 9.61, 45.8, 46.134),
        _p(23.936, 2.5650, 60.408, 9.98, 45.5, 47.544),
        _p(23.963, 2.6488, 62.932, 10.36, 44.9, 48.751),
        _p(23.960, 2.7128, 64.609, 10.74, 44.4, 49.980),
        _p(23.993, 2.9873, 71.076, 11.53, 43.6, 52.617),
        _p(23.994, 3.1195, 74.205, 12.30, 42.3, 54.519),
        _p(23.949, 3.2104, 76.173, 12.64, 41.5, 55.000),
        _p(23.873, 3.2809, 77.740, 12.99, 41.0, 55.835),
        _p(23.962, 3.4649, 82.309, 13.39, 40.6, 56.873),
        _p(24.014, 3.5122, 83.733, 13.80, 40.0, 57.713),
        _p(23.998, 3.6295, 86.568, 14.14, 39.2, 58.022),
        _p(24.064, 3.7226, 89.044, 14.59, 38.5, 58.769),
        _p(23.930, 3.8304, 90.962, 14.91, 37.3, 58.226),
        _p(24.077, 3.9608, 94.710, 15.26, 37.2, 59.405),
        _p(24.081, 4.0589, 97.081, 15.67, 36.3, 59.477),
        _p(24.002, 4.1486, 98.960, 16.09, 35.2, 59.245),
        _p(24.000, 4.3283, 103.24, 16.44, 34.6, 59.538),
        _p(24.008, 4.3664, 104.25, 16.71, 33.8, 59.123),
        _p(24.060, 4.4767, 107.14, 17.17, 32.7, 58.830),
        _p(24.085, 4.6217, 110.69, 17.54, 31.7, 58.239),
        _p(24.025, 4.7002, 112.40, 17.86, 30.6, 57.166),
    ),
)


GF43X10_10 = GF43MotorSpec(
    model="GF43X10-10",
    gear_ratio=10.0,
    rated_voltage_v=24.0,
    rated_bus_current_a=2.3,
    rated_phase_current_a=2.0,
    rated_power_w=40.0,
    rated_output_speed_rpm=172.0,
    rated_output_torque_nm=2.2,
    peak_power_w=70.0,
    peak_output_torque_nm=7.0,
    listed_torque_constant_nm_per_a=0.095,
    mass_kg=0.300,
    outer_diameter_m=0.057,
    length_m=0.0459,
    gearbox_backlash_arcmin=7.5,
    communication="CAN",
    source_pages=(10, 11, 12, 13),
    bench_points=(
        _p(23.911, 0.3690, 8.8131, 0.160, 217.1, 3.6414, 41.3),
        _p(23.908, 0.5135, 12.247, 0.306, 214.5, 6.8809, 56.2),
        _p(23.907, 0.6447, 15.407, 0.436, 212.6, 9.7099, 63.0),
        _p(23.905, 0.7685, 18.364, 0.563, 210.3, 12.398, 67.5),
        _p(23.904, 0.8885, 21.177, 0.685, 208.7, 14.971, 70.7),
        _p(23.937, 1.0119, 24.252, 0.801, 207.3, 17.393, 71.7),
        _p(24.004, 1.1321, 27.182, 0.928, 205.3, 19.942, 73.4),
        _p(23.996, 1.2528, 30.064, 1.048, 203.4, 22.325, 74.3),
        _p(23.994, 1.3754, 33.033, 1.174, 201.4, 24.750, 74.9),
        _p(23.997, 1.4907, 35.761, 1.292, 199.4, 26.965, 75.4),
        _p(24.000, 1.6086, 38.632, 1.415, 197.3, 29.224, 75.6),
        _p(23.995, 1.7277, 41.467, 1.535, 195.2, 31.371, 75.7),
        _p(23.992, 1.8497, 44.330, 1.651, 193.1, 33.395, 75.3),
        _p(23.993, 1.9632, 47.176, 1.776, 191.1, 35.536, 75.3),
        _p(23.994, 2.0855, 50.013, 1.887, 188.8, 37.286, 74.6),
        _p(23.987, 2.2036, 52.903, 2.016, 186.7, 39.410, 74.5),
        _p(23.985, 2.3215, 55.635, 2.134, 184.6, 41.258, 74.2),
        _p(23.983, 2.4424, 58.519, 2.248, 182.4, 42.939, 73.4),
        _p(23.982, 2.5564, 61.315, 2.368, 180.1, 44.668, 72.8),
        _p(23.981, 2.6746, 64.093, 2.500, 177.9, 46.553, 72.6),
        _p(23.981, 2.7940, 66.918, 2.617, 175.4, 48.054, 71.8),
        _p(23.970, 2.9193, 69.936, 2.730, 172.9, 49.418, 70.7),
        _p(23.960, 3.0290, 72.522, 2.850, 170.6, 50.915, 70.2),
        _p(23.963, 3.1488, 75.404, 2.966, 168.1, 52.213, 69.2),
        _p(23.958, 3.2666, 78.132, 3.091, 165.6, 53.602, 68.6),
        _p(23.952, 3.3808, 80.983, 3.200, 163.3, 54.710, 67.6),
        _p(23.949, 3.4959, 83.701, 3.319, 160.7, 55.849, 66.7),
        _p(23.947, 3.6162, 86.614, 3.436, 158.1, 56.861, 65.6),
        _p(23.945, 3.7267, 89.319, 3.556, 155.3, 57.808, 64.7),
        _p(23.944, 3.8480, 92.057, 3.668, 152.7, 58.629, 63.7),
        _p(23.942, 3.9642, 94.832, 3.794, 149.7, 59.472, 62.7),
        _p(23.941, 4.0813, 97.667, 3.911, 147.0, 60.190, 61.6),
        _p(23.941, 4.1995, 100.48, 4.017, 144.3, 60.682, 60.4),
        _p(23.943, 4.3170, 103.30, 4.136, 141.2, 61.128, 59.2),
        _p(23.939, 4.4339, 106.05, 4.243, 138.1, 61.360, 57.9),
        _p(23.940, 4.5527, 108.92, 4.374, 134.8, 61.722, 56.7),
        _p(23.941, 4.6695, 111.72, 4.481, 131.3, 61.590, 55.1),
        _p(23.937, 4.7903, 114.56, 4.593, 128.4, 61.763, 53.9),
        _p(23.936, 4.9093, 117.44, 4.695, 124.7, 61.286, 52.2),
        _p(23.937, 5.0318, 120.38, 4.814, 120.8, 60.877, 50.6),
        _p(23.934, 5.1546, 123.27, 4.923, 117.2, 60.434, 49.0),
        _p(23.932, 5.2782, 126.20, 5.041, 113.6, 59.975, 47.5),
        _p(23.930, 5.3982, 129.01, 5.149, 109.2, 58.852, 45.6),
        _p(23.928, 5.5253, 132.11, 5.266, 105.2, 58.004, 43.9),
        _p(23.931, 5.6548, 135.17, 5.373, 101.1, 56.868, 42.1),
    ),
)

GF43_MOTOR_SPECS = {
    GF43X10_10.model: GF43X10_10,
    GF43X40_10.model: GF43X40_10,
}
