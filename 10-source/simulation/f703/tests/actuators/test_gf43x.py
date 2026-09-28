"""Contracts for the manual-bounded GF43 actuator model."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.actuators.gf43x import (
    GF43_MOTOR_SPECS,
    GF43X10_10,
    GF43X40_10,
    GF43X40_10_EMPTY_LOAD_FIT,
    GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S,
    assess_static_operating_point,
    gf43x40_motoring_torque_limit,
    gf43x40_torque_bounds,
    interpolate_bench_point_by_torque,
    mechanical_power_w,
    rad_s_to_rpm,
    rpm_to_rad_s,
    static_torque_magnitude_limit,
)


@pytest.mark.parametrize(
    ("spec", "ratio", "speed", "rated_torque", "peak_torque", "mass", "backlash"),
    [
        (GF43X10_10, 10.0, 172.0, 2.2, 7.0, 0.300, 7.5),
        (GF43X40_10, 40.0, 43.0, 8.9, 23.5, 0.362, 15.0),
    ],
)
def test_nameplate_contract(
    spec,
    ratio: float,
    speed: float,
    rated_torque: float,
    peak_torque: float,
    mass: float,
    backlash: float,
) -> None:
    spec.validate()
    assert GF43_MOTOR_SPECS[spec.model] is spec
    assert spec.gear_ratio == ratio
    assert spec.rated_voltage_v == 24.0
    assert spec.rated_bus_current_a == 2.3
    assert spec.rated_phase_current_a == 2.0
    assert spec.rated_power_w == 40.0
    assert spec.peak_power_w == 70.0
    assert spec.rated_output_speed_rpm == speed
    assert spec.rated_output_torque_nm == rated_torque
    assert spec.peak_output_torque_nm == peak_torque
    assert spec.listed_torque_constant_nm_per_a == 0.095
    assert spec.mass_kg == mass
    assert spec.outer_diameter_m == 0.057
    assert spec.gearbox_backlash_arcmin == backlash
    assert spec.communication == "CAN"


def test_output_units_and_rated_mechanical_power_are_consistent() -> None:
    for spec in (GF43X10_10, GF43X40_10):
        assert rad_s_to_rpm(spec.rated_output_speed_rad_s) == pytest.approx(
            spec.rated_output_speed_rpm
        )
        rated_mechanical_power = mechanical_power_w(
            spec.rated_output_torque_nm,
            spec.rated_output_speed_rad_s,
        )
        assert rated_mechanical_power == pytest.approx(spec.rated_power_w, rel=0.011)

    assert GF43X10_10.gearbox_backlash_rad == pytest.approx(np.deg2rad(0.125))
    assert GF43X40_10.gearbox_backlash_rad == pytest.approx(np.deg2rad(0.25))


def test_static_envelope_uses_torque_then_power_limit() -> None:
    zero_and_rated_speed = np.asarray([0.0, GF43X40_10.rated_output_speed_rad_s])
    rated = static_torque_magnitude_limit(zero_and_rated_speed, GF43X40_10, duty="rated")
    assert rated[0] == pytest.approx(8.9)
    assert rated[1] == pytest.approx(40.0 / rpm_to_rad_s(43.0))

    peak_transition = 70.0 / 23.5
    peak = static_torque_magnitude_limit(
        np.asarray([0.0, peak_transition, 2.0 * peak_transition]),
        GF43X40_10,
        duty="peak",
    )
    np.testing.assert_allclose(peak, [23.5, 23.5, 11.75])


def test_x40_hybrid_motoring_envelope_uses_evidence_endpoint() -> None:
    speeds = np.asarray([rpm_to_rad_s(value) for value in (0.0, 30.6, 46.7, 55.0)])
    limits = gf43x40_motoring_torque_limit(speeds)
    np.testing.assert_allclose(limits, [23.5, 17.86, 8.89, 1.14], atol=1.0e-9)

    endpoint = GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S
    assert gf43x40_motoring_torque_limit(endpoint) == pytest.approx(0.0)
    assert gf43x40_motoring_torque_limit(rpm_to_rad_s(100.0)) == pytest.approx(0.0)


def test_x40_hybrid_bounds_separate_motoring_braking_and_zero_speed() -> None:
    speed = np.asarray([rpm_to_rad_s(100.0), -rpm_to_rad_s(100.0), 0.0])
    lower, upper = gf43x40_torque_bounds(speed)
    np.testing.assert_allclose(lower, [-8.9, 0.0, -23.5])
    np.testing.assert_allclose(upper, [0.0, 8.9, 23.5])


def test_operating_point_assessment_reports_selected_duty() -> None:
    rated = assess_static_operating_point(8.9, rpm_to_rad_s(43.0), GF43X40_10)
    assert rated.torque_utilization == pytest.approx(1.0)
    assert rated.power_utilization == pytest.approx(1.0019, abs=0.002)
    assert rated.within_envelope

    strict_rated = assess_static_operating_point(
        8.9,
        rpm_to_rad_s(43.0),
        GF43X40_10,
        relative_tolerance=0.0,
    )
    assert not strict_rated.within_envelope

    peak = assess_static_operating_point(8.9, rpm_to_rad_s(43.0), GF43X40_10, duty="peak")
    assert peak.within_envelope
    assert peak.power_utilization < 1.0


def test_bench_tables_preserve_manual_boundaries_and_efficiency_semantics() -> None:
    assert len(GF43X40_10.bench_points) == 29
    assert GF43X40_10.measured_torque_range_nm == (1.14, 17.86)
    assert GF43X40_10.measured_speed_range_rpm == (30.6, 55.0)
    assert GF43X40_10.bench_points[0].calculated_efficiency_percent is None
    assert GF43X40_10.bench_points[1].calculated_efficiency_percent == pytest.approx(92.6, abs=0.1)

    assert len(GF43X10_10.bench_points) == 45
    assert GF43X10_10.measured_torque_range_nm == (0.160, 5.373)
    assert GF43X10_10.measured_speed_range_rpm == (101.1, 217.1)
    assert GF43X10_10.bench_points[0].reported_efficiency_percent == 41.3
    assert GF43X10_10.bench_points[-1].reported_efficiency_percent == 42.1


def test_bench_rows_are_internally_consistent_with_printed_precision() -> None:
    for spec in (GF43X10_10, GF43X40_10):
        for point in spec.bench_points:
            assert point.input_power_w == pytest.approx(
                point.voltage_v * point.bus_current_a,
                abs=1.0,
            )
            assert point.output_power_w == pytest.approx(
                mechanical_power_w(point.output_torque_nm, rpm_to_rad_s(point.output_speed_rpm)),
                abs=0.1,
            )
            if point.reported_efficiency_percent is not None:
                assert point.calculated_efficiency_percent == pytest.approx(
                    point.reported_efficiency_percent,
                    abs=0.1,
                )


def test_bench_interpolation_is_exact_inside_domain_and_rejects_extrapolation() -> None:
    exact = interpolate_bench_point_by_torque(8.89, GF43X40_10)
    assert exact is GF43X40_10.bench_points[7]

    interpolated = interpolate_bench_point_by_torque((8.17 + 8.89) / 2.0, GF43X40_10)
    assert interpolated.output_speed_rpm == pytest.approx((47.5 + 46.7) / 2.0)
    assert interpolated.bus_current_a == pytest.approx((2.0618 + 2.2320) / 2.0)
    assert interpolated.reported_efficiency_percent is None

    with pytest.raises(ValueError, match="limited to measured torque"):
        interpolate_bench_point_by_torque(0.0, GF43X40_10)
    with pytest.raises(ValueError, match="limited to measured torque"):
        interpolate_bench_point_by_torque(7.0, GF43X10_10)


def test_unsupported_dynamic_parameters_remain_explicitly_unknown() -> None:
    unknown_fields = (
        "phase_resistance_ohm",
        "phase_inductance_h",
        "pole_pairs",
        "rotor_inertia_kg_m2",
        "gearbox_efficiency",
        "encoder_resolution_bits",
        "current_control_bandwidth_hz",
        "position_control_bandwidth_hz",
        "torque_response_time_s",
        "peak_torque_duration_s",
        "thermal_resistance_c_per_w",
        "thermal_capacity_j_per_c",
        "maximum_winding_temperature_c",
        "can_cycle_time_s",
    )
    for spec in (GF43X10_10, GF43X40_10):
        assert all(getattr(spec, field_name) is None for field_name in unknown_fields)


def test_empty_load_fit_is_a_hashed_three_motor_nominal_and_interpolates_kd() -> None:
    fit = GF43X40_10_EMPTY_LOAD_FIT
    assert GF43X40_10.empty_load_fit is fit
    assert fit.measured_motor_ids == (13, 14, 15)
    assert fit.source_sha256 == "f2d042627a2b16143e2e5bd1564f7a5b69430009f356c84d491628d7f07e89eb"
    assert fit.effective_inertia_kg_m2 == pytest.approx(0.028035586440512583)
    assert fit.symmetric_coulomb_nm == pytest.approx(0.2896676302784623)
    assert fit.symmetric_viscous_nm_s_rad == pytest.approx(0.029525530930526823)

    response = fit.response_for_kd(0.5)
    assert response.time_constant_s == pytest.approx(0.09062897031588783)
    assert response.command_gain == pytest.approx(0.518870345362054)
    assert fit.response_for_kd(0.0).kd == pytest.approx(0.2)
    assert fit.response_for_kd(5.0).kd == pytest.approx(4.0)


def test_empty_load_fit_preserves_directional_friction_sign() -> None:
    fit = GF43X40_10_EMPTY_LOAD_FIT
    speeds = np.asarray([-1.0, 0.0, 1.0])
    friction = fit.friction_torque(speeds)
    assert friction[0] < 0.0
    assert friction[1] == pytest.approx(0.0)
    assert friction[2] > 0.0
    assert abs(friction[0]) > abs(friction[2])
