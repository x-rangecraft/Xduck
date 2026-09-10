#!/usr/bin/env python3
"""Build the deployable empty-load J4340 parameter file from fitted/raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MOTORS = (13, 14, 15)
MEASURED_NO_LOAD_PEAK_SPEED_RAD_S = 5.809523809523809

# Exact GF43X40-10 table transcribed from pages 9-10 of 4340电机参数.pdf.
# Columns: voltage V, bus current A, input power W, torque N*m, speed rpm,
# output power W. Efficiency is derived from the two power columns.
GF43X40_10_PERFORMANCE = (
    (24.019, 0.0000, 0.000, 1.14, 55.0, 6.5501),
    (24.020, 1.0833, 25.576, 4.35, 52.0, 23.682),
    (23.834, 1.2612, 29.517, 5.08, 50.7, 26.958),
    (23.958, 1.5342, 36.155, 6.23, 49.5, 32.243),
    (23.916, 1.7728, 41.923, 7.36, 48.3, 37.217),
    (23.985, 1.9510, 46.466, 7.79, 48.3, 39.430),
    (24.034, 2.0618, 48.908, 8.17, 47.5, 40.592),
    (23.959, 2.2320, 53.154, 8.89, 46.7, 43.464),
    (23.865, 2.4408, 57.695, 9.61, 45.8, 46.134),
    (23.936, 2.5650, 60.408, 9.98, 45.5, 47.544),
    (23.963, 2.6488, 62.932, 10.36, 44.9, 48.751),
    (23.960, 2.7128, 64.609, 10.74, 44.4, 49.980),
    (23.993, 2.9873, 71.076, 11.53, 43.6, 52.617),
    (23.994, 3.1195, 74.205, 12.30, 42.3, 54.519),
    (23.949, 3.2104, 76.173, 12.64, 41.5, 55.000),
    (23.873, 3.2809, 77.740, 12.99, 41.0, 55.835),
    (23.962, 3.4649, 82.309, 13.39, 40.6, 56.873),
    (24.014, 3.5122, 83.733, 13.80, 40.0, 57.713),
    (23.998, 3.6295, 86.568, 14.14, 39.2, 58.022),
    (24.064, 3.7226, 89.044, 14.59, 38.5, 58.769),
    (23.930, 3.8304, 90.962, 14.91, 37.3, 58.226),
    (24.077, 3.9608, 94.710, 15.26, 37.2, 59.405),
    (24.081, 4.0589, 97.081, 15.67, 36.3, 59.477),
    (24.002, 4.1486, 98.960, 16.09, 35.2, 59.245),
    (24.000, 4.3283, 103.24, 16.44, 34.6, 59.538),
    (24.008, 4.3664, 104.25, 16.71, 33.8, 59.123),
    (24.060, 4.4767, 107.14, 17.17, 32.7, 58.830),
    (24.085, 4.6217, 110.69, 17.54, 31.7, 58.239),
    (24.025, 4.7002, 112.40, 17.86, 30.6, 57.166),
)


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def data_range(values: list[float], floor_fraction: float = 0.15, nonnegative: bool = True) -> list[float]:
    values_array = np.asarray(values, dtype=float)
    center = float(np.median(values_array))
    motor_sigma = float(1.4826 * np.median(np.abs(values_array - center)))
    delta = max(2.0 * motor_sigma, floor_fraction * abs(center))
    low = center - delta
    if nonnegative:
        low = max(0.0, low)
    return [low, center + delta]


def fit_peak_region_extrapolation() -> dict[str, object]:
    """Extrapolate the uncovered 17.86-23.5 N*m region from the exact table.

    A cubic gives the best full-table fit and tail holdout prediction among
    first-, second- and third-order polynomials.  Both the cubic nominal and
    quadratic comparison curve are shifted to pass through the final measured
    table point, avoiding a discontinuity at 17.86 N*m.
    """
    torque = np.asarray([row[3] for row in GF43X40_10_PERFORMANCE], dtype=float)
    speed_rpm = np.asarray([row[4] for row in GF43X40_10_PERFORMANCE], dtype=float)
    cubic = np.polyfit(torque, speed_rpm, 3)
    quadratic = np.polyfit(torque, speed_rpm, 2)
    cubic_fit = np.polyval(cubic, torque)
    rmse = float(np.sqrt(np.mean((cubic_fit - speed_rpm) ** 2)))
    r_squared = float(
        1.0
        - np.sum((cubic_fit - speed_rpm) ** 2)
        / np.sum((speed_rpm - np.mean(speed_rpm)) ** 2)
    )
    cubic_offset = float(speed_rpm[-1] - np.polyval(cubic, torque[-1]))
    quadratic_offset = float(speed_rpm[-1] - np.polyval(quadratic, torque[-1]))
    extension_torque = np.asarray(
        [17.86, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.5], dtype=float
    )
    nominal_rpm = np.polyval(cubic, extension_torque) + cubic_offset
    comparison_rpm = np.polyval(quadratic, extension_torque) + quadratic_offset
    nominal_rad_s = nominal_rpm * 2.0 * np.pi / 60.0
    if np.any(nominal_rpm < 0.0) or np.any(np.diff(nominal_rpm) > 0.0):
        raise RuntimeError("peak-region speed extrapolation must be nonnegative and monotonic")
    if np.any(extension_torque * nominal_rad_s > 70.0):
        raise RuntimeError("peak-region extrapolation exceeds the 70 W peak-power constraint")
    points = []
    for tau, rpm, alternate_rpm in zip(
        extension_torque, nominal_rpm, comparison_rpm, strict=True
    ):
        rad_s = float(rpm * 2.0 * np.pi / 60.0)
        points.append(
            {
                "torque_Nm": float(tau),
                "speed_rpm_nominal": float(rpm),
                "speed_rad_s_nominal": rad_s,
                "mechanical_power_W_nominal": float(tau * rad_s),
                "speed_rpm_model_form_range": [
                    float(min(rpm, alternate_rpm)),
                    float(max(rpm, alternate_rpm)),
                ],
            }
        )
    return {
        "status": "model extrapolation, not measured data",
        "method": "endpoint-continuous cubic polynomial; quadratic-vs-cubic spread used as model-form uncertainty",
        "source_torque_range_Nm": [float(torque[0]), float(torque[-1])],
        "extrapolated_torque_range_Nm": [float(torque[-1]), 23.5],
        "cubic_coefficients_rpm_descending_power": [float(value) for value in cubic],
        "endpoint_offset_rpm": cubic_offset,
        "full_table_fit_rmse_rpm": rmse,
        "full_table_fit_r_squared": r_squared,
        "constraints": {
            "physical_peak_torque_Nm": 23.5,
            "peak_power_W": 70.0,
            "monotonic_nonincreasing_speed": True,
        },
        "points": points,
    }


def build_torque_speed_envelope(
    peak_region_extrapolation: dict[str, object],
) -> dict[str, object]:
    """Build the runtime inverse lookup: speed -> accelerating torque limit."""
    pairs = [(0.0, 23.5), (MEASURED_NO_LOAD_PEAK_SPEED_RAD_S, 0.0)]
    pairs.extend(
        (float(point["speed_rad_s_nominal"]), float(point["torque_Nm"]))
        for point in peak_region_extrapolation["points"]
    )
    pairs.extend(
        (float(row[4] * 2.0 * np.pi / 60.0), float(row[3]))
        for row in GF43X40_10_PERFORMANCE
    )

    # Equal rounded speeds occur in the manufacturer table.  Keep the larger
    # supported torque so the inverse lookup remains single-valued.
    by_speed: dict[float, float] = {}
    for speed, torque in pairs:
        key = round(speed, 12)
        by_speed[key] = max(torque, by_speed.get(key, 0.0))
    ordered = sorted((speed, torque) for speed, torque in by_speed.items())
    speeds = np.asarray([point[0] for point in ordered], dtype=float)
    torques = np.asarray([point[1] for point in ordered], dtype=float)
    if np.any(np.diff(speeds) <= 0.0) or np.any(np.diff(torques) > 1e-9):
        raise RuntimeError("torque-speed envelope must have increasing speed and decreasing torque")

    return {
        "interpretation": "maximum same-direction accelerating torque at the current absolute output speed",
        "interpolation": "piecewise linear; clamp below first point; the final point and all higher speeds return zero",
        "speed_rad_s_ascending": [float(value) for value in speeds],
        "max_accelerating_torque_Nm": [float(value) for value in torques],
        "maximum_speed_rad_s": float(speeds[-1]),
        "physical_peak_torque_Nm": 23.5,
        "braking_rule": "do not apply the speed envelope to opposite-direction braking torque; still apply physical, thermal, overload and structural limits",
        "source": "GF43X40-10 manufacturer table plus constrained high-torque extrapolation and measured no-load peak-speed endpoint",
    }


def load_high_pulse_table(raw_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    per_motor: dict[str, object] = {}
    aggregate_rows: list[dict[str, float | str]] = []
    for motor in MOTORS:
        paths = sorted(raw_dir.glob(f"torque_pulse_kd4_to_tmax_m{motor}_r1_*.csv"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one TMAX pulse file for motor {motor}, found {len(paths)}")
        data = pd.read_csv(paths[0], low_memory=False)
        data = data[(data.reply_received == 1) & (data.tau_ff != 0)].copy()
        rows = []
        for command, group in data.groupby("tau_ff"):
            command = float(command)
            peak_torque = float(group.tau_feedback.max() if command > 0 else group.tau_feedback.min())
            peak_speed = float(group.qd_feedback.max() if command > 0 else group.qd_feedback.min())
            row = {
                "command_Nm": command,
                "feedback_peak_Nm": peak_torque,
                "feedback_median_Nm": float(group.tau_feedback.median()),
                "speed_peak_rad_s": peak_speed,
                "peak_feedback_to_command_ratio": peak_torque / command,
                "samples": int(len(group)),
            }
            rows.append(row)
            aggregate_rows.append({"motor": str(motor), **row})
        per_motor[str(motor)] = {"source": paths[0].name, "points": rows}

    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate = []
    for command, group in aggregate_frame.groupby("command_Nm"):
        aggregate.append(
            {
                "command_Nm": float(command),
                "feedback_peak_median_Nm": float(group.feedback_peak_Nm.median()),
                "feedback_peak_min_Nm": float(group.feedback_peak_Nm.min()),
                "feedback_peak_max_Nm": float(group.feedback_peak_Nm.max()),
                "speed_peak_median_rad_s": float(group.speed_peak_rad_s.median()),
                "peak_feedback_to_command_ratio_median": float(
                    group.peak_feedback_to_command_ratio.median()
                ),
            }
        )
    return per_motor, {"points": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default="data/processed/empty_load_fit_rounds_1_2.json")
    parser.add_argument("--kd4-fit", default="data/processed/torque_pulse_kd4_fit.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/j4340_empty_fit.json")
    args = parser.parse_args()

    fit = json.loads(Path(args.fit).read_text(encoding="utf-8"))
    kd4 = json.loads(Path(args.kd4_fit).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir)
    high_per_motor, high_aggregate = load_high_pulse_table(raw_dir)
    peak_region_extrapolation = fit_peak_region_extrapolation()
    torque_speed_envelope = build_torque_speed_envelope(peak_region_extrapolation)
    datasheet_performance = [
        {
            "voltage_V": voltage,
            "bus_current_A": current,
            "input_power_W": input_power,
            "torque_Nm": torque,
            "speed_rpm": speed_rpm,
            "speed_rad_s": speed_rpm * 2.0 * np.pi / 60.0,
            "output_power_W": output_power,
            "efficiency_percent": (
                None if input_power == 0.0 else 100.0 * output_power / input_power
            ),
        }
        for voltage, current, input_power, torque, speed_rpm, output_power
        in GF43X40_10_PERFORMANCE
    ]

    mechanical_keys = (
        "J_eff_kg_m2",
        "coulomb_positive_Nm",
        "coulomb_negative_Nm",
        "viscous_positive_Nm_s_rad",
        "viscous_negative_Nm_s_rad",
        "torque_offset_Nm",
    )
    individual: dict[str, object] = {}
    nominal: dict[str, float] = {}
    randomization: dict[str, list[float]] = {}
    for key in mechanical_keys:
        values = [float(fit["motors"][str(motor)]["mechanical"][key]) for motor in MOTORS]
        nominal[key] = median(values)
        randomization[key] = data_range(values, nonnegative=key != "torque_offset_Nm")

    speed_positive = [
        float(fit["motors"][str(motor)]["velocity_tracking"]["positive"]["gain"])
        for motor in MOTORS
    ]
    speed_negative = [
        float(fit["motors"][str(motor)]["velocity_tracking"]["negative"]["gain"])
        for motor in MOTORS
    ]
    nominal["velocity_gain_positive"] = median(speed_positive)
    nominal["velocity_gain_negative"] = median(speed_negative)
    randomization["velocity_gain_positive"] = data_range(speed_positive)
    randomization["velocity_gain_negative"] = data_range(speed_negative)

    tau_low = [
        float(fit["motors"][str(motor)]["actuator_dynamics"]["time_constant_s_median"])
        for motor in MOTORS
    ]
    gain_low = [
        float(fit["motors"][str(motor)]["actuator_dynamics"]["command_gain_median"])
        for motor in MOTORS
    ]
    tau_kd4 = [float(kd4["motors"][str(motor)]["time_constant_s"]) for motor in MOTORS]
    gain_kd4 = [float(kd4["motors"][str(motor)]["command_gain"]) for motor in MOTORS]

    latency_medians = [
        float(fit["motors"][str(motor)]["communication"]["latency_median_ms"]) / 1000.0
        for motor in MOTORS
    ]
    loss_rates = [
        float(fit["motors"][str(motor)]["communication"]["loss_rate"])
        for motor in MOTORS
    ]
    position_biases = [
        float(fit["motors"][str(motor)]["disabled"]["q_feedback"]["median"])
        for motor in MOTORS
    ]
    velocity_biases = [
        float(fit["motors"][str(motor)]["disabled"]["qd_feedback"]["median"])
        for motor in MOTORS
    ]
    torque_biases = [
        float(fit["motors"][str(motor)]["disabled"]["tau_feedback"]["median"])
        for motor in MOTORS
    ]

    for motor in MOTORS:
        source = fit["motors"][str(motor)]
        individual[str(motor)] = {
            "feedback_can_id": motor + 16,
            "mechanical": source["mechanical"],
            "velocity_tracking": source["velocity_tracking"],
            "communication": source["communication"],
            "sensor_disabled": source["disabled"],
            "actuator_kd_0p2": source["actuator_dynamics"],
            "actuator_kd_4": kd4["motors"][str(motor)],
            "high_pulse_kd_4": high_per_motor[str(motor)],
        }

    model = {
        "schema_version": 1,
        "model": "GF43X40-10 / DM-J4340-10 24V CAN-observable equivalent",
        "data_scope": {
            "velocity_rounds": [1, 2],
            "small_torque_pulse_rounds": [1, 2, 3],
            "high_torque_pulse_rounds": [1],
            "motor_ids": list(MOTORS),
        },
        "fixed_specification": {
            "nominal_bus_voltage_V": 24.0,
            "datasheet_test_voltage_range_V": [23.834, 24.085],
            "working_voltage_range_status": "not specified in the GF43X40-10 table",
            "gear_ratio": 40.0,
            "rated_bus_current_A": 2.3,
            "rated_phase_current_A": 2.0,
            "rated_power_W": 40.0,
            "peak_power_W": 70.0,
            "rated_torque_Nm": 8.9,
            "mechanical_peak_torque_Nm": 23.5,
            "rated_velocity_rpm": 43.0,
            "rated_velocity_rad_s": 4.50294947014537,
            "maximum_speed_observed_in_table_rpm": 55.0,
            "maximum_speed_observed_in_table_rad_s": 5.759586531581287,
            "maximum_speed_observed_on_real_motors_rad_s": MEASURED_NO_LOAD_PEAK_SPEED_RAD_S,
            "maximum_speed_observed_on_real_motors_rpm": MEASURED_NO_LOAD_PEAK_SPEED_RAD_S * 60.0 / (2.0 * np.pi),
            "motor_torque_constant_Nm_A": 0.095,
            "torque_constant_note": "manufacturer motor constant; do not treat as output torque per bus amp",
            "motor_mass_kg": 0.362,
            "backlash_arcmin": 15.0,
            "backlash_rad": 0.004363323129985824,
            "motor_size_mm": "57 x 56.5",
        },
        "observed_protocol": {
            "classic_can_bitrate": 500000,
            "control_rate_hz": 200.0,
            "control_mode": "MIT",
            "pmax_rad": 12.5,
            "vmax_rad_s": 10.0,
            "tmax_mapping_Nm": 28.0,
            "kp_range": [0.0, 500.0],
            "kd_range": [0.0, 5.0],
        },
        "simulation_limits": {
            "velocity_command_and_feedback_mapping_rad_s": [-10.0, 10.0],
            "torque_command_and_feedback_mapping_Nm": [-28.0, 28.0],
            "mujoco_actuator_force_range_Nm": [-23.5, 23.5],
            "note": (
                "Vmax/Tmax are the ranges reported by the connected motors and are "
                "used for MIT frame encoding, decoding and command clipping. They do "
                "not prove steady mechanical capability at those limits. Physical "
                "output is capped by the GF43X40-10 peak-torque specification."
            ),
        },
        "torque_speed_capability": torque_speed_envelope,
        "datasheet_gf43x40_10_24V": {
            "source": "4340电机参数.pdf pages 7-10",
            "quality": "exact transcription of the manufacturer table; efficiency derived from tabulated powers",
            "performance_table": datasheet_performance,
            "coverage": {
                "torque_Nm": [1.14, 17.86],
                "speed_rpm": [30.6, 55.0],
                "note": "The performance table does not cover the specified 23.5 N*m peak."
            },
            "peak_region_extrapolation": peak_region_extrapolation,
        },
        "nominal_empty_load": {
            **nominal,
            "actuator_time_constant_kd_0p2_s": median(tau_low),
            "actuator_command_gain_kd_0p2": median(gain_low),
            "actuator_time_constant_kd_4_s": median(tau_kd4),
            "actuator_command_gain_kd_4": median(gain_kd4),
            "communication_latency_s": median(latency_medians),
            "packet_loss_probability": median(loss_rates),
            "velocity_bias_rad_s": median(velocity_biases),
            "torque_feedback_bias_Nm": median(torque_biases),
        },
        "sensor": {
            "position_observed_static_values_rad": position_biases,
            "position_bias_status": "requires a known mechanical reference; current values are shaft positions",
            "velocity_bias_rad_s": median(velocity_biases),
            "torque_feedback_bias_Nm": median(torque_biases),
            "position_noise_std_upper_bound_rad": 25.0 / 65535.0 / 2.0,
            "velocity_noise_std_upper_bound_rad_s": 20.0 / 4095.0 / 2.0,
            "torque_noise_std_upper_bound_Nm": 56.0 / 4095.0 / 2.0,
            "note": "static MAD was zero because feedback did not cross a CAN quantization step",
        },
        "domain_randomization": {
            **randomization,
            "actuator_time_constant_s": data_range(tau_low),
            "command_gain": data_range(gain_low),
            "communication_delay_s": [0.0, 0.010],
            "packet_loss_probability": [min(loss_rates), max(loss_rates)],
            "torque_speed_curve": {
                "base_lookup": "torque_speed_capability",
                "torque_axis_scale_range": [0.9, 1.0],
                "speed_axis_scale_range": [
                    5.677655677655678 / MEASURED_NO_LOAD_PEAK_SPEED_RAD_S,
                    1.0,
                ],
                "peak_region_shape_blend_range": [0.0, 1.0],
                "peak_power_W_range": [63.0, 70.0],
                "transform": "(omega_i, tau_i) -> (s_omega*omega_i, s_tau*tau_i), then apply tau <= P_peak/max(abs(omega), eps)",
                "derived_rated_torque_Nm": "8.9 * s_tau",
                "derived_peak_torque_Nm": "23.5 * s_tau",
                "derived_rated_velocity_rad_s": "4.50294947014537 * s_omega",
                "derived_maximum_velocity_rad_s": "5.809523809523809 * s_omega",
                "note": "rated and peak torque/speed are derived from one sampled curve and must not be sampled independently",
            },
            "velocity_mapping_limit_rad_s": [10.0, 10.0],
            "torque_mapping_limit_Nm": [28.0, 28.0],
            "voltage_V": [23.8, 24.2],
        },
        "high_pulse_kd_4_50ms": {
            "interpretation": "historical transient command-response lookup, not steady torque capability; 24 and 28 N*m commands exceed the later-confirmed GF43X40-10 physical peak of 23.5 N*m",
            "aggregate": high_aggregate,
        },
        "individual_motors": individual,
        "unidentified": {
            "steady_torque_speed_capability": "manufacturer table covers 1.14-17.86 N*m; 17.86-23.5 N*m now has a constrained cubic extrapolation but still requires controlled-load validation",
            "bus_current_and_efficiency": "exact manufacturer lookup is available; per-motor validation requires synchronized power-supply current",
            "motor_thermal_coefficients": "not provided for GF43X40-10 in this PDF; requires sustained runs from known ambient plus cooling traces",
            "mos_thermal_coefficients": "requires sustained runs with ambient temperature and MOS temperature logging",
            "overload_accumulation_and_recovery": "requires repeated loaded over-rated pulses",
            "absolute_position_bias": "requires a known output-shaft reference fixture",
            "allowed_working_voltage_range": "only 24 V rated voltage and 23.834-24.085 V test points are provided",
            "hardware_encoder_resolution": "the PDF says dual encoders but does not state bit depth for GF43X40-10",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
