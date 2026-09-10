#!/usr/bin/env python3
"""Fit CAN-observable empty-load parameters from campaign rounds 1 and 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


MOTORS = (13, 14, 15)
STATIC = ("disabled_static_r1_*.csv", "disabled_static_r2_*.csv", "disabled_static_r3_*.csv")
ROUND1 = {
    13: ("velocity_low_m13_r1_*.csv", "velocity_mid_m13_r1_*.csv", "velocity_high_m13_r1_*.csv"),
    14: ("velocity_low_m14_r1_retry_*.csv", "velocity_mid_m14_r1_*.csv", "velocity_high_m14_r1_*.csv"),
    15: ("velocity_low_m15_r1_*.csv", "velocity_mid_m15_r1_*.csv", "velocity_high_m15_r1_*.csv"),
}
ROUND2 = {
    13: ("velocity_full_m13_r2_*.csv", "velocity_full_m13_r2_tail_recovery_safe_*.csv"),
    14: ("velocity_full_m14_r2_*.csv", "velocity_full_m14_r2_tail_recovery_safe_*.csv"),
    15: ("velocity_full_m15_r2_*.csv",),
}
ZERO = {
    13: "enabled_zero_m13_r1_retry_*.csv",
    14: "enabled_zero_m14_r1_*.csv",
    15: "enabled_zero_m15_r1_*.csv",
}
PULSE = {motor: f"torque_pulse_m{motor}_r[123]_*.csv" for motor in MOTORS}


def resolve(root: Path, patterns: tuple[str, ...] | str) -> list[Path]:
    if isinstance(patterns, str):
        patterns = (patterns,)
    result: list[Path] = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no file matches {pattern}")
        result.extend(matches)
    return list(dict.fromkeys(result))


def read(paths: list[Path], motor: int | None = None, valid_only: bool = True) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if motor is not None:
        data = data[data.motor_serial_or_index == motor]
    if valid_only:
        data = data[data.reply_received == 1]
    return data.copy()


def robust(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    med = float(np.median(x))
    return {
        "median": med,
        "robust_sigma": float(1.4826 * np.median(np.abs(x - med))),
        "samples": int(x.size),
    }


def communication(rows: pd.DataFrame) -> dict[str, float | int]:
    total = len(rows)
    replies = int((rows.reply_received == 1).sum())
    lat = robust(rows.latency_ms)
    return {
        "commands": total,
        "replies": replies,
        "loss_rate": (total - replies) / total,
        "latency_median_ms": lat["median"],
        "latency_robust_sigma_ms": lat["robust_sigma"],
    }


def tracking(data: pd.DataFrame) -> dict[str, object]:
    data["command_step"] = data.groupby("source_file").qd_des.diff().abs()
    steady = data[(data.qd_des.abs() >= 0.2) & (data.command_step < 2e-3)]
    result: dict[str, object] = {"samples": int(len(steady))}
    for label, sign in (("positive", 1), ("negative", -1)):
        part = steady[steady.qd_des * sign > 0]
        x, y = part.qd_des.to_numpy(float), part.qd_feedback.to_numpy(float)
        design = np.column_stack((x, np.ones_like(x)))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        residual = y - pred
        denominator = np.sum((y - y.mean()) ** 2)
        result[label] = {
            "gain": float(coef[0]),
            "offset_rad_s": float(coef[1]),
            "rmse_rad_s": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1 - np.sum(residual**2) / denominator) if denominator else math.nan,
            "samples": int(len(part)),
        }
    return result


def acceleration(data: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for _, part in data.groupby("source_file", sort=False):
        part = part.sort_values("timestamp_rx_monotonic_ns").copy()
        dt = float(np.median(np.diff(part.timestamp_rx_monotonic_ns)) / 1e9)
        window = min(41, len(part) // 2 * 2 - 1)
        if window >= 7 and dt > 0:
            part["qdd"] = savgol_filter(part.qd_feedback, window, 3, deriv=1, delta=dt)
            outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def mechanical(data: pd.DataFrame) -> dict[str, object]:
    data = acceleration(data)
    data = data[(data.qd_feedback.abs() < 6.1) & (data.qdd.abs() < 80)].iloc[::5]
    v, qdd, y = (data[name].to_numpy(float) for name in ("qd_feedback", "qdd", "tau_feedback"))
    positive = v >= 0
    design = np.column_stack(
        (
            qdd,
            np.tanh(v / 0.05) * positive,
            np.tanh(v / 0.05) * (~positive),
            np.maximum(v, 0),
            np.minimum(v, 0),
            np.ones_like(v),
        )
    )
    initial = [0.01, 0.2, 0.2, 0.02, 0.02, 0]
    lower = [0, 0, 0, 0, 0, -5]
    upper = [10, 10, 10, 10, 10, 5]
    fit = least_squares(lambda beta: design @ beta - y, initial, bounds=(lower, upper), loss="soft_l1")
    pred = design @ fit.x
    names = (
        "J_eff_kg_m2", "coulomb_positive_Nm", "coulomb_negative_Nm",
        "viscous_positive_Nm_s_rad", "viscous_negative_Nm_s_rad", "torque_offset_Nm",
    )
    result = {name: float(value) for name, value in zip(names, fit.x)}
    result.update(
        rmse_Nm=float(np.sqrt(np.mean((y - pred) ** 2))),
        samples=int(len(y)),
        note="CAN-observable effective fit; no external torque sensor",
    )
    return result


def dynamics(data: pd.DataFrame) -> dict[str, object]:
    estimates = []
    for _, part in data.groupby("source_file", sort=False):
        part = part.sort_values("timestamp_rx_monotonic_ns")
        dt = float(np.median(np.diff(part.timestamp_rx_monotonic_ns)) / 1e9)
        u = (part.kd * (part.qd_des - part.qd_feedback) + part.tau_ff).to_numpy(float)
        y = part.tau_feedback.to_numpy(float)
        candidates = []
        for delay in range(21):
            target = y[delay + 1 :]
            design = np.column_stack((y[delay:-1], u[: -(delay + 1)], np.ones(len(target))))
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
            rmse = float(np.sqrt(np.mean((target - design @ beta) ** 2)))
            candidates.append((rmse, delay, beta))
        rmse, delay, beta = min(candidates, key=lambda row: row[0])
        a, b, _ = map(float, beta)
        tau = -dt / math.log(a) if 0 < a < 1 else math.nan
        gain = b / (1 - a) if 0 < a < 1 else math.nan
        estimates.append((delay * dt, tau, gain, rmse))
    values = np.asarray(estimates)
    return {
        "delay_s_median": float(np.nanmedian(values[:, 0])),
        "time_constant_s_median": float(np.nanmedian(values[:, 1])),
        "command_gain_median": float(np.nanmedian(values[:, 2])),
        "rmse_Nm_median": float(np.nanmedian(values[:, 3])),
        "runs": len(estimates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed/empty_load_fit_rounds_1_2.json")
    args = parser.parse_args()
    root, output = Path(args.input), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    static_all = read(resolve(root, STATIC))
    result: dict[str, object] = {
        "schema_version": 1,
        "source_rounds": [1, 2],
        "torque_pulse_rounds": [1, 2, 3],
        "excluded": ["velocity_full_m13_r3_*", "acceptance tests", "failed enabled-zero run"],
        "motors": {},
    }
    for motor in MOTORS:
        static = static_all[static_all.motor_serial_or_index == motor]
        dynamic_paths = resolve(root, ROUND1[motor]) + resolve(root, ROUND2[motor])
        dynamic_raw = read(dynamic_paths, valid_only=False)
        dynamic = dynamic_raw[dynamic_raw.reply_received == 1].copy()
        zero = read(resolve(root, ZERO[motor]), motor)
        pulse = read(resolve(root, PULSE[motor]), motor)
        result["motors"][str(motor)] = {
            "dynamic_sources": [path.name for path in dynamic_paths],
            "disabled": {name: robust(static[name]) for name in ("q_feedback", "qd_feedback", "tau_feedback")},
            "enabled_zero": {name: robust(zero[name]) for name in ("q_feedback", "qd_feedback", "tau_feedback")},
            "communication": communication(dynamic_raw),
            "velocity_tracking": tracking(dynamic),
            "mechanical": mechanical(dynamic),
            "actuator_dynamics": dynamics(pulse),
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
