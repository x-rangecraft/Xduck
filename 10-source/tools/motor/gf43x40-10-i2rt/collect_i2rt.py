#!/usr/bin/env python3
"""Timestamped I2RT CAN data collection for GF43X40-10 bench identification.

Disabled collection is the default and cannot create motor torque.  Any mode
that enables a motor requires both ``--arm`` and exactly one selected motor.
All exits attempt to send a disable command to every selected motor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import struct
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import can

from i2rt_protocol import Register, parse_ids
from probe_i2rt_canable import (
    CAN_BITRATE,
    build_disable_request,
    build_register_request,
    choose_device,
    decode_register_message,
    discover_devices,
    open_bus,
)


PROFILE_REGISTERS = (
    Register(0x0E, "sw_ver", "u32"),
    Register(0x08, "can_id", "u32"),
    Register(0x07, "master_id", "u32"),
    Register(0x0A, "control_mode", "u32"),
    Register(0x15, "pmax", "f32", "rad"),
    Register(0x16, "vmax", "f32", "rad/s"),
    Register(0x17, "tmax", "f32", "N*m"),
)

CSV_FIELDS = (
    "experiment_id", "run_id", "motor_serial_or_index", "joint_name", "test_condition",
    "sample_index", "sequence", "timestamp_tx", "timestamp_rx", "timestamp_tx_monotonic_ns",
    "timestamp_rx_monotonic_ns", "latency_ms", "requested_rate_hz", "q_des", "qd_des", "kp",
    "kd", "tau_ff", "q_feedback", "qd_feedback", "tau_feedback", "motor_temperature",
    "mos_temperature", "driver_state", "fault_state", "power_supply_voltage", "power_supply_current",
    "robot_base_pose", "robot_base_velocity", "contact_state", "tx_can_id", "tx_dlc",
    "tx_data_hex", "rx_can_id", "rx_dlc", "rx_data_hex", "reply_received",
)


@dataclass(frozen=True)
class MotorProfile:
    motor_id: int
    feedback_id: int
    sw_ver: int
    control_mode: int
    pmax: float
    vmax: float
    tmax: float


@dataclass(frozen=True)
class Command:
    q_des: float = 0.0
    qd_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tau_ff: float = 0.0
    special: str | None = None


def float_to_uint(value: float, minimum: float, maximum: float, bits: int) -> int:
    if not math.isfinite(value):
        raise ValueError("MIT command values must be finite")
    value = min(max(value, minimum), maximum)
    return round((value - minimum) * ((1 << bits) - 1) / (maximum - minimum))


def uint_to_float(value: int, minimum: float, maximum: float, bits: int) -> float:
    return value * (maximum - minimum) / ((1 << bits) - 1) + minimum


def pack_mit_command(profile: MotorProfile, command: Command) -> can.Message:
    if profile.control_mode != 1:
        raise ValueError(f"motor {profile.motor_id} is not in MIT mode")
    p = float_to_uint(command.q_des, -profile.pmax, profile.pmax, 16)
    v = float_to_uint(command.qd_des, -profile.vmax, profile.vmax, 12)
    kp = float_to_uint(command.kp, 0.0, 500.0, 12)
    kd = float_to_uint(command.kd, 0.0, 5.0, 12)
    tau = float_to_uint(command.tau_ff, -profile.tmax, profile.tmax, 12)
    data = bytes(
        [
            p >> 8,
            p & 0xFF,
            v >> 4,
            ((v & 0x0F) << 4) | (kp >> 8),
            kp & 0xFF,
            kd >> 4,
            ((kd & 0x0F) << 4) | (tau >> 8),
            tau & 0xFF,
        ]
    )
    return can.Message(arbitration_id=profile.motor_id, data=data, is_extended_id=False)


def decode_feedback(message: can.Message, profile: MotorProfile) -> dict[str, float | int] | None:
    data = bytes(message.data)
    if (
        not message.is_rx
        or message.is_extended_id
        or message.is_error_frame
        or message.is_remote_frame
        or message.arbitration_id != profile.feedback_id
        or len(data) != 8
        or (data[0] & 0x0F) != (profile.motor_id & 0x0F)
    ):
        return None
    p = (data[1] << 8) | data[2]
    v = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0x0F) << 8) | data[5]
    driver_state = data[0] >> 4
    return {
        "q_feedback": uint_to_float(p, -profile.pmax, profile.pmax, 16),
        "qd_feedback": uint_to_float(v, -profile.vmax, profile.vmax, 12),
        "tau_feedback": uint_to_float(tau, -profile.tmax, profile.tmax, 12),
        "mos_temperature": data[6],
        "motor_temperature": data[7],
        "driver_state": driver_state,
        "fault_state": driver_state if driver_state >= 8 else 0,
    }


def discover_profiles(
    bus: can.BusABC, motor_ids: list[int], timeout: float = 0.12, retries: int = 1
) -> list[MotorProfile]:
    profiles: list[MotorProfile] = []
    for motor_id in motor_ids:
        values: dict[str, int | float] = {}
        response_id: int | None = None
        for register in PROFILE_REGISTERS:
            reply = read_profile_register(bus, motor_id, register, timeout, retries)
            if reply is None:
                raise RuntimeError(f"motor ID {motor_id:#x} did not answer register {register.name}")
            value, response_id, _ = reply
            values[register.name] = value
        assert response_id is not None
        profile = MotorProfile(
            motor_id=int(values["can_id"]),
            feedback_id=response_id,
            sw_ver=int(values["sw_ver"]),
            control_mode=int(values["control_mode"]),
            pmax=float(values["pmax"]),
            vmax=float(values["vmax"]),
            tmax=float(values["tmax"]),
        )
        if profile.motor_id != motor_id:
            raise RuntimeError(f"requested ID {motor_id:#x}, motor reported {profile.motor_id:#x}")
        profiles.append(profile)
    return profiles


def safe_recv(bus: can.BusABC, timeout: float) -> can.Message | None:
    """Ignore malformed short USB packets occasionally emitted by candleLight."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return bus.recv(timeout=remaining)
        except struct.error:
            continue


def read_profile_register(
    bus: can.BusABC, motor_id: int, register: Register, timeout: float, retries: int
) -> tuple[int | float, int, str] | None:
    for _ in range(retries + 1):
        bus.send(build_register_request(motor_id, register.address))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            message = safe_recv(bus, remaining)
            if message is None:
                break
            decoded = decode_register_message(message, motor_id, register)
            if decoded is not None:
                value, raw = decoded
                return value, message.arbitration_id, raw
    return None


def drain(bus: can.BusABC) -> None:
    while True:
        try:
            if bus.recv(timeout=0) is None:
                return
        except struct.error:
            continue


def send_disable(bus: can.BusABC, profiles: Sequence[MotorProfile]) -> None:
    for profile in profiles:
        bus.send(build_disable_request(profile.motor_id, profile.control_mode))


def send_enable(bus: can.BusABC, profiles: Sequence[MotorProfile]) -> None:
    payload = bytes([0xFF] * 7 + [0xFC])
    for profile in profiles:
        bus.send(can.Message(arbitration_id=profile.motor_id, data=payload, is_extended_id=False))


def receive_feedback(
    bus: can.BusABC, profile: MotorProfile, deadline_ns: int
) -> tuple[can.Message | None, dict[str, float | int] | None, int | None, int | None]:
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return None, None, None, None
        message = safe_recv(bus, remaining_ns / 1e9)
        rx_mono_ns = time.perf_counter_ns()
        rx_wall_ns = time.time_ns()
        if message is None:
            return None, None, None, None
        decoded = decode_feedback(message, profile)
        if decoded is not None:
            return message, decoded, rx_mono_ns, rx_wall_ns


def make_command_fn(
    args: argparse.Namespace, profile: MotorProfile, q_hold: float = 0.0
) -> Callable[[float], Command]:
    if args.condition in ("disabled_static", "enabled_zero"):
        return lambda elapsed: Command(special="disable") if args.condition == "disabled_static" else Command()
    if args.condition == "velocity_scan":
        ramp, hold, settle = args.ramp, args.hold, args.settle
        speeds = [float(value) for value in args.speeds.split(",") if value.strip()]
        targets = [sign * speed for speed in speeds for sign in (1.0, -1.0)]
        segment = ramp + hold + ramp + settle

        def velocity_command(elapsed: float) -> Command:
            index = min(int(elapsed / segment), len(targets) - 1)
            local = elapsed - index * segment
            target = targets[index]
            if local < ramp:
                velocity = target * 0.5 * (1.0 - math.cos(math.pi * local / ramp))
            elif local < ramp + hold:
                velocity = target
            elif local < 2 * ramp + hold:
                x = (local - ramp - hold) / ramp
                velocity = target * 0.5 * (1.0 + math.cos(math.pi * x))
            else:
                velocity = 0.0
            return Command(qd_des=velocity, kd=args.kd)

        return velocity_command
    if args.condition == "torque_pulse":
        pulse, rest = args.pulse, args.rest
        torques = [float(value) for value in args.torques.split(",") if value.strip()]
        targets = [sign * torque for torque in torques for sign in (1.0, -1.0)]
        segment = pulse + rest

        def pulse_command(elapsed: float) -> Command:
            index = min(int(elapsed / segment), len(targets) - 1)
            local = elapsed - index * segment
            tau = targets[index] if local < pulse else 0.0
            return Command(q_des=q_hold, kp=args.kp, kd=args.kd, tau_ff=tau)

        return pulse_command
    raise ValueError(f"unsupported condition: {args.condition}")


def computed_duration(args: argparse.Namespace) -> float:
    if args.duration is not None:
        return args.duration
    if args.condition == "velocity_scan":
        return 2 * len([x for x in args.speeds.split(",") if x.strip()]) * (
            2 * args.ramp + args.hold + args.settle
        )
    if args.condition == "torque_pulse":
        return 2 * len([x for x in args.torques.split(",") if x.strip()]) * (args.pulse + args.rest)
    return 60.0


def validate_args(args: argparse.Namespace) -> None:
    if args.rate <= 0 or args.rate > 500:
        raise ValueError("--rate must be in (0, 500] Hz")
    if computed_duration(args) <= 0:
        raise ValueError("duration must be positive")
    if args.reply_timeout <= 0:
        raise ValueError("--reply-timeout must be positive")
    if not 0.0 <= args.kp <= 500.0:
        raise ValueError("--kp must be in [0, 500]")
    if not 0.0 <= args.kd <= 5.0:
        raise ValueError("--kd must be in [0, 5]")
    moving = args.condition != "disabled_static"
    if moving and not args.arm:
        raise ValueError(f"{args.condition} requires --arm")
    ids = parse_ids(args.ids)
    if moving and len(ids) != 1:
        raise ValueError("enabled experiments require exactly one motor ID")
    if args.condition == "velocity_scan":
        speeds = [float(x) for x in args.speeds.split(",") if x.strip()]
        if not speeds or any(speed <= 0 or speed > 5.8095 for speed in speeds):
            raise ValueError("velocity speeds must be in (0, 5.8095] rad/s")
        if not 0 < args.kd <= 1.0:
            raise ValueError("initial velocity scan requires --kd in (0, 1]")
    if args.condition == "torque_pulse":
        torques = [float(x) for x in args.torques.split(",") if x.strip()]
        # The connected motors report TMAX=28 N*m for MIT mapping, while the
        # GF43X40-10 datasheet specifies a 23.5 N*m physical peak.
        torque_limit = 23.5 if args.high_torque_arm else 1.0
        if not torques or any(torque <= 0 or torque > torque_limit for torque in torques):
            raise ValueError(f"pulse magnitudes must be in (0, {torque_limit}] N*m")
        if any(torque > 1.0 for torque in torques) and not args.high_torque_arm:
            raise ValueError("torque above 1 N*m requires --high-torque-arm")
        if not 0.05 <= args.pulse <= (0.1 if max(torques) > 8.9 else 0.3):
            raise ValueError("--pulse must be 0.05..0.3 seconds")


def collect(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    validate_args(args)
    motor_ids = parse_ids(args.ids)
    duration = computed_duration(args)
    rows = discover_devices()
    device_row = choose_device(rows, args.serial)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_utc = datetime.now(timezone.utc)
    experiment_id = args.experiment_id or f"{args.condition}_{start_utc:%Y%m%dT%H%M%SZ}"
    run_id = args.run_id or uuid.uuid4().hex[:12]
    stem = f"{experiment_id}_{run_id}"
    csv_path = output_dir / f"{stem}.csv"
    metadata_path = output_dir / f"{stem}.metadata.json"
    counters = {"commands_sent": 0, "replies_received": 0, "missed_replies": 0, "fault_replies": 0}

    with open_bus(device_row, args.bitrate) as bus:
        # Configuration discovery is not part of the realtime loop.  Give it a
        # longer timeout so an occasional malformed USB packet cannot prevent
        # an otherwise healthy motor from passing the preflight check.
        profiles = discover_profiles(bus, motor_ids, max(args.reply_timeout, 0.1), 2)
        if any(profile.control_mode != 1 for profile in profiles):
            raise RuntimeError("this collector currently requires MIT mode (control_mode=1)")
        metadata: dict[str, object] = {
            "schema_version": 2,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_condition": args.condition,
            "start_time_utc": start_utc.isoformat(),
            "duration_s": duration,
            "requested_rate_hz": args.rate,
            "bitrate": args.bitrate,
            "classic_can": True,
            "adapter": {k: v for k, v in device_row.items() if k != "device"},
            "motors": [asdict(profile) for profile in profiles],
            "cold_hot_state": args.thermal_state,
            "ambient_temperature_deg_c": args.ambient_temperature,
            "joint_names": args.joint_names,
            "notes": args.notes,
            "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
            "command_parameters": {
                "speeds": args.speeds,
                "torques": args.torques,
                "kp": args.kp,
                "kd": args.kd,
                "ramp_s": args.ramp,
                "hold_s": args.hold,
                "settle_s": args.settle,
                "pulse_s": args.pulse,
                "rest_s": args.rest,
                "high_torque_arm": args.high_torque_arm,
            },
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        armed = args.condition != "disabled_static"
        try:
            if armed:
                send_enable(bus, profiles)
                time.sleep(0.05)
            drain(bus)
            hold_positions: dict[int, float] = {}
            if armed and args.condition == "torque_pulse" and args.kp > 0.0:
                # Read each motor's actual position with zero gains/torque, then
                # hold that position while the requested feed-forward pulse is
                # applied. This avoids pulling the shaft toward protocol q=0.
                for profile in profiles:
                    bus.send(pack_mit_command(profile, Command()))
                    deadline_ns = time.perf_counter_ns() + round(max(args.reply_timeout, 0.05) * 1e9)
                    _, decoded, _, _ = receive_feedback(bus, profile, deadline_ns)
                    if decoded is None:
                        raise RuntimeError(
                            f"motor ID {profile.motor_id} did not answer the hold-position preflight"
                        )
                    hold_positions[profile.motor_id] = float(decoded["q_feedback"])
                metadata["command_parameters"]["hold_positions_rad"] = hold_positions
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            command_fns = {
                profile.motor_id: make_command_fn(
                    args, profile, hold_positions.get(profile.motor_id, 0.0)
                )
                for profile in profiles
            }
            period_ns = round(1e9 / args.rate)
            start_ns = time.perf_counter_ns()
            total_samples = max(1, round(duration * args.rate))
            next_tick_ns = start_ns
            sample_index = 0
            sequence = 0
            consecutive_misses = 0
            joint_names = [name.strip() for name in args.joint_names.split(",")]
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                while sample_index < total_samples:
                    now_ns = time.perf_counter_ns()
                    if now_ns < next_tick_ns:
                        time.sleep((next_tick_ns - now_ns) / 1e9)
                    elapsed = (next_tick_ns - start_ns) / 1e9
                    for motor_index, profile in enumerate(profiles):
                        command = command_fns[profile.motor_id](elapsed)
                        if command.special == "disable":
                            message = build_disable_request(profile.motor_id, profile.control_mode)
                        else:
                            message = pack_mit_command(profile, command)
                        tx_mono_ns = time.perf_counter_ns()
                        tx_wall_ns = time.time_ns()
                        bus.send(message)
                        counters["commands_sent"] += 1
                        sequence += 1
                        deadline_ns = tx_mono_ns + round(args.reply_timeout * 1e9)
                        reply, decoded, rx_mono_ns, rx_wall_ns = receive_feedback(bus, profile, deadline_ns)
                        got_reply = decoded is not None and reply is not None
                        counters["replies_received" if got_reply else "missed_replies"] += 1
                        consecutive_misses = 0 if got_reply else consecutive_misses + 1
                        if got_reply and int(decoded["fault_state"]) != 0:
                            counters["fault_replies"] += 1
                        row = {field: "" for field in CSV_FIELDS}
                        row.update(
                            {
                                "experiment_id": experiment_id,
                                "run_id": run_id,
                                "motor_serial_or_index": profile.motor_id,
                                "joint_name": joint_names[motor_index] if motor_index < len(joint_names) else "",
                                "test_condition": args.condition,
                                "sample_index": sample_index,
                                "sequence": sequence,
                                "timestamp_tx": tx_wall_ns / 1e9,
                                "timestamp_tx_monotonic_ns": tx_mono_ns,
                                "requested_rate_hz": args.rate,
                                "q_des": command.q_des,
                                "qd_des": command.qd_des,
                                "kp": command.kp,
                                "kd": command.kd,
                                "tau_ff": command.tau_ff,
                                "tx_can_id": message.arbitration_id,
                                "tx_dlc": message.dlc,
                                "tx_data_hex": bytes(message.data[: message.dlc]).hex(" "),
                                "reply_received": int(got_reply),
                            }
                        )
                        if got_reply:
                            assert reply is not None and rx_mono_ns is not None and rx_wall_ns is not None
                            row.update(decoded)
                            row.update(
                                {
                                    "timestamp_rx": rx_wall_ns / 1e9,
                                    "timestamp_rx_monotonic_ns": rx_mono_ns,
                                    "latency_ms": (rx_mono_ns - tx_mono_ns) / 1e6,
                                    "rx_can_id": reply.arbitration_id,
                                    "rx_dlc": reply.dlc,
                                    "rx_data_hex": bytes(reply.data).hex(" "),
                                }
                            )
                            if (
                                int(decoded["fault_state"]) != 0
                                or float(decoded["motor_temperature"]) >= args.max_motor_temperature
                                or float(decoded["mos_temperature"]) >= args.max_mos_temperature
                                or abs(float(decoded["qd_feedback"])) > args.max_speed
                            ):
                                writer.writerow(row)
                                csv_file.flush()
                                raise RuntimeError("safety limit or motor fault triggered; motors disabled")
                        writer.writerow(row)
                        if armed and consecutive_misses >= args.max_consecutive_misses:
                            csv_file.flush()
                            raise RuntimeError("consecutive feedback loss limit reached; motors disabled")
                    sample_index += 1
                    next_tick_ns = start_ns + sample_index * period_ns
                    if sample_index % max(1, round(args.rate)) == 0:
                        csv_file.flush()
        finally:
            send_disable(bus, profiles)

    end_utc = datetime.now(timezone.utc)
    metadata.update({"end_time_utc": end_utc.isoformat(), "counters": counters, "csv_file": csv_path.name})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, metadata_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect synchronized GF43X40-10 data over I2RT CAN")
    parser.add_argument("condition", choices=("disabled_static", "enabled_zero", "velocity_scan", "torque_pulse"))
    parser.add_argument("--ids", default="13,14,15", help="motor IDs; motion conditions require exactly one")
    parser.add_argument("--bitrate", type=int, choices=(250000, 500000, 800000, 1000000), default=500000)
    parser.add_argument("--serial", help="CANable USB serial number")
    parser.add_argument("--duration", type=float, help="seconds; generated sequence length when omitted")
    parser.add_argument("--rate", type=float, default=200.0, help="per-motor command/sample rate in Hz")
    parser.add_argument("--reply-timeout", type=float, default=0.002, help="feedback wait per command in seconds")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    parser.add_argument("--joint-names", default="motor_13,motor_14,motor_15")
    parser.add_argument("--thermal-state", choices=("cold", "warm", "hot", "unknown"), default="unknown")
    parser.add_argument("--ambient-temperature", type=float)
    parser.add_argument("--notes", default="")
    parser.add_argument("--arm", action="store_true", help="required for every condition that enables a motor")
    parser.add_argument("--high-torque-arm", action="store_true", help="explicitly allow pulses above 1 N*m, up to the GF43X40-10 physical peak of 23.5 N*m")
    parser.add_argument("--max-consecutive-misses", type=int, default=3)
    parser.add_argument("--kp", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument("--speeds", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--ramp", type=float, default=1.5)
    parser.add_argument("--hold", type=float, default=5.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--torques", default="0.2,0.5,1.0")
    parser.add_argument("--pulse", type=float, default=0.2)
    parser.add_argument("--rest", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=6.2)
    parser.add_argument("--max-motor-temperature", type=float, default=100.0)
    parser.add_argument("--max-mos-temperature", type=float, default=100.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path, metadata_path, metadata = collect(args)
    except (ValueError, RuntimeError, can.CanError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counters = metadata["counters"]
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(json.dumps(counters, ensure_ascii=False))
    return 0 if counters["replies_received"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
