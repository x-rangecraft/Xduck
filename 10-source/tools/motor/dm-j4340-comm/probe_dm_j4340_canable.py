#!/usr/bin/env python3
"""Safe I2RT protocol probe for CANable/candleLight (gs_usb) adapters."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from typing import Sequence

try:
    import can
    import usb.core
except ImportError as exc:
    print(
        'error: missing dependency; install with: python3 -m pip install -r requirements.txt',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from probe_dm_j4340 import Register, parse_ids, print_human


CANABLE_VID = 0x1D50
CANABLE_PID = 0x606F
CAN_BITRATE = 1_000_000
CAN_REQUEST_ID = 0x7FF

REGISTERS: tuple[Register, ...] = (
    Register(0x0E, "sw_ver", "u32"),
    Register(0x08, "can_id", "u32"),
    Register(0x07, "master_id", "u32"),
    Register(0x0A, "control_mode", "u32"),
    Register(0x15, "pmax", "f32", "rad"),
    Register(0x16, "vmax", "f32", "rad/s"),
    Register(0x17, "tmax", "f32", "N*m"),
    Register(0x23, "can_bitrate", "u32"),
)
MODE_NAMES = {1: "MIT", 2: "position_velocity", 3: "velocity", 4: "position_force"}
MODE_ID_OFFSETS = {1: 0x000, 2: 0x100, 3: 0x200, 4: 0x300}


def discover_devices() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, device in enumerate(
        usb.core.find(find_all=True, idVendor=CANABLE_VID, idProduct=CANABLE_PID) or []
    ):
        try:
            product = device.product
        except (ValueError, usb.core.USBError):
            product = None
        try:
            serial_number = device.serial_number
        except (ValueError, usb.core.USBError):
            serial_number = None
        rows.append(
            {
                "index": index,
                "product": product,
                "serial_number": serial_number,
                "bus": device.bus,
                "address": device.address,
                "device": device,
            }
        )
    return rows


def public_device_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if key != "device"} for row in rows]


def choose_device(rows: list[dict[str, object]], serial_number: str | None) -> dict[str, object]:
    if serial_number:
        matches = [row for row in rows if row["serial_number"] == serial_number]
        if len(matches) != 1:
            raise RuntimeError(f"CANable serial number not found: {serial_number}")
        return matches[0]
    if len(rows) == 1:
        return rows[0]
    if not rows:
        raise RuntimeError("no CANable candleLight device (1d50:606f) found")
    raise RuntimeError("multiple CANable devices found; choose one with --serial")


def open_bus(device_row: dict[str, object], bitrate: int = CAN_BITRATE) -> "can.BusABC":
    # gs-usb 0.3.1 tries to detach a kernel driver whenever PyUSB reports one.
    # On macOS libusb reports the generic IOUSBHost stack as active, but trying
    # to detach it is both unnecessary and rejected with LIBUSB_ERROR_ACCESS.
    if sys.platform == "darwin":
        usb.core.Device.is_kernel_driver_active = lambda self, interface: False
    device = device_row["device"]
    return can.Bus(
        interface="gs_usb",
        channel=getattr(device, "product", 0),
        bus=getattr(device, "bus"),
        address=getattr(device, "address"),
        bitrate=bitrate,
    )


def build_register_request(motor_id: int, register: int) -> "can.Message":
    # I2RT section 6.1: parameter reads use DLC=4, not a padded 8-byte payload.
    data = motor_id.to_bytes(2, "little") + bytes([0x33, register])
    return can.Message(arbitration_id=CAN_REQUEST_ID, data=data, is_extended_id=False)


def build_disable_request(motor_id: int, control_mode: int = 1) -> "can.Message":
    try:
        arbitration_id = MODE_ID_OFFSETS[control_mode] + motor_id
    except KeyError as exc:
        raise ValueError(f"unsupported I2RT control mode: {control_mode}") from exc
    return can.Message(
        arbitration_id=arbitration_id,
        data=bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]),
        is_extended_id=False,
    )


def uint_to_float(value: int, minimum: float, maximum: float, bits: int) -> float:
    return value * (maximum - minimum) / ((1 << bits) - 1) + minimum


def decode_status_message(message: "can.Message", motor_id: int) -> dict[str, object] | None:
    data = bytes(message.data)
    if (
        not message.is_rx
        or message.is_extended_id
        or message.is_error_frame
        or message.is_remote_frame
        or len(data) != 8
    ):
        return None
    # D0 overlays ERR in the high nibble, so only the low node-ID nibble is
    # available in this compact status payload.
    if (data[0] & 0x0F) != (motor_id & 0x0F):
        return None
    position_raw = (data[1] << 8) | data[2]
    velocity_raw = (data[3] << 4) | (data[4] >> 4)
    torque_raw = ((data[4] & 0x0F) << 8) | data[5]
    return {
        "requested_can_id": motor_id,
        "response_can_id": message.arbitration_id,
        "error": data[0] >> 4,
        "position_rad": uint_to_float(position_raw, -12.5, 12.5, 16),
        "velocity_rad_s": uint_to_float(velocity_raw, -45.0, 45.0, 12),
        "torque_nm": uint_to_float(torque_raw, -18.0, 18.0, 12),
        "mos_temperature_deg_c": data[6],
        "motor_temperature_deg_c": data[7],
        "raw": f"{message.arbitration_id:03X}#{data.hex().upper()}",
    }


def disable_scan(bus: "can.BusABC", motor_ids: list[int], timeout: float) -> list[dict[str, object]]:
    """Try the documented disable command at each possible control-mode CAN ID."""
    results: list[dict[str, object]] = []
    while bus.recv(timeout=0) is not None:
        pass
    for motor_id in motor_ids:
        for control_mode in MODE_ID_OFFSETS:
            request = build_disable_request(motor_id, control_mode)
            bus.send(request)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                message = bus.recv(timeout=remaining)
                if message is None:
                    break
                status = decode_status_message(message, motor_id)
                if status is not None:
                    status["command"] = "disable"
                    status["control_mode"] = control_mode
                    status["request_can_id"] = request.arbitration_id
                    results.append(status)
                    break
            if results and results[-1].get("requested_can_id") == motor_id:
                break
    return results


def passive_listen(bus: "can.BusABC", duration: float) -> list[dict[str, object]]:
    """Capture real bus traffic without transmitting any CAN frame."""
    frames: list[dict[str, object]] = []
    deadline = time.monotonic() + duration
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        message = bus.recv(timeout=remaining)
        if message is None or not message.is_rx:
            continue
        frames.append(
            {
                "timestamp": message.timestamp,
                "can_id": message.arbitration_id,
                "extended": message.is_extended_id,
                "error_frame": message.is_error_frame,
                "remote_frame": message.is_remote_frame,
                "dlc": message.dlc,
                "data_hex": bytes(message.data).hex(" "),
            }
        )
    return frames


def trace_register_requests(
    bus: "can.BusABC", motor_ids: list[int], timeout: float
) -> dict[str, object]:
    """Send one safe SW-version read per ID and expose every adapter event."""
    events: list[dict[str, object]] = []
    submitted = 0
    while bus.recv(timeout=0) is not None:
        pass
    for motor_id in motor_ids:
        request = build_register_request(motor_id, 0x0E)
        bus.send(request)
        submitted += 1
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            message = bus.recv(timeout=remaining)
            if message is None:
                break
            data = bytes(message.data)
            events.append(
                {
                    "direction": "rx" if message.is_rx else "tx_echo",
                    "can_id": message.arbitration_id,
                    "dlc": message.dlc,
                    "data_hex": data.hex(" "),
                    "error_frame": message.is_error_frame,
                    "extended": message.is_extended_id,
                }
            )
    return {
        "usb_submissions": submitted,
        "tx_echoes": sum(event["direction"] == "tx_echo" for event in events),
        "rx_data_frames": sum(
            event["direction"] == "rx" and not event["error_frame"] for event in events
        ),
        "error_frames": sum(bool(event["error_frame"]) for event in events),
        "events": events,
    }


def decode_register_message(
    message: "can.Message", motor_id: int, register: Register
) -> tuple[int | float, str] | None:
    data = bytes(message.data)
    # gs_usb reports the adapter's transmit acknowledgement as a message with
    # is_rx=False.  Its CAN ID and payload equal our request and must never be
    # mistaken for a motor reply.
    if not message.is_rx or message.is_extended_id or len(data) != 8 or data[2] != 0x33:
        return None
    if int.from_bytes(data[0:2], "little") != motor_id or data[3] != register.address:
        return None
    if register.value_type == "u32":
        value: int | float = int.from_bytes(data[4:8], "little")
    elif register.value_type == "f32":
        value = struct.unpack("<f", data[4:8])[0]
    else:
        raise ValueError(f"unsupported register type: {register.value_type}")
    raw = f"{message.arbitration_id:03X}#{data.hex().upper()}"
    return value, raw


def read_register(
    bus: "can.BusABC", motor_id: int, register: Register, timeout: float, retries: int
) -> tuple[int | float, int, str] | None:
    request = build_register_request(motor_id, register.address)
    for _ in range(retries + 1):
        bus.send(request)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                message = bus.recv(timeout=remaining)
            except struct.error:
                continue
            if message is None:
                break
            decoded = decode_register_message(message, motor_id, register)
            if decoded is not None:
                value, raw = decoded
                return value, message.arbitration_id, raw
    return None


def probe(
    bus: "can.BusABC", motor_ids: list[int], timeout: float, retries: int
) -> list[dict[str, object]]:
    while True:
        try:
            if bus.recv(timeout=0) is None:
                break
        except struct.error:
            continue
    results: list[dict[str, object]] = []
    for motor_id in motor_ids:
        registers: dict[str, object] = {}
        raw_replies: list[str] = []
        result: dict[str, object] = {
            "requested_can_id": motor_id,
            "online": False,
            "registers": registers,
            "raw_replies": raw_replies,
        }
        for register in REGISTERS:
            reply = read_register(bus, motor_id, register, timeout, retries)
            if reply is None:
                registers[register.name] = None
                continue
            value, response_can_id, raw = reply
            result["online"] = True
            result["response_can_id"] = response_can_id
            registers[register.name] = value
            raw_replies.append(raw)
        mode = registers.get("control_mode")
        if isinstance(mode, int):
            result["control_mode_name"] = MODE_NAMES.get(mode, f"unknown({mode})")
        results.append(result)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely probe I2RT motors through CANable candleLight/gs_usb."
    )
    parser.add_argument("--ids", default="1,2,3", help="motor CAN IDs (default: 1,2,3; ranges accepted)")
    parser.add_argument("--serial", help="CANable USB serial number; optional when only one is connected")
    parser.add_argument(
        "--bitrate",
        type=int,
        choices=(250000, 500000, 800000, 1000000),
        default=CAN_BITRATE,
        help="classic CAN bitrate (default: 1000000)",
    )
    parser.add_argument("--timeout", type=float, default=0.12, help="reply timeout per attempt in seconds")
    parser.add_argument("--retries", type=int, default=1, help="retries after the first request")
    parser.add_argument("--list-adapters", action="store_true", help="list matching USB adapters and exit")
    parser.add_argument("--disable-scan", action="store_true", help="send disable at all four I2RT mode IDs")
    parser.add_argument("--trace-tx", action="store_true", help="trace USB submissions, TX echoes and RX/error frames")
    parser.add_argument("--listen", type=float, metavar="SECONDS", help="passively listen without transmitting")
    parser.add_argument("--json", action="store_true", help="print JSON, including raw CAN replies")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        rows = discover_devices()
        if args.list_adapters:
            print(json.dumps(public_device_rows(rows), ensure_ascii=False, indent=2))
            return 0
        if args.timeout <= 0 or args.retries < 0 or (args.listen is not None and args.listen <= 0):
            raise ValueError("--timeout must be positive and --retries cannot be negative")
        device_row = choose_device(rows, args.serial)
        with open_bus(device_row, args.bitrate) as bus:
            if args.listen is not None:
                frames = passive_listen(bus, args.listen)
                print(json.dumps(frames, ensure_ascii=False, indent=2))
                return 0 if frames else 1
            motor_ids = parse_ids(args.ids)
            if any(not 1 <= motor_id <= 31 for motor_id in motor_ids):
                raise ValueError("I2RT node IDs must be in the documented range 1..31")
            if args.trace_tx:
                trace = trace_register_requests(bus, motor_ids, args.timeout)
                print(json.dumps(trace, ensure_ascii=False, indent=2))
                return 0
            if args.disable_scan:
                statuses = disable_scan(bus, motor_ids, args.timeout)
                print(json.dumps(statuses, ensure_ascii=False, indent=2))
                return 0 if statuses else 1
            results = probe(bus, motor_ids, args.timeout, args.retries)
    except (ValueError, RuntimeError, can.CanError, usb.core.USBError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(
            f"Adapter: {device_row['product']} SN={device_row['serial_number']} "
            f"@ {args.bitrate} bit/s (read-only probe)"
        )
        print_human(results)
    return 0 if any(result["online"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
