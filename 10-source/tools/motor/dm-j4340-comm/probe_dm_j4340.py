#!/usr/bin/env python3
"""Read-only communication probe for DM-J4340-2EC via Damiao USB-to-CAN.

The adapter is the serial (U2CAN) model used by Damiao's public Python SDK.
This program only sends CAN register-read requests.  It never enables a motor,
writes a register, stores parameters, or changes the zero position.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # Allow protocol self-tests without pyserial installed.
    serial = None
    list_ports = None


SERIAL_BAUD = 921_600
CAN_REQUEST_ID = 0x7FF

# Damiao U2CAN serial bridge framing, matching the vendor Python SDK.
TX_TEMPLATE = bytes(
    [
        0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00,
        0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ]
)


@dataclass(frozen=True)
class Register:
    address: int
    name: str
    value_type: str
    unit: str = ""


REGISTERS: tuple[Register, ...] = (
    Register(0x0E, "sw_ver", "u32"),
    Register(0x24, "sub_ver", "u32"),
    Register(0x08, "can_id", "u32"),
    Register(0x07, "master_id", "u32"),
    Register(0x0A, "control_mode", "u32"),
    Register(0x15, "pmax", "f32", "rad"),
    Register(0x16, "vmax", "f32", "rad/s"),
    Register(0x17, "tmax", "f32", "N*m"),
    Register(0x3C, "bus_voltage", "f32", "V"),
    Register(0x3D, "mos_temperature", "f32", "degC"),
    Register(0x3E, "motor_temperature", "f32", "degC"),
)

MODE_NAMES = {0: "unknown", 1: "MIT", 2: "position_velocity", 3: "velocity", 4: "position_force"}


@dataclass(frozen=True)
class AdapterFrame:
    command: int
    can_id: int
    data: bytes
    raw: bytes


@dataclass(frozen=True)
class RegisterReply:
    requested_motor_id: int
    response_can_id: int
    motor_id: int
    register: int
    value: int | float
    raw_hex: str


class FrameStream:
    """Incremental parser for 16-byte U2CAN receive frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[AdapterFrame]:
        self._buffer.extend(chunk)
        while True:
            start = self._buffer.find(b"\xAA")
            if start < 0:
                self._buffer.clear()
                return
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 16:
                return
            if self._buffer[15] != 0x55:
                del self._buffer[0]
                continue
            raw = bytes(self._buffer[:16])
            del self._buffer[:16]
            yield AdapterFrame(
                command=raw[1],
                can_id=int.from_bytes(raw[3:7], "little"),
                data=raw[7:15],
                raw=raw,
            )


def build_can_tx_frame(can_id: int, data: bytes) -> bytes:
    if not 0 <= can_id <= 0x7FF:
        raise ValueError(f"standard CAN ID out of range: {can_id:#x}")
    if len(data) != 8:
        raise ValueError("a classic CAN data frame must contain exactly 8 bytes")
    frame = bytearray(TX_TEMPLATE)
    frame[13:15] = can_id.to_bytes(2, "little")
    frame[21:29] = data
    return bytes(frame)


def build_register_read(motor_id: int, register: int) -> bytes:
    if not 0 <= motor_id <= 0x7FF:
        raise ValueError(f"motor CAN ID out of range: {motor_id:#x}")
    if not 0 <= register <= 0xFF:
        raise ValueError(f"register out of range: {register:#x}")
    payload = motor_id.to_bytes(2, "little") + bytes([0x33, register, 0, 0, 0, 0])
    return build_can_tx_frame(CAN_REQUEST_ID, payload)


def decode_register_reply(
    frame: AdapterFrame, requested_motor_id: int, register: Register
) -> RegisterReply | None:
    data = frame.data
    if frame.command != 0x11 or len(data) != 8 or data[2] != 0x33:
        return None
    motor_id = int.from_bytes(data[0:2], "little")
    if motor_id != requested_motor_id or data[3] != register.address:
        return None
    value: int | float
    if register.value_type == "u32":
        value = int.from_bytes(data[4:8], "little")
    elif register.value_type == "f32":
        value = struct.unpack("<f", data[4:8])[0]
    else:
        raise ValueError(f"unsupported register type: {register.value_type}")
    return RegisterReply(
        requested_motor_id=requested_motor_id,
        response_can_id=frame.can_id,
        motor_id=motor_id,
        register=register.address,
        value=value,
        raw_hex=frame.raw.hex(" "),
    )


def parse_ids(text: str) -> list[int]:
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text, 0), int(last_text, 0)
            if last < first:
                raise ValueError(f"invalid descending ID range: {item}")
            result.extend(range(first, last + 1))
        else:
            result.append(int(item, 0))
    unique = list(dict.fromkeys(result))
    if not unique or any(not 0 <= value <= 0x7FF for value in unique):
        raise ValueError("motor IDs must be in the standard CAN range 0..0x7ff")
    return unique


def serial_port_rows() -> list[dict[str, str | None]]:
    if list_ports is None:
        return []
    return [
        {
            "device": port.device,
            "description": port.description,
            "vid": f"0x{port.vid:04x}" if port.vid is not None else None,
            "pid": f"0x{port.pid:04x}" if port.pid is not None else None,
            "serial_number": port.serial_number,
        }
        for port in list_ports.comports()
    ]


def choose_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port
    ports = serial_port_rows()
    likely = [
        row["device"]
        for row in ports
        if row["device"]
        and any(token in row["device"].lower() for token in ("usbmodem", "usbserial", "wchusbserial", "ttyacm"))
    ]
    if len(likely) == 1:
        return str(likely[0])
    if not likely:
        raise RuntimeError("no likely USB serial port found; use --list-ports, then pass --port")
    raise RuntimeError(f"multiple USB serial ports found ({', '.join(map(str, likely))}); pass --port")


def wait_for_reply(
    device: "serial.Serial",
    parser: FrameStream,
    motor_id: int,
    register: Register,
    timeout: float,
) -> RegisterReply | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiting = device.in_waiting
        chunk = device.read(waiting or 1)
        for frame in parser.feed(chunk):
            reply = decode_register_reply(frame, motor_id, register)
            if reply is not None:
                return reply
    return None


def read_register(
    device: "serial.Serial",
    parser: FrameStream,
    motor_id: int,
    register: Register,
    timeout: float,
    retries: int,
) -> RegisterReply | None:
    request = build_register_read(motor_id, register.address)
    for _ in range(retries + 1):
        device.write(request)
        device.flush()
        reply = wait_for_reply(device, parser, motor_id, register, timeout)
        if reply is not None:
            return reply
    return None


def probe(
    device: "serial.Serial",
    motor_ids: Iterable[int],
    timeout: float,
    retries: int,
) -> list[dict[str, object]]:
    parser = FrameStream()
    results: list[dict[str, object]] = []
    for motor_id in motor_ids:
        record: dict[str, object] = {"requested_can_id": motor_id, "online": False, "registers": {}}
        registers: dict[str, object] = record["registers"]  # type: ignore[assignment]
        for register in REGISTERS:
            reply = read_register(device, parser, motor_id, register, timeout, retries)
            if reply is None:
                registers[register.name] = None
                continue
            record["online"] = True
            record["response_can_id"] = reply.response_can_id
            registers[register.name] = reply.value
        mode = registers.get("control_mode")
        if isinstance(mode, int):
            record["control_mode_name"] = MODE_NAMES.get(mode, f"unknown({mode})")
        results.append(record)
    return results


def print_human(results: Sequence[dict[str, object]]) -> None:
    for result in results:
        motor_id = int(result["requested_can_id"])
        if not result["online"]:
            print(f"ID {motor_id:#05x}: no reply")
            continue
        registers = result["registers"]
        assert isinstance(registers, dict)
        print(
            f"ID {motor_id:#05x}: ONLINE, feedback CAN ID={int(result['response_can_id']):#05x}, "
            f"mode={result.get('control_mode_name', 'unknown')}"
        )
        for register in REGISTERS:
            value = registers.get(register.name)
            suffix = f" {register.unit}" if value is not None and register.unit else ""
            if isinstance(value, float):
                print(f"  {register.name:18s} {value:10.4f}{suffix}")
            else:
                print(f"  {register.name:18s} {str(value):>10s}{suffix}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely probe DM-J4340 motors through a Damiao serial USB-to-CAN adapter."
    )
    parser.add_argument("--port", help="serial port, e.g. /dev/cu.usbmodem1101 or /dev/ttyACM0")
    parser.add_argument("--ids", default="1,2,3", help="motor CAN IDs; supports comma lists/ranges (default: 1,2,3)")
    parser.add_argument("--timeout", type=float, default=0.12, help="reply timeout per attempt in seconds")
    parser.add_argument("--retries", type=int, default=1, help="retries after the first request")
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a human-readable report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.list_ports:
        print(json.dumps(serial_port_rows(), ensure_ascii=False, indent=2))
        return 0
    if serial is None:
        print("error: pyserial is missing; install with: python3 -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.retries < 0:
        print("error: --timeout must be positive and --retries cannot be negative", file=sys.stderr)
        return 2
    try:
        motor_ids = parse_ids(args.ids)
        port = choose_port(args.port)
        with serial.Serial(port, SERIAL_BAUD, timeout=min(args.timeout, 0.02)) as device:
            device.reset_input_buffer()
            results = probe(device, motor_ids, args.timeout, args.retries)
    except (ValueError, RuntimeError, OSError, serial.SerialException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"Port: {port} @ {SERIAL_BAUD} baud (read-only probe)")
        print_human(results)
    return 0 if any(result["online"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
