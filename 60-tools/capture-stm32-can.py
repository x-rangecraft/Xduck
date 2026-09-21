#!/usr/bin/env python3
"""Capture selected STM32 motor commands, CAN sends and feedback through DAP.

Requires a matching H7DM_CAN_CAPTURE=ON firmware image already installed.
Only the dedicated RAM control words are written; the MCU is not halted,
reset or flashed by this tool.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "10-source/stm32/stm32-h7-dm/CtrBoard-H7_ALL"
DEFAULT_ELF = ROOT / "30-artifacts/stm32/stm32-h7-dm/imu-phase-2026-09-21/CtrBoard-H7_ALL.elf"
DEFAULT_OUTPUT = ROOT / "50-logs/test/motor/gf43x40-10-i2rt/data/can-capture"
INTERFACE = PROJECT / "cmake/openocd/dm-cmsis-dap.cfg"
HEADER = struct.Struct("<12I")
EVENT = struct.Struct("<IHBB8s")
COMMAND = struct.Struct("<IHBBiiiHH")
MAGIC = 0x314E4143
EVENT_CAPACITY = 18432
COMMAND_CAPACITY = 768
SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


def openocd(*commands):
    args = ["openocd", "-f", str(INTERFACE), "-f", "target/stm32h7x.cfg",
            "-c", "adapter speed 500", "-c", "init"]
    for command in commands:
        args.extend(("-c", command))
    args.extend(("-c", "shutdown"))
    result = subprocess.run(args, text=True, capture_output=True, timeout=30, check=False)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"OpenOCD failed ({result.returncode}):\n{output[-2500:]}")
    return output


def read_words(address, count):
    output = openocd(f"mdw 0x{address:08x} {count}")
    words = []
    for line in output.splitlines():
        match = re.match(r"^0x[0-9a-fA-F]{8}:\s*(.*)$", line.strip())
        if match:
            words.extend(int(word, 16) for word in re.findall(r"\b[0-9a-fA-F]{8}\b", match[1]))
    if len(words) != count:
        raise RuntimeError(f"expected {count} RAM words; got {len(words)}:\n{output[-1500:]}")
    return words


def symbol_address(elf):
    output = subprocess.check_output(["arm-none-eabi-nm", "--defined-only", str(elf)], text=True)
    for line in output.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]+)\s+\S\s+g_can_capture", line)
        if match:
            return int(match[1], 16)
    raise RuntimeError("ELF has no g_can_capture symbol; build with -DH7DM_CAN_CAPTURE=ON")


def read_header(address):
    names = ("magic", "version", "state", "event_count", "event_dropped",
             "command_count", "command_dropped", "start_tick_ms", "start_cycle",
             "core_hz", "window_ms", "motor_mask")
    header = dict(zip(names, read_words(address, len(names))))
    if header["magic"] != MAGIC or header["version"] != 3:
        raise RuntimeError("capture RAM does not match this tool; verify the installed firmware")
    if (header["event_count"] > EVENT_CAPACITY or
            header["command_count"] > COMMAND_CAPACITY or
            header["window_ms"] > 7000 or
            not 0 < header["motor_mask"] <= 0x3FFF):
        raise RuntimeError(f"invalid capture header: {header}")
    return header


def dump_array(start_address, byte_count, output):
    with output.open("wb") as whole:
        for offset in range(0, byte_count, 16384):
            size = min(16384, byte_count - offset)
            chunk = output.parent / f"chunk-{offset:06x}.bin"
            if not SAFE_PATH.fullmatch(str(chunk)):
                raise RuntimeError("temporary path contains unsupported OpenOCD characters")
            for attempt in range(3):
                try:
                    openocd(f"dump_image {{{chunk}}} 0x{start_address + offset:08x} {size}")
                    if chunk.stat().st_size != size:
                        raise RuntimeError("short DAP read")
                    break
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    if attempt == 2:
                        raise
                    time.sleep(0.3)
            whole.write(chunk.read_bytes())
            chunk.unlink()


def scaled(raw, low, high, bits):
    return low + raw * (high - low) / ((1 << bits) - 1)


def decode_tx(data):
    p = (data[0] << 8) | data[1]
    v = (data[2] << 4) | (data[3] >> 4)
    kp = ((data[3] & 15) << 8) | data[4]
    kd = (data[5] << 4) | (data[6] >> 4)
    torque = ((data[6] & 15) << 8) | data[7]
    return (scaled(p, -12.5, 12.5, 16), scaled(v, -10, 10, 12),
            scaled(kp, 0, 500, 12), scaled(kd, 0, 5, 12),
            scaled(torque, -28, 28, 12))


def decode_rx(data):
    p = (data[1] << 8) | data[2]
    v = (data[3] << 4) | (data[4] >> 4)
    torque = ((data[4] & 15) << 8) | data[5]
    return (scaled(p, -12.5, 12.5, 16), scaled(v, -10, 10, 12),
            scaled(torque, -28, 28, 12), (data[0] >> 4) & 15)


def parse_commands(raw, header):
    if len(raw) != header["command_count"] * COMMAND.size:
        raise RuntimeError("command dump length does not match header")
    commands = []
    for cycles, seq, motor, reserved, p, v, torque, kp, kd in COMMAND.iter_unpack(raw):
        if not 1 <= motor <= 14 or reserved != 0:
            raise RuntimeError("invalid command record")
        commands.append({"time_us": cycles * 1_000_000 / header["core_hz"],
                         "motor_id": motor, "command_seq": seq,
                         "target_p_rad": p / 1000, "target_v_rad_s": v / 1000,
                         "target_kp": kp / 100, "target_kd": kd / 1000,
                         "target_ff_nm": torque / 1000})
    return commands


def parse_events(raw, header):
    if len(raw) != header["event_count"] * EVENT.size:
        raise RuntimeError("event dump length does not match header")
    events = []
    for cycles, seq, motor, kind, data in EVENT.iter_unpack(raw):
        if not 1 <= motor <= 14 or kind not in (1, 2):
            raise RuntimeError("invalid CAN event record")
        events.append({"time_us": cycles * 1_000_000 / header["core_hz"],
                       "motor_id": motor, "command_seq": seq,
                       "kind": "tx_enqueued" if kind == 1 else "rx",
                       "can_port": 1 if motor <= 5 else (2 if motor <= 10 else 3),
                       "can_id": f"0x{motor if kind == 1 else motor + 0x10:03X}",
                       "data_hex": data.hex().upper(), "data": data})
    return events


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(folder, commands, events):
    command_columns = ("time_us", "motor_id", "command_seq", "target_p_rad",
                       "target_v_rad_s", "target_kp", "target_kd", "target_ff_nm")
    event_columns = ("time_us", "motor_id", "command_seq", "kind", "can_port",
                     "can_id", "data_hex", "position_rad", "velocity_rad_s",
                     "kp", "kd", "torque_nm", "motor_state")
    sample_columns = ("motor_id", "command_seq", "command_received_us",
                      "target_p_rad", "target_v_rad_s", "target_kp", "target_kd",
                      "target_ff_nm", "can_tx_us", "can_tx_p_rad", "can_tx_v_rad_s",
                      "can_tx_kp", "can_tx_kd", "can_tx_ff_nm", "feedback_rx_us",
                      "feedback_p_rad", "feedback_v_rad_s", "feedback_torque_est_nm",
                      "feedback_motor_state", "tx_to_feedback_us", "feedback_match")
    write_csv(folder / "commands.csv", command_columns, commands)

    decoded_events = []
    for event in events:
        row = {key: value for key, value in event.items() if key != "data"}
        if event["kind"] == "tx_enqueued":
            p, v, kp, kd, torque = decode_tx(event["data"])
            row.update(position_rad=p, velocity_rad_s=v, kp=kp, kd=kd, torque_nm=torque)
        else:
            p, v, torque, state = decode_rx(event["data"])
            row.update(position_rad=p, velocity_rad_s=v, torque_nm=torque,
                       motor_state=state)
        decoded_events.append(row)
    write_csv(folder / "events.csv", event_columns, decoded_events)

    by_key = {}
    for command in commands:
        by_key.setdefault((command["motor_id"], command["command_seq"]), []).append(command)
    by_motor = {}
    for event in events:
        by_motor.setdefault(event["motor_id"], []).append(event)

    samples = []
    for motor, motor_events in by_motor.items():
        motor_events.sort(key=lambda event: event["time_us"])
        for i, event in enumerate(motor_events):
            if event["kind"] != "tx_enqueued":
                continue
            command_candidates = by_key.get((motor, event["command_seq"]), [])
            command = next((item for item in reversed(command_candidates)
                            if item["time_us"] <= event["time_us"]), None)
            p, v, kp, kd, torque = decode_tx(event["data"])
            row = {"motor_id": motor, "command_seq": event["command_seq"],
                   "can_tx_us": event["time_us"], "can_tx_p_rad": p,
                   "can_tx_v_rad_s": v, "can_tx_kp": kp, "can_tx_kd": kd,
                   "can_tx_ff_nm": torque, "feedback_match": "none"}
            if command is not None:
                row.update(command_received_us=command["time_us"],
                           target_p_rad=command["target_p_rad"],
                           target_v_rad_s=command["target_v_rad_s"],
                           target_kp=command["target_kp"], target_kd=command["target_kd"],
                           target_ff_nm=command["target_ff_nm"])
            for later in motor_events[i + 1:]:
                if later["kind"] == "tx_enqueued":
                    break
                if later["kind"] == "rx":
                    fp, fv, ft, state = decode_rx(later["data"])
                    row.update(feedback_rx_us=later["time_us"], feedback_p_rad=fp,
                               feedback_v_rad_s=fv, feedback_torque_est_nm=ft,
                               feedback_motor_state=state,
                               tx_to_feedback_us=later["time_us"] - event["time_us"],
                               feedback_match="first_rx_before_next_tx")
                    break
            samples.append(row)
    samples.sort(key=lambda row: row["can_tx_us"])
    write_csv(folder / "samples.csv", sample_columns, samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ms", type=int, default=3000,
                        help="capture time in milliseconds (default: 3000)")
    parser.add_argument("--motors", default="2,7,12",
                        help="comma-separated motor IDs, default: 2,7,12")
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF,
                        help="exact ELF corresponding to the installed capture firmware")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.duration_ms <= 7000:
        parser.error("--duration-ms must be between 1 and 7000")
    try:
        motors = [int(part.strip()) for part in args.motors.split(",")]
    except ValueError:
        parser.error("--motors must be comma-separated integers")
    if not motors or any(motor < 1 or motor > 14 for motor in motors) or len(set(motors)) != len(motors):
        parser.error("--motors must contain unique IDs from 1 to 14")
    motor_mask = sum(1 << (motor - 1) for motor in motors)
    elf = args.elf.resolve(strict=True)
    address = symbol_address(elf)
    dev_id = read_words(0x5C001000, 1)[0] & 0xFFF
    if dev_id != 0x483:
        raise RuntimeError(f"connected MCU ID is 0x{dev_id:03x}, expected STM32H723 (0x483)")
    before = read_header(address)
    if before["state"] not in (0, 3):
        raise RuntimeError(f"capture is already armed or running (state={before['state']})")

    openocd(f"mww 0x{address + 44:08x} {motor_mask}",
            f"mww 0x{address + 40:08x} {args.duration_ms}",
            f"mww 0x{address + 8:08x} 1")
    deadline = time.monotonic() + args.duration_ms / 1000 + 2.0
    while time.monotonic() < deadline:
        time.sleep(min(0.15, args.duration_ms / 1000 + 0.02))
        header = read_header(address)
        if header["state"] == 3:
            break
    else:
        raise RuntimeError("capture did not finish; no selected motor traffic or STM32 reset")
    if (header["window_ms"] != args.duration_ms or header["motor_mask"] != motor_mask or
            header["core_hz"] == 0):
        raise RuntimeError(f"capture configuration changed during recording: {header}")

    temp_root = ROOT / "90-temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="can-capture-", dir=temp_root) as temp_name:
        temp = Path(temp_name)
        event_path = temp / "events.bin"
        command_path = temp / "commands.bin"
        dump_array(address + HEADER.size, header["event_count"] * EVENT.size, event_path)
        dump_array(address + HEADER.size + EVENT_CAPACITY * EVENT.size,
                   header["command_count"] * COMMAND.size, command_path)
        commands = parse_commands(command_path.read_bytes(), header)
        events = parse_events(event_path.read_bytes(), header)
        write_outputs(temp, commands, events)
        metadata = dict(header)
        metadata.update({"selected_motor_ids": motors, "event_size": EVENT.size,
                         "command_size": COMMAND.size, "event_capacity": EVENT_CAPACITY,
                         "command_capacity": COMMAND_CAPACITY,
                         "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
                         "timestamp": "microseconds relative to capture start, converted from STM32 DWT cycles",
                         "target_semantics": "received and submitted to async motor command slot; may be superseded before application",
                         "tx_semantics": "accepted into FDCAN TX FIFO; bus completion is not proven",
                         "feedback_match": "first RX after TX and before next TX for that motor; not a causal guarantee",
                         "feedback_torque_scale_nm": [-28, 28],
                         "feedback_torque_meaning": "motor CAN torque estimate; raw phase current is not available"})
        (temp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if destination.exists():
            raise RuntimeError(f"output directory already exists: {destination}")
        shutil.move(str(temp), str(destination))
    print(f"Saved {header['event_count']} CAN events and {header['command_count']} targets; "
          f"dropped {header['event_dropped']} events, {header['command_dropped']} targets: {destination}")


if __name__ == "__main__":
    main()
