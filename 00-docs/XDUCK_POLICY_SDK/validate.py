#!/usr/bin/env python3
"""Validate one Xduck policy.py + model.onnx pair through the production worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
HOME = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
]


def neutral_frame(sequence):
    return {
        "sequence": sequence,
        "gateway_tick_ms": sequence * 20,
        "dt": 0.02,
        "joint_names": JOINT_NAMES,
        "positions": HOME,
        "velocities": [0.0] * 15,
        "motor_torques_nm": [0.0] * 15,
        "motor_flags": [0] * 15,
        "motor_temperatures_c": [25.0] * 15,
        "motor_feedback_age_ms": [0] * 15,
        "attitude_rpy": [0.0] * 3,
        "imu": {
            "gyro": [0.0] * 3,
            "gravity": [0.0, 0.0, -1.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        "command": {
            "twist": [0.0] * 3,
            "head": [0.0] * 4,
            "body": [0.0] * 3,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the same contract and full-chain checks used by Xduck robotd."
    )
    parser.add_argument("policy", nargs="?", type=Path, default=HERE / "policy.py")
    parser.add_argument("model", nargs="?", type=Path, default=HERE / "model.onnx")
    args = parser.parse_args()
    for role, path in (("policy.py", args.policy), ("model.onnx", args.model)):
        if not path.is_file():
            parser.error(f"{role} not found: {path}")

    requests = [
        {"action": "reset", "frame": neutral_frame(0)},
        *({"action": "step", "frame": neutral_frame(index)} for index in range(1, 4)),
    ]
    command = [
        sys.executable,
        "-I",
        "-u",
        "-c",
        (HERE / "_policy_worker.py").read_text(),
        str(args.policy.resolve()),
        str(args.model.resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            input="".join(json.dumps(request) + "\n" for request in requests),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SystemExit("FAIL: policy pipeline exceeded the 15 second developer timeout") from error

    replies = []
    for line in result.stdout.splitlines():
        try:
            replies.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"FAIL: worker returned non-JSON output: {line!r}") from error
    for reply in replies:
        if not reply.get("ok"):
            raise SystemExit(f"FAIL: {reply.get('error', reply)}\n{result.stderr}")
    if result.returncode != 0 or len(replies) != 5:
        raise SystemExit(
            f"FAIL: worker exit={result.returncode}, replies={len(replies)}\n{result.stderr}"
        )
    contract = replies[0].get("contract")
    final = replies[-1]
    if not final.get("ready") or set(final.get("targets", {})) != set(JOINT_NAMES) - {"mouth"}:
        raise SystemExit(f"FAIL: incomplete final MIT targets: {final}")

    print("PASS: policy.py + model.onnx completed reset and three full inference cycles")
    print(f"period_us: {contract['period_us']}")
    print("inputs:", ", ".join(
        f"{name} {port['dtype']} {port['shape']}" for name, port in contract["inputs"].items()
    ))
    print("outputs:", ", ".join(
        f"{name} {port['dtype']} {port['shape']}" for name, port in contract["outputs"].items()
    ))
    print("controlled joints:", len(final["targets"]))


if __name__ == "__main__":
    main()
