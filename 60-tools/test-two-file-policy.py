#!/usr/bin/env python3
"""Exercise the production policy worker with the bundled 61→14 example; no motor IO."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
MICRODUCK = ROOT / "10-source/rk3566/microduck"
worker = MICRODUCK / "robotd/src/custom_policy_worker.py"
policy = MICRODUCK / "policies/two_file_policy_example.py"
model = MICRODUCK / "policies/alpha_walking.onnx"

joint_names = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
home = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530]
frame = {
    "sequence": 1, "gateway_tick_ms": 20, "dt": 0.02,
    "joint_names": joint_names, "positions": home, "velocities": [0.0] * 15,
    "motor_torques_nm": [0.0] * 15, "motor_flags": [0] * 15,
    "motor_temperatures_c": [25.0] * 15, "motor_feedback_age_ms": [0] * 15,
    "attitude_rpy": [0.0] * 3,
    "imu": {"gyro": [0.0] * 3, "gravity": [0.0, 0.0, -1.0], "quat": [1.0, 0.0, 0.0, 0.0]},
    "command": {"twist": [0.0] * 3, "head": [0.0] * 4, "body": [0.0] * 3},
}


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise AssertionError(f"worker exited without reply: {process.stderr.read()}")
    reply = json.loads(line)
    assert reply["ok"], reply
    return reply


def start(policy_path, model_path):
    return subprocess.Popen(
        [sys.executable, "-I", "-u", "-c", worker.read_text(), str(policy_path), str(model_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


process = start(policy, model)
try:
    ready = receive(process)
    assert ready["contract"]["inputs"]["obs"]["shape"] == [1, 61]
    for action in ("reset", "step", "step"):
        process.stdin.write(json.dumps({"action": action, "frame": frame}) + "\n")
        process.stdin.flush()
        reply = receive(process)
    assert reply["ready"]
    assert set(reply["targets"]) == set(joint_names) - {"mouth"}
    assert all(set(target) == {"position", "velocity", "torque_ff", "kp", "kd"}
               for target in reply["targets"].values())
finally:
    process.kill()
    process.wait()
with tempfile.TemporaryDirectory() as directory:
    from onnx import TensorProto, checker, helper

    directory = Path(directory)
    dynamic_model = directory / "dynamic.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["obs_out"]),
         helper.make_node("Identity", ["hidden"], ["hidden_out"])],
        "two-file-dynamic",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 96]),
         helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("obs_out", TensorProto.FLOAT, [None, 96]),
         helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, [1, 4])],
    )
    generated = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    checker.check_model(generated)
    dynamic_model.write_bytes(generated.SerializeToString())
    controlled = [name for name in joint_names if name != "mouth"]
    dynamic_policy = directory / "policy.py"
    dynamic_policy.write_text(f'''import numpy as np
CONTROLLED = {controlled!r}
class Policy:
    def describe(self):
        return {{"api_version":1,"period_us":40000,"required_sources":["joints"],
          "inputs":{{"obs":{{"dtype":"float32","shape":[1,96]}},"hidden":{{"dtype":"float32","shape":[1,4]}}}},
          "outputs":{{"obs_out":{{"dtype":"float32","shape":[1,96]}},"hidden_out":{{"dtype":"float32","shape":[1,4]}}}},
          "controlled_joints":CONTROLLED}}
    def reset(self, robot_info, first_frame): self.count=0
    def preprocess(self, frame, feedback):
        self.count += 1
        return {{"obs":np.full((1,96),self.count,dtype=np.float32),"hidden":np.zeros((1,4),dtype=np.float32)}}
    def postprocess(self, outputs, frame):
        return {{name:{{"position":float(outputs["obs_out"][0,index]/100),"velocity":0.0,"torque_ff":0.0,"kp":20.0,"kd":1.0}} for index,name in enumerate(CONTROLLED)}}
''')
    process = start(dynamic_policy, dynamic_model)
    try:
        ready = receive(process)
        assert ready["contract"]["inputs"]["obs"]["shape"] == [1, 96]
        assert set(ready["contract"]["inputs"]) == {"obs", "hidden"}
        process.stdin.write(json.dumps({"action": "reset", "frame": frame}) + "\n")
        process.stdin.flush(); receive(process)
        process.stdin.write(json.dumps({"action": "step", "frame": frame}) + "\n")
        process.stdin.flush(); reply = receive(process)
        assert len(reply["targets"]) == 14
    finally:
        process.kill(); process.wait()

print("Two-file policy: 61-D compatibility, 96-D dynamic shape, named multi-IO and 14-joint MIT output passed")
