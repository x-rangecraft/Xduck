#!/usr/bin/env python3
"""Exercise the production policy worker with the bundled 61→14 example; no motor IO."""
import json
import importlib.util
import os
import numpy as np
from pathlib import Path
import pwd
import resource
import subprocess
import sys
import tempfile


ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
MICRODUCK = ROOT if (ROOT / "Cargo.toml").is_file() else ROOT / "10-source/rk3566/microduck"
SDK = ROOT / "00-docs/XDUCK_POLICY_SDK"
worker = MICRODUCK / "robotd/src/custom_policy_worker.py"
policy = MICRODUCK / "robotd/src/default_locomotion_policy.py"
SANDBOX = os.environ.get("XDUCK_POLICY_SANDBOX") == "1"
model = SDK / "model.onnx" if (SDK / "model.onnx").is_file() else MICRODUCK / "policies/alpha_walking.onnx"
if (SDK / "_policy_worker.py").is_file() and worker.read_bytes() != (SDK / "_policy_worker.py").read_bytes():
    raise AssertionError("SDK validator worker has drifted from robotd's production worker")
if (SDK / "policy.py").is_file() and policy.read_bytes() != (SDK / "policy.py").read_bytes():
    raise AssertionError("SDK default policy has drifted from robotd's built-in default")

joint_names = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
home = [0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
        0.3491, 0.3491, 0.0, 0.0, 0.0]
frame = {
    "sequence": 1, "gateway_tick_ms": 20, "dt": 0.02,
    "joint_names": joint_names, "positions": home, "velocities": [0.0] * 15,
    "motor_torques_nm": [0.0] * 15, "motor_flags": [0] * 15,
    "motor_temperatures_c": [25.0] * 15, "motor_feedback_age_ms": [0] * 15,
    "attitude_rpy": [0.0] * 3,
    "imu": {"gyro": [0.0] * 3, "gravity": [0.0, 0.0, -1.0], "quat": [1.0, 0.0, 0.0, 0.0]},
    "command": {"twist": [0.0] * 3, "head": [0.0] * 4, "body": [0.0] * 3},
    "policy_context": {"action": "walk", "phase": None, "body_active": False},
}


def load_consumer(filename):
    path = MICRODUCK / "robotd/src" / filename
    spec = importlib.util.spec_from_file_location(f"xduck_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Command-observation semantics live in each consumer, not in robotd's scheduler.
semantic_frame = dict(frame)
semantic_frame["command"] = {
    "twist": [0.2, -0.3, 0.4],
    "head": [0.5, -0.6, 0.7, -0.8],
    "body": [0.9, -1.0, 1.1],
}
locomotion = load_consumer("default_locomotion_policy.py")
semantic_frame["policy_context"] = {"action": "walk", "phase": None, "body_active": False}
assert np.allclose(locomotion.build_command_obs(semantic_frame),
                   [0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.0, 0.0, 0.9, -1.0, 1.1, 0.0])
semantic_frame["policy_context"] = {"action": "stand", "phase": None, "body_active": True}
assert np.allclose(locomotion.build_command_obs(semantic_frame),
                   [0.0, 0.0, 0.0, 0.5, -0.6, 0.7, -0.8, 0.0, 0.0, 0.9, -1.0, 1.1, 0.0])
ground_pick = load_consumer("default_ground_pick_policy.py")
semantic_frame["policy_context"] = {"action": "ground_pick", "phase": 0.25, "body_active": False}
assert np.allclose(ground_pick.build_command_obs(semantic_frame), [0.0, 1.0, *([0.0] * 11)], atol=1e-6)
for filename, action_name in (
    ("default_kick_left_policy.py", "kick_left"),
    ("default_kick_right_policy.py", "kick_right"),
    ("default_roulade_policy.py", "roulade"),
):
    consumer = load_consumer(filename)
    semantic_frame["policy_context"] = {"action": action_name, "phase": None, "body_active": False}
    assert np.array_equal(consumer.build_command_obs(semantic_frame), np.zeros(13, dtype=np.float32))
sitstand = load_consumer("default_sitstand_policy.py")
semantic_frame["policy_context"] = {"action": "sit", "phase": None, "body_active": False}
assert np.array_equal(sitstand.build_command_obs(semantic_frame),
                      np.asarray([1.0, *([0.0] * 12)], dtype=np.float32))
semantic_frame["policy_context"] = {"action": "rise", "phase": None, "body_active": False}
assert np.array_equal(sitstand.build_command_obs(semantic_frame), np.zeros(13, dtype=np.float32))


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise AssertionError(f"worker exited without reply: {process.stderr.read()}")
    reply = json.loads(line)
    assert reply["ok"], reply
    return reply


def start(policy_path, model_path, stand_path=None):
    command = [sys.executable, "-I", "-u", "-c", worker.read_text(),
               str(policy_path), str(model_path)]
    if stand_path is not None:
        command.append(str(stand_path))
    preexec_fn = None
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if SANDBOX:
        identity = pwd.getpwnam("robot-policy")
        command = [
            "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-net",
            "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--ro-bind", "/", "/",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run",
            "--chdir", "/", "--", "/usr/bin/setpriv", "--reuid", str(identity.pw_uid),
            "--regid", str(identity.pw_gid), "--clear-groups", "--no-new-privs",
            "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all", *command,
        ]

        def limits():
            for limit, value in ((resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_NPROC, 16),
                                 (resource.RLIMIT_FSIZE, 128 * 1024 * 1024),
                                 (resource.RLIMIT_AS, 1536 * 1024 * 1024),
                                 (resource.RLIMIT_CORE, 0)):
                resource.setrlimit(limit, (value, value))

        preexec_fn = limits
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, preexec_fn=preexec_fn,
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

# Every ONNX shipped in the daemon release must satisfy its slot-specific default consumer.
default_consumers = {
    "alpha_walking.onnx": ("default_locomotion_policy.py", "walk", None),
    "roller.onnx": ("default_locomotion_policy.py", "walk", None),
    "alpha_stand.onnx": ("default_locomotion_policy.py", "stand", None),
    "alpha_sitstand.onnx": ("default_sitstand_policy.py", "sit", None),
    "alpha_ground_pick.onnx": ("default_ground_pick_policy.py", "ground_pick", 0.25),
    "roller_crouch.onnx": ("default_ground_pick_policy.py", "ground_pick", 0.25),
    "ball_kick_left.onnx": ("default_kick_left_policy.py", "kick_left", None),
    "ball_kick_right.onnx": ("default_kick_right_policy.py", "kick_right", None),
    "roulade.onnx": ("default_roulade_policy.py", "roulade", None),
}
bundled_processes = []
try:
    for bundled_model in sorted((MICRODUCK / "policies").glob("*.onnx")):
        policy_name, action_name, phase = default_consumers[bundled_model.name]
        bundled_policy = MICRODUCK / "robotd/src" / policy_name
        policy_frame = dict(frame)
        policy_frame["policy_context"] = {
            "action": action_name, "phase": phase, "body_active": False,
        }
        process = start(bundled_policy, bundled_model)
        bundled_processes.append(process)
        try:
            receive(process)
            for action in ("reset", "step"):
                process.stdin.write(json.dumps({"action": action, "frame": policy_frame}) + "\n")
                process.stdin.flush()
                reply = receive(process)
            assert reply["ready"], bundled_model
            assert len(reply["targets"]) == 14, bundled_model
        except Exception:
            raise AssertionError(f"bundled policy failed: {bundled_model}\n{process.stderr.read()}")
finally:
    for process in bundled_processes:
        process.kill()
        process.wait()

if SANDBOX:
    print("Two-file policy sandbox: every bundled ONNX loaded concurrently within RLIMIT_NPROC=16")
    sys.exit(0)
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
        return {{"api_version":2,"period_us":40000,"required_sources":["joints"],
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
    v1_policy = directory / "policy-v1.py"
    v1_policy.write_text(dynamic_policy.read_text().replace('"api_version":2', '"api_version":1'))
    process = start(v1_policy, dynamic_model)
    try:
        rejected = json.loads(process.stdout.readline())
        assert not rejected["ok"] and "v1" in rejected["error"], rejected
    finally:
        process.kill(); process.wait()

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

    # Distinct networks make wrong routing and stale per-slot feedback observable. The next
    # model consumes the immediately preceding model's raw output and filtered target.
    pair_models = []
    for name, increment in (("walk", 1.0), ("stand", 10.0)):
        pair_graph = helper.make_graph(
            [helper.make_node("Add", ["obs", "increment"], ["actions"])], name,
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 14])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
            [helper.make_tensor("increment", TensorProto.FLOAT, [1, 14], [increment] * 14)],
        )
        generated = helper.make_model(pair_graph, opset_imports=[helper.make_opsetid("", 17)])
        generated.ir_version = 10
        checker.check_model(generated)
        path = directory / f"{name}.onnx"
        path.write_bytes(generated.SerializeToString())
        pair_models.append(path)
    paired_policy = directory / "shared.py"
    paired_policy.write_text(f'''import numpy as np
CONTROLLED = {controlled!r}
class Policy:
    def describe(self):
        return {{"api_version":2,"period_us":20000,"required_sources":["joints"],
          "inputs":{{"obs":{{"dtype":"float32","shape":[1,14]}}}},
          "outputs":{{"actions":{{"dtype":"float32","shape":[1,14]}}}},
          "controlled_joints":CONTROLLED}}
    def reset(self, robot_info, first_frame):
        self.last_action = np.zeros((1,14), dtype=np.float32)
        self.filtered = 0.0
    def preprocess(self, frame, feedback):
        if feedback is not None:
            assert np.array_equal(feedback["outputs"]["actions"], self.last_action)
            assert feedback["requested_targets"][CONTROLLED[0]]["position"] == self.filtered
        return {{"obs": self.last_action}}
    def postprocess(self, outputs, frame):
        self.last_action = outputs["actions"].copy()
        self.filtered = 0.5 * float(self.last_action[0,0]) + 0.5 * self.filtered
        return {{name:{{"position":self.filtered,"velocity":0.0,"torque_ff":0.0,"kp":20.0,"kd":1.0}} for name in CONTROLLED}}
''')
    process = start(paired_policy, *pair_models)
    try:
        receive(process)
        process.stdin.write(json.dumps({"action": "reset", "frame": frame}) + "\n")
        process.stdin.flush(); receive(process)
        raw = filtered = 0.0
        for slot in ("walk", "stand", "walk", "stand", "stand", "walk"):
            raw += 1.0 if slot == "walk" else 10.0
            filtered = 0.5 * raw + 0.5 * filtered
            pair_frame = dict(frame)
            pair_frame["policy_context"] = {
                "action": slot, "phase": None, "body_active": False,
            }
            process.stdin.write(json.dumps({"action": "step", "model": slot, "frame": pair_frame}) + "\n")
            process.stdin.flush()
            reply = receive(process)
            assert reply["targets"][controlled[0]]["position"] == filtered, (slot, reply, filtered)
        process.stdin.write(json.dumps({"action": "reset", "frame": frame}) + "\n")
        process.stdin.flush(); receive(process)
        stand_frame = dict(frame)
        stand_frame["policy_context"] = {
            "action": "stand", "phase": None, "body_active": False,
        }
        process.stdin.write(json.dumps({"action": "step", "model": "stand", "frame": stand_frame}) + "\n")
        process.stdin.flush()
        assert receive(process)["targets"][controlled[0]]["position"] == 5.0
    finally:
        process.kill(); process.wait()

    # Reject either incompatible peer before readiness, never silently run a one-sided pair.
    for models in ((pair_models[0], dynamic_model), (dynamic_model, pair_models[1])):
        process = start(paired_policy, *models)
        try:
            reply = json.loads(process.stdout.readline())
            assert not reply["ok"] and "incompatible" in reply["error"], reply
        finally:
            process.kill(); process.wait()

print("Two-file policy: bundled ONNX, dynamic multi-IO, shared walk/stand feedback/filter, reset and incompatible-peer rejection passed")
