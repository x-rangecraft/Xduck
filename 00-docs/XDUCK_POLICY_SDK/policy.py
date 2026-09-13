"""Two-file policy API v1 example for alpha_walking.onnx.

Copy this file and the ONNX before editing. It is a developer example, not an automatic
replacement for the built-in controller.
"""
import numpy as np


JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
CONTROLLED = [name for name in JOINTS if name != "mouth"]
HOME = np.asarray([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)
CONTROLLED_INDEX = [index for index, name in enumerate(JOINTS) if name != "mouth"]


class Policy:
    def describe(self):
        return {
            "api_version": 1,
            "period_us": 20_000,
            "required_sources": ["joints", "imu", "command"],
            "inputs": {"obs": {"dtype": "float32", "shape": [1, 61]}},
            "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
            "controlled_joints": CONTROLLED,
        }

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS:
            raise ValueError("joint order does not match this example")
        self.last_action = np.zeros(14, dtype=np.float32)
        self.previous_targets = None

    def preprocess(self, frame, feedback):
        positions = np.asarray(frame["positions"], dtype=np.float32)[CONTROLLED_INDEX]
        velocities = np.asarray(frame["velocities"], dtype=np.float32)[CONTROLLED_INDEX]
        command = frame["command"]
        body = command["body"]
        command_block = np.asarray([
            *command["twist"], *command["head"], 0.0, 0.0,
            body[0], body[1], body[2], 0.0,
        ], dtype=np.float32)
        obs = np.concatenate([
            np.asarray(frame["imu"]["gyro"], dtype=np.float32),
            np.asarray(frame["imu"]["gravity"], dtype=np.float32),
            positions - HOME[CONTROLLED_INDEX],
            velocities,
            self.last_action,
            command_block,
        ])
        return {"obs": obs.reshape(1, 61)}

    def postprocess(self, outputs, frame):
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        raw = HOME[CONTROLLED_INDEX] + 0.9 * action
        if self.previous_targets is not None:
            filtered = raw.copy()
            for slot, joint_index in enumerate(CONTROLLED_INDEX):
                alpha = 0.5 if 5 <= joint_index < 9 else 0.7
                filtered[slot] = alpha * raw[slot] + (1.0 - alpha) * self.previous_targets[slot]
            raw = filtered
        self.last_action = action.copy()
        self.previous_targets = raw.copy()
        return {
            name: {
                "position": float(raw[slot]),
                "velocity": 0.0,
                "torque_ff": 0.0,
                "kp": 60.0,
                "kd": 4.0,
            }
            for slot, name in enumerate(CONTROLLED)
        }
