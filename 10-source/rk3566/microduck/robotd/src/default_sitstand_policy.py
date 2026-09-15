"""Built-in API-v2 consumer for the shared Sit and Rise model."""
import numpy as np


JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
CONTROLLED = [name for name in JOINTS if name != "mouth"]
HOME = np.asarray([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.0, 0.0873, 0.4579,
                   0.0049, -0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0], dtype=np.float32)
CONTROLLED_INDEX = [index for index, name in enumerate(JOINTS) if name != "mouth"]
ACTION_SCALE = 0.9
KP = 60.0
KD = 4.0


def build_command_obs(frame):
    action = frame["policy_context"]["action"]
    if action == "sit":
        return np.asarray([1.0, *([0.0] * 12)], dtype=np.float32)
    if action == "rise":
        return np.zeros(13, dtype=np.float32)
    raise ValueError("sitstand consumer requires sit or rise action")


class Policy:
    def describe(self):
        return {"api_version": 2, "period_us": 20_000,
                "required_sources": ["joints", "imu", "command", "policy_context"],
                "inputs": {"obs": {"dtype": "float32", "shape": [1, 61]}},
                "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
                "controlled_joints": CONTROLLED}

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS: raise ValueError("joint order mismatch")
        self.last_action = np.zeros(14, dtype=np.float32)
        self.previous_targets = None

    def preprocess(self, frame, feedback):
        positions = np.asarray(frame["positions"], dtype=np.float32)[CONTROLLED_INDEX]
        velocities = np.asarray(frame["velocities"], dtype=np.float32)[CONTROLLED_INDEX]
        obs = np.concatenate([np.asarray(frame["imu"]["gyro"], dtype=np.float32),
                              np.asarray(frame["imu"]["gravity"], dtype=np.float32),
                              positions - HOME[CONTROLLED_INDEX], velocities, self.last_action,
                              build_command_obs(frame)])
        return {"obs": obs.reshape(1, 61)}

    def postprocess(self, outputs, frame):
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        raw = HOME[CONTROLLED_INDEX] + ACTION_SCALE * action
        if self.previous_targets is not None:
            filtered = raw.copy()
            for slot, joint_index in enumerate(CONTROLLED_INDEX):
                alpha = 0.5 if 10 <= joint_index < 14 else 0.7
                filtered[slot] = alpha * raw[slot] + (1.0 - alpha) * self.previous_targets[slot]
            raw = filtered
        self.last_action = action.copy(); self.previous_targets = raw.copy()
        return {name: {"position": float(raw[slot]), "velocity": 0.0, "torque_ff": 0.0,
                       "kp": KP, "kd": KD} for slot, name in enumerate(CONTROLLED)}
