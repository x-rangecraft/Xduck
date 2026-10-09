"""API-v2 shared Walk/Stand consumer for XDuck mean-only flex320 v3."""
import numpy as np

JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
CONTROLLED = [name for name in JOINTS if name != "mouth"]
TRAINING_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
TRAINING_INDEX = [JOINTS.index(name) for name in TRAINING_JOINTS]

REFERENCE = np.asarray([
    0.0, 0.0, 0.384, 0.0, -0.384,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.384, 0.0, 0.384,
], dtype=np.float32)
TARGET_LOW = np.asarray([
    -1.19, -0.11, -2.99, -1.19, -1.49, -0.29, -1.19,
    -1.99, -0.49, -0.49, -1.19, -2.99, -1.49, -1.29,
], dtype=np.float32)
TARGET_HIGH = np.asarray([
    0.49, 1.49, 2.99, 1.39, 1.29, 1.29, 1.19,
    1.99, 0.49, 1.19, 0.19, 1.99, 1.29, 1.49,
], dtype=np.float32)

# model_1279 HistoricalRigid action-to-radian mapping in training joint order.
WALK_ACTION_SCALE = np.asarray([
    0.12, 0.03666666666666667, 0.12, 0.12, 0.12,
    0.09666666666666666, 0.12, 0.12, 0.12,
    0.12, 0.03666666666666667, 0.12, 0.12, 0.12,
], dtype=np.float32)
# The paired Stand 599 policy was trained with a uniform 0.12-rad mapping.
STAND_ACTION_SCALE = np.full(14, 0.12, dtype=np.float32)

KP = np.asarray(
    [90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90],
    dtype=np.float32,
)
KD = np.asarray(
    [2.5, 2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5,
     2.5, 2.5, 2.5, 2.5, 3.0, 3.0],
    dtype=np.float32,
)

CONTROL_DT = 0.02
ACTION_FILTER_ALPHA = 1.0


def command_obs(frame):
    action = frame["policy_context"]["action"]
    if action == "stand":
        return np.zeros(13, dtype=np.float32)
    if action != "walk":
        raise ValueError("v3 locomotion policy requires walk or stand action")
    command = frame["command"]
    twist = [0.0,0.0,0.0] if frame["policy_context"]["body_active"] else command["twist"]
    return np.asarray([
        *twist,
        *command["head"],
        0.0, 0.0, *command["body"], 0.0,
    ], dtype=np.float32)


class Policy:
    def describe(self):
        return {
            "api_version": 2,
            "period_us": 20_000,
            "required_sources": ["joints", "imu", "command", "policy_context"],
            "inputs": {"obs": {"dtype": "float32", "shape": [1,61]}},
            "outputs": {"actions": {"dtype": "float32", "shape": [1,14]}},
            "controlled_joints": CONTROLLED,
        }

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS:
            raise ValueError("joint order does not match the XDuck API-v2 contract")
        self.last_physical_offset = np.zeros(14, dtype=np.float32)
        velocity = np.asarray(first_frame["velocities"], dtype=np.float32)
        self.previous_velocity = velocity[TRAINING_INDEX].copy()

    @staticmethod
    def action_scale(frame):
        action = frame["policy_context"]["action"]
        if action == "walk":
            return WALK_ACTION_SCALE
        if action == "stand":
            return STAND_ACTION_SCALE
        raise ValueError("v3 locomotion policy requires walk or stand action")

    def preprocess(self, frame, feedback):
        del feedback
        positions = np.asarray(frame["positions"], dtype=np.float32)[TRAINING_INDEX]
        current_velocity = np.asarray(frame["velocities"], dtype=np.float32)[TRAINING_INDEX]
        delayed_velocity = self.previous_velocity
        self.previous_velocity = current_velocity.copy()
        scale = self.action_scale(frame)
        last_action = self.last_physical_offset/scale
        obs = np.concatenate([
            np.asarray(frame["imu"]["gyro"], dtype=np.float32),
            np.asarray(frame["imu"]["gravity"], dtype=np.float32),
            positions-REFERENCE,
            delayed_velocity,
            last_action,
            command_obs(frame),
        ]).astype(np.float32)
        return {"obs": obs.reshape(1,61)}

    def postprocess(self, outputs, frame):
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        scale = self.action_scale(frame)
        physical_offset = scale*action
        target = np.clip(REFERENCE+physical_offset, TARGET_LOW, TARGET_HIGH)
        # Keep raw, unclipped physical offset so the next policy can recover
        # its own raw-action units across Walk/Stand scale changes.
        self.last_physical_offset = physical_offset.copy()
        by_name = {}
        for slot, name in enumerate(TRAINING_JOINTS):
            by_name[name] = {
                "position": float(target[slot]),
                "velocity": 0.0,
                "torque_ff": 0.0,
                "kp": float(KP[slot]),
                "kd": float(KD[slot]),
            }
        return {name: by_name[name] for name in CONTROLLED}
