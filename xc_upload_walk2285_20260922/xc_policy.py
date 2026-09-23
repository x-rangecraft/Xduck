"""API-v2 consumer shared by the XDuck 1.1.2 Walk and Stand slots."""
import numpy as np

# robotd frame order: left leg, right leg, head, mouth.
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
CONTROLLED = [name for name in JOINTS if name != "mouth"]

# Training/action order: left leg, head, right leg.
TRAINING_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
TRAINING_INDEX = [JOINTS.index(name) for name in TRAINING_JOINTS]

# GAIT_REFERENCE (Walk) and STAND (Stand) are numerically identical.
REFERENCE = np.asarray([
    0.0, 0.0, 0.384, 0.0, -0.384,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.384, 0.0, 0.384,
], dtype=np.float32)

# V1.1.2 hard limits with the same 0.01 rad margin as both training envs.
TARGET_LOW = np.asarray([
    -1.19, -0.11, -2.99, -1.19, -1.49, -0.29, -1.19,
    -1.99, -0.49, -0.49, -1.19, -2.99, -1.49, -1.29,
], dtype=np.float32)
TARGET_HIGH = np.asarray([
    0.49, 1.49, 2.99, 1.39, 1.29, 1.29, 1.19,
    1.99, 0.49, 1.19, 0.19, 1.99, 1.29, 1.49,
], dtype=np.float32)

# Exact per-joint gains from both selected checkpoints' run_config.json.
KP = np.asarray(
    [90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90],
    dtype=np.float32,
)
KD = np.asarray(
    [2.5, 2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5,
     2.5, 2.5, 2.5, 2.5, 3.0, 3.0],
    dtype=np.float32,
)

ACTION_SCALE = 0.12
ACTION_FILTER_ALPHA = 1.0  # Both simulation configs have no action filter.
CONTROL_DT = 0.02


def build_command_obs(frame):
    """Build the training 13D command from API-v2's unencoded command fields."""
    context = frame["policy_context"]
    action = context["action"]
    if action not in ("walk", "stand"):
        raise ValueError("locomotion consumer requires walk or stand action")
    if action == "stand":
        # XDuckWalkStopEnv fixes the entire 13D command to zero.
        return np.zeros(13, dtype=np.float32)
    command = frame["command"]
    twist = [0.0, 0.0, 0.0] if context["body_active"] else command["twist"]
    # body from API is [z, roll, pitch]; training uses [x,y,z,roll,pitch,yaw].
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
            "inputs": {"obs": {"dtype": "float32", "shape": [1, 61]}},
            "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
            "controlled_joints": CONTROLLED,
        }

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS:
            raise ValueError("joint order does not match the XDuck API-v2 contract")
        self.last_action = np.zeros(14, dtype=np.float32)
        first_velocity = np.asarray(first_frame["velocities"], dtype=np.float32)
        self.previous_velocity = first_velocity[TRAINING_INDEX].copy()

    def preprocess(self, frame, feedback):
        del feedback
        positions = np.asarray(frame["positions"], dtype=np.float32)[TRAINING_INDEX]
        current_velocity = np.asarray(frame["velocities"], dtype=np.float32)[TRAINING_INDEX]
        # Training delays native motor velocity feedback by exactly one 20 ms frame.
        delayed_velocity = self.previous_velocity
        self.previous_velocity = current_velocity.copy()
        obs = np.concatenate([
            np.asarray(frame["imu"]["gyro"], dtype=np.float32),
            np.asarray(frame["imu"]["gravity"], dtype=np.float32),
            positions - REFERENCE,
            delayed_velocity,
            self.last_action,
            build_command_obs(frame),
        ]).astype(np.float32)
        return {"obs": obs.reshape(1, 61)}

    def postprocess(self, outputs, frame):
        del frame
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        # ONNX emits raw action. Scaling occurs exactly once here.
        target = np.clip(
            REFERENCE + ACTION_SCALE*action,
            TARGET_LOW,
            TARGET_HIGH,
        )
        self.last_action = action.copy()  # raw, unfiltered and unclipped history
        by_name = {}
        for slot, name in enumerate(TRAINING_JOINTS):
            by_name[name] = {
                "position": float(target[slot]),
                "velocity": 0.0,
                "torque_ff": 0.0,
                "kp": float(KP[slot]),
                "kd": float(KD[slot]),
            }
        # Return platform order; controlled_joints itself is order-independent.
        return {name: by_name[name] for name in CONTROLLED}
