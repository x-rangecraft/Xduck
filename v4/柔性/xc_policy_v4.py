"""API-v2 v4: original-reference flex_adapt_50 Walk; new-reference Stand 650."""
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

WALK_REFERENCE = np.asarray([
    0.0, 0.0, 0.384, 0.0, -0.384,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.384, 0.0, 0.384,
], dtype=np.float32)
STAND_REFERENCE = np.asarray([
    0.0, 0.08, 0.394, -0.061, -0.455,
    0.019, 0.06, 0.0, 0.0,
    0.0, -0.08, -0.394, 0.061, 0.455,
], dtype=np.float32)
TARGET_LOW = np.asarray([
    -1.19, -0.11, -2.99, -1.19, -1.49, -0.29, -1.19,
    -1.99, -0.49, -0.49, -1.19, -2.99, -1.49, -1.29,
], dtype=np.float32)
TARGET_HIGH = np.asarray([
    0.49, 1.49, 2.99, 1.39, 1.29, 1.29, 1.19,
    1.99, 0.49, 1.19, 0.19, 1.99, 1.29, 1.49,
], dtype=np.float32)

# flex_adapt_50 action-to-radian mapping in training joint order.
WALK_ACTION_SCALE = np.asarray([
    0.12, 0.03666666666666667, 0.12, 0.12, 0.12,
    0.09666666666666666, 0.12, 0.12, 0.12,
    0.12, 0.03666666666666667, 0.12, 0.12, 0.12,
], dtype=np.float32)
# Stand 650 uses its new reference and a uniform 0.12-rad mapping.
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
        raise ValueError("v4 locomotion policy requires walk or stand action")
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
        self.last_raw_action = np.zeros(14, dtype=np.float32)
        self.last_absolute_target = None
        self.last_mode = None
        velocity = np.asarray(first_frame["velocities"], dtype=np.float32)
        self.previous_velocity = velocity[TRAINING_INDEX].copy()

    @staticmethod
    def action_scale(frame):
        action = frame["policy_context"]["action"]
        if action == "walk":
            return WALK_ACTION_SCALE
        if action == "stand":
            return STAND_ACTION_SCALE
        raise ValueError("v4 locomotion policy requires walk or stand action")

    @staticmethod
    def reference(frame):
        action = frame["policy_context"]["action"]
        if action == "walk":
            return WALK_REFERENCE
        if action == "stand":
            return STAND_REFERENCE
        raise ValueError("v4 locomotion policy requires walk or stand action")

    def preprocess(self, frame, feedback):
        del feedback
        positions = np.asarray(frame["positions"], dtype=np.float32)[TRAINING_INDEX]
        current_velocity = np.asarray(frame["velocities"], dtype=np.float32)[TRAINING_INDEX]
        delayed_velocity = self.previous_velocity
        self.previous_velocity = current_velocity.copy()
        scale = self.action_scale(frame)
        reference = self.reference(frame)
        mode = frame["policy_context"]["action"]
        if self.last_mode is None:
            last_action = np.zeros(14, dtype=np.float32)
        elif mode == self.last_mode:
            last_action = self.last_raw_action
        else:
            # Preserve the preceding unclipped physical target while expressing
            # history in the newly selected model's reference and action units.
            last_action = (self.last_absolute_target-reference)/scale
        obs = np.concatenate([
            np.asarray(frame["imu"]["gyro"], dtype=np.float32),
            np.asarray(frame["imu"]["gravity"], dtype=np.float32),
            positions-reference,
            delayed_velocity,
            last_action,
            command_obs(frame),
        ]).astype(np.float32)
        return {"obs": obs.reshape(1,61)}

    def postprocess(self, outputs, frame):
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        scale = self.action_scale(frame)
        absolute_target = self.reference(frame)+scale*action
        target = np.clip(absolute_target, TARGET_LOW, TARGET_HIGH)
        # Training history retains raw actions, even when targets saturate.
        self.last_raw_action = action.copy()
        self.last_absolute_target = absolute_target.copy()
        self.last_mode = frame["policy_context"]["action"]
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
