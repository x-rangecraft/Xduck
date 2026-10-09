"""API-v2 consumer for XDuck V1.1.2 five-pose calibrated model_63 Sit/Rise policy."""
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

# STAND reference from the supplied training configuration's XDuck V1.1.2 task.
REFERENCE = np.asarray([
    0.0, 0.0, 0.384, 0.0, -0.384,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.384, 0.0, 0.384,
], dtype=np.float32)

# V1.1.2 hard limits with 0.01 rad safety margin.
TARGET_LOW = np.asarray([
    -1.19, -0.11, -2.99, -1.19, -1.49, -0.29, -1.19,
    -1.99, -0.49, -0.49, -1.19, -2.99, -1.49, -1.29,
], dtype=np.float32)
TARGET_HIGH = np.asarray([
    0.49, 1.49, 2.99, 1.39, 1.29, 1.29, 1.19,
    1.99, 0.49, 1.19, 0.19, 1.99, 1.29, 1.49,
], dtype=np.float32)

KP = np.asarray(
    [90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90],
    dtype=np.float32,
)
KD = np.asarray(
    [2.5, 2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5,
     2.5, 2.5, 2.5, 2.5, 3.0, 3.0],
    dtype=np.float32,
)

ACTION_SCALE = 1.0
CONTROL_DT = 0.02
TRANSITION_SECONDS = 4.0
ACTION_FILTER_ALPHA = 1.0  # CorrectiveTimingActor owns its learned action-history correction.


class Policy:
    def describe(self):
        return {
            "api_version": 2,
            "period_us": 20_000,
            "required_sources": ["joints", "imu", "policy_context"],
            "inputs": {"obs": {"dtype": "float32", "shape": [1, 62]}},
            "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
            "controlled_joints": CONTROLLED,
        }

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS:
            raise ValueError("joint order does not match the XDuck API-v2 contract")
        self._start_action(first_frame)

    def _start_action(self, frame):
        action = frame["policy_context"]["action"]
        if action not in ("sit", "rise"):
            raise ValueError("SitStand policy requires sit or rise action")
        self.action = action
        self.command = 1.0 if action == "sit" else 0.0
        self.phase = 0.0
        self.last_action = np.zeros(14, dtype=np.float32)
        velocity = np.asarray(frame["velocities"], dtype=np.float32)
        self.previous_velocity = velocity[TRAINING_INDEX].copy()

    def preprocess(self, frame, feedback):
        del feedback
        # Normal action entries reset the instance, but robotd's atomic import
        # preheater may exercise both directions on one instance. Make the
        # observed direction change equivalent to a fresh action-cycle reset.
        if frame["policy_context"]["action"] != self.action:
            self._start_action(frame)
        positions = np.asarray(frame["positions"], dtype=np.float32)[TRAINING_INDEX]
        current_velocity = np.asarray(frame["velocities"], dtype=np.float32)[TRAINING_INDEX]
        delayed_velocity = self.previous_velocity
        self.previous_velocity = current_velocity.copy()
        command = np.zeros(13, dtype=np.float32)
        command[0] = self.command
        obs = np.concatenate([
            np.asarray(frame["imu"]["gyro"], dtype=np.float32),
            np.asarray(frame["imu"]["gravity"], dtype=np.float32),
            positions-REFERENCE,
            delayed_velocity,
            self.last_action,
            command,
            np.asarray([self.phase], dtype=np.float32),
        ]).astype(np.float32)
        dt = float(frame["dt"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("frame.dt must be positive and finite")
        self.phase = min(1.0, self.phase + dt/TRANSITION_SECONDS)
        return {"obs": obs.reshape(1, 62)}

    def postprocess(self, outputs, frame):
        del frame
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        target = np.clip(
            REFERENCE + ACTION_SCALE*action,
            TARGET_LOW,
            TARGET_HIGH,
        )
        self.last_action = action.copy()
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
