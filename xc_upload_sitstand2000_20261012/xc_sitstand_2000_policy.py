"""API-v2 incremental Sit/Rise consumer for model_1999 (2000 PPO updates)."""
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

# Exact values supplied in policy_io.json, in training order.
DEFAULT_ANGLES = np.asarray([
    0, 0, .594845, .005, -.589848, -.25, -.349066, 0, 0,
    0, 0, -.594845, -.005, .589848,
], dtype=np.float64)
TARGET_LOW = np.asarray([
    -1.19, -.11, -2.7791550000000003, -1.185, -1.6958480000000002,
    -.54, -1.539066, -1.99, -.49, -.49, -1.19,
    -3.2008449999999997, -1.4949999999999999, -1.084152,
], dtype=np.float64)
TARGET_HIGH = np.asarray([
    .49, 1.49, 3.2008449999999997, 1.3949999999999998, 1.084152,
    1.04, .840934, 1.99, .49, 1.19, .19,
    1.7791550000000003, 1.2850000000000001, 1.6958480000000002,
], dtype=np.float64)
INCREMENT = np.asarray([
    .015, .010, .080, .060, .075, .110, .025, .025, .010,
    .015, .010, .080, .060, .075,
], dtype=np.float64)
HISTORY_SCALE = .12
KP = np.asarray([90,90,75,100,90,75,90,75,90,90,90,75,100,90], dtype=np.float64)
KD = np.asarray([2.5,2.5,2.5,3,3,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3,3], dtype=np.float64)
CONTROL_DT = .02


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
            raise ValueError("Joint order does not match the API-v2 contract")
        # Training reset seeds the integrator from actual measured positions.
        positions = np.asarray(first_frame["positions"], dtype=np.float64)
        velocities = np.asarray(first_frame["velocities"], dtype=np.float64)
        self.target = positions[TRAINING_INDEX].copy()
        self.previous_velocity = velocities[TRAINING_INDEX].copy()
        if not np.isfinite(self.target).all() or not np.isfinite(self.previous_velocity).all():
            raise ValueError("Initial joint feedback must be finite")

    def preprocess(self, frame, feedback):
        del feedback
        mode = frame["policy_context"]["action"]
        if mode not in ("sit", "rise"):
            raise ValueError("Incremental SitStand requires sit or rise")
        q = np.asarray(frame["positions"], dtype=np.float64)[TRAINING_INDEX]
        qd = np.asarray(frame["velocities"], dtype=np.float64)[TRAINING_INDEX]
        commands = np.zeros(13, dtype=np.float64)
        commands[0] = 1.0 if mode == "sit" else 0.0
        # Head channels are desired offsets from DEFAULT_ANGLES, in radians.
        commands[3:7] = np.asarray(frame["command"]["head"], dtype=np.float64)
        # Body and other velocity commands were zero in this training task.
        history = (self.target-DEFAULT_ANGLES)/HISTORY_SCALE
        obs = np.concatenate((
            np.asarray(frame["imu"]["gyro"], dtype=np.float64),
            np.asarray(frame["imu"]["gravity"], dtype=np.float64),
            q-DEFAULT_ANGLES, self.previous_velocity, history, commands,
        )).astype(np.float32)
        self.previous_velocity = qd.copy()
        return {"obs": obs.reshape(1,61)}

    def postprocess(self, outputs, frame):
        del frame
        # Tanh is evaluated in the network's float32 action dtype, matching
        # NumPy simulation decoding; target integration uses float64 state.
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        if not np.isfinite(action).all():
            raise ValueError("Model output must be finite")
        self.target = np.clip(self.target+INCREMENT*np.tanh(action), TARGET_LOW, TARGET_HIGH)
        # This is already the absolute target. No extra .12 scale, reference
        # addition, EMA or phase schedule may be applied by the caller.
        by_name = {
            name: {"position": float(self.target[i]), "velocity": 0.0,
                   "torque_ff": 0.0, "kp": float(KP[i]), "kd": float(KD[i])}
            for i, name in enumerate(TRAINING_JOINTS)
        }
        return {name: by_name[name] for name in CONTROLLED}
