"""API-v2 B handoff model_2499 consumer: 2500 updates; no head governor.
Standalone NumPy consumer. Keep state across sit/rise commands.
Use current joint velocity feedback; no extra policy-frame velocity delay.
"""

import numpy as np

JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
]
CONTROLLED = [name for name in JOINTS if name != "mouth"]
TRAINING_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
TRAINING_INDEX = [JOINTS.index(name) for name in TRAINING_JOINTS]

# Exact values supplied in policy_io.json, in training order.
DEFAULT_ANGLES = np.asarray(
    [
        0,
        0.07999999821186066,
        0.39399999380111694,
        -0.061000000685453415,
        -0.45500001311302185,
        0.01899999938905239,
        0.05999999865889549,
        0,
        0,
        0,
        -0.07999999821186066,
        -0.39399999380111694,
        0.061000000685453415,
        0.45500001311302185,
    ],
    dtype=np.float32,
)
TARGET_LOW = np.asarray(
    [
        -1.19,
        -0.11000000178813935,
        -2.990000003198883,
        -1.1900000006854534,
        -1.4900000161130218,
        -0.2900000006109476,
        -1.1900000013411045,
        -1.99,
        -0.49,
        -0.49,
        -1.1899999982118608,
        -2.989999996801117,
        -1.4899999993145465,
        -1.2899999838869782,
    ],
    dtype=np.float32,
)
TARGET_HIGH = np.asarray(
    [
        0.49,
        1.4899999982118606,
        2.989999996801117,
        1.3899999993145467,
        1.2899999838869782,
        1.2899999993890523,
        1.1899999986588954,
        1.99,
        0.49,
        1.19,
        0.19000000178813936,
        1.9900000031988831,
        1.2900000006854535,
        1.4900000161130218,
    ],
    dtype=np.float32,
)
INCREMENT = np.asarray(
    [
        0.015,
        0.010,
        0.080,
        0.060,
        0.075,
        0.110,
        0.025,
        0.025,
        0.010,
        0.015,
        0.010,
        0.080,
        0.060,
        0.075,
    ],
    dtype=np.float32,
)
HISTORY_SCALE = 0.12
KP = np.asarray([90, 90, 75, 100, 90, 75, 90, 75, 90, 90, 90, 75, 100, 90], dtype=np.float32)
KD = np.asarray([2.5, 2.5, 2.5, 3, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3, 3], dtype=np.float32)
CONTROL_DT = 0.02


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
            raise ValueError("Joint order does not match the API-v2 contract")
        # Training reset seeds the integrator from actual measured positions.
        positions = np.asarray(first_frame["positions"], dtype=np.float32)
        velocities = np.asarray(first_frame["velocities"], dtype=np.float32)
        # The supplied reference_adapter.initialize_target also clips the
        # measured initial target to the policy bounds before integration.
        self.target = np.clip(positions[TRAINING_INDEX], TARGET_LOW, TARGET_HIGH)
        self.previous_velocity = velocities[TRAINING_INDEX].copy()
        if not np.isfinite(self.target).all() or not np.isfinite(self.previous_velocity).all():
            raise ValueError("Initial joint feedback must be finite")

    def preprocess(self, frame, feedback):
        if feedback is not None:
            targets = feedback.get("requested_targets", {})
            if "requested_targets" in feedback and not all(
                name in targets for name in TRAINING_JOINTS
            ):
                raise ValueError("Handoff feedback must contain all 14 requested targets")
            if all(name in targets for name in TRAINING_JOINTS):
                target = np.asarray(
                    [targets[name]["position"] for name in TRAINING_JOINTS], dtype=np.float32
                )
                if not np.isfinite(target).all():
                    raise ValueError("Previous commanded targets must be finite")
                self.target = np.clip(target, TARGET_LOW, TARGET_HIGH)
        mode = frame["policy_context"]["action"]
        if mode not in ("sit", "rise"):
            raise ValueError("Incremental SitStand requires sit or rise")
        q = np.asarray(frame["positions"], dtype=np.float32)[TRAINING_INDEX]
        qd = np.asarray(frame["velocities"], dtype=np.float32)[TRAINING_INDEX]
        commands = np.zeros(13, dtype=np.float32)
        commands[0] = 1.0 if mode == "sit" else 0.0
        # Head channels are desired offsets from DEFAULT_ANGLES, in radians.
        commands[3:7] = np.asarray(frame["command"]["head"], dtype=np.float32)
        # Body and other velocity commands were zero in this training task.
        history = (self.target - DEFAULT_ANGLES) / HISTORY_SCALE
        obs = np.concatenate(
            (
                np.asarray(frame["imu"]["gyro"], dtype=np.float32),
                np.asarray(frame["imu"]["gravity"], dtype=np.float32),
                q - DEFAULT_ANGLES,
                qd,
                history,
                commands,
            )
        ).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Observation must be finite")
        self.previous_velocity = qd.copy()
        return {"obs": obs.reshape(1, 61)}

    def postprocess(self, outputs, frame):
        del frame
        # Tanh is evaluated in the network's float32 action dtype, matching
        # supplied reference_adapter decoding; target state is also float32.
        action = np.asarray(outputs["actions"], dtype=np.float32).reshape(14)
        if not np.isfinite(action).all():
            raise ValueError("Model output must be finite")
        self.target = np.clip(self.target + INCREMENT * np.tanh(action), TARGET_LOW, TARGET_HIGH)
        # This is already the absolute target. No extra .12 scale, reference
        # addition, EMA or phase schedule may be applied by the caller.
        by_name = {
            name: {
                "position": float(self.target[i]),
                "velocity": 0.0,
                "torque_ff": 0.0,
                "kp": float(KP[i]),
                "kd": float(KD[i]),
            }
            for i, name in enumerate(TRAINING_JOINTS)
        }
        return {name: by_name[name] for name in CONTROLLED}
