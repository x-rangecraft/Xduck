"""XDuck API v2 consumer for F703 Walk + R2 Stand (COM35, HOME384).

Pair only with the packaged F703 Walk and R2 Stand ONNX files (obs -> actions, float32, 61 -> 14).
Commands and experiment durations come from the platform, never this file.
The 10 ms actuator response and smooth friction belong to the simulated plant;
we do not add artificial response lag, friction or sensor noise on hardware.
Request guards run at 50 Hz here, versus 500 Hz for the sim's limit governor.
Nominal torque projection is a request guard, not a hardware torque guarantee.
"""

import numpy as np

JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
CONTROLLED = JOINTS[:-1]
TRAINING_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
TRAINING_INDEX = [JOINTS.index(name) for name in TRAINING_JOINTS]
REFERENCE = np.asarray([
    0, 0, .384, 0, -.384, 0, 0, 0, 0, 0, 0, -.384, 0, .384,
], dtype=np.float32)
HARD_LIMITS = np.asarray([
    [-1.2, .5], [-.12, 1.5], [-3, 3], [-1.2, 1.4], [-1.5, 1.3],
    [-.3, 1.3], [-1.2, 1.2], [-2, 2], [-.5, .5],
    [-.5, 1.2], [-1.2, .2], [-3, 2], [-1.5, 1.3], [-1.3, 1.5],
], dtype=np.float32)
SAFE_LOW = HARD_LIMITS[:, 0] + np.float32(np.deg2rad(5))
SAFE_HIGH = HARD_LIMITS[:, 1] - np.float32(np.deg2rad(5))
WALK_LOW = SAFE_LOW.copy()
WALK_HIGH = SAFE_HIGH.copy()
WALK_LOW[10] = HARD_LIMITS[10, 0] + np.deg2rad(10)
WALK_HIGH[1] = HARD_LIMITS[1, 1] - np.deg2rad(10)

# Owner: unilab.actuators.gf43x, GF43X40_10 nominal measured motoring envelope.
CURVE_SPEED = np.asarray([
    0, 1.102076109287, 1.335039450506, 1.771432309589, 2.170334154035,
    2.534009883643, 2.864724398216, 3.164742597554, 3.204424506662,
    3.319616237293, 3.424335992413, 3.539527723044, 3.62330352714,
    3.686135380212, 3.801327110844, 3.895574890451, 3.906046865963,
    4.031710572107, 4.105014400691, 4.188790204786, 4.251622057858,
    4.293509959906, 4.345869837466, 4.429645641562, 4.565781323217,
    4.649557127313, 4.701917004873, 4.764748857945, 4.79616478448,
    4.890412564088, 4.974188368184, 5.05796417228, 5.183627878423,
    5.309291584567, 5.445427266222, 5.759586531581, 5.809523809523809,
], dtype=np.float64)
CURVE_TORQUE = np.asarray([
    23.5, 23.5, 23, 22, 21, 20, 19, 18, 17.86, 17.54, 17.17, 16.71,
    16.44, 16.09, 15.67, 15.26, 14.91, 14.59, 14.14, 13.8, 13.39,
    12.99, 12.64, 12.3, 11.53, 10.74, 10.36, 9.98, 9.61, 8.89,
    8.17, 7.79, 6.23, 5.08, 4.35, 1.14, 0,
], dtype=np.float64)


def finite_array(value, shape, label):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def frame_joints(frame):
    if frame["joint_names"] != JOINTS:
        raise ValueError("joint order does not match XDuck API v2")
    q = finite_array(frame["positions"], (15,), "positions")[TRAINING_INDEX]
    dq = finite_array(frame["velocities"], (15,), "velocities")[TRAINING_INDEX]
    return q, dq


def slot(frame):
    action = frame["policy_context"]["action"]
    if action not in ("walk", "stand"):
        raise ValueError("this consumer requires the walk or stand slot")
    return action


def build_command_obs(frame):
    command = frame["command"]
    twist = finite_array(command["twist"], (3,), "twist")
    head = finite_array(command["head"], (4,), "head")
    body = finite_array(command["body"], (3,), "body")
    if slot(frame) == "stand":
        return np.zeros(13, dtype=np.float32)
    if frame["policy_context"]["body_active"]:
        twist = np.zeros(3, dtype=np.float32)
    return np.concatenate((twist, head, [0, 0], body, [0])).astype(np.float32)


def nominal_torque_bounds(velocity):
    speed = np.abs(velocity)
    nominal = np.interp(speed, CURVE_SPEED, CURVE_TORQUE).astype(np.float32)
    power_bound = np.divide(70., speed, out=np.full_like(speed, np.inf), where=speed > 0)
    upper = np.minimum(23.5, power_bound)
    taper = np.where(speed <= CURVE_SPEED[-2], 1., np.clip(
        (CURVE_SPEED[-1] - speed) / (CURVE_SPEED[-1] - CURVE_SPEED[-2]), 0., 1.,
    ))
    motoring = np.minimum(nominal, upper * taper)
    low = np.where(velocity > 0, -8.9, np.where(velocity < 0, -motoring, -23.5))
    high = np.where(velocity > 0, motoring, np.where(velocity < 0, 8.9, 23.5))
    return low.astype(np.float32) * .98, high.astype(np.float32) * .98


def project_request(action, q, dq, action_slot):
    kp, kd = 30., 3. if action_slot == "walk" else 2.
    # Training action_scale=1.0, no action low-pass filter.
    requested = np.clip(REFERENCE + action, HARD_LIMITS[:, 0], HARD_LIMITS[:, 1])
    torque_low, torque_high = nominal_torque_bounds(dq)
    torque = np.clip((requested - q) * kp - dq * kd, torque_low, torque_high)
    projected = q + (torque + kd * dq) / kp
    low, high = (WALK_LOW, WALK_HIGH) if action_slot == "walk" else (SAFE_LOW, SAFE_HIGH)
    safe = np.clip(projected, low, high)
    predicted = q + .25 * dq
    upper_risk = np.where(dq > 0, np.maximum(predicted - high, 0), 0)
    lower_risk = np.where(dq < 0, np.maximum(low - predicted, 0), 0)
    return np.clip(safe + lower_risk - upper_risk, low, high), kp, kd


class Policy:
    def describe(self):
        return {
            "api_version": 2,
            "period_us": 20_000,
            "required_sources": ["joints", "imu", "command", "policy_context"],
            "inputs": {"obs": {"dtype": "float32", "shape": [1, 61]}},
            "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
            "controlled_joints": list(CONTROLLED),
        }

    def reset(self, robot_info, first_frame):
        if robot_info["joint_names"] != JOINTS:
            raise ValueError("joint order does not match XDuck API v2")
        frame_joints(first_frame)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.steps = 0

    def preprocess(self, frame, feedback):
        del feedback  # Actual motion comes from this frame, never requested targets.
        q, dq = frame_joints(frame)
        obs = np.concatenate((
            finite_array(frame["imu"]["gyro"], (3,), "gyro"),
            finite_array(frame["imu"]["gravity"], (3,), "gravity"),
            q - REFERENCE, dq, self.last_action, build_command_obs(frame),
        )).astype(np.float32)
        # Platform IMU is already in trunk coordinates. No extra mounting rotation.
        return {"obs": obs.reshape(1, 61)}

    def postprocess(self, outputs, frame):
        action = finite_array(outputs["actions"], (1, 14), "action")[0].copy()
        q, dq = frame_joints(frame)
        action_slot = slot(frame)
        if action_slot == "walk":
            max_delta = .01 if self.steps < 10 else .04
            indices = [0, 9]
            action[indices] = self.last_action[indices] + np.clip(
                action[indices] - self.last_action[indices], -max_delta, max_delta,
            )
        target, kp, kd = project_request(action, q, dq, action_slot)
        if not np.isfinite(target).all():
            raise ValueError("non-finite MIT position target")
        # Observation history stores hip-yaw-governed action, not projected target.
        # Stand stores raw action; shared slot changes keep history and startup age.
        self.last_action = action.copy()
        self.steps += 1
        by_name = {
            name: {"position": float(target[i]), "velocity": 0., "torque_ff": 0.,
                   "kp": kp, "kd": kd}
            for i, name in enumerate(TRAINING_JOINTS)
        }
        return {name: by_name[name] for name in CONTROLLED}
