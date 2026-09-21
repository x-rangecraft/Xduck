"""Selected GF walk reward terms with legacy names and scaling conventions."""

import numpy as np


def _robot(env):
    return env.scene["robot"]


def _command(env):
    return env.command_manager.get_command("walk")


def _moving(env):
    cmd = _command(env)
    return np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2]) > 0.01


def pose(env):
    joint_rel = _robot(env).data.joint_pos - _robot(env).data.default_joint_pos
    leg_ids = np.array((0, 1, 2, 3, 4, 9, 10, 11, 12, 13))
    standing_std = np.asarray([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
    walking_std = np.asarray([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
    std = np.where(_moving(env)[:, None], walking_std, standing_std)
    return np.exp(-np.mean(np.square(joint_rel[:, leg_ids] / std), axis=1))


def upright(env, std=0.22360679774997896):
    up = -_robot(env).data.projected_gravity_b
    return np.exp(-np.sum(np.square(up[:, :2]), axis=1) / std**2)


def track_linear_velocity(env, std=0.4571651780264985):
    velocity = _robot(env).data.root_link_lin_vel_b
    cmd = _command(env)
    error = np.column_stack((velocity[:, :2] - cmd[:, :2], velocity[:, 2]))
    return np.exp(-np.sum(np.square(error), axis=1) / std**2)


def track_angular_velocity(env, std=0.4891159880445185):
    gyro = _robot(env).data.root_link_ang_vel_b
    cmd = _command(env)
    error = np.column_stack((gyro[:, :2], gyro[:, 2] - cmd[:, 2]))
    return np.exp(-np.sum(np.square(error), axis=1) / std**2)


def body_ang_vel(env):
    # Legacy uses trunk world-frame angular velocity. Entity root transform is
    # used here; compare body and root axes before parity acceptance.
    velocity = _robot(env).data.root_link_ang_vel_w
    return np.sum(np.square(velocity[:, :2]), axis=1)


def action_rate_l2(env):
    manager = env.action_manager
    return np.sum(np.square(manager.action - manager.prev_action), axis=1)


def dof_pos_limits(env):
    data = _robot(env).data
    q = data.joint_pos
    bounds = data.soft_joint_pos_limits
    return np.sum(np.maximum(bounds[:, 0] - q, 0.0) + np.maximum(q - bounds[:, 1], 0.0), axis=1)


def head_pose_tracking(env):
    data = _robot(env).data
    error = data.joint_pos[:, 5:9] - data.default_joint_pos[:, 5:9] - _command(env)[:, 3:7]
    return np.mean(np.exp(-np.square(error / 0.5)), axis=1)


class AngularMomentum:
    def __init__(self, cfg, env):
        del cfg
        self.sensor = env.scene.bind_sensor_data(("root_angmom",))

    def __call__(self, env):
        del env
        return np.sum(np.square(self.sensor.read()), axis=1)


class SelfCollisions:
    def __init__(self, cfg, env):
        del cfg
        self.sensor = env.scene.bind_sensor_data(("self_collision_found",))

    def __call__(self, env):
        del env
        return self.sensor.read()[:, 0]


def foot_slip(env):
    velocity = env.walk_foot_velocity
    return np.sum(np.sum(np.square(velocity[:, :, :2]), axis=2) * env.walk_contact, axis=1) * _moving(env)


def foot_clearance(env, target=0.0418):
    speed = np.linalg.norm(env.walk_foot_velocity[:, :, :2], axis=2)
    height = np.abs(env.walk_foot_height - target)
    return np.sum(height * speed, axis=1) * _moving(env)


def foot_swing_height(env, target=0.0418):
    error = env.walk_peak_foot_height / target - 1.0
    return np.sum(np.square(error) * env.walk_first_contact, axis=1) * _moving(env)


def air_time(env, minimum=0.180710403685012, maximum=0.4337049688440288):
    within = (env.walk_air_time > minimum) & (env.walk_air_time < maximum) & ~env.walk_contact
    return np.sum(within, axis=1) * _moving(env)


def zero(env):
    return np.zeros(env.num_envs, dtype=np.float32)


def both_airborne(env):
    return (~np.any(env.walk_contact, axis=1)).astype(np.float32)


def target_limit_l2(env, margin=0.025):
    term = env.action_manager.get_term("joint_pos")
    value = term.processed_action
    residual = value - np.clip(value, term._low + margin, term._high - margin)
    return np.sum(np.square(residual / 0.12), axis=1)


def target_limit_l1(env, margin=0.025):
    term = env.action_manager.get_term("joint_pos")
    value = term.processed_action
    residual = value - np.clip(value, term._low + margin, term._high - margin)
    return np.sum(np.abs(residual / 0.12), axis=1)


def joint_limit_margin_l1(env, margin=0.015):
    data = _robot(env).data
    q = data.joint_pos
    bounds = data.soft_joint_pos_limits
    violation = np.maximum(bounds[:, 0] + margin - q, 0.0)
    violation += np.maximum(q - bounds[:, 1] + margin, 0.0)
    return np.sum(violation / 0.12, axis=1)


REWARD_FUNCTIONS = {
    "pose": pose,
    "upright": upright,
    "foot_slip": foot_slip,
    "self_collisions": SelfCollisions,
    "air_time": air_time,
    "body_ang_vel": body_ang_vel,
    "angular_momentum": AngularMomentum,
    "track_linear_velocity": track_linear_velocity,
    "track_angular_velocity": track_angular_velocity,
    "action_rate_l2": action_rate_l2,
    "foot_clearance": foot_clearance,
    "foot_swing_height": foot_swing_height,
    "head_pose_tracking": head_pose_tracking,
    "body_pose_tracking": zero,
    "head_pose_bias": zero,
    "dof_pos_limits": dof_pos_limits,
    "command_progress": zero,
    "both_airborne": both_airborne,
    "target_limit_l2": target_limit_l2,
    "target_limit_l1": target_limit_l1,
    "joint_limit_margin_l1": joint_limit_margin_l1,
}
