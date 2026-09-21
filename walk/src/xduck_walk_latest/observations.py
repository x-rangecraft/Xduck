"""Checkpoint-shape observation terms for the XDuck walking policy."""

import numpy as np


def _robot(env):
    return env.scene["robot"]


def _action(env):
    return env.action_manager.get_term("joint_pos")


class WalkPolicyObservation:
    """Legacy 61D field order with 20 ms velocity lag and 0/20 ms IMU lag."""

    def __init__(self, cfg, env):
        del cfg
        self._history_imu = np.zeros((2, env.num_envs, 6), dtype=np.float32)
        self._history_vel = np.zeros((2, env.num_envs, 14), dtype=np.float32)
        self._cursor = 0
        self._fresh = np.ones(env.num_envs, dtype=bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._fresh[ids] = True

    def __call__(self, env, noise_level: float = 2.0):
        robot = _robot(env)
        action = _action(env)
        position = robot.data.joint_pos
        velocity = robot.data.joint_vel
        position = action.motor.quantize_position_feedback(position)
        velocity = action.motor.quantize_velocity_feedback(velocity)
        gyro = robot.data.root_link_ang_vel_b
        gravity = robot.data.projected_gravity_b
        imu = np.concatenate((gyro, gravity), axis=1)
        fresh = self._fresh
        if np.any(fresh):
            self._history_imu[:, fresh] = imu[fresh][None]
            self._history_vel[:, fresh] = velocity[fresh][None]
            self._fresh[fresh] = False
        cursor = self._cursor
        self._history_imu[cursor] = imu
        self._history_vel[cursor] = velocity
        rows = np.arange(env.num_envs)
        imu_lag = env.rng.integers(0, 2, size=env.num_envs)
        delayed_imu = self._history_imu[(cursor - imu_lag) % 2, rows]
        delayed_vel = self._history_vel[(cursor - 1) % 2, rows]
        self._cursor = (cursor + 1) % 2

        def corrupt(value, scale):
            if noise_level <= 0:
                return value
            return value + env.rng.uniform(-1.0, 1.0, value.shape) * (noise_level * scale)

        result = np.concatenate((
            corrupt(delayed_imu[:, :3], 0.020751433915982238),
            corrupt(delayed_imu[:, 3:], 0.01),
            corrupt(position - robot.data.default_joint_pos, 0.001),
            corrupt(delayed_vel, 0.17292861596651865),
            env.action_manager.action,
            env.command_manager.get_command("walk"),
        ), axis=1).astype(np.float32)
        if result.shape != (env.num_envs, 61):
            raise ValueError(f"walk actor observation must be (num_envs, 61), got {result.shape}")
        return result


class WalkCriticObservation:
    """Privileged 76D tensor in the legacy checkpoint's field order."""

    def __init__(self, cfg, env):
        del cfg
        self._foot = env.scene.bind_sensor_data((
            "left_foot_pos", "right_foot_pos", "left_foot_found", "right_foot_found",
            "left_foot_contact", "right_foot_contact",
        ))

    def __call__(self, env):
        robot = _robot(env)
        command = env.command_manager.get_command("walk")
        foot = self._foot.read()
        if foot.shape != (env.num_envs, 14):
            raise ValueError(f"walk foot sensor contract must be 14D, got {foot.shape}")
        left_z, right_z = foot[:, 2:3], foot[:, 5:6]
        contact = foot[:, 6:8]
        force = foot[:, 8:14]
        # Flat-ground approximation. Legacy critic uses raycast clearance and
        # per-ms contact history; sensor timing parity is a migration gate.
        height = np.maximum(np.concatenate((left_z, right_z), axis=1) - 0.04, 0.0)
        air_time = env.walk_air_time
        result = np.concatenate((
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
            robot.data.projected_gravity_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel,
            env.action_manager.action,
            command[:, :3], height, air_time, contact, force,
            command[:, 3:7], command[:, 7:13],
        ), axis=1).astype(np.float32)
        if result.shape != (env.num_envs, 76):
            raise ValueError(f"walk critic observation must be (num_envs, 76), got {result.shape}")
        return result
