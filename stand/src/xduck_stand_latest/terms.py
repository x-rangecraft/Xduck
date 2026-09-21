"""Task-owned reset, observation, reward, termination, and metric terms."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from unilab.envs.mdp import resolve_env_ids
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg
from unilab.utils.rotation import np_quat_apply, np_quat_mul, np_yaw_to_quat

_HANDOFF_EVENT = "handoff_reset"


class HandoffReset(ManagerTermBase):
    """Cold-load upright walking states and stage selected rows at reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        path = Path(cfg.params["bank_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing handoff bank: {path}")
        with np.load(path, allow_pickle=False) as data:
            self.qpos = data["qpos"].copy()
            self.qvel = data["qvel"].copy()
            self.actions = data["actions"].copy()
            self.metadata = json.loads(str(data["metadata"].item()))
        n = len(self.qpos)
        if n == 0 or self.qpos.shape != (n, 21) or self.qvel.shape != (n, 20):
            raise ValueError("Handoff bank must contain qpos[N,21] and qvel[N,20]")
        if self.actions.shape != (n, 14):
            raise ValueError("Handoff bank must contain actions[N,14]")
        if not all(np.isfinite(a).all() for a in (self.qpos, self.qvel, self.actions)):
            raise ValueError("Handoff bank contains non-finite state")
        if self.metadata.get("version") != 1:
            raise ValueError("Unsupported handoff bank version")
        robot = env.scene["robot"]
        action = env.cfg.actions["joint_pos"]
        if tuple(self.metadata["joint_names"]) != robot.joint_names:
            raise ValueError("Handoff bank joint order differs from this task")
        if not np.allclose(
            self.metadata["default_angles"], robot.data.default_joint_pos[0], atol=1e-6
        ):
            raise ValueError("Handoff bank action reference differs from this task")
        if not np.isclose(self.metadata["action_scale_rad"], action.scale):
            raise ValueError("Handoff bank action scale differs from this task")
        for key in ("joint_kp", "joint_kd"):
            cfg_values = getattr(action, "kp" if key == "joint_kp" else "kd")
            if not np.allclose(self.metadata[key], cfg_values):
                raise ValueError(f"Handoff bank {key} differs from this task")
        if not np.isclose(self.metadata["ctrl_dt"], env.step_dt) or not np.isclose(
            self.metadata["sim_dt"], env.physics_dt
        ):
            raise ValueError("Handoff bank control/physics step differs from this task")
        self.last_actions = np.zeros((env.num_envs, 14), dtype=np.float32)
        self.selected_indices = np.zeros(env.num_envs, dtype=np.int64)

    def __call__(self, env: Any, env_ids: np.ndarray | None, **params: Any) -> None:
        del params
        ids = resolve_env_ids(env, env_ids)
        selected = env.rng.integers(len(self.qpos), size=len(ids))
        qpos = self.qpos[selected].copy()
        qvel = self.qvel[selected].copy()
        yaw = env.rng.uniform(-math.pi, math.pi, len(ids))
        rotation = np_yaw_to_quat(yaw)
        qpos[:, :2] += np.asarray(env.scene.env_origins)[ids, :2]
        qpos[:, 3:7] = np_quat_mul(rotation, qpos[:, 3:7])
        qvel[:, :3] = np_quat_apply(rotation, qvel[:, :3])
        robot = env.scene["robot"]
        robot.write_root_state_to_sim(
            np.concatenate((qpos[:, :7], qvel[:, :6]), axis=1), env_ids=ids
        )
        robot.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=ids)
        self.last_actions[ids] = self.actions[selected]
        self.selected_indices[ids] = selected

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        """Seed policy history after ActionManager clears reset rows."""
        ids = resolve_env_ids(self._env, env_ids)
        action = self._env.action_manager.get_term("joint_pos")
        action.seed_previous_action(ids, self.last_actions[ids])


def _handoff(env: Any) -> HandoffReset:
    term = env.event_manager.get_term_cfg(_HANDOFF_EVENT).func
    if not isinstance(term, HandoffReset):
        raise TypeError("handoff_reset must use HandoffReset")
    return term


def zero_command_13(env: Any) -> np.ndarray:
    """The stop actor's 13 command slots are always zero."""
    return np.zeros((env.num_envs, 13), dtype=np.float32)


def handoff_last_action(env: Any) -> np.ndarray:
    """Preserve the walking raw action in the first reset observation."""
    current = env.action_manager.action
    seeded = _handoff(env).last_actions
    return np.where(env.episode_length_buf[:, None] == 0, seeded, current)


def handoff_action_rate_l2(env: Any) -> np.ndarray:
    """Charge the first post-handoff action against the incoming walk action."""
    current = env.action_manager.action
    previous = np.where(
        env.episode_length_buf[:, None] <= 1,
        _handoff(env).last_actions,
        env.action_manager.prev_action,
    )
    return np.sum(np.square(current - previous), axis=1)


def stop_failure(
    env: Any,
    *,
    max_tilt_deg: float = 55.0,
    min_height_ratio: float = 0.6,
) -> np.ndarray:
    """Terminate at the old stop task's tilt or root-height limits."""
    robot = env.scene["robot"]
    quat = robot.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    default_height = robot.data.default_root_state[:, 2]
    return (upright < math.cos(math.radians(max_tilt_deg))) | (
        robot.data.root_link_pos_w[:, 2] < default_height * min_height_ratio
    )


class SettledState(ManagerTermBase):
    """Track a continuous one-second stand with both feet supported."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        super().__init__(env)
        self._duration = np.zeros(env.num_envs, dtype=np.float32)
        self._lin_max = float(cfg.params.get("linear_speed", 0.03))
        self._ang_max = float(cfg.params.get("angular_speed", 0.2))
        self._joint_max = float(cfg.params.get("joint_speed_rms", 0.15))
        self._hold_time = float(cfg.params.get("hold_seconds", 1.0))
        self._contact_threshold = float(cfg.params.get("contact_threshold", 0.1))
        self._contacts = env.scene.bind_sensor_data(
            ("left_foot_contact", "right_foot_contact")
        )

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._duration[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: Any, **params: Any) -> np.ndarray:
        del params
        robot = env.scene["robot"]
        velocity = robot.data.root_link_lin_vel_b
        gyro = robot.data.root_link_ang_vel_b
        joint_rms = np.sqrt(np.mean(np.square(robot.data.joint_vel), axis=1))
        contact = self._contacts.read()
        both_contact = np.all(
            contact.reshape(env.num_envs, 2, -1).max(axis=2) > self._contact_threshold,
            axis=1,
        )
        quat = robot.data.root_link_quat_w
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        settled = (
            (np.linalg.norm(velocity, axis=1) < self._lin_max)
            & (np.linalg.norm(gyro, axis=1) < self._ang_max)
            & (joint_rms < self._joint_max)
            & both_contact
            & (upright > math.cos(math.radians(15.0)))
        )
        self._duration[:] = np.where(settled, self._duration + env.step_dt, 0.0)
        return np.asarray(self._duration >= self._hold_time, dtype=np.float32)
