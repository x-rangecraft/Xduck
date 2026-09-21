"""Manager-Based XDuck walking task over the published UniSim backend.

This is a runnable integration probe. The original training recipe's rewards,
randomization, sensor delays, contact history, and curriculum still require a
term-by-term migration before resumed checkpoint training can be called parity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unilab.envs.mdp import (
    bad_orientation,
    geom_friction,
    joint_armature,
    push_by_setting_velocity,
    reset_scene_to_default,
    time_out,
)
from unilab.managers.event_manager import EventTermCfg
from unilab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from unilab.managers.reward_manager import RewardTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.managers.termination_manager import TerminationTermCfg

from . import rewards as reward_terms
from .actions import GF43X40ActionCfg
from .assets import scene_asset_path
from .commands import XDuckWalkCommandCfg
from .observations import WalkCriticObservation, WalkPolicyObservation

JOINTS = (
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "neck_pitch_joint",
    "head_pitch_joint", "head_yaw_joint", "head_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint",
)
ACTUATORS = tuple(name.removesuffix("_joint") for name in JOINTS)


def _contract() -> dict:
    return json.loads((Path(scene_asset_path()).with_name("xc_contract.json")).read_text())


def reset_walk_history(env, env_ids):
    ids = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.intp)
    env.walk_air_time[ids] = 0.0
    env.walk_contact[ids] = False
    env.walk_first_contact[ids] = False
    env.walk_foot_velocity[ids] = 0.0
    env.walk_foot_height[ids] = 0.0
    env.walk_peak_foot_height[ids] = 0.0
    env.walk_history_fresh[ids] = True


def velocity_tracking(env) -> np.ndarray:
    """Minimal nonzero reward for integration smoke; not original recipe."""
    measured = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    command = env.command_manager.get_command("walk")[:, :2]
    return np.exp(-np.sum((measured - command) ** 2, axis=1) / 0.25)


def walk_cfg() -> ManagerBasedRlEnvCfg:
    contract = _contract()
    scene = SceneCfg(
        model_file=scene_asset_path(),
        default_keyframe_name="GAIT_REFERENCE",
        entities={"robot": EntityCfg(
            root_body_name="base_link", joint_names=JOINTS,
            actuator_names=ACTUATORS,
            geom_names=("left_foot_collision", "right_foot_collision"),
        )},
    )
    return ManagerBasedRlEnvCfg(
        scene=scene, sim_dt=0.001, ctrl_dt=0.02, max_episode_seconds=20.0,
        observations={
            "policy": ObservationGroupCfg(terms={
                "legacy_order": ObservationTermCfg(func=WalkPolicyObservation),
            }),
            "critic": ObservationGroupCfg(terms={
                "legacy_order": ObservationTermCfg(func=WalkCriticObservation),
            }),
        },
        policy_observation_group="policy", critic_observation_group="critic",
        actions={"joint_pos": GF43X40ActionCfg(
            entity_name="robot", actuator_names=JOINTS, preserve_order=True,
            scale=contract["action_scale"],
            kp=tuple(contract["kp"]), kd=tuple(contract["kd"]),
            target_low=tuple(contract["target_low"]),
            target_high=tuple(contract["target_high"]),
        )},
        commands={"walk": XDuckWalkCommandCfg(resampling_time_range=(4.3370496884, 11.5654658358))},
        events={
            "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "reset_walk_history": EventTermCfg(func=reset_walk_history, mode="reset"),
            "foot_friction": EventTermCfg(
                func=geom_friction, mode="reset",
                params={"ranges": (0.4, 1.6), "operation": "abs",
                        "asset_cfg": SceneEntityCfg(name="robot", geom_names=(".*foot_collision",))},
            ),
            "armature": EventTermCfg(
                func=joint_armature, mode="reset",
                params={"ranges": (0.8, 1.2), "operation": "scale",
                        "asset_cfg": SceneEntityCfg(name="robot", joint_names=(".*",))},
            ),
            "velocity_push": EventTermCfg(
                func=push_by_setting_velocity, mode="interval",
                interval_range_s=(4.3370496884, 8.6740993769),
                params={"velocity_range": {
                    "x": (-0.4337049688, 0.4337049688),
                    "y": (-0.4337049688, 0.4337049688),
                    "z": (0.0, 0.0), "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                }},
            ),
        },
        rewards={name: RewardTermCfg(func=reward_terms.REWARD_FUNCTIONS[name], weight=float(weight))
                 for name, weight in json.loads((Path(scene_asset_path()).with_name("selected_reward.json")).read_text())["scales"].items()},
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "bad_orientation": TerminationTermCfg(
                func=bad_orientation, params={"limit_angle": np.deg2rad(70.0)}
            ),
        },
    )


class GF43X40WalkEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, backend, num_envs):
        self.walk_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self.walk_history_fresh = np.ones(num_envs, dtype=bool)
        self.walk_contact = np.zeros((num_envs, 2), dtype=bool)
        self.walk_first_contact = np.zeros((num_envs, 2), dtype=bool)
        self.walk_foot_velocity = np.zeros((num_envs, 2, 3), dtype=np.float32)
        self.walk_foot_height = np.zeros((num_envs, 2), dtype=np.float32)
        self.walk_peak_foot_height = np.zeros((num_envs, 2), dtype=np.float32)
        super().__init__(cfg, backend, num_envs)
        self._walk_foot_view = self.scene.bind_sensor_data((
            "left_foot_pos", "right_foot_pos", "left_foot_vel", "right_foot_vel",
            "left_foot_found", "right_foot_found",
        ))

    def update_state(self, state):
        # Complete final substep's feedback before the observation manager runs.
        self.action_manager.get_term("joint_pos").finish_control_step()
        foot = self._walk_foot_view.read()
        contact = foot[:, 12:14] > 0.0
        self.walk_first_contact[:] = contact & ~self.walk_contact & ~self.walk_history_fresh[:, None]
        self.walk_air_time[:] = np.where(contact, 0.0, self.walk_air_time + self.step_dt)
        self.walk_foot_velocity[:] = foot[:, 6:12].reshape(self.num_envs, 2, 3)
        self.walk_foot_height[:] = np.maximum(foot[:, [2, 5]] - 0.04, 0.0)
        self.walk_peak_foot_height[:] = np.maximum(self.walk_peak_foot_height, self.walk_foot_height)
        self.walk_contact[:] = contact
        self.walk_history_fresh[:] = False
        return super().update_state(state)


def make_walk_env(
    cfg: ManagerBasedRlEnvCfg | None = None,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> GF43X40WalkEnv:
    """Build the public backend and Manager-Based runtime for GF walking."""
    if backend_type != "mujoco":
        raise NotImplementedError("GF43X40 task currently supports MuJoCo only")
    cfg = walk_cfg() if cfg is None else cfg
    cfg.validate()
    apply_env_cpu_runtime(cfg.cpu_ids)
    kwargs = env_backend_kwargs(cfg)
    backend = create_backend(
        backend_type, cfg.scene, num_envs, cfg.sim_dt,
        base_name="base_link", body_state_required=True,
        refresh_pre_step_body_state=False,
        **kwargs,
    )
    try:
        return GF43X40WalkEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise
