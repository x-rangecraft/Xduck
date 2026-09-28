"""Original-size Micro Duck forward-roll task mapped to UniLab contracts.

The task follows the upstream ``Mjlab-Roulade-Flat-MicroDuck`` state recipe:
supported forward-rotation progress, an over-the-head contact latch, landing
rewards that open only after roll completion, and reverse-curriculum mid-roll
resets.  It keeps the shared 61D observation and 14D action layout so the
resulting actor can occupy the upstream roulade policy slot.  Head return is a
deployment-handoff acceptance metric, not an extra training reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck.walk import (
    MICRODUCK_ACTOR_OBS_DIM,
    NUM_MICRODUCK_ACTIONS,
    MicroDuckCommands,
    MicroDuckDomainRandConfig,
    MicroDuckRewardConfig,
    MicroDuckSensor,
    MicroDuckWalkDomainRandomizationProvider,
    MicroDuckWalkFlatCfg,
    MicroDuckWalkFlatEnv,
)
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_quat_from_euler_xyz

_LEG_JOINT_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
_HEAD_PITCH_INDICES = np.asarray((5, 6), dtype=np.intp)
_HEAD_YAW_ROLL_INDICES = np.asarray((7, 8), dtype=np.intp)
_HEAD_TOP_AXIS = np.asarray((0.882, 0.0, 0.471), dtype=np.float64)
_HEAD_TOP_DOWN_MIN = 0.3
_HEAD_LATCH_LO = np.deg2rad(20.0)
_HEAD_LATCH_HI = np.deg2rad(170.0)
_FLAT_FULL = 0.5
_FLAT_ZERO = 0.866
MICRODUCK_ROULADE_CRITIC_OBS_DIM = 74

_TUCK_OVERRIDES = {
    2: -1.15,
    3: 1.25,
    4: 1.05,
    5: -1.0,
    6: 1.0,
    11: 1.15,
    12: -1.25,
    13: -1.05,
}


@dataclass
class MicroDuckRouladeSensor(MicroDuckSensor):
    critic_feet: tuple[str, str] = ("roulade_left_foot_state", "roulade_right_foot_state")
    head_ground: tuple[str, str, str] = (
        "roulade_head_top_contact",
        "roulade_jaw_contact",
        "roulade_head_bottom_contact",
    )
    support_ground: tuple[str, ...] = (
        "left_foot_contact",
        "right_foot_contact",
        "roulade_trunk_contact",
        "roulade_left_hip_contact",
        "roulade_right_hip_contact",
        "roulade_left_leg_contact",
        "roulade_right_leg_contact",
        "roulade_head_top_contact",
        "roulade_jaw_contact",
        "roulade_head_bottom_contact",
    )
    self_collision: tuple[str, ...] = (
        "self_left_trunk_contact",
        "self_right_trunk_contact",
        "self_leg_leg_contact",
    )


@dataclass
class MicroDuckRouladeCommands(MicroDuckCommands):
    """Zero command block retained for deployment-compatible 61D observations."""

    vel_limit: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rel_standing_envs: float = 1.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    head_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 4, [0.0] * 4])
    body_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 6, [0.0] * 6])


@dataclass
class MicroDuckRouladeDomainRandConfig(MicroDuckDomainRandConfig):
    push_robots: bool = False
    randomize_body_mass: bool = True
    body_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.003, 0.003])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.003, 0.003])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.003, 0.003])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.7, 1.3])
    randomize_dof_armature: bool = True
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class MicroDuckRouladeCurriculumConfig:
    enabled: bool = True
    play_mode: bool = False
    num_steps_per_iteration: int = 24
    spawn_stages: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "standing_prob": 0.50, "midroll_prob": 0.50},
            {"step": 3000 * 24, "standing_prob": 0.65, "midroll_prob": 0.35},
            {"step": 6000 * 24, "standing_prob": 0.80, "midroll_prob": 0.20},
        ]
    )
    play_standing_prob: float = 1.0
    play_midroll_prob: float = 0.0

    def spawn_probabilities(self, step: int) -> tuple[float, float]:
        if self.play_mode:
            return self.play_standing_prob, self.play_midroll_prob
        stages = self.spawn_stages if self.enabled else self.spawn_stages[-1:]
        active = stages[0]
        previous_step = -1
        for stage in stages:
            stage_step = int(stage["step"])
            if stage_step <= previous_step:
                raise ValueError("roulade spawn curriculum steps must be strictly increasing")
            previous_step = stage_step
            if step < stage_step:
                break
            active = stage
        standing = float(active["standing_prob"])
        midroll = float(active["midroll_prob"])
        if standing < 0.0 or midroll < 0.0 or standing + midroll <= 0.0:
            raise ValueError("roulade spawn probabilities must be non-negative with positive sum")
        total = standing + midroll
        return standing / total, midroll / total

    def reference_step(self, physical_step: int) -> int:
        """Map sequential collection to upstream's 24-step PPO iteration clock."""
        if self.num_steps_per_iteration < 24 or self.num_steps_per_iteration % 24:
            raise ValueError("Roulade collection must use complete 24-step segments")
        return physical_step * 24 // self.num_steps_per_iteration


@dataclass
class MicroDuckRouladeRewardConfig(MicroDuckRewardConfig):
    target_angle: float = 2.0 * np.pi
    max_paid_rate: float = 5.0
    overspeed_omega_max: float = 7.0
    stand_height: float = 0.115
    landing_gate_lo: float = float(np.deg2rad(260.0))
    landing_gate_hi: float = float(np.deg2rad(330.0))
    rise_gate_lo: float = float(np.deg2rad(180.0))
    rise_gate_hi: float = float(np.deg2rad(260.0))
    success_height: float = 0.105
    success_tilt_deg: float = 15.0
    success_head_pitch_tol: float = float(np.deg2rad(10.0))
    success_head_yaw_roll_tol: float = float(np.deg2rad(5.0))
    actuator_force_limit: float = 0.96
    action_rate_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": -0.1},
            {"step": 1500 * 24, "weight": -0.2},
            {"step": 3000 * 24, "weight": -0.4},
        ]
    )
    arrival_damping_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": 0.0},
            {"step": 2500 * 24, "weight": -0.025},
            {"step": 3500 * 24, "weight": -0.05},
        ]
    )
    gentle_landing_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": -0.002},
            {"step": 2500 * 24, "weight": -0.005},
        ]
    )
    torque_rate_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": 0.0},
            {"step": 2500 * 24, "weight": -5.0e-4},
            {"step": 3500 * 24, "weight": -1.0e-3},
        ]
    )


@registry.envcfg("MicroDuckRouladeFlat")
@dataclass
class MicroDuckRouladeFlatCfg(MicroDuckWalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck" / "scene_roulade.xml")
        )
    )
    sensor: MicroDuckRouladeSensor = field(default_factory=MicroDuckRouladeSensor)  # type: ignore[assignment]
    commands: MicroDuckRouladeCommands = field(default_factory=MicroDuckRouladeCommands)
    reward_config: MicroDuckRouladeRewardConfig | None = None
    domain_rand: MicroDuckRouladeDomainRandConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckRouladeDomainRandConfig
    )
    curriculum: MicroDuckRouladeCurriculumConfig = field(
        default_factory=MicroDuckRouladeCurriculumConfig
    )
    max_episode_seconds: float = 5.0
    reset_base_qvel_limit: float = 0.0
    standing_z_range: tuple[float, float] = (0.11, 0.12)
    standing_tilt_max: float = float(np.deg2rad(5.0))
    forward_vel_range: tuple[float, float] = (0.0, 0.0)
    midroll_pitch_range: tuple[float, float] = (
        float(np.deg2rad(50.0)),
        float(np.deg2rad(340.0)),
    )
    midroll_z_range: tuple[float, float] = (0.05, 0.10)
    midroll_omega_range: tuple[float, float] = (0.0, 3.0)
    tuck_factor_range: tuple[float, float] = (0.3, 1.0)
    joint_noise_std: float = 0.08

    def validate(self) -> None:
        super().validate()
        self.curriculum.spawn_probabilities(0)
        for name in (
            "standing_z_range",
            "forward_vel_range",
            "midroll_pitch_range",
            "midroll_z_range",
            "midroll_omega_range",
            "tuck_factor_range",
        ):
            low, high = getattr(self, name)
            if high < low:
                raise ValueError(f"{name} must satisfy low <= high")


class MicroDuckRouladeDomainRandomizationProvider(MicroDuckWalkDomainRandomizationProvider):
    def _compute_reset_obs(
        self,
        env,
        env_ids,
        info_updates,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
    ):
        # Reset can be a subset. Sample physical state using these exact ids.
        contact, force = env._critic_foot_state(env_ids)
        info_updates["roulade_critic_linvel"] = np_quat_apply_inverse(
            env._backend.get_base_quat()[env_ids],
            env._backend.get_base_lin_vel()[env_ids],
        )
        info_updates["roulade_critic_contact"] = contact
        info_updates["roulade_critic_force"] = force
        info_updates["roulade_critic_air_time"] = np.zeros_like(contact)
        return env._compute_obs(
            info_updates,
            gyro,
            env._projected_gravity()[env_ids],
            dof_pos,
            dof_vel,
        )

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        return (
            env._base_body_mass,
            env._base_geom_friction,
            env._ground_geom_id,
            env._base_dof_armature,
        )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        del env, step_counter
        return None

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        _, midroll_prob = env.cfg.curriculum.spawn_probabilities(
            env.cfg.curriculum.reference_step(env.step_counter)
        )
        is_mid = np.random.uniform(size=num_reset) < midroll_prob

        yaw = np.random.uniform(-np.pi, np.pi, size=num_reset)
        pitch = np.random.uniform(
            -env.cfg.standing_tilt_max,
            env.cfg.standing_tilt_max,
            size=num_reset,
        )
        pitch[is_mid] = np.random.uniform(*env.cfg.midroll_pitch_range, size=int(np.sum(is_mid)))
        roll = np.random.uniform(
            -env.cfg.standing_tilt_max,
            env.cfg.standing_tilt_max,
            size=num_reset,
        )
        plan.qpos[:, 3:7] = np_quat_from_euler_xyz(roll, pitch, yaw)
        plan.qpos[:, 2] = np.random.uniform(*env.cfg.standing_z_range, size=num_reset)
        plan.qpos[is_mid, 2] = np.random.uniform(*env.cfg.midroll_z_range, size=int(np.sum(is_mid)))
        plan.qvel[:] = 0.0

        if np.any(is_mid):
            mid_count = int(np.sum(is_mid))
            tuck = np.random.uniform(*env.cfg.tuck_factor_range, size=mid_count)
            joints = np.tile(env.default_angles, (mid_count, 1))
            for joint_index, target in _TUCK_OVERRIDES.items():
                home = env.default_angles[joint_index]
                joints[:, joint_index] = home + tuck * (target - home)
            joints += np.random.normal(0.0, env.cfg.joint_noise_std, size=joints.shape)
            if env._joint_range is not None:
                joints = np.clip(joints, env._joint_range[:, 0], env._joint_range[:, 1])
            plan.qpos[is_mid, 7:] = joints

            omega = np.random.uniform(*env.cfg.midroll_omega_range, size=mid_count)
            # MuJoCo free-joint qvel[4] produces body +Y rotation for this
            # model independently of spawn yaw (locked by the task contract
            # test).  Positive body +Y is the upstream forward-roll sign.
            plan.qvel[is_mid, 4] = omega

        is_standing = ~is_mid
        if np.any(is_standing) and env.cfg.forward_vel_range[1] > 0.0:
            speed = np.random.uniform(*env.cfg.forward_vel_range, size=int(np.sum(is_standing)))
            plan.qvel[is_standing, 0] = np.cos(yaw[is_standing]) * speed
            plan.qvel[is_standing, 1] = np.sin(yaw[is_standing]) * speed

        spawn_angle = np.where(is_mid, pitch, 0.0).astype(get_global_dtype())
        zeros = np.zeros(num_reset, dtype=get_global_dtype())
        plan.info_updates.update(
            {
                "roulade_reset_midroll": is_mid,
                "roulade_accum": spawn_angle.copy(),
                "roulade_max": spawn_angle.copy(),
                "roulade_paid": spawn_angle.copy(),
                "roulade_head_latch": is_mid.copy(),
                "roulade_progress_signal": zeros.copy(),
                "roulade_supported": np.zeros(num_reset, dtype=bool),
                "roulade_head_contact": np.zeros(num_reset, dtype=bool),
                "roulade_prev_vz": plan.qvel[:, 2].astype(get_global_dtype()),
                "roulade_vertical_accel": zeros.copy(),
                "roulade_prev_torque": np.zeros(
                    (num_reset, NUM_MICRODUCK_ACTIONS), dtype=get_global_dtype()
                ),
                "roulade_prev_torque_valid": np.zeros(num_reset, dtype=bool),
            }
        )
        return plan


@registry.env("MicroDuckRouladeFlat", sim_backend="mujoco")
class MicroDuckRouladeFlatEnv(MicroDuckWalkFlatEnv):
    _cfg: MicroDuckRouladeFlatCfg

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": MICRODUCK_ACTOR_OBS_DIM, "critic": MICRODUCK_ROULADE_CRITIC_OBS_DIM}

    def build_symmetry_augmentation(self, *, device: str):
        from .symmetry import MicroDuckSymmetryAugmentation

        return MicroDuckSymmetryAugmentation(device=device)

    def _critic_foot_state(self, env_ids=None):
        samples = [
            np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype()).reshape(
                self._num_envs, 4
            )
            for name in self._cfg.sensor.critic_feet
        ]
        data = np.stack(samples, axis=1)
        if env_ids is not None:
            data = data[env_ids]
        return (data[:, :, 0] > 0).astype(get_global_dtype()), data[:, :, 1:4].reshape(-1, 6)

    def _compute_obs(self, info, gyro, projected_gravity, dof_pos, dof_vel):
        obs = super()._compute_obs(info, gyro, projected_gravity, dof_pos, dof_vel)
        force = info["roulade_critic_force"]
        # Match upstream term order, including noise-free proprioception and
        # signed-log world-frame net forces. Never concatenate this into actor obs.
        obs["critic"] = np.concatenate(
            [
                info["roulade_critic_linvel"],
                gyro,
                projected_gravity,
                dof_pos - self.default_angles,
                dof_vel,
                info.get("current_actions", np.zeros_like(dof_pos)),
                info["commands"],
                info["roulade_critic_air_time"],
                info["roulade_critic_contact"],
                np.sign(force) * np.log1p(np.abs(force)),
                info["head_commands"],
                info["body_commands"],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return obs

    def __init__(
        self,
        cfg: MicroDuckRouladeFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            **env_backend_kwargs(cfg),
        )
        LocomotionBaseEnv.__init__(self, cfg, backend, num_envs)
        if self._num_action != NUM_MICRODUCK_ACTIONS:
            raise ValueError(
                f"Micro Duck roulade requires {NUM_MICRODUCK_ACTIONS} actuators, "
                f"got {self._num_action}"
            )
        self._reward_cfg = cfg.reward_config
        self._joint_range = self._backend.get_joint_range()
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(cfg.asset.foot_site_offsets, dtype=get_global_dtype())
        self._head_body_ids = self._backend.get_body_ids(("jaw_soft",))
        self._base_body_mass = self._backend.get_body_mass()
        self._base_geom_friction = self._backend.get_geom_friction()
        self._ground_geom_id = self._backend.get_geom_id(cfg.asset.ground)
        self._base_dof_armature = self._backend.get_dof_armature()
        self._actuator_kp, self._actuator_kd = self._backend.get_actuator_gains()
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(MicroDuckRouladeDomainRandomizationProvider())

    @property
    def _roulade_reward_cfg(self) -> MicroDuckRouladeRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckRouladeRewardConfig):
            raise TypeError("MicroDuckRouladeFlat requires MicroDuckRouladeRewardConfig")
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "roulade_progress": self._progress,
            "roulade_overspeed": self._overspeed,
            "roulade_head_pivot": self._head_pivot,
            "roulade_landing_composite": self._landing_composite,
            "roulade_upright_after_roll": self._upright_after_roll,
            "roulade_height_after_roll": self._height_after_roll,
            "roulade_landing_sharp": self._landing_sharp,
            "roulade_stand_shortfall": self._stand_shortfall,
            "roulade_rise_velocity": self._rise_velocity,
            "roulade_sagittal": self._sagittal,
            "roulade_lateral_vel": self._lateral_velocity,
            "roulade_flatness": self._flatness,
            "self_collisions": self._self_collisions,
            "body_ang_vel": rewards.ang_vel_xy,
            "angular_momentum": self._angular_momentum,
            "arrival_damping": self._arrival_damping,
            "gentle_landing": self._gentle_landing,
            "joint_torque_rate": self._joint_torque_rate,
            "action_rate": rewards.action_rate,
        }

    def _contact_any(self, names: tuple[str, ...]) -> np.ndarray:
        contacts = []
        for name in names:
            value = np.asarray(self._backend.get_sensor_data(name)).reshape(self._num_envs, -1)
            contacts.append(np.any(value != 0.0, axis=1))
        return np.any(np.stack(contacts, axis=1), axis=1)

    @staticmethod
    def _lateral_axis_z(quat: np.ndarray) -> np.ndarray:
        return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])

    def _head_top_down(self) -> np.ndarray:
        quat = self._backend.get_body_quat_w(self._head_body_ids)[:, 0, :]
        axis = np.broadcast_to(_HEAD_TOP_AXIS, (self._num_envs, 3))
        axis_world = np_quat_apply(quat, axis)
        return axis_world[:, 2] < -_HEAD_TOP_DOWN_MIN

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        x = np.clip(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _completion_gate(self, info: dict[str, Any], low: float, high: float) -> np.ndarray:
        # Gate landing/rise rewards on the current signed supported progress.
        # The frontier is deliberately irreversible for dense progress payout,
        # but must not keep the landing annuity open after a reverse unwind.
        progress = np.asarray(info["roulade_accum"], dtype=get_global_dtype())
        gate = self._smoothstep((progress - low) / max(high - low, 1.0e-6))
        return np.asarray(gate * info["roulade_head_latch"], dtype=get_global_dtype())

    def _update_roulade_state(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        base_quat: np.ndarray,
        base_lin_vel_w: np.ndarray,
    ) -> None:
        cfg = self._roulade_reward_cfg
        supported = self._contact_any(self._cfg.sensor.support_ground)
        head_contact = self._contact_any(self._cfg.sensor.head_ground)
        flatness = np.abs(self._lateral_axis_z(base_quat))
        flat_gate = self._smoothstep((_FLAT_ZERO - flatness) / (_FLAT_ZERO - _FLAT_FULL))
        omega_forward = np.nan_to_num(gyro[:, 1], nan=0.0)
        delta = omega_forward * self._cfg.ctrl_dt * supported * flat_gate
        accum = np.asarray(info["roulade_accum"], dtype=get_global_dtype()) + delta
        frontier = np.maximum(info["roulade_max"], accum)

        top_down = self._head_top_down()
        latch_window = (accum > _HEAD_LATCH_LO) & (accum < _HEAD_LATCH_HI)
        latch = info["roulade_head_latch"] | (head_contact & latch_window & top_down)

        new_paid = np.minimum(frontier, cfg.target_angle)
        old_paid = np.minimum(info["roulade_paid"], cfg.target_angle)
        paid_delta = np.clip(new_paid - old_paid, 0.0, cfg.max_paid_rate * self._cfg.ctrl_dt)
        info["roulade_accum"] = np.asarray(accum, dtype=get_global_dtype())
        info["roulade_max"] = np.asarray(frontier, dtype=get_global_dtype())
        info["roulade_paid"] = np.maximum(info["roulade_paid"], new_paid).astype(get_global_dtype())
        info["roulade_head_latch"] = latch
        info["roulade_progress_signal"] = np.asarray(
            paid_delta / (self._cfg.ctrl_dt * cfg.target_angle), dtype=get_global_dtype()
        )
        info["roulade_supported"] = supported
        info["roulade_head_contact"] = head_contact

        vz = np.asarray(base_lin_vel_w[:, 2], dtype=get_global_dtype())
        info["roulade_vertical_accel"] = np.abs(
            (vz - info["roulade_prev_vz"]) / self._cfg.ctrl_dt
        ).astype(get_global_dtype())
        info["roulade_prev_vz"] = vz

    def _progress(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["roulade_progress_signal"], dtype=get_global_dtype())

    def _overspeed(self, ctx: RewardContext) -> np.ndarray:
        excess = np.maximum(
            np.abs(ctx.gyro[:, 1]) - self._roulade_reward_cfg.overspeed_omega_max, 0.0
        )
        return np.asarray(np.square(excess), dtype=get_global_dtype())

    def _head_pivot(self, ctx: RewardContext) -> np.ndarray:
        angle = np.asarray(ctx.info["roulade_accum"])
        window = (angle > np.deg2rad(30.0)) & (angle < np.deg2rad(240.0))
        rate = np.clip(ctx.gyro[:, 1] / 2.0, 0.0, 1.0)
        top_score = 0.3 + 0.7 * self._head_top_down().astype(get_global_dtype())
        return np.asarray(ctx.info["roulade_head_contact"] * window * rate * top_score)

    def _pose_score(self, ctx: RewardContext, std: float) -> np.ndarray:
        error = ctx.dof_pos[:, _LEG_JOINT_INDICES] - ctx.default_angles[_LEG_JOINT_INDICES]
        return np.exp(-np.mean(np.square(error), axis=1) / (std * std))

    def _head_is_home(self, dof_pos: np.ndarray) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        error = np.abs(dof_pos - self.default_angles[None, :])
        pitch_ok = np.all(
            error[:, _HEAD_PITCH_INDICES] <= cfg.success_head_pitch_tol,
            axis=1,
        )
        yaw_roll_ok = np.all(
            error[:, _HEAD_YAW_ROLL_INDICES] <= cfg.success_head_yaw_roll_tol,
            axis=1,
        )
        return pitch_ok & yaw_roll_ok

    def _tilt_sq(self) -> np.ndarray:
        quat = self._backend.get_base_quat()
        return 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))

    def _landing_composite(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        height = np.exp(-np.square((ctx.base_height - cfg.stand_height) / 0.04))
        upright = np.exp(-self._tilt_sq() / 0.40**2)
        gate = self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        return np.asarray(height * upright * self._pose_score(ctx, 0.40) * gate)

    def _upright_after_roll(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        assert ctx.gravity is not None
        gate = self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        return np.asarray(np.maximum(-ctx.gravity[:, 2], 0.0) * gate)

    def _height_after_roll(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        score = np.exp(-np.square((ctx.base_height - cfg.stand_height) / 0.04))
        return np.asarray(
            score * self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        )

    def _landing_sharp(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        height = np.exp(-np.square((ctx.base_height - cfg.stand_height) / 0.015))
        upright = np.exp(-self._tilt_sq() / 0.30**2)
        return np.asarray(
            height
            * upright
            * self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        )

    def _stand_shortfall(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        shortfall = np.maximum(cfg.stand_height - ctx.base_height, 0.0)
        return np.asarray(
            shortfall * self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        )

    def _rise_velocity(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        gate = self._completion_gate(ctx.info, cfg.rise_gate_lo, cfg.rise_gate_hi)
        active = ctx.base_height < cfg.stand_height + 0.01
        return np.asarray(np.maximum(ctx.info["base_lin_vel_w"][:, 2], 0.0) * active * gate)

    def _sagittal(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(np.square(ctx.gyro[:, 0]) + np.square(ctx.gyro[:, 2]))

    def _lateral_velocity(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(np.square(ctx.linvel[:, 1]))

    def _flatness(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        return np.asarray(np.square(self._lateral_axis_z(self._backend.get_base_quat())))

    def _self_collisions(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        contacts = [
            np.asarray(self._backend.get_sensor_data(name)).reshape(self._num_envs, -1)
            for name in self._cfg.sensor.self_collision
        ]
        return np.asarray(
            np.sum([np.any(value != 0.0, axis=1) for value in contacts], axis=0),
            dtype=get_global_dtype(),
        )

    def _joint_torque_rate(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["roulade_torque_rate"], dtype=get_global_dtype())

    def _angular_momentum(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        momentum = np.asarray(self._backend.get_sensor_data("root_angmom")).reshape(
            self._num_envs, -1
        )
        return np.asarray(np.sum(np.square(momentum), axis=1))

    def _arrival_damping(self, ctx: RewardContext) -> np.ndarray:
        height_gate = self._smoothstep((ctx.base_height - 0.09) / 0.02)
        assert ctx.gravity is not None
        tilt_deg = np.rad2deg(np.arccos(np.clip(-ctx.gravity[:, 2], -1.0, 1.0)))
        tilt_gate = self._smoothstep((45.0 - tilt_deg) / 25.0)
        return np.asarray(np.sum(np.square(ctx.gyro[:, :2]), axis=1) * height_gate * tilt_gate)

    def _gentle_landing(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["roulade_vertical_accel"], dtype=get_global_dtype())

    @staticmethod
    def _schedule_weight(
        default: float,
        stages: list[dict[str, float | int]],
        step: int,
    ) -> float:
        value = float(default)
        previous = -1
        for stage in stages:
            stage_step = int(stage["step"])
            if stage_step <= previous:
                raise ValueError("roulade reward curriculum steps must be strictly increasing")
            previous = stage_step
            if step < stage_step:
                break
            value = float(stage["weight"])
        return value

    def _reward_scales_for_step(self) -> dict[str, float]:
        cfg = self._roulade_reward_cfg
        scales = {name: float(value) for name, value in cfg.scales.items()}
        if not self._cfg.curriculum.enabled or self._cfg.curriculum.play_mode:
            return scales
        schedules = {
            "action_rate": cfg.action_rate_schedule,
            "arrival_damping": cfg.arrival_damping_schedule,
            "gentle_landing": cfg.gentle_landing_schedule,
            "joint_torque_rate": cfg.torque_rate_schedule,
        }
        for name, stages in schedules.items():
            if name in scales:
                scales[name] = self._schedule_weight(
                    scales[name], stages, self._cfg.curriculum.reference_step(self.step_counter)
                )
        return scales

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_quat = self._backend.get_base_quat()
        base_lin_vel_w = self._backend.get_base_lin_vel()
        base_height = self._backend.get_base_pos()[:, 2]
        state.info["base_lin_vel_w"] = np.asarray(base_lin_vel_w, dtype=get_global_dtype())
        contact, force = self._critic_foot_state()
        state.info["roulade_critic_linvel"] = np_quat_apply_inverse(base_quat, base_lin_vel_w)
        state.info["roulade_critic_contact"] = contact
        state.info["roulade_critic_force"] = force
        state.info["roulade_critic_air_time"] = np.where(
            contact > 0,
            0.0,
            state.info["roulade_critic_air_time"] + self._cfg.ctrl_dt,
        ).astype(get_global_dtype())
        self._update_foot_contact_history(state.info)
        self._update_roulade_state(state.info, gyro, base_quat, base_lin_vel_w)
        target = (
            state.info["current_actions"] * self._cfg.control_config.action_scale
            + self.default_angles
        )
        torque = self._actuator_kp * (target - dof_pos) - self._actuator_kd * dof_vel
        torque = np.clip(
            torque,
            -self._roulade_reward_cfg.actuator_force_limit,
            self._roulade_reward_cfg.actuator_force_limit,
        )
        previous_torque = state.info["roulade_prev_torque"]
        torque_delta = torque - previous_torque
        torque_delta[~state.info["roulade_prev_torque_valid"]] = 0.0
        state.info["roulade_torque_rate"] = np.sum(np.square(torque_delta), axis=1)
        state.info["roulade_prev_torque"] = np.asarray(torque, dtype=get_global_dtype())
        state.info["roulade_prev_torque_valid"][:] = True

        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_scales_for_step(),
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        finite = (
            np.isfinite(obs["obs"]).all(axis=1)
            & np.isfinite(obs["critic"]).all(axis=1)
            & np.isfinite(reward)
            & np.isfinite(base_height)
        )
        current_progress = np.asarray(state.info["roulade_accum"], dtype=get_global_dtype())
        frontier_progress = np.asarray(state.info["roulade_max"], dtype=get_global_dtype())
        state.info.setdefault("log", {})["roulade/mean_progress_rad"] = float(
            np.mean(current_progress)
        )
        state.info["log"]["roulade/frontier_progress_rad"] = float(
            np.mean(frontier_progress)
        )
        state.info["log"]["roulade/frontier_giveback_rad"] = float(
            np.mean(np.maximum(frontier_progress - current_progress, 0.0))
        )
        state.info["log"]["roulade/head_latch_rate"] = float(
            np.mean(state.info["roulade_head_latch"])
        )
        landing_ready = (
            current_progress >= self._roulade_reward_cfg.landing_gate_lo
        ) & state.info["roulade_head_latch"]
        state.info["log"]["roulade/landing_gate_rate"] = float(np.mean(landing_ready))
        state.info["log"]["roulade/full_rotation_rate"] = float(
            np.mean(current_progress >= self._roulade_reward_cfg.target_angle)
        )
        head_home = self._head_is_home(dof_pos)
        state.info["log"]["roulade/head_home_rate"] = float(np.mean(head_home))
        tilt_deg = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
        success = (
            (current_progress >= self._roulade_reward_cfg.landing_gate_hi)
            & state.info["roulade_head_latch"]
            & (base_height >= self._roulade_reward_cfg.success_height)
            & (tilt_deg <= self._roulade_reward_cfg.success_tilt_deg)
        )
        state.info["log"]["roulade/success_rate"] = float(np.mean(success))
        return state.replace(obs=obs, reward=reward, terminated=~finite)


assert MICRODUCK_ACTOR_OBS_DIM == 61
