"""Forward-roll task for the enlarged Micro Duck V1.0.6 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MICRODUCK_ENLARGED_ACTOR_OBS_DIM,
    NUM_MICRODUCK_ENLARGED_ACTIONS,
    MicroDuckEnlargedCommands,
    MicroDuckEnlargedRewardConfig,
    MicroDuckEnlargedSensor,
)
from unilab.envs.locomotion.microduck_enlarged_106.roulade_reference import (
    RouladeReference,
    RouladeReferenceConfig,
)
from unilab.envs.locomotion.microduck_enlarged_106.tasks import (
    MicroDuckEnlarged106WalkDomainRandConfig,
    MicroDuckEnlarged106WalkDomainRandomizationProvider,
    MicroDuckEnlarged106WalkFlatCfg,
    MicroDuckEnlarged106WalkFlatEnv,
)
from unilab.utils.rotation import (
    np_quat_apply,
    np_quat_apply_inverse,
    np_quat_from_euler_xyz,
)

_LEG_JOINT_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
_HEAD_PITCH_INDICES = np.asarray((5, 6), dtype=np.intp)
_HEAD_YAW_ROLL_INDICES = np.asarray((7, 8), dtype=np.intp)
_HEAD_TOP_AXIS = np.asarray((0.882, 0.0, 0.471), dtype=np.float64)
_HEAD_TOP_DOWN_MIN = 0.3
_HEAD_LATCH_LO = np.deg2rad(20.0)
_HEAD_LATCH_HI = np.deg2rad(170.0)
_FLAT_FULL = 0.5
_FLAT_ZERO = 0.866
MICRODUCK_ENLARGED_106_ROULADE_CRITIC_OBS_DIM = 74

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
class MicroDuckEnlarged106RouladeSensor(MicroDuckEnlargedSensor):
    critic_feet: tuple[str, str] = (
        "roulade_left_foot_state",
        "roulade_right_foot_state",
    )
    head_ground: tuple[str, str, str] = (
        "roulade_head_top_contact",
        "roulade_jaw_contact",
        "roulade_head_bottom_contact",
    )
    support_ground: tuple[str, ...] = (
        "left_foot_contact",
        "right_foot_contact",
        "roulade_trunk_contact",
        "roulade_head_top_contact",
        "roulade_jaw_contact",
        "roulade_head_bottom_contact",
    )


@dataclass
class MicroDuckEnlarged106RouladeCommands(MicroDuckEnlargedCommands):
    """Zero command block retained for the compatible 61D actor input."""

    vel_limit: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rel_standing_envs: float = 1.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    head_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 4, [0.0] * 4])
    body_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 6, [0.0] * 6])


@dataclass
class MicroDuckEnlarged106RouladeDomainRandConfig(MicroDuckEnlarged106WalkDomainRandConfig):
    push_robots: bool = False
    randomize_body_mass: bool = True
    body_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.97, 1.03])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_dof_armature: bool = True
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.924, 1.25])


@dataclass
class MicroDuckEnlarged106RouladeCurriculumConfig:
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
                raise ValueError("roulade spawn curriculum steps must increase strictly")
            previous_step = stage_step
            if step < stage_step:
                break
            active = stage
        standing = float(active["standing_prob"])
        midroll = float(active["midroll_prob"])
        if standing < 0.0 or midroll < 0.0 or standing + midroll <= 0.0:
            raise ValueError("roulade spawn probabilities require a positive sum")
        total = standing + midroll
        return standing / total, midroll / total

    def reference_step(self, physical_step: int) -> int:
        if self.num_steps_per_iteration < 24 or self.num_steps_per_iteration % 24:
            raise ValueError("roulade collection requires complete 24-step segments")
        return physical_step * 24 // self.num_steps_per_iteration


@dataclass
class MicroDuckEnlarged106RouladeRewardConfig(MicroDuckEnlargedRewardConfig):
    target_angle: float = 2.0 * np.pi
    max_paid_rate: float = 5.0
    overspeed_omega_max: float = 7.0
    stand_height: float = 0.2446
    landing_height_std: float = 0.080
    landing_height_sharp_std: float = 0.030
    landing_gate_lo: float = float(np.deg2rad(260.0))
    landing_gate_hi: float = float(np.deg2rad(330.0))
    rise_gate_lo: float = float(np.deg2rad(180.0))
    rise_gate_hi: float = float(np.deg2rad(260.0))
    success_height: float = 0.210
    success_tilt_deg: float = 15.0
    success_head_pitch_tol: float = float(np.deg2rad(10.0))
    success_head_yaw_roll_tol: float = float(np.deg2rad(5.0))
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


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_106" / "scene_roulade.xml")
    )


@registry.envcfg("MicroDuckEnlarged106RouladeFlat")
@dataclass
class MicroDuckEnlarged106RouladeFlatCfg(MicroDuckEnlarged106WalkFlatCfg):
    scene: SceneCfg = field(default_factory=_scene)
    sensor: MicroDuckEnlarged106RouladeSensor = field(  # type: ignore[assignment]
        default_factory=MicroDuckEnlarged106RouladeSensor
    )
    commands: MicroDuckEnlarged106RouladeCommands = field(
        default_factory=MicroDuckEnlarged106RouladeCommands
    )
    reward_config: MicroDuckEnlarged106RouladeRewardConfig | None = None
    domain_rand: MicroDuckEnlarged106RouladeDomainRandConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckEnlarged106RouladeDomainRandConfig
    )
    curriculum: MicroDuckEnlarged106RouladeCurriculumConfig = field(
        default_factory=MicroDuckEnlarged106RouladeCurriculumConfig
    )
    max_episode_seconds: float = 5.0
    reset_base_qvel_limit: float = 0.0
    standing_z_range: tuple[float, float] = (0.235, 0.250)
    standing_tilt_max: float = float(np.deg2rad(5.0))
    forward_vel_range: tuple[float, float] = (0.0, 0.0)
    midroll_pitch_range: tuple[float, float] = (
        float(np.deg2rad(50.0)),
        float(np.deg2rad(340.0)),
    )
    midroll_z_range: tuple[float, float] = (0.10, 0.21)
    midroll_omega_range: tuple[float, float] = (0.0, 3.0)
    tuck_factor_range: tuple[float, float] = (0.3, 1.0)
    joint_noise_std: float = 0.08
    policy_action_clip: float | None = None
    reference_motion: RouladeReferenceConfig = field(default_factory=RouladeReferenceConfig)

    def validate(self) -> None:
        super().validate()
        self.curriculum.spawn_probabilities(0)
        if self.policy_action_clip is not None and self.policy_action_clip <= 0.0:
            raise ValueError("policy_action_clip must be positive or null")
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


class MicroDuckEnlarged106RouladeDomainRandomizationProvider(
    MicroDuckEnlarged106WalkDomainRandomizationProvider
):
    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del linvel, gravity
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
            plan.qvel[is_mid, 4] = np.random.uniform(*env.cfg.midroll_omega_range, size=mid_count)

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
                    (num_reset, NUM_MICRODUCK_ENLARGED_ACTIONS),
                    dtype=get_global_dtype(),
                ),
                "roulade_prev_torque_valid": np.zeros(num_reset, dtype=bool),
            }
        )
        return plan


@registry.env("MicroDuckEnlarged106RouladeFlat", sim_backend="mujoco")
class MicroDuckEnlarged106RouladeFlatEnv(MicroDuckEnlarged106WalkFlatEnv):
    _cfg: MicroDuckEnlarged106RouladeFlatCfg

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": MICRODUCK_ENLARGED_ACTOR_OBS_DIM,
            "critic": MICRODUCK_ENLARGED_106_ROULADE_CRITIC_OBS_DIM,
        }

    def build_symmetry_augmentation(self, *, device: str):
        from unilab.envs.locomotion.microduck.symmetry import (
            MicroDuckSymmetryAugmentation,
        )

        return MicroDuckSymmetryAugmentation(device=device)

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged106RouladeDomainRandomizationProvider:
        return MicroDuckEnlarged106RouladeDomainRandomizationProvider()

    def __init__(
        self,
        cfg: MicroDuckEnlarged106RouladeFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        super().__init__(cfg, num_envs, backend_type)
        self._joint_range = self._backend.get_joint_range()
        self._head_body_ids = self._backend.get_body_ids(("jaw_soft",))
        self._actuator_kp, self._actuator_kd = self._backend.get_actuator_gains()
        self._reference_motion = RouladeReference(cfg.reference_motion)

    def _reference_pose(self, ctx: RewardContext) -> np.ndarray:
        return self._reference_motion.score(ctx)

    def apply_action(self, action: np.ndarray, state: NpEnvState) -> np.ndarray:
        clip = self._cfg.policy_action_clip
        if clip is None:
            return super().apply_action(action, state)
        raw_action = np.asarray(action, dtype=get_global_dtype())
        state.info["roulade_action_clip_rate"] = np.mean(
            np.abs(raw_action) > clip, axis=1
        ).astype(get_global_dtype())
        return super().apply_action(np.clip(raw_action, -clip, clip), state)

    @property
    def _roulade_reward_cfg(self) -> MicroDuckEnlarged106RouladeRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckEnlarged106RouladeRewardConfig):
            raise TypeError("V1.0.6 roulade requires its task reward config")
        return cfg

    def _critic_foot_state(
        self, env_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = [
            np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype()).reshape(
                self._num_envs, 4
            )
            for name in self._cfg.sensor.critic_feet
        ]
        data = np.stack(samples, axis=1)
        if env_ids is not None:
            data = data[env_ids]
        return (
            (data[:, :, 0] > 0).astype(get_global_dtype()),
            data[:, :, 1:4].reshape(-1, 6),
        )

    def _compute_obs(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(info, gyro, projected_gravity, dof_pos, dof_vel)
        force = info["roulade_critic_force"]
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

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "roulade_reference_pose": self._reference_pose,
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
            "body_ang_vel": rewards.ang_vel_xy,
            "angular_momentum": self._angular_momentum,
            "arrival_damping": self._arrival_damping,
            "gentle_landing": self._gentle_landing,
            "joint_torque_rate": self._joint_torque_rate,
            "action_rate": rewards.action_rate,
            "motor_peak_envelope": self._motor_peak_envelope,
            "motor_peak_envelope_max": self._motor_peak_envelope_max,
            "motor_joint_target_limit": self._motor_joint_target_limit,
            "motor_rated_torque": self._motor_rated_torque_penalty,
            "motor_rated_power": self._motor_rated_power_penalty,
            "motor_speed": self._motor_speed_penalty,
            "motor_evidence_speed": self._motor_evidence_speed_penalty,
            "motor_evidence_speed_max": self._motor_evidence_speed_max_penalty,
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
        return np_quat_apply(quat, axis)[:, 2] < -_HEAD_TOP_DOWN_MIN

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        x = np.clip(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _completion_gate(self, info: dict[str, Any], low: float, high: float) -> np.ndarray:
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
        delta = np.nan_to_num(gyro[:, 1], nan=0.0) * self._cfg.ctrl_dt
        accum = np.asarray(info["roulade_accum"], dtype=get_global_dtype())
        accum = accum + delta * supported * flat_gate
        frontier = np.maximum(info["roulade_max"], accum)
        latch_window = (accum > _HEAD_LATCH_LO) & (accum < _HEAD_LATCH_HI)
        latch = info["roulade_head_latch"] | (head_contact & latch_window & self._head_top_down())
        new_paid = np.minimum(frontier, cfg.target_angle)
        old_paid = np.minimum(info["roulade_paid"], cfg.target_angle)
        paid_delta = np.clip(new_paid - old_paid, 0.0, cfg.max_paid_rate * self._cfg.ctrl_dt)
        info["roulade_accum"] = np.asarray(accum, dtype=get_global_dtype())
        info["roulade_max"] = np.asarray(frontier, dtype=get_global_dtype())
        info["roulade_paid"] = np.maximum(info["roulade_paid"], new_paid).astype(get_global_dtype())
        info["roulade_head_latch"] = latch
        info["roulade_progress_signal"] = np.asarray(
            paid_delta / (self._cfg.ctrl_dt * cfg.target_angle),
            dtype=get_global_dtype(),
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
            np.abs(ctx.gyro[:, 1]) - self._roulade_reward_cfg.overspeed_omega_max,
            0.0,
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
        pitch_ok = np.all(error[:, _HEAD_PITCH_INDICES] <= cfg.success_head_pitch_tol, axis=1)
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
        height = np.exp(-np.square((ctx.base_height - cfg.stand_height) / cfg.landing_height_std))
        upright = np.exp(-self._tilt_sq() / 0.40**2)
        gate = self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        return np.asarray(height * upright * self._pose_score(ctx, 0.40) * gate)

    def _upright_after_roll(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        cfg = self._roulade_reward_cfg
        gate = self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        return np.asarray(np.maximum(-ctx.gravity[:, 2], 0.0) * gate)

    def _height_after_roll(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        score = np.exp(-np.square((ctx.base_height - cfg.stand_height) / cfg.landing_height_std))
        return np.asarray(
            score * self._completion_gate(ctx.info, cfg.landing_gate_lo, cfg.landing_gate_hi)
        )

    def _landing_sharp(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._roulade_reward_cfg
        height = np.exp(
            -np.square((ctx.base_height - cfg.stand_height) / cfg.landing_height_sharp_std)
        )
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
        active = ctx.base_height < cfg.stand_height + 0.02
        return np.asarray(np.maximum(ctx.info["base_lin_vel_w"][:, 2], 0.0) * active * gate)

    def _sagittal(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(np.square(ctx.gyro[:, 0]) + np.square(ctx.gyro[:, 2]))

    def _lateral_velocity(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(np.square(ctx.linvel[:, 1]))

    def _flatness(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        return np.asarray(np.square(self._lateral_axis_z(self._backend.get_base_quat())))

    def _joint_torque_rate(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["roulade_torque_rate"], dtype=get_global_dtype())

    def _angular_momentum(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        momentum = np.asarray(self._backend.get_sensor_data("root_angmom")).reshape(
            self._num_envs, -1
        )
        return np.asarray(np.sum(np.square(momentum), axis=1))

    def _arrival_damping(self, ctx: RewardContext) -> np.ndarray:
        height_gate = self._smoothstep((ctx.base_height - 0.18) / 0.04)
        assert ctx.gravity is not None
        tilt_deg = np.rad2deg(np.arccos(np.clip(-ctx.gravity[:, 2], -1.0, 1.0)))
        tilt_gate = self._smoothstep((45.0 - tilt_deg) / 25.0)
        return np.asarray(np.sum(np.square(ctx.gyro[:, :2]), axis=1) * height_gate * tilt_gate)

    def _gentle_landing(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(ctx.info["roulade_vertical_accel"], dtype=get_global_dtype())

    @staticmethod
    def _schedule_weight(default: float, stages: list[dict[str, float | int]], step: int) -> float:
        value = float(default)
        previous = -1
        for stage in stages:
            stage_step = int(stage["step"])
            if stage_step <= previous:
                raise ValueError("roulade reward curriculum steps must increase strictly")
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
        step = self._cfg.curriculum.reference_step(self.step_counter)
        for name, stages in schedules.items():
            if name in scales:
                scales[name] = self._schedule_weight(scales[name], stages, step)
        return scales

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_motor_speed_metrics(dof_vel, state.info)
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

        torque = np.asarray(
            state.info.get(
                "motor_effective_torque_nm",
                self._actuator_kp
                * (
                    state.info["current_actions"] * self._cfg.control_config.action_scale
                    + self.default_angles
                    - dof_pos
                )
                - self._actuator_kd * dof_vel,
            ),
            dtype=get_global_dtype(),
        )
        torque_delta = torque - state.info["roulade_prev_torque"]
        torque_delta[~state.info["roulade_prev_torque_valid"]] = 0.0
        state.info["roulade_torque_rate"] = np.sum(np.square(torque_delta), axis=1)
        state.info["roulade_prev_torque"] = torque.copy()
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
        self._control_step += 1
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        finite = (
            np.isfinite(obs["obs"]).all(axis=1)
            & np.isfinite(obs["critic"]).all(axis=1)
            & np.isfinite(reward)
            & np.isfinite(base_height)
        )
        current = np.asarray(state.info["roulade_accum"], dtype=get_global_dtype())
        frontier = np.asarray(state.info["roulade_max"], dtype=get_global_dtype())
        log = state.info.setdefault("log", {})
        log["roulade/mean_progress_rad"] = float(np.mean(current))
        log["roulade/frontier_progress_rad"] = float(np.mean(frontier))
        log["roulade/frontier_giveback_rad"] = float(np.mean(np.maximum(frontier - current, 0.0)))
        log["roulade/head_latch_rate"] = float(np.mean(state.info["roulade_head_latch"]))
        landing_ready = (current >= self._roulade_reward_cfg.landing_gate_lo) & state.info[
            "roulade_head_latch"
        ]
        log["roulade/landing_gate_rate"] = float(np.mean(landing_ready))
        log["roulade/full_rotation_rate"] = float(
            np.mean(current >= self._roulade_reward_cfg.target_angle)
        )
        log["roulade/head_home_rate"] = float(np.mean(self._head_is_home(dof_pos)))
        tilt_deg = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
        success = (
            (current >= self._roulade_reward_cfg.landing_gate_hi)
            & state.info["roulade_head_latch"]
            & (base_height >= self._roulade_reward_cfg.success_height)
            & (tilt_deg <= self._roulade_reward_cfg.success_tilt_deg)
        )
        log["roulade/success_rate"] = float(np.mean(success))
        return state.replace(obs=obs, reward=reward, terminated=~finite)


assert MICRODUCK_ENLARGED_ACTOR_OBS_DIM == 61


__all__ = [
    "MICRODUCK_ENLARGED_106_ROULADE_CRITIC_OBS_DIM",
    "MicroDuckEnlarged106RouladeCurriculumConfig",
    "MicroDuckEnlarged106RouladeDomainRandConfig",
    "MicroDuckEnlarged106RouladeDomainRandomizationProvider",
    "MicroDuckEnlarged106RouladeFlatCfg",
    "MicroDuckEnlarged106RouladeFlatEnv",
    "MicroDuckEnlarged106RouladeRewardConfig",
    "MicroDuckEnlarged106RouladeSensor",
]
