"""Commanded sit/stand training for the original Micro Duck.

The task mirrors the public ``alpha_sitstand.onnx`` contract while keeping the
implementation native to UniLab's NumPy environment API:

* ``commands[:, 0]`` is a binary posture flag (0 = stand, 1 = sit), with the
  other two twist entries fixed at zero;
* the actor keeps the walking policy's 61-dimensional observation layout and
  14-dimensional position-offset action layout;
* the target used by the reward is a slowly slewed blend of HOME and the
  measured stable sitting pose, while the observation still exposes the raw
  binary command used by deployment;
* resets are an even mixture of standing and sitting states, independent of
  the command sampled for that episode;
* tilt receives an explicit penalty.  The official policy stays well inside
  this envelope, while a direct penalty prevents PPO from treating a
  sideways seated pose as a valid solution without changing the upstream
  task's non-fall-termination contract.

The full-collision scene is shared with the existing ``MicroDuckVelStand``
owner.  This makes seated contacts physical without changing the public robot
asset or the shared simulator interfaces.
"""

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
from unilab.envs.locomotion.microduck.walk import (
    MICRODUCK_JOINT_NAMES,
    NUM_MICRODUCK_ACTIONS,
    MicroDuckCommands,
    MicroDuckRewardConfig,
    MicroDuckWalkDomainRandomizationProvider,
    MicroDuckWalkFlatCfg,
    MicroDuckWalkFlatEnv,
)

# These values are measured on the original full-collision Micro Duck scene and
# are deliberately kept as named constants: they are part of the upstream
# sitstand task contract, not generic locomotion defaults.
STAND_Z = 0.115
SIT_Z = 0.060
POSTURE_RAMP_S = 2.0

# Servo-index -> absolute joint angle.  The four head joints are intentionally
# absent; the head remains commandable through the existing head-command block.
SITTING_TARGET_OVERRIDES: dict[int, float] = {
    1: 0.0,
    2: -0.4079,
    3: 1.35,
    4: 0.0,
    10: 0.0,
    11: 0.4079,
    12: -1.35,
    13: 0.0,
}

SITSTAND_LEG_JOINT_INDICES = np.asarray(
    (0, 1, 2, 3, 4, 9, 10, 11, 12, 13),
    dtype=np.intp,
)
SITSTAND_HEAD_JOINT_INDICES = np.asarray((5, 6, 7, 8), dtype=np.intp)


@dataclass
class MicroDuckSitStandCommands(MicroDuckCommands):
    """Binary posture command carried in the upstream twist observation slot."""

    vel_limit: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sit_prob: float = 0.5
    rel_standing_envs: float = 0.0
    rel_forward_envs: float = 0.0
    rel_turn_in_place_envs: float = 0.0
    body_limit: list[list[float]] = field(default_factory=lambda: [[0.0] * 6, [0.0] * 6])


@dataclass
class MicroDuckSitStandRewardConfig(MicroDuckRewardConfig):
    """Reward parameters for the commanded two-posture task.

    Terms whose function already returns a negative penalty use positive
    weights in the task YAML.  This sign convention is important for the
    descent/rise speed caps and the vertical-acceleration shock cost.
    """

    scales: dict[str, float]
    posture_pose_std: float = 0.5
    posture_height_std: float = 0.04
    posture_height_sharp_std: float = 0.015
    max_descent_speed: float = 0.05
    max_rise_speed: float = 0.08
    rise_bootstrap_max_height: float = 0.125
    rise_bootstrap_max_speed: float = 0.08
    upright_tall_low: float = 0.075
    upright_tall_high: float = 0.10
    stillness_band_full: float = 0.012
    stillness_band_zero: float = 0.03
    stillness_vel_std: float = 0.05
    stillness_tilt_full_deg: float = 25.0
    stillness_tilt_zero_deg: float = 60.0
    composite_height_std: float = 0.03
    composite_upright_std: float = 0.40
    composite_pose_std: float = 0.40
    composite_head_std: float = 0.40


@dataclass
class MicroDuckSitStandDomainRandomizationProvider(MicroDuckWalkDomainRandomizationProvider):
    """Reset and dwell-command provider for the sitstand owner."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = np.zeros((num_reset, 3), dtype=get_global_dtype())
        sit_prob = float(env.cfg.commands.sit_prob)
        if not 0.0 <= sit_prob <= 1.0:
            raise ValueError("Micro Duck sit_prob must be within [0, 1]")
        commands[:, 0] = (np.random.uniform(size=num_reset) < sit_prob).astype(get_global_dtype())
        return commands

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        # The body command occupies six observation slots for protocol parity,
        # but the official sitstand task does not command body pose.
        updates["body_commands"] = np.zeros((num_reset, 6), dtype=get_global_dtype())
        updates["posture_blend"] = np.zeros((num_reset,), dtype=get_global_dtype())
        updates["sitstand_target_blend"] = np.zeros((num_reset,), dtype=get_global_dtype())
        updates["sitstand_reset_mode"] = np.zeros((num_reset,), dtype=np.int8)
        updates["sitstand_reset_is_sitting"] = np.zeros((num_reset,), dtype=bool)
        updates["previous_base_vz"] = np.zeros((num_reset,), dtype=get_global_dtype())
        return updates

    @staticmethod
    def _upright_quaternions(
        num: int,
        tilt_max_rad: float,
    ) -> np.ndarray:
        """Return random-yaw quaternions with bounded pitch and roll."""

        yaw = np.random.uniform(-np.pi, np.pi, size=num)
        pitch = np.random.uniform(-tilt_max_rad, tilt_max_rad, size=num)
        roll = np.random.uniform(-tilt_max_rad, tilt_max_rad, size=num)
        cy = np.cos(0.5 * yaw)
        sy = np.sin(0.5 * yaw)
        cp = np.cos(0.5 * pitch)
        sp = np.sin(0.5 * pitch)
        cr = np.cos(0.5 * roll)
        sr = np.sin(0.5 * roll)
        return np.asarray(
            np.stack(
                (
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ),
                axis=1,
            ),
            dtype=get_global_dtype(),
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        if num_reset == 0:
            return plan

        sit_prob = float(env.cfg.sitting_reset_prob)
        stand_prob = float(env.cfg.standing_reset_prob)
        if sit_prob < 0.0 or stand_prob < 0.0 or sit_prob + stand_prob <= 0.0:
            raise ValueError(
                "Micro Duck sitstand reset probabilities must be non-negative "
                "and have a positive sum"
            )
        sitting = np.random.uniform(size=num_reset) < sit_prob / (sit_prob + stand_prob)

        plan.qpos[:, 2] = np.where(
            sitting,
            np.random.uniform(
                float(env.cfg.sitting_z_min),
                float(env.cfg.sitting_z_max),
                size=num_reset,
            ),
            np.random.uniform(
                float(env.cfg.standing_z_min),
                float(env.cfg.standing_z_max),
                size=num_reset,
            ),
        )
        plan.qpos[:, 3:7] = self._upright_quaternions(
            num_reset,
            np.deg2rad(float(env.cfg.reset_tilt_max_deg)),
        )
        # Starting with zero generalized velocity is essential for a stable
        # seated contact state and matches the upstream ground-state event.
        plan.qvel[:, :] = 0.0

        sit_indices = np.flatnonzero(sitting)
        if sit_indices.size:
            for joint_index, angle in SITTING_TARGET_OVERRIDES.items():
                plan.qpos[sit_indices, 7 + int(joint_index)] = float(angle)
            noise_std = float(env.cfg.sitting_joint_noise_std)
            if noise_std < 0.0:
                raise ValueError("sitting_joint_noise_std must be non-negative")
            if noise_std > 0.0:
                plan.qpos[sit_indices, 7:] += np.random.normal(
                    0.0,
                    noise_std,
                    size=(sit_indices.size, len(MICRODUCK_JOINT_NAMES)),
                ).astype(get_global_dtype())

        reset_mode = sitting.astype(np.int8)
        initial_blend = np.clip(
            (float(env.cfg.stand_z) - plan.qpos[:, 2])
            / max(float(env.cfg.stand_z) - float(env.cfg.sit_z), 1e-6),
            0.0,
            1.0,
        ).astype(get_global_dtype())
        plan.info_updates["posture_blend"] = initial_blend
        plan.info_updates["sitstand_target_blend"] = initial_blend.copy()
        plan.info_updates["sitstand_reset_mode"] = reset_mode
        plan.info_updates["sitstand_reset_is_sitting"] = sitting
        plan.info_updates["previous_base_vz"] = np.zeros((num_reset,), dtype=get_global_dtype())
        return plan


@registry.envcfg("MicroDuckSitStandFlat")
@dataclass
class MicroDuckSitStandFlatCfg(MicroDuckWalkFlatCfg):
    """Flat full-collision sit/stand task for the original Micro Duck."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck" / "scene_velstand_flat.xml")
        )
    )
    commands: MicroDuckSitStandCommands = field(default_factory=MicroDuckSitStandCommands)
    reward_config: MicroDuckSitStandRewardConfig | None = None
    max_episode_seconds: float = 12.0
    twist_resample_s: tuple[float, float] = (3.5, 6.5)
    sit_z: float = SIT_Z
    stand_z: float = STAND_Z
    posture_ramp_s: float = POSTURE_RAMP_S
    sitting_reset_prob: float = 0.5
    standing_reset_prob: float = 0.5
    sitting_z_min: float = 0.060
    sitting_z_max: float = 0.075
    standing_z_min: float = 0.110
    standing_z_max: float = 0.120
    reset_tilt_max_deg: float = 8.0
    sitting_joint_noise_std: float = 0.10
    descent_speed_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": 10.0},
            {"step": 500 * 24, "weight": 20.0},
        ]
    )
    rise_speed_curriculum: list[dict[str, float | int]] = field(
        default_factory=lambda: [
            {"step": 0, "weight": 0.0},
            {"step": 1500 * 24, "weight": 5.0},
            {"step": 2500 * 24, "weight": 10.0},
        ]
    )


@registry.env("MicroDuckSitStandFlat", sim_backend="mujoco")
class MicroDuckSitStandFlatEnv(MicroDuckWalkFlatEnv):
    """Two-way sit/stand policy environment with the upstream I/O contract."""

    _cfg: MicroDuckSitStandFlatCfg

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckSitStandDomainRandomizationProvider:
        return MicroDuckSitStandDomainRandomizationProvider()

    @property
    def _sitstand_reward_cfg(self) -> MicroDuckSitStandRewardConfig:
        cfg = self._reward_cfg
        if not isinstance(cfg, MicroDuckSitStandRewardConfig):
            raise TypeError("MicroDuckSitStandFlat requires MicroDuckSitStandRewardConfig")
        return cfg

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "posture_pose_legs": self._posture_pose_legs,
            "head_pose_tracking": self._head_pose_tracking,
            "posture_pose_l1": self._posture_pose_l1,
            "posture_height": self._posture_height_gaussian,
            "posture_height_sharp": self._posture_height_sharp,
            "posture_height_l1": self._posture_height_l1,
            "rise_bootstrap": self._rise_bootstrap,
            "descent_speed": self._descent_speed,
            "rise_speed": self._rise_speed,
            "gentle_motion": self._gentle_motion,
            "upright_linear": self._upright_linear,
            "upright_while_tall": self._upright_while_tall,
            "tilt_penalty": self._tilt_penalty,
            "posture_stillness": self._posture_stillness,
            "posture_composite": self._posture_composite,
            "action_rate_l2": rewards.action_rate,
            # Keep the UniLab walking spelling available for small downstream
            # tools that switch between the two owners programmatically.
            "action_rate": rewards.action_rate,
            "body_ang_vel": rewards.ang_vel_xy,
        }

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _posture_blend(self, info: dict[str, Any]) -> np.ndarray:
        blend = info.get("posture_blend", info.get("sitstand_target_blend"))
        if blend is None:
            blend = np.asarray(info["commands"])[:, 0]
        return np.asarray(
            np.clip(
                np.nan_to_num(blend, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            ),
            dtype=get_global_dtype(),
        )

    def _advance_posture_blend(self, info: dict[str, Any]) -> np.ndarray:
        """Slew the internal target toward the raw binary posture command."""

        blend = self._posture_blend(info)
        flag = np.asarray(info["commands"], dtype=get_global_dtype())[:, 0]
        flag = np.clip(np.nan_to_num(flag, nan=0.0), 0.0, 1.0)
        ramp_s = float(self._cfg.posture_ramp_s)
        if ramp_s <= 0.0:
            blend = flag
        else:
            max_step = float(self._cfg.ctrl_dt) / ramp_s
            blend = blend + np.clip(flag - blend, -max_step, max_step)
        blend = np.asarray(np.clip(blend, 0.0, 1.0), dtype=get_global_dtype())
        info["posture_blend"] = blend
        info["sitstand_target_blend"] = blend.copy()
        return blend

    # Descriptive alias used by task-level diagnostics.
    def _update_posture_blend(self, info: dict[str, Any]) -> np.ndarray:
        return self._advance_posture_blend(info)

    def _posture_target(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        blend = self._posture_blend(info)
        stand = np.broadcast_to(self.default_angles, (self._num_envs, self._num_action)).copy()
        sit = stand.copy()
        for joint_index, angle in SITTING_TARGET_OVERRIDES.items():
            sit[:, int(joint_index)] = float(angle)
        target = stand + blend[:, None] * (sit - stand)
        return blend, np.asarray(target, dtype=get_global_dtype())

    def _posture_height_target(self, info: dict[str, Any]) -> np.ndarray:
        blend = self._posture_blend(info)
        return np.asarray(
            float(self._cfg.stand_z) + blend * (float(self._cfg.sit_z) - float(self._cfg.stand_z)),
            dtype=get_global_dtype(),
        )

    @staticmethod
    def _world_linvel_from_info(info: dict[str, Any], num_envs: int) -> np.ndarray:
        value = info.get("base_world_linvel")
        if value is None:
            return np.zeros((num_envs, 3), dtype=get_global_dtype())
        return np.asarray(
            np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )

    def _posture_pose_legs(self, ctx: RewardContext) -> np.ndarray:
        _, target = self._posture_target(ctx.info)
        error = ctx.dof_pos[:, SITSTAND_LEG_JOINT_INDICES] - target[:, SITSTAND_LEG_JOINT_INDICES]
        error = np.nan_to_num(error, nan=0.0, posinf=1e3, neginf=-1e3)
        std = float(self._sitstand_reward_cfg.posture_pose_std)
        if std <= 0.0:
            raise ValueError("posture_pose_std must be positive")
        return np.asarray(
            np.exp(-np.mean(np.square(error / std), axis=1)),
            dtype=get_global_dtype(),
        )

    def _head_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        # Reuse the walking implementation's exact HOME-relative head command
        # convention, including the four [neck, pitch, yaw, roll] slots.
        return super()._head_pose_tracking(ctx)

    def _posture_pose_l1(self, ctx: RewardContext) -> np.ndarray:
        _, target = self._posture_target(ctx.info)
        error = ctx.dof_pos[:, SITSTAND_LEG_JOINT_INDICES] - target[:, SITSTAND_LEG_JOINT_INDICES]
        error = np.nan_to_num(error, nan=0.0, posinf=1e3, neginf=-1e3)
        return np.asarray(-np.mean(np.abs(error), axis=1), dtype=get_global_dtype())

    def _height_error(self, ctx: RewardContext) -> np.ndarray:
        target = self._posture_height_target(ctx.info)
        actual = np.asarray(
            np.nan_to_num(ctx.base_height, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        return actual - target

    def _posture_height_gaussian(self, ctx: RewardContext) -> np.ndarray:
        std = float(self._sitstand_reward_cfg.posture_height_std)
        if std <= 0.0:
            raise ValueError("posture_height_std must be positive")
        error = self._height_error(ctx)
        return np.asarray(np.exp(-np.square(error / std)), dtype=get_global_dtype())

    def _posture_height_sharp(self, ctx: RewardContext) -> np.ndarray:
        std = float(self._sitstand_reward_cfg.posture_height_sharp_std)
        if std <= 0.0:
            raise ValueError("posture_height_sharp_std must be positive")
        error = self._height_error(ctx)
        return np.asarray(np.exp(-np.square(error / std)), dtype=get_global_dtype())

    def _posture_height_l1(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(-np.abs(self._height_error(ctx)), dtype=get_global_dtype())

    def _rise_bootstrap(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        vz = np.clip(velocity[:, 2], 0.0, None)
        if cfg.rise_bootstrap_max_speed > 0.0:
            vz = np.minimum(vz, cfg.rise_bootstrap_max_speed)
        base_height = np.asarray(
            np.nan_to_num(ctx.base_height, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        flag = np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 0]
        return np.asarray(
            vz * (base_height < cfg.rise_bootstrap_max_height) * (1.0 - np.clip(flag, 0.0, 1.0)),
            dtype=get_global_dtype(),
        )

    def _descent_speed(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        return np.asarray(
            -np.clip(-velocity[:, 2] - self._sitstand_reward_cfg.max_descent_speed, 0.0, None),
            dtype=get_global_dtype(),
        )

    def _rise_speed(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        return np.asarray(
            -np.clip(velocity[:, 2] - self._sitstand_reward_cfg.max_rise_speed, 0.0, None),
            dtype=get_global_dtype(),
        )

    def _gentle_motion(self, ctx: RewardContext) -> np.ndarray:
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        current_vz = velocity[:, 2]
        previous = np.asarray(
            ctx.info.get("previous_base_vz", current_vz),
            dtype=get_global_dtype(),
        )
        previous = np.nan_to_num(previous, nan=0.0, posinf=0.0, neginf=0.0)
        acceleration = (current_vz - previous) / max(float(self._cfg.ctrl_dt), 1e-6)
        steps = np.asarray(
            ctx.info.get("steps", np.ones((ctx.num_envs,), dtype=np.uint32)),
        )
        acceleration = np.where(steps <= 0, 0.0, acceleration)
        return np.asarray(-np.abs(acceleration), dtype=get_global_dtype())

    def _upright_cosine(self, ctx: RewardContext) -> np.ndarray:
        gravity = ctx.gravity
        if gravity is None:
            return np.ones((ctx.num_envs,), dtype=get_global_dtype())
        gravity = np.asarray(gravity, dtype=get_global_dtype())
        return np.clip(np.nan_to_num(-gravity[:, 2], nan=1.0), -1.0, 1.0)

    def _upright_linear(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(self._upright_cosine(ctx), dtype=get_global_dtype())

    def _upright_while_tall(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        actual = np.asarray(
            np.nan_to_num(ctx.base_height, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        gate = self._smoothstep(
            (actual - cfg.upright_tall_low)
            / max(cfg.upright_tall_high - cfg.upright_tall_low, 1e-6)
        )
        return np.asarray(self._upright_cosine(ctx) * gate, dtype=get_global_dtype())

    def _tilt_penalty(self, ctx: RewardContext) -> np.ndarray:
        """Penalize loss of upright support without terminating the task."""

        return np.asarray(-(1.0 - self._upright_cosine(ctx)), dtype=get_global_dtype())

    def _posture_stillness(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        velocity = self._world_linvel_from_info(ctx.info, ctx.num_envs)
        speed = np.linalg.norm(velocity, axis=1)
        vel_std = max(cfg.stillness_vel_std, 1e-6)
        velocity_score = np.exp(-np.square(speed / vel_std))

        height_error = np.abs(self._height_error(ctx))
        height_t = (cfg.stillness_band_zero - height_error) / max(
            cfg.stillness_band_zero - cfg.stillness_band_full, 1e-6
        )
        height_gate = self._smoothstep(height_t)

        cos_tilt = self._upright_cosine(ctx)
        cos_full = np.cos(np.deg2rad(cfg.stillness_tilt_full_deg))
        cos_zero = np.cos(np.deg2rad(cfg.stillness_tilt_zero_deg))
        tilt_gate = self._smoothstep((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6))

        flag = np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 0]
        blend = self._posture_blend(ctx.info)
        ramp_done = (np.abs(np.clip(flag, 0.0, 1.0) - blend) < 0.02).astype(get_global_dtype())
        return np.asarray(
            velocity_score * height_gate * tilt_gate * ramp_done,
            dtype=get_global_dtype(),
        )

    def _posture_composite(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._sitstand_reward_cfg
        height_error = self._height_error(ctx)
        height_score = np.exp(-np.square(height_error / max(cfg.composite_height_std, 1e-6)))

        cos_tilt = self._upright_cosine(ctx)
        tilt_sq = np.clip(1.0 - cos_tilt, 0.0, 2.0)
        upright_score = np.exp(
            -tilt_sq / max(cfg.composite_upright_std * cfg.composite_upright_std, 1e-6)
        )

        _, target = self._posture_target(ctx.info)
        joint_error = (
            ctx.dof_pos[:, SITSTAND_LEG_JOINT_INDICES] - target[:, SITSTAND_LEG_JOINT_INDICES]
        )
        joint_error = np.nan_to_num(joint_error, nan=0.0, posinf=1e3, neginf=-1e3)
        pose_score = np.exp(
            -np.mean(np.square(joint_error), axis=1)
            / max(cfg.composite_pose_std * cfg.composite_pose_std, 1e-6)
        )

        head = np.asarray(
            ctx.info.get(
                "head_commands",
                np.zeros((ctx.num_envs, SITSTAND_HEAD_JOINT_INDICES.size)),
            ),
            dtype=get_global_dtype(),
        )
        actual_head = (
            ctx.dof_pos[:, SITSTAND_HEAD_JOINT_INDICES]
            - ctx.default_angles[SITSTAND_HEAD_JOINT_INDICES]
        )
        head_error = np.nan_to_num(actual_head - head, nan=0.0, posinf=1e3, neginf=-1e3)
        head_score = np.exp(
            -np.mean(np.square(head_error), axis=1)
            / max(cfg.composite_head_std * cfg.composite_head_std, 1e-6)
        )
        return np.asarray(
            height_score * upright_score * pose_score * head_score, dtype=get_global_dtype()
        )

    def _reward_scales_for_step(self) -> dict[str, float]:
        scales = {name: float(value) for name, value in self._reward_cfg.scales.items()}

        def curriculum_weight(name: str, stages: list[dict[str, float | int]]) -> None:
            if name not in scales:
                return
            weight = scales[name]
            for stage in stages:
                if self.step_counter < int(stage["step"]):
                    break
                weight = float(stage["weight"])
            scales[name] = weight

        curriculum_weight("action_rate_l2", self._cfg.action_rate_curriculum)
        curriculum_weight("action_rate", self._cfg.action_rate_curriculum)
        curriculum_weight("descent_speed", self._cfg.descent_speed_curriculum)
        curriculum_weight("rise_speed", self._cfg.rise_speed_curriculum)
        return scales

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        base_world_linvel = np.asarray(self._backend.get_base_lin_vel(), dtype=get_global_dtype())
        state.info["base_height"] = np.asarray(
            np.nan_to_num(base_pos[:, 2], nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        state.info["base_world_linvel"] = np.asarray(
            np.nan_to_num(base_world_linvel, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=get_global_dtype(),
        )
        self._advance_posture_blend(state.info)

        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_scales_for_step(),
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

        state.info["previous_base_vz"] = state.info["base_world_linvel"][:, 2].copy()
        self._resample_commands_for_next_step(state.info)
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        for value in obs.values():
            np.nan_to_num(value, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(reward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        arrays = (linvel, gyro, gravity, dof_pos, dof_vel, base_pos, base_quat, base_world_linvel)
        invalid = np.zeros((self._num_envs,), dtype=bool)
        for value in arrays:
            finite = np.isfinite(np.asarray(value)).reshape(self._num_envs, -1).all(axis=1)
            invalid |= ~finite
        return state.replace(obs=obs, reward=reward, terminated=invalid)


__all__ = [
    "MICRODUCK_SIT_Z",
    "MICRODUCK_STAND_Z",
    "POSTURE_RAMP_S",
    "SITSTAND_HEAD_JOINT_INDICES",
    "SITSTAND_LEG_JOINT_INDICES",
    "SITTING_TARGET_OVERRIDES",
    "SIT_Z",
    "STAND_Z",
    "MicroDuckSitStandCommands",
    "MicroDuckSitStandDomainRandomizationProvider",
    "MicroDuckSitStandFlatCfg",
    "MicroDuckSitStandFlatEnv",
    "MicroDuckSitStandRewardConfig",
]


MICRODUCK_SIT_Z = SIT_Z
MICRODUCK_STAND_Z = STAND_Z
