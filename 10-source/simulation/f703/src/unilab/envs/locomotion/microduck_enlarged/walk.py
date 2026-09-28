"""Flat-ground locomotion for the enlarged Micro Duck task family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from unilab.actuators.gf43x import (
    GF43X10_10,
    GF43X40_10,
    GF43MotorSpec,
    gf43x40_torque_bounds,
    static_torque_magnitude_limit,
)
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
    Sensor,
)
from unilab.envs.locomotion.common.commands import Commands, zero_small_xy_commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

MICRODUCK_ENLARGED_JOINT_NAMES = (
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
)
NUM_MICRODUCK_ENLARGED_ACTIONS = len(MICRODUCK_ENLARGED_JOINT_NAMES)
MICRODUCK_ENLARGED_ACTOR_OBS_DIM = 61
_LEG_JOINT_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13), dtype=np.intp)
_HEAD_JOINT_INDICES = np.asarray((5, 6, 7, 8), dtype=np.intp)

# This is the actuator order in the supplied V1.0.4 MJCF.  Keep it explicit:
# the two GF43 variants have very different output envelopes, and silently
# broadcasting one motor's limits to all 14 joints would invalidate training.
MICRODUCK_ENLARGED_104_MOTOR_SPECS: tuple[GF43MotorSpec, ...] = (
    GF43X10_10,
    GF43X40_10,
    GF43X10_10,
    GF43X10_10,
    GF43X10_10,
    GF43X40_10,
    GF43X10_10,
    GF43X10_10,
    GF43X10_10,
    GF43X10_10,
    GF43X40_10,
    GF43X10_10,
    GF43X10_10,
    GF43X10_10,
)


@dataclass
class MicroDuckEnlargedSensor(Sensor):
    local_linvel: str = "imu_lin_vel"
    gyro: str = "imu_ang_vel"
    upvector: str = "imu_upvector"
    feet_contact: tuple[str, str] = ("left_foot_contact", "right_foot_contact")


@dataclass
class MicroDuckEnlargedAsset:
    base_name: str = "trunk_base"
    ground: str = "floor"
    foot_body_names: tuple[str, str] = ("ankle_left", "ankle_right")
    # Local foot-site offsets copied from the scaled MJCF, not the original robot.
    foot_site_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -0.03777488276, -0.02234204138),
        (0.0, -0.03777488276, -0.02234204138),
    )


@dataclass
class MicroDuckEnlargedControlConfig(PdControlConfig):
    """Position-target control for the enlarged Micro Duck baseline.

    The imported MJCF keeps its source ``kp=3.4818``.  A read-only settle
    sweep showed that gain cannot hold the supplied STAND pose, while kp=30
    remains below 1.4 N m in the same three-second trial.  The override is
    task-owned and intentionally does not claim real firmware parity.  The
    Version owners additionally apply the configured GF43 output envelope below.
    """

    action_scale: float = 0.25
    Kp: float = 30.0
    Kd: float = 0.0


@dataclass
class MicroDuckEnlargedMotorConstraintConfig:
    """Runtime use of the documented GF43 nameplate envelope.

    The supplied GF43 data is sufficient for an instantaneous
    constant-torque/constant-power envelope.  It is not sufficient to derive
    a phase-current, winding-temperature, or peak-duration model, so those
    quantities deliberately do not appear here.
    """

    enabled: bool = False
    hard_duty: str = "peak"
    hard_limit_margin: float = 0.98
    rated_speed_soft_limit: bool = True
    envelope_model: str = "constant_power"
    braking_duty: str = "rated"

    def validate(self) -> None:
        if self.hard_duty not in ("rated", "peak"):
            raise ValueError("motor hard_duty must be 'rated' or 'peak'")
        if self.envelope_model not in ("constant_power", "gf43x40_hybrid"):
            raise ValueError("motor envelope_model must be 'constant_power' or 'gf43x40_hybrid'")
        if self.braking_duty not in ("rated", "peak"):
            raise ValueError("motor braking_duty must be 'rated' or 'peak'")
        if not np.isfinite(self.hard_limit_margin) or not 0.0 < self.hard_limit_margin <= 1.0:
            raise ValueError("motor hard_limit_margin must be in (0, 1]")


@dataclass
class MicroDuckEnlargedIdentifiedDynamicsConfig:
    """Opt-in use of the measured GF43 empty-load equivalent dynamics.

    The fit is implemented in the owner layer so the shared ``SimBackend``
    contract stays backend-neutral.  ``first_order_response`` and
    ``asymmetric_friction`` are independent switches for controlled ablation;
    both are enabled by the V1.0.5 task configuration.
    """

    enabled: bool = False
    first_order_response: bool = True
    # Optional loaded-joint hypothesis.  This leaves the empty-load reference
    # fit untouched and permits task-specific robustness sampling.
    response_time_constant_range_s: tuple[float, float] | None = None
    response_command_gain: float | None = None
    response_command_gain_range: tuple[float, float] | None = None
    asymmetric_friction: bool = True
    randomize_friction_parameters: bool = False
    packet_loss_probability_range: tuple[float, float] | None = None
    friction_transition_speed_rad_s: float = 0.01

    def validate(self) -> None:
        if self.response_time_constant_range_s is not None:
            lower, upper = (float(value) for value in self.response_time_constant_range_s)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper < lower:
                raise ValueError(
                    "identified dynamics response_time_constant_range_s must "
                    "be a positive ordered pair"
                )
        if self.response_command_gain is not None and (
            not np.isfinite(self.response_command_gain) or self.response_command_gain <= 0.0
        ):
            raise ValueError(
                "identified dynamics response_command_gain must be finite and positive"
            )
        if self.response_command_gain_range is not None:
            lower, upper = (float(value) for value in self.response_command_gain_range)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper < lower:
                raise ValueError(
                    "identified dynamics response_command_gain_range must be "
                    "a positive ordered pair"
                )
        if self.packet_loss_probability_range is not None:
            lower, upper = (float(value) for value in self.packet_loss_probability_range)
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower < 0.0
                or upper < lower
                or upper > 1.0
            ):
                raise ValueError(
                    "identified dynamics packet_loss_probability_range must be "
                    "an ordered pair within [0, 1]"
                )
        if (
            not np.isfinite(self.friction_transition_speed_rad_s)
            or self.friction_transition_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "identified dynamics friction_transition_speed_rad_s must be finite and positive"
            )


@dataclass
class MicroDuckEnlargedBootstrapActuatorConfig:
    """Idealized passive dynamics for learning a Phase-0 policy.

    The GF43 datasheet specifies output limits but not reflected rotor inertia,
    viscous damping, or joint Coulomb friction.  This opt-in profile removes
    those unverified effects while retaining MJCF joint limits and actuator
    force ranges.  It is a training bootstrap, not a deployment model.
    """

    enabled: bool = False
    armature: float = 0.001
    damping: float = 0.0
    frictionloss: float = 0.0

    def validate(self) -> None:
        for name in ("armature", "damping", "frictionloss"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"bootstrap actuator {name} must be finite and non-negative")


@dataclass
class MicroDuckEnlargedNoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.001
    scale_joint_vel: float = 0.25
    scale_gyro: float = 0.03
    scale_gravity: float = 0.01


@dataclass
class MicroDuckEnlargedCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.4, -0.3, -1.0], [0.4, 0.3, 1.0]]
    )
    rel_standing_envs: float = 0.10
    rel_forward_envs: float = 0.25
    rel_turn_in_place_envs: float = 0.15
    head_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.05, -0.05, -0.07, -0.015],
            [0.05, 0.05, 0.07, 0.015],
        ]
    )
    body_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.008, -0.008, -0.008, -0.05, -0.05, -0.05],
            [0.008, 0.008, 0.008, 0.05, 0.05, 0.05],
        ]
    )


@dataclass
class MicroDuckEnlargedRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.1
    angular_tracking_sigma: float = 0.5
    base_height_target: float = 0.183
    min_base_height: float = 0.10
    max_tilt_deg: float = 70.0
    foot_contact_force_threshold: float = 1.0
    standing_leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.1, 0.05, 0.15, 0.15, 0.1, 0.1, 0.05, 0.15, 0.15, 0.1]
    )
    walking_leg_pose_stds: list[float] = field(
        default_factory=lambda: [0.3, 0.05, 0.4, 0.4, 0.25, 0.3, 0.05, 0.4, 0.4, 0.25]
    )
    walking_command_threshold: float = 0.01
    upright_std: float = float(np.sqrt(0.05))
    head_pose_std: float = 0.5
    # Price only sustained head-pose error, not unavoidable gait oscillation.
    head_pose_bias_tau_s: float = 1.0
    head_pose_bias_schedule_steps: list[int] = field(default_factory=lambda: [0])
    head_pose_bias_schedule_scales: list[float] = field(default_factory=lambda: [0.0])
    # The official alpha_walking policy uses ~0.18--0.20 s swing times on the
    # original robot.  Scale the timing window by sqrt(46 / 29) for this model.
    swing_time_min: float = 0.15
    swing_time_max: float = 0.38
    swing_time_target: float = 0.24
    swing_time_std: float = 0.06
    short_swing_threshold: float = 0.16
    gait_command_threshold: float = 0.01
    base_height_std: float = 0.012
    foot_height_target: float = 0.0317
    foot_regularization_command_threshold: float = 0.01
    # Official action-rate curriculum, expressed in environment control steps
    # (24 rollout steps per PPO iteration).
    action_rate_schedule_steps: list[int] = field(
        default_factory=lambda: [0, 12000, 18000, 24000, 30000, 36000]
    )
    action_rate_schedule_scales: list[float] = field(
        default_factory=lambda: [-0.1, -0.2, -0.4, -0.6, -0.8, -1.0]
    )


@dataclass
class MicroDuckEnlargedDomainRandConfig(DomainRandConfig):
    push_body_name: str | None = "trunk_base"


@registry.envcfg("MicroDuckEnlargedWalkFlat")
@dataclass
class MicroDuckEnlargedWalkFlatCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged" / "scene_flat.xml")
        )
    )
    sensor: MicroDuckEnlargedSensor = field(default_factory=MicroDuckEnlargedSensor)  # type: ignore[assignment]
    asset: MicroDuckEnlargedAsset = field(default_factory=MicroDuckEnlargedAsset)
    control_config: MicroDuckEnlargedControlConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckEnlargedControlConfig
    )
    motor_constraints: MicroDuckEnlargedMotorConstraintConfig = field(
        default_factory=MicroDuckEnlargedMotorConstraintConfig
    )
    identified_dynamics: MicroDuckEnlargedIdentifiedDynamicsConfig = field(
        default_factory=MicroDuckEnlargedIdentifiedDynamicsConfig
    )
    bootstrap_actuator: MicroDuckEnlargedBootstrapActuatorConfig = field(
        default_factory=MicroDuckEnlargedBootstrapActuatorConfig
    )
    noise_config: MicroDuckEnlargedNoiseConfig = field(  # type: ignore[assignment]
        default_factory=MicroDuckEnlargedNoiseConfig
    )
    commands: MicroDuckEnlargedCommands = field(default_factory=MicroDuckEnlargedCommands)
    reward_config: MicroDuckEnlargedRewardConfig | None = None
    domain_rand: MicroDuckEnlargedDomainRandConfig = field(
        default_factory=MicroDuckEnlargedDomainRandConfig
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 10.0
    reset_base_qvel_limit: float = 0.05


class MicroDuckEnlargedWalkDomainRandomizationProvider(LocomotionDRProvider):
    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
        commands = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, 3)),
            dtype=get_global_dtype(),
        )
        zero_small_xy_commands(commands, threshold=0.08)

        forward_fraction = float(env.cfg.commands.rel_forward_envs)
        max_forward = max(abs(float(low[0])), abs(float(high[0])))
        if forward_fraction > 0.0 and max_forward > 0.0:
            forward_mask = np.random.uniform(size=(num_reset,)) < min(forward_fraction, 1.0)
            commands[forward_mask, 0] = np.clip(
                np.abs(commands[forward_mask, 0]), min(0.3, max_forward), max_forward
            )
            commands[forward_mask, 1:] = 0.0

        standing_fraction = float(env.cfg.commands.rel_standing_envs)
        if standing_fraction > 0.0:
            standing_mask = np.random.uniform(size=(num_reset,)) < min(standing_fraction, 1.0)
            commands[standing_mask] = 0.0

        turn_fraction = float(env.cfg.commands.rel_turn_in_place_envs)
        if turn_fraction <= 0.0:
            return commands
        turn_mask = np.random.uniform(size=(num_reset,)) < min(turn_fraction, 1.0)
        num_turn = int(np.sum(turn_mask))
        if num_turn == 0:
            return commands
        yaw_limit = max(abs(float(low[2])), abs(float(high[2])))
        signs = np.where(np.random.uniform(size=(num_turn,)) < 0.5, -1.0, 1.0)
        magnitudes = np.random.uniform(0.4 * yaw_limit, yaw_limit, size=(num_turn,))
        commands[turn_mask, :2] = 0.0
        commands[turn_mask, 2] = signs * magnitudes
        return np.asarray(commands, dtype=get_global_dtype())

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        head_limit = np.asarray(env.cfg.commands.head_limit, dtype=get_global_dtype())
        body_limit = np.asarray(env.cfg.commands.body_limit, dtype=get_global_dtype())
        return {
            "head_commands": np.asarray(
                np.random.uniform(head_limit[0], head_limit[1], size=(num_reset, 4)),
                dtype=get_global_dtype(),
            ),
            "body_commands": np.asarray(
                np.random.uniform(body_limit[0], body_limit[1], size=(num_reset, 6)),
                dtype=get_global_dtype(),
            ),
            "current_air_time": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "current_contact_time": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "last_air_time": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "current_peak_foot_height": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "peak_foot_height_at_landing": np.zeros((num_reset, 2), dtype=get_global_dtype()),
            "first_foot_contact": np.zeros((num_reset, 2), dtype=bool),
        }

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        """Mirror randomized actuator gains into episode info.

        MuJoCo keeps reset-time Kp/Kd overrides in its per-environment worker
        models.  The GF43 projection runs in the owner before each physics
        substep, so it must use the exact same sampled gains rather than the
        nominal gains in the backend's template model.
        """

        plan = super().build_reset_plan(env, env_ids)
        randomization = plan.randomization
        if randomization is not None and randomization.kp is not None:
            plan.info_updates["actuator_kp"] = np.asarray(
                randomization.kp, dtype=get_global_dtype()
            ).copy()
        if randomization is not None and randomization.kd is not None:
            plan.info_updates["actuator_kd"] = np.asarray(
                randomization.kd, dtype=get_global_dtype()
            ).copy()
        return plan

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
        return env._compute_obs(
            info_updates,
            gyro,
            env._projected_gravity()[env_ids],
            dof_pos,
            dof_vel,
        )


@registry.env("MicroDuckEnlargedWalkFlat", sim_backend="mujoco")
class MicroDuckEnlargedWalkFlatEnv(LocomotionBaseEnv):
    """Independent enlarged-robot task; it does not import the original replica."""

    _cfg: MicroDuckEnlargedWalkFlatCfg
    # Version owners may override this mapping while retaining the common
    # observation, reset, reward, and envelope implementation.  The default
    # is the established V1.0.4 mixed X10/X40 layout.
    MOTOR_SPECS: ClassVar[tuple[GF43MotorSpec, ...]] = MICRODUCK_ENLARGED_104_MOTOR_SPECS
    # Mechanical revisions may rename joints without changing actuator order
    # or policy I/O. Version owners override this tuple at their owner layer.
    JOINT_NAMES: ClassVar[tuple[str, ...]] = MICRODUCK_ENLARGED_JOINT_NAMES

    def _apply_bootstrap_actuator_parameters(self, backend: Any, cfg: Any) -> None:
        """Remove passive terms not identified by the manufacturer datasheet."""

        if backend.backend_type != "mujoco":
            raise NotImplementedError(
                "bootstrap actuator overrides currently require the MuJoCo backend"
            )
        dof_indices = np.asarray(
            backend.get_joint_dof_indices(self.JOINT_NAMES), dtype=np.intp
        )
        model = backend.model
        model.dof_armature[dof_indices] = float(cfg.armature)
        model.dof_damping[dof_indices] = float(cfg.damping)
        model.dof_frictionloss[dof_indices] = float(cfg.frictionloss)

    def _apply_identified_model_parameters(self, backend: Any) -> None:
        """Install the fitted passive parameters before the backend materializes.

        MuJoCo's public backend contract intentionally has no generic damping
        or friction setter.  The owner therefore edits the not-yet-materialized
        MuJoCo model and keeps the custom friction in the pre-step callback;
        this also prevents double-counting the fitted losses.
        """

        if backend.backend_type != "mujoco":
            raise NotImplementedError(
                "identified GF43 empty-load dynamics currently require the MuJoCo backend"
            )
        if backend.num_actuators != len(self.MOTOR_SPECS):
            raise ValueError(
                "identified GF43 dynamics require one fit per actuator, got "
                f"{backend.num_actuators} actuators and {len(self.MOTOR_SPECS)} specs"
            )
        fits = []
        for spec in self.MOTOR_SPECS:
            fit = spec.empty_load_fit
            if fit is None:
                raise ValueError(
                    f"{spec.model} has no CAN-observable empty-load fit; "
                    "do not enable identified dynamics for this motor map"
                )
            fit.validate()
            fits.append(fit)

        dof_indices = np.asarray(
            backend.get_joint_dof_indices(self.JOINT_NAMES), dtype=np.intp
        )
        if dof_indices.shape != (len(fits),):
            raise ValueError(
                "identified GF43 dynamics could not resolve one joint DoF per actuator"
            )
        model = backend.model
        model.dof_armature[dof_indices] = np.asarray(
            [fit.effective_inertia_kg_m2 for fit in fits], dtype=np.float64
        )
        # Friction is applied directionally by ``GF43EmptyLoadFit`` below.
        # Zeroing MuJoCo's symmetric passive terms avoids applying it twice.
        model.dof_damping[dof_indices] = 0.0
        model.dof_frictionloss[dof_indices] = 0.0
        self._identified_fit_by_joint = tuple(fits)

    def __init__(
        self,
        cfg: MicroDuckEnlargedWalkFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        identified_cfg = cfg.identified_dynamics
        if not isinstance(identified_cfg, MicroDuckEnlargedIdentifiedDynamicsConfig):
            raise TypeError(
                "identified_dynamics must be a MicroDuckEnlargedIdentifiedDynamicsConfig"
            )
        identified_cfg.validate()
        bootstrap_cfg = cfg.bootstrap_actuator
        if not isinstance(bootstrap_cfg, MicroDuckEnlargedBootstrapActuatorConfig):
            raise TypeError("bootstrap_actuator must be a MicroDuckEnlargedBootstrapActuatorConfig")
        bootstrap_cfg.validate()
        if bootstrap_cfg.enabled and identified_cfg.enabled:
            raise ValueError(
                "bootstrap actuator dynamics and identified dynamics are mutually exclusive"
            )
        self._identified_dynamics_enabled = bool(identified_cfg.enabled)
        self._identified_fit_by_joint = ()
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains=cfg.control_config.position_gains(),
            **env_backend_kwargs(cfg),
        )
        if bootstrap_cfg.enabled:
            self._apply_bootstrap_actuator_parameters(backend, bootstrap_cfg)
        elif self._identified_dynamics_enabled:
            if not cfg.motor_constraints.enabled:
                raise ValueError("identified GF43 dynamics require motor_constraints.enabled=true")
            self._apply_identified_model_parameters(backend)
        self._identified_torque_state = np.zeros(
            (num_envs, backend.num_actuators), dtype=get_global_dtype()
        )
        response_range = identified_cfg.response_time_constant_range_s
        response_nominal = 0.0
        if response_range is not None:
            response_nominal = 0.5 * (float(response_range[0]) + float(response_range[1]))
        self._identified_response_time_s = np.full(
            (num_envs, backend.num_actuators),
            response_nominal,
            dtype=get_global_dtype(),
        )
        command_gain_nominal = (
            float(identified_cfg.response_command_gain)
            if identified_cfg.response_command_gain is not None
            else 1.0
        )
        self._identified_command_gain = np.full(
            (num_envs, backend.num_actuators),
            command_gain_nominal,
            dtype=get_global_dtype(),
        )
        self._identified_packet_loss_probability = np.zeros(
            (num_envs, backend.num_actuators), dtype=get_global_dtype()
        )
        self._identified_coulomb_positive_nm = np.zeros(
            (num_envs, backend.num_actuators), dtype=get_global_dtype()
        )
        self._identified_coulomb_negative_nm = np.zeros_like(self._identified_coulomb_positive_nm)
        self._identified_viscous_positive_nm_s_rad = np.zeros_like(
            self._identified_coulomb_positive_nm
        )
        self._identified_viscous_negative_nm_s_rad = np.zeros_like(
            self._identified_coulomb_positive_nm
        )
        for joint_index, fit in enumerate(self._identified_fit_by_joint):
            self._identified_coulomb_positive_nm[:, joint_index] = fit.coulomb_positive_nm
            self._identified_coulomb_negative_nm[:, joint_index] = fit.coulomb_negative_nm
            self._identified_viscous_positive_nm_s_rad[:, joint_index] = (
                fit.viscous_positive_nm_s_rad
            )
            self._identified_viscous_negative_nm_s_rad[:, joint_index] = (
                fit.viscous_negative_nm_s_rad
            )
        super().__init__(cfg, backend, num_envs)
        if self._num_action != len(self.JOINT_NAMES):
            raise ValueError(
                f"enlarged Micro Duck requires {len(self.JOINT_NAMES)} actuators, "
                f"got {self._num_action}"
            )
        self._reward_cfg = cfg.reward_config
        self._foot_body_ids = self._backend.get_body_ids(cfg.asset.foot_body_names)
        self._foot_site_offsets = np.asarray(cfg.asset.foot_site_offsets, dtype=get_global_dtype())
        self._control_step = 0
        self._enable_reward_log = True
        self._init_motor_constraints()
        self._init_reward_functions()
        self._init_domain_randomization(self._make_domain_randomization_provider())

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the fitted actuator state with the physical episode state."""

        result = super().reset(env_indices)
        if self._identified_dynamics_enabled:
            indices = np.asarray(env_indices, dtype=np.intp)
            self._identified_torque_state[indices] = 0.0
            response_range = self._cfg.identified_dynamics.response_time_constant_range_s
            if response_range is not None:
                lower, upper = (float(value) for value in response_range)
                self._identified_response_time_s[indices] = np.random.uniform(
                    lower,
                    upper,
                    size=(len(indices), self._num_action),
                ).astype(get_global_dtype())
            command_gain_range = self._cfg.identified_dynamics.response_command_gain_range
            if command_gain_range is not None:
                lower, upper = (float(value) for value in command_gain_range)
                self._identified_command_gain[indices] = np.random.uniform(
                    lower,
                    upper,
                    size=(len(indices), self._num_action),
                ).astype(get_global_dtype())
            packet_loss_range = self._cfg.identified_dynamics.packet_loss_probability_range
            if packet_loss_range is not None:
                lower, upper = (float(value) for value in packet_loss_range)
                self._identified_packet_loss_probability[indices] = np.random.uniform(
                    lower,
                    upper,
                    size=(len(indices), self._num_action),
                ).astype(get_global_dtype())
            if self._cfg.identified_dynamics.randomize_friction_parameters:
                for joint_index, fit in enumerate(self._identified_fit_by_joint):
                    self._identified_coulomb_positive_nm[indices, joint_index] = np.random.uniform(
                        *fit.coulomb_positive_range_nm, size=len(indices)
                    )
                    self._identified_coulomb_negative_nm[indices, joint_index] = np.random.uniform(
                        *fit.coulomb_negative_range_nm, size=len(indices)
                    )
                    self._identified_viscous_positive_nm_s_rad[indices, joint_index] = (
                        np.random.uniform(*fit.viscous_positive_range_nm_s_rad, size=len(indices))
                    )
                    self._identified_viscous_negative_nm_s_rad[indices, joint_index] = (
                        np.random.uniform(*fit.viscous_negative_range_nm_s_rad, size=len(indices))
                    )
        return result

    def _active_actuator_gains(self, info: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        """Return the gains actually installed for each vectorized episode."""

        compiled_kp, compiled_kd = self._backend.get_actuator_gains()
        kp = np.broadcast_to(
            np.asarray(compiled_kp, dtype=get_global_dtype()).reshape(1, -1),
            (self._num_envs, self._num_action),
        )
        kd = np.broadcast_to(
            np.asarray(compiled_kd, dtype=get_global_dtype()).reshape(1, -1),
            (self._num_envs, self._num_action),
        )
        if info is not None and "actuator_kp" in info:
            kp = np.asarray(info["actuator_kp"], dtype=get_global_dtype())
        if info is not None and "actuator_kd" in info:
            kd = np.asarray(info["actuator_kd"], dtype=get_global_dtype())
        expected_shape = (self._num_envs, self._num_action)
        if kp.shape != expected_shape or kd.shape != expected_shape:
            raise RuntimeError(
                "active actuator gains must have shape "
                f"{expected_shape}, got kp={kp.shape}, kd={kd.shape}"
            )
        if (
            not np.isfinite(kp).all()
            or np.any(kp <= 0.0)
            or not np.isfinite(kd).all()
            or np.any(kd < 0.0)
        ):
            raise ValueError("active actuator gains require finite Kp > 0 and Kd >= 0")
        return kp, kd

    def _init_motor_constraints(self) -> None:
        """Validate and cache the configured actuator envelope metadata."""
        cfg = self._cfg.motor_constraints
        if not isinstance(cfg, MicroDuckEnlargedMotorConstraintConfig):
            raise TypeError("motor_constraints must be a MicroDuckEnlargedMotorConstraintConfig")
        cfg.validate()
        self._motor_constraints_enabled = bool(cfg.enabled)
        self._motor_specs = self.MOTOR_SPECS
        if not self._motor_constraints_enabled:
            return
        if self._num_action != len(self._motor_specs):
            raise ValueError(
                "enlarged GF43 motor map expects "
                f"{len(self._motor_specs)} actuators, got {self._num_action}"
            )
        for spec in self._motor_specs:
            spec.validate()
        self._motor_rated_torque_limits = np.asarray(
            [spec.rated_output_torque_nm for spec in self._motor_specs],
            dtype=get_global_dtype(),
        )
        self._motor_peak_torque_limits = np.asarray(
            [spec.peak_output_torque_nm for spec in self._motor_specs],
            dtype=get_global_dtype(),
        )
        self._motor_rated_power_limits = np.asarray(
            [spec.rated_power_w for spec in self._motor_specs],
            dtype=get_global_dtype(),
        )
        self._motor_rated_speed_limits = np.asarray(
            [spec.rated_output_speed_rad_s for spec in self._motor_specs],
            dtype=get_global_dtype(),
        )
        self._motor_evidence_speed_limits = np.asarray(
            [
                max(abs(value) for value in spec.measured_speed_range_rpm) * (2.0 * np.pi / 60.0)
                for spec in self._motor_specs
            ],
            dtype=get_global_dtype(),
        )
        # Re-project before every MuJoCo substep so a velocity increase within
        # one 20 ms control interval cannot invalidate the power envelope.
        if self._motor_constraints_enabled or self._identified_dynamics_enabled:
            self._backend.set_pre_step_control(self._motor_pre_step_control)

    def _motor_pre_step_control(self, backend: Any, ctrl: np.ndarray) -> np.ndarray:
        """Apply the GF43 target projection at the simulator substep rate."""
        target = np.asarray(ctrl, dtype=get_global_dtype())
        qpos = np.asarray(backend.get_dof_pos(), dtype=get_global_dtype())
        qvel = np.asarray(backend.get_dof_vel(), dtype=get_global_dtype())
        kp, kd = self._active_actuator_gains(None if self._state is None else self._state.info)
        joint_range = np.asarray(backend.get_joint_range(), dtype=get_global_dtype())
        target = np.clip(target, joint_range[None, :, 0], joint_range[None, :, 1])
        requested_torque = (target - qpos) * kp - qvel * kd
        torque_lower, torque_upper = self._motor_torque_bounds_for_velocity(
            qvel, duty=self._cfg.motor_constraints.hard_duty
        )
        margin = float(self._cfg.motor_constraints.hard_limit_margin)
        torque_lower *= margin
        torque_upper *= margin
        feasible_torque = np.clip(requested_torque, torque_lower, torque_upper)

        if self._identified_dynamics_enabled:
            target = self._identified_motor_control(
                qpos=qpos,
                qvel=qvel,
                kp=kp,
                kd=kd,
                feasible_torque=feasible_torque,
                torque_lower=torque_lower,
                torque_upper=torque_upper,
            )
        else:
            target = qpos + (feasible_torque + kd * qvel) / kp
        if self._state is not None:
            self._state.info["motor_substep_target_rad"] = target.copy()
            self._state.info["motor_substep_torque_lower_nm"] = torque_lower.copy()
            self._state.info["motor_substep_torque_upper_nm"] = torque_upper.copy()
            self._state.info["motor_substep_torque_limit_nm"] = np.maximum(
                np.abs(torque_lower), np.abs(torque_upper)
            )
        return target

    def _identified_motor_control(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        feasible_torque: np.ndarray,
        torque_lower: np.ndarray,
        torque_upper: np.ndarray,
    ) -> np.ndarray:
        """Convert a position-controller request through the measured fit.

        The controller request is treated as an output-shaft torque command.
        The fitted command gain and first-order time constant produce a motor
        torque state, then the fitted direction-dependent losses are subtracted
        before mapping the net torque back to MuJoCo's position actuator.  This
        realizes the empty-load equivalent without changing the shared backend.
        """

        fit_cfg = self._cfg.identified_dynamics
        if not self._identified_fit_by_joint:
            raise RuntimeError("identified dynamics were enabled without fitted motor parameters")
        expected_shape = (self._num_envs, self._num_action)
        if kp.shape != expected_shape or kd.shape != expected_shape:
            raise ValueError(
                "identified dynamics expected per-environment gains with shape "
                f"{expected_shape}, got kp={kp.shape}, kd={kd.shape}"
            )

        time_constants = np.empty_like(kd)
        command_gains = np.empty_like(kd)
        for joint_index, fit in enumerate(self._identified_fit_by_joint):
            kd_points = np.asarray([point.kd for point in fit.response_points], dtype=np.float64)
            time_constants[:, joint_index] = np.interp(
                kd[:, joint_index],
                kd_points,
                [point.time_constant_s for point in fit.response_points],
            )
            command_gains[:, joint_index] = np.interp(
                kd[:, joint_index],
                kd_points,
                [point.command_gain for point in fit.response_points],
            )
        if fit_cfg.response_time_constant_range_s is not None:
            time_constants = self._identified_response_time_s
        if fit_cfg.response_command_gain_range is not None:
            command_gains = self._identified_command_gain
        elif fit_cfg.response_command_gain is not None:
            command_gains.fill(float(fit_cfg.response_command_gain))
        if fit_cfg.first_order_response:
            alpha = -np.expm1(-float(self._cfg.sim_dt) / np.maximum(time_constants, 1.0e-6))
            self._identified_torque_state += alpha * (
                command_gains * feasible_torque - self._identified_torque_state
            )
        else:
            self._identified_torque_state[:] = command_gains * feasible_torque

        if fit_cfg.asymmetric_friction:
            positive = qvel >= 0.0
            coulomb = np.where(
                positive,
                self._identified_coulomb_positive_nm,
                self._identified_coulomb_negative_nm,
            )
            viscous = np.where(
                positive,
                self._identified_viscous_positive_nm_s_rad,
                self._identified_viscous_negative_nm_s_rad,
            )
            friction = (
                coulomb * np.tanh(qvel / float(fit_cfg.friction_transition_speed_rad_s))
                + viscous * qvel
            )
        else:
            friction = np.empty_like(qvel)
            for joint_index, fit in enumerate(self._identified_fit_by_joint):
                symmetric_speed = qvel[:, joint_index]
                friction[:, joint_index] = (
                    fit.symmetric_coulomb_nm
                    * np.tanh(symmetric_speed / float(fit_cfg.friction_transition_speed_rad_s))
                    + fit.symmetric_viscous_nm_s_rad * symmetric_speed
                )

        net_torque = self._identified_torque_state - friction
        net_torque = np.clip(net_torque, torque_lower, torque_upper)
        target = qpos + (net_torque + kd * qvel) / kp
        if self._state is not None:
            self._state.info["motor_fit_torque_state_nm"] = self._identified_torque_state.copy()
            self._state.info["motor_fit_friction_nm"] = friction.copy()
            self._state.info["motor_fit_net_torque_nm"] = net_torque.copy()
            self._state.info["motor_fit_time_constant_s"] = np.broadcast_to(
                time_constants, (self._num_envs, self._num_action)
            ).copy()
            self._state.info["motor_fit_command_gain"] = command_gains.copy()
            self._state.info["motor_packet_loss_probability"] = (
                self._identified_packet_loss_probability.copy()
            )
        return target

    def _motor_limits_for_velocity(self, dof_vel: np.ndarray, *, duty: str) -> np.ndarray:
        """Return per-actuator GF43 torque limits at the current output speed."""
        velocity = np.asarray(dof_vel, dtype=get_global_dtype())
        if velocity.shape != (self._num_envs, self._num_action):
            raise ValueError(
                "motor velocity must have shape "
                f"{(self._num_envs, self._num_action)}, got {velocity.shape}"
            )
        limits = np.empty_like(velocity)
        for joint_index, spec in enumerate(self._motor_specs):
            limits[:, joint_index] = static_torque_magnitude_limit(
                velocity[:, joint_index],
                spec,
                duty=duty,  # type: ignore[arg-type]
            )
        return np.asarray(limits, dtype=get_global_dtype())

    def _motor_torque_bounds_for_velocity(
        self, dof_vel: np.ndarray, *, duty: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return output-side directional torque bounds for every motor.

        Legacy task owners retain the symmetric constant-power approximation.
        V1.0.6 opts into the X40 evidence-bounded hybrid model explicitly so
        this change cannot silently alter older mixed-motor task families.
        """

        velocity = np.asarray(dof_vel, dtype=get_global_dtype())
        if velocity.shape != (self._num_envs, self._num_action):
            raise ValueError(
                "motor velocity must have shape "
                f"{(self._num_envs, self._num_action)}, got {velocity.shape}"
            )
        if self._cfg.motor_constraints.envelope_model == "constant_power":
            magnitude = self._motor_limits_for_velocity(velocity, duty=duty)
            return -magnitude, magnitude

        lower = np.empty_like(velocity)
        upper = np.empty_like(velocity)
        for joint_index, spec in enumerate(self._motor_specs):
            if spec.model != GF43X40_10.model:
                raise ValueError(
                    "gf43x40_hybrid envelope requires every configured motor "
                    f"to be GF43X40-10, got {spec.model} at index {joint_index}"
                )
            braking_torque = (
                spec.rated_output_torque_nm
                if self._cfg.motor_constraints.braking_duty == "rated"
                else spec.peak_output_torque_nm
            )
            joint_lower, joint_upper = gf43x40_torque_bounds(
                velocity[:, joint_index], braking_torque_nm=braking_torque
            )
            lower[:, joint_index] = joint_lower
            upper[:, joint_index] = joint_upper
        return (
            np.asarray(lower, dtype=get_global_dtype()),
            np.asarray(upper, dtype=get_global_dtype()),
        )

    def _motor_metric(self, info: dict[str, Any], name: str) -> np.ndarray:
        return np.asarray(
            info.get(name, np.zeros(self._num_envs, dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )

    def _motor_peak_envelope(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_peak_envelope_excess")

    def _motor_peak_envelope_max(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_peak_envelope_max_excess")

    def _motor_joint_target_limit(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_joint_target_limit_excess")

    def _motor_rated_torque_penalty(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_rated_torque_excess")

    def _motor_rated_power_penalty(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_rated_power_excess")

    def _motor_speed_penalty(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_speed_excess")

    def _motor_evidence_speed_penalty(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_evidence_speed_excess")

    def _motor_evidence_speed_max_penalty(self, ctx: RewardContext) -> np.ndarray:
        return self._motor_metric(ctx.info, "motor_evidence_speed_max_excess")

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlargedWalkDomainRandomizationProvider:
        """Return the task-owned reset/randomization provider."""
        return MicroDuckEnlargedWalkDomainRandomizationProvider()

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Project policy position targets through the configured GF43 envelope.

        The XML ``forcerange`` remains the final simulator safety net.  When
        the motor constraint is enabled, this owner performs the same
        projection before physics: first to the joint range and then to the
        per-motor peak torque/power limit at the current output speed.  The
        raw request and the resulting utilization signals stay in ``info`` so
        PPO can learn to avoid relying on that projection.
        """
        action = np.asarray(actions, dtype=get_global_dtype())
        expected_shape = (self._num_envs, self._num_action)
        if action.shape != expected_shape:
            raise ValueError(f"action must have shape {expected_shape}, got {action.shape}")
        if not self._motor_constraints_enabled:
            return super().apply_action(action, state)

        control = self._cfg.control_config
        scale = float(control.action_scale)
        kp, kd = self._active_actuator_gains(state.info)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("motor-constrained position control requires action_scale > 0")

        qpos = np.asarray(self.get_dof_pos(), dtype=get_global_dtype())
        qvel = np.asarray(self.get_dof_vel(), dtype=get_global_dtype())
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            raise RuntimeError("motor-constrained position control requires joint limits")
        joint_range = np.asarray(joint_range, dtype=get_global_dtype())
        if joint_range.shape != (self._num_action, 2):
            raise RuntimeError(
                "motor-constrained position control requires one range per actuator, "
                f"got {joint_range.shape}"
            )

        requested_target = self.default_angles[None, :] + action * scale
        joint_target = np.clip(requested_target, joint_range[None, :, 0], joint_range[None, :, 1])
        requested_torque = (joint_target - qpos) * kp - qvel * kd
        torque_lower, torque_upper = self._motor_torque_bounds_for_velocity(
            qvel, duty=self._cfg.motor_constraints.hard_duty
        )
        margin = float(self._cfg.motor_constraints.hard_limit_margin)
        torque_lower *= margin
        torque_upper *= margin
        feasible_torque = np.clip(requested_torque, torque_lower, torque_upper)
        projected_target = qpos + (feasible_torque + kd * qvel) / kp
        effective_torque = (projected_target - qpos) * kp - qvel * kd
        effective_action = (projected_target - self.default_angles[None, :]) / scale

        requested_power = np.abs(requested_torque * qvel)
        requested_limit = np.where(requested_torque >= 0.0, torque_upper, -torque_lower)
        if self._cfg.motor_constraints.envelope_model == "gf43x40_hybrid":
            # The evidence-bounded motoring limit intentionally reaches zero
            # above 55.48 rpm.  Normalize the absolute excess by physical peak
            # torque there; dividing by an epsilon would create a meaningless
            # multi-million reward penalty and destabilize later training.
            hard_excess = (
                np.maximum(np.abs(requested_torque) - requested_limit, 0.0)
                / self._motor_peak_torque_limits[None, :]
            )
        else:
            hard_excess = np.where(
                np.abs(requested_torque) > 0.0,
                np.maximum(
                    np.abs(requested_torque) / np.maximum(requested_limit, 1e-6) - 1.0,
                    0.0,
                ),
                0.0,
            )
        rated_torque_excess = np.maximum(
            np.abs(requested_torque) / self._motor_rated_torque_limits[None, :] - 1.0,
            0.0,
        )
        rated_power_excess = np.maximum(
            requested_power / self._motor_rated_power_limits[None, :] - 1.0,
            0.0,
        )
        previous_policy_action = np.asarray(
            state.info.get("current_actions", np.zeros_like(action)),
            dtype=get_global_dtype(),
        ).copy()
        previous_effective_action = np.asarray(
            state.info.get("motor_effective_actions", np.zeros_like(action)),
            dtype=get_global_dtype(),
        ).copy()
        state.info["policy_actions"] = action.copy()
        state.info["motor_effective_actions"] = effective_action.copy()
        state.info["motor_requested_target_rad"] = requested_target.copy()
        state.info["motor_projected_target_rad"] = projected_target.copy()
        state.info["motor_requested_torque_nm"] = requested_torque.copy()
        state.info["motor_effective_torque_nm"] = effective_torque.copy()
        state.info["motor_torque_lower_nm"] = torque_lower.copy()
        state.info["motor_torque_upper_nm"] = torque_upper.copy()
        state.info["motor_peak_limit_nm"] = requested_limit.copy()
        state.info["motor_joint_target_limit_excess"] = np.sum(
            np.abs(requested_target - joint_target), axis=1
        ).astype(get_global_dtype())
        state.info["motor_peak_envelope_excess"] = np.mean(hard_excess, axis=1).astype(
            get_global_dtype()
        )
        state.info["motor_peak_envelope_max_excess"] = np.max(hard_excess, axis=1).astype(
            get_global_dtype()
        )
        state.info["motor_rated_torque_excess"] = np.mean(rated_torque_excess, axis=1).astype(
            get_global_dtype()
        )
        state.info["motor_rated_power_excess"] = np.mean(rated_power_excess, axis=1).astype(
            get_global_dtype()
        )
        effective_limit = np.where(effective_torque >= 0.0, torque_upper, -torque_lower)
        state.info["motor_torque_utilization_peak"] = np.max(
            np.divide(
                np.abs(effective_torque),
                effective_limit,
                out=np.zeros_like(effective_torque),
                # The evidence taper reaches an exact zero-torque endpoint.
                # Ignore float32 reconstruction residue at that endpoint
                # instead of reporting it as a >100% utilization event.
                where=effective_limit > 1e-5,
            ),
            axis=1,
        ).astype(get_global_dtype())
        state.info["motor_torque_utilization_rated"] = np.max(
            np.abs(effective_torque) / self._motor_rated_torque_limits[None, :], axis=1
        ).astype(get_global_dtype())
        # Physics receives the projected action, while observation/action-rate
        # history retains the raw network output.  This matches the official
        # runtime contract and prevents PPO from hiding an excessive request
        # behind the actuator projection.
        exec_action = (
            previous_effective_action
            if self._cfg.control_config.simulate_action_latency
            else effective_action
        )
        packet_loss_probability = self._identified_packet_loss_probability
        if np.any(packet_loss_probability > 0.0):
            packet_loss = np.random.uniform(size=expected_shape) < packet_loss_probability
            exec_action = np.where(packet_loss, previous_effective_action, exec_action)
            state.info["motor_packet_loss"] = packet_loss
        state.info["last_actions"] = previous_policy_action
        state.info["current_actions"] = action.copy()
        return exec_action * scale + self.default_angles

    def _update_motor_speed_metrics(self, dof_vel: np.ndarray, info: dict[str, Any]) -> None:
        if not self._motor_constraints_enabled:
            return
        if not self._cfg.motor_constraints.rated_speed_soft_limit:
            info["motor_speed_excess"] = np.zeros(self._num_envs, dtype=get_global_dtype())
            info["motor_speed_utilization_rated"] = np.zeros(
                self._num_envs, dtype=get_global_dtype()
            )
            info["motor_evidence_speed_excess"] = np.zeros(self._num_envs, dtype=get_global_dtype())
            info["motor_evidence_speed_max_excess"] = np.zeros(
                self._num_envs, dtype=get_global_dtype()
            )
            info["motor_speed_utilization_evidence"] = np.zeros(
                self._num_envs, dtype=get_global_dtype()
            )
            return
        speed_excess = np.maximum(
            np.abs(dof_vel) / self._motor_rated_speed_limits[None, :] - 1.0,
            0.0,
        )
        info["motor_speed_excess"] = np.mean(speed_excess, axis=1).astype(get_global_dtype())
        info["motor_speed_utilization_rated"] = np.max(
            np.abs(dof_vel) / self._motor_rated_speed_limits[None, :], axis=1
        ).astype(get_global_dtype())
        evidence_speed_excess = np.maximum(
            np.abs(dof_vel) / self._motor_evidence_speed_limits[None, :] - 1.0,
            0.0,
        )
        info["motor_evidence_speed_excess"] = np.mean(evidence_speed_excess, axis=1).astype(
            get_global_dtype()
        )
        info["motor_evidence_speed_max_excess"] = np.max(evidence_speed_excess, axis=1).astype(
            get_global_dtype()
        )
        info["motor_speed_utilization_evidence"] = np.max(
            np.abs(dof_vel) / self._motor_evidence_speed_limits[None, :], axis=1
        ).astype(get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + projected gravity(3) + q-q_home(14) + dq(14)
        # + previous action(14) + [twist(3), head(4), body(6)] = 61.
        return {"obs": MICRODUCK_ENLARGED_ACTOR_OBS_DIM}

    def _projected_gravity(self) -> np.ndarray:
        gravity = np.zeros((self._num_envs, 3), dtype=get_global_dtype())
        gravity[:, 2] = -1.0
        return np.asarray(
            np_quat_apply_inverse(self._backend.get_base_quat(), gravity),
            dtype=get_global_dtype(),
        )

    def _compute_obs(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise = self._cfg.noise_config
        gyro_observed = gyro + np.asarray(
            info.get("gyro_bias_rad_s", np.zeros_like(gyro)),
            dtype=get_global_dtype(),
        )
        joint_offset = (
            dof_pos
            - self.default_angles
            + np.asarray(
                info.get("joint_position_bias_rad", np.zeros_like(dof_pos)),
                dtype=get_global_dtype(),
            )
        )
        dof_vel_observed = dof_vel + np.asarray(
            info.get("joint_velocity_bias_rad_s", np.zeros_like(dof_vel)),
            dtype=get_global_dtype(),
        )
        last_actions = info.get("current_actions", np.zeros_like(joint_offset))
        actor = np.concatenate(
            [
                self._obs_noise(gyro_observed, noise.scale_gyro),
                self._obs_noise(projected_gravity, noise.scale_gravity),
                self._obs_noise(joint_offset, noise.scale_joint_angle),
                self._obs_noise(dof_vel_observed, noise.scale_joint_vel),
                last_actions,
                info["commands"],
                info["head_commands"],
                info["body_commands"],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        if actor.shape[1] != MICRODUCK_ENLARGED_ACTOR_OBS_DIM:
            raise RuntimeError(
                "enlarged Micro Duck actor observation must be "
                f"{MICRODUCK_ENLARGED_ACTOR_OBS_DIM}D, got {actor.shape[1]}"
            )
        return {"obs": actor}

    def _init_reward_functions(self) -> None:
        self._reward_fns = {
            "track_linear_velocity": self._track_linear_velocity,
            "track_angular_velocity": self._track_angular_velocity,
            "upright": self._upright_tracking,
            "leg_pose": self._leg_pose_tracking,
            "head_pose": self._head_pose_tracking,
            "head_pose_bias": self._head_pose_bias_penalty,
            "single_support": self._single_support,
            "swing_time": self._swing_time,
            "landing_quality": self._landing_quality,
            "short_swing": self._short_swing,
            "flight": self._flight,
            "double_support": self._double_support,
            "base_height": self._base_height_tracking,
            "foot_clearance": self._foot_clearance,
            "foot_slip": self._foot_slip,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
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

    def _track_linear_velocity(self, ctx: RewardContext) -> np.ndarray:
        error = np.sum(np.square(ctx.info["commands"][:, :2] - ctx.linvel[:, :2]), axis=1)
        return np.asarray(
            np.exp(-error / self._reward_cfg.tracking_sigma), dtype=get_global_dtype()
        )

    def _track_angular_velocity(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 2] - ctx.gyro[:, 2])
        return np.asarray(
            np.exp(-error / self._reward_cfg.angular_tracking_sigma), dtype=get_global_dtype()
        )

    def _upright_tracking(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        error = np.sum(np.square(ctx.gravity[:, :2]), axis=1)
        return np.asarray(
            np.exp(-error / self._reward_cfg.upright_std**2), dtype=get_global_dtype()
        )

    def _leg_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        standing_std = np.asarray(self._reward_cfg.standing_leg_pose_stds, dtype=get_global_dtype())
        walking_std = np.asarray(self._reward_cfg.walking_leg_pose_stds, dtype=get_global_dtype())
        expected_shape = (_LEG_JOINT_INDICES.size,)
        if standing_std.shape != expected_shape or walking_std.shape != expected_shape:
            raise ValueError("enlarged Micro Duck leg pose std lists must contain ten values")
        if np.any(standing_std <= 0.0) or np.any(walking_std <= 0.0):
            raise ValueError("enlarged Micro Duck leg pose std values must be positive")
        command = ctx.info["commands"]
        speed = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        std = np.where(
            (speed < self._reward_cfg.walking_command_threshold)[:, None],
            standing_std[None, :],
            walking_std[None, :],
        )
        error = ctx.dof_pos[:, _LEG_JOINT_INDICES] - ctx.default_angles[_LEG_JOINT_INDICES]
        return np.asarray(
            np.exp(-np.mean(np.square(error / std), axis=1)), dtype=get_global_dtype()
        )

    def _head_pose_tracking(self, ctx: RewardContext) -> np.ndarray:
        actual = ctx.dof_pos[:, _HEAD_JOINT_INDICES] - ctx.default_angles[_HEAD_JOINT_INDICES]
        error = actual - ctx.info["head_commands"]
        return np.asarray(
            np.mean(np.exp(-np.square(error / self._reward_cfg.head_pose_std)), axis=1),
            dtype=get_global_dtype(),
        )

    def _head_pose_bias_penalty(self, ctx: RewardContext) -> np.ndarray:
        """Penalize the one-second EMA of persistent head tracking error."""

        tau_s = float(self._reward_cfg.head_pose_bias_tau_s)
        if not np.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("head_pose_bias_tau_s must be finite and positive")
        actual = ctx.dof_pos[:, _HEAD_JOINT_INDICES] - ctx.default_angles[_HEAD_JOINT_INDICES]
        error = actual - ctx.info["head_commands"]
        previous = np.asarray(
            ctx.info.get("head_pose_bias_ema", np.zeros_like(error)),
            dtype=get_global_dtype(),
        )
        if previous.shape != error.shape:
            raise RuntimeError("head_pose_bias_ema must have shape (num_envs, 4)")
        previous = np.where(np.asarray(ctx.info["steps"])[:, None] == 0, 0.0, previous)
        alpha = min(1.0, float(self._cfg.ctrl_dt) / tau_s)
        ema = (1.0 - alpha) * previous + alpha * error
        ctx.info["head_pose_bias_ema"] = np.asarray(ema, dtype=get_global_dtype())
        return np.asarray(-np.mean(np.abs(ema), axis=1), dtype=get_global_dtype())

    def _gait_active(self, info: dict[str, Any]) -> np.ndarray:
        command = np.asarray(info["commands"], dtype=get_global_dtype())
        command_norm = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        return command_norm > self._reward_cfg.gait_command_threshold

    def _support_masks(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contact = np.asarray(info["foot_contact"], dtype=bool)
        if contact.shape != (self._num_envs, 2):
            raise RuntimeError("enlarged Micro Duck foot_contact must have shape (N, 2)")
        single = np.logical_xor(contact[:, 0], contact[:, 1])
        double = np.logical_and(contact[:, 0], contact[:, 1])
        flight = ~np.logical_or(contact[:, 0], contact[:, 1])
        return single, double, flight

    def _single_support(self, ctx: RewardContext) -> np.ndarray:
        single, _, _ = self._support_masks(ctx.info)
        return np.asarray(single & self._gait_active(ctx.info), dtype=get_global_dtype())

    def _swing_time(self, ctx: RewardContext) -> np.ndarray:
        air_time = np.asarray(ctx.info["current_air_time"], dtype=get_global_dtype())
        in_window = (air_time > self._reward_cfg.swing_time_min) & (
            air_time < self._reward_cfg.swing_time_max
        )
        contact = np.asarray(ctx.info["foot_contact"], dtype=bool)
        # Reward only the one airborne foot while the other foot supports the
        # body.  The rejected policy exploited the old sum by lifting both feet.
        valid_swing = np.logical_xor(contact[:, 0], contact[:, 1]) & np.any(
            in_window & ~contact, axis=1
        )
        return np.asarray(valid_swing & self._gait_active(ctx.info), dtype=get_global_dtype())

    def _flight(self, ctx: RewardContext) -> np.ndarray:
        _, _, flight = self._support_masks(ctx.info)
        return np.asarray(flight & self._gait_active(ctx.info), dtype=get_global_dtype())

    def _landing_quality(self, ctx: RewardContext) -> np.ndarray:
        first_contact = np.asarray(ctx.info["first_foot_contact"], dtype=bool)
        last_air_time = np.asarray(ctx.info["last_air_time"], dtype=get_global_dtype())
        error = (last_air_time - self._reward_cfg.swing_time_target) / (
            self._reward_cfg.swing_time_std
        )
        quality = np.exp(-np.square(error)) * first_contact
        return np.asarray(
            np.sum(quality, axis=1) * self._gait_active(ctx.info), dtype=get_global_dtype()
        )

    def _short_swing(self, ctx: RewardContext) -> np.ndarray:
        first_contact = np.asarray(ctx.info["first_foot_contact"], dtype=bool)
        last_air_time = np.asarray(ctx.info["last_air_time"], dtype=get_global_dtype())
        threshold = self._reward_cfg.short_swing_threshold
        shortfall = np.clip((threshold - last_air_time) / threshold, 0.0, 1.0)
        return np.asarray(
            np.sum(shortfall * first_contact, axis=1) * self._gait_active(ctx.info),
            dtype=get_global_dtype(),
        )

    def _double_support(self, ctx: RewardContext) -> np.ndarray:
        _, double, _ = self._support_masks(ctx.info)
        return np.asarray(double & self._gait_active(ctx.info), dtype=get_global_dtype())

    def _base_height_tracking(self, ctx: RewardContext) -> np.ndarray:
        error = (ctx.base_height - ctx.base_height_target) / self._reward_cfg.base_height_std
        return np.asarray(np.exp(-np.square(error)), dtype=get_global_dtype())

    def _update_action_rate_curriculum(self) -> None:
        steps = self._reward_cfg.action_rate_schedule_steps
        scales = self._reward_cfg.action_rate_schedule_scales
        if len(steps) != len(scales) or not steps:
            raise ValueError("action-rate schedule steps/scales must have equal non-zero length")
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("action-rate schedule steps must be strictly increasing")
        index = int(np.searchsorted(np.asarray(steps), self._control_step, side="right") - 1)
        self._reward_cfg.scales["action_rate"] = float(scales[max(index, 0)])

    def _update_head_pose_bias_curriculum(self) -> None:
        if "head_pose_bias" not in self._reward_cfg.scales:
            return
        steps = self._reward_cfg.head_pose_bias_schedule_steps
        scales = self._reward_cfg.head_pose_bias_schedule_scales
        if len(steps) != len(scales) or not steps:
            raise ValueError("head-pose-bias schedule steps/scales must have equal non-zero length")
        if steps[0] != 0 or any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("head-pose-bias schedule must start at zero and increase strictly")
        if not np.isfinite(scales).all() or any(scale < 0.0 for scale in scales):
            raise ValueError("head-pose-bias schedule scales must be finite and non-negative")
        index = int(np.searchsorted(np.asarray(steps), self._control_step, side="right") - 1)
        self._reward_cfg.scales["head_pose_bias"] = float(scales[max(index, 0)])

    def _foot_regularization_active(self, info: dict[str, Any]) -> np.ndarray:
        command = info["commands"]
        command_norm = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
        return command_norm > self._reward_cfg.foot_regularization_command_threshold

    def _foot_clearance(self, ctx: RewardContext) -> np.ndarray:
        height = np.asarray(ctx.info["foot_height"], dtype=get_global_dtype())
        velocity = np.asarray(ctx.info["foot_site_lin_vel_w"], dtype=get_global_dtype())
        speed_xy = np.linalg.norm(velocity[:, :, :2], axis=2)
        error = np.abs(height - self._reward_cfg.foot_height_target)
        active = self._foot_regularization_active(ctx.info)
        return np.asarray(np.sum(error * speed_xy, axis=1) * active, dtype=get_global_dtype())

    def _foot_slip(self, ctx: RewardContext) -> np.ndarray:
        contact = np.asarray(ctx.info["foot_contact"], dtype=bool)
        velocity = np.asarray(ctx.info["foot_site_lin_vel_w"], dtype=get_global_dtype())
        slip_sq = np.sum(np.square(velocity[:, :, :2]), axis=2)
        active = self._foot_regularization_active(ctx.info)
        return np.asarray(np.sum(slip_sq * contact, axis=1) * active, dtype=get_global_dtype())

    def _foot_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        body_pos, body_quat = self._backend.get_body_pose_w(self._foot_body_ids)
        body_lin_vel, body_ang_vel = self._backend.get_body_vel_w(self._foot_body_ids)
        offsets = np.broadcast_to(self._foot_site_offsets, (self._num_envs, 2, 3))
        rotated_offsets = np_quat_apply(body_quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(
            self._num_envs, 2, 3
        )
        site_pos = body_pos + rotated_offsets
        site_lin_vel = body_lin_vel + np.cross(body_ang_vel, rotated_offsets)
        return (
            np.asarray(site_pos[:, :, 2], dtype=get_global_dtype()),
            np.asarray(site_lin_vel, dtype=get_global_dtype()),
        )

    def _update_foot_contact_history(self, info: dict[str, Any]) -> None:
        sensor_values = []
        for sensor_name in self._cfg.sensor.feet_contact:
            value = np.asarray(
                self._backend.get_sensor_data(sensor_name), dtype=get_global_dtype()
            ).reshape(self._num_envs, -1)
            if value.shape[1] < 3:
                raise RuntimeError(
                    f"enlarged Micro Duck contact sensor {sensor_name!r} must expose a force vector"
                )
            sensor_values.append(np.linalg.norm(value[:, :3], axis=1))
        contact = np.stack(sensor_values, axis=1) > self._reward_cfg.foot_contact_force_threshold

        zeros = np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        air_time = np.asarray(info.get("current_air_time", zeros), dtype=get_global_dtype())
        contact_time = np.asarray(info.get("current_contact_time", zeros), dtype=get_global_dtype())
        if air_time.shape != zeros.shape or contact_time.shape != zeros.shape:
            raise RuntimeError("enlarged Micro Duck foot contact timers must have shape (N, 2)")

        foot_height, foot_site_lin_vel = self._foot_kinematics()
        peak_height = np.asarray(
            info.get("current_peak_foot_height", zeros), dtype=get_global_dtype()
        )
        last_air_time = np.asarray(info.get("last_air_time", zeros), dtype=get_global_dtype())
        first_contact = contact & (air_time > 0.0)
        peak_height = np.where(~contact, np.maximum(peak_height, foot_height), peak_height)
        info["foot_contact"] = contact
        info["foot_height"] = foot_height
        info["foot_site_lin_vel_w"] = foot_site_lin_vel
        info["first_foot_contact"] = first_contact
        info["last_air_time"] = np.where(first_contact, air_time, last_air_time).astype(
            get_global_dtype()
        )
        info["peak_foot_height_at_landing"] = np.where(first_contact, peak_height, 0.0).astype(
            get_global_dtype()
        )
        info["current_peak_foot_height"] = np.where(first_contact, 0.0, peak_height).astype(
            get_global_dtype()
        )
        info["current_air_time"] = np.where(contact, 0.0, air_time + self._cfg.ctrl_dt).astype(
            get_global_dtype()
        )
        info["current_contact_time"] = np.where(
            contact, contact_time + self._cfg.ctrl_dt, 0.0
        ).astype(get_global_dtype())

    def _reward_context(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            gravity=gravity,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_motor_speed_metrics(dof_vel, state.info)
        self._update_foot_contact_history(state.info)
        self._update_action_rate_curriculum()
        self._update_head_pose_bias_curriculum()
        tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
        terminated = np.logical_or(
            tilt > np.deg2rad(self._reward_cfg.max_tilt_deg),
            self._backend.get_base_pos()[:, 2] < self._reward_cfg.min_base_height,
        )
        ctx = self._reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=state.info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )
        self._control_step += 1
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)
