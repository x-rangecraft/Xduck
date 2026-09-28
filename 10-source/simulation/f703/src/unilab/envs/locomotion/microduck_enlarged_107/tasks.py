"""Standing and walking owners for the supplied V1.0.7 all-X40 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.actuators.gf43x import (
    GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S,
    GF43X40_10_MOTORING_SPEED_RAD_S,
    gf43x40_motoring_torque_limit,
)
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MICRODUCK_ENLARGED_JOINT_NAMES,
)
from unilab.envs.locomotion.microduck_enlarged_106.tasks import (
    MicroDuckEnlarged106StandFlatCfg,
    MicroDuckEnlarged106StandFlatEnv,
    MicroDuckEnlarged106WalkDomainRandConfig,
    MicroDuckEnlarged106WalkDomainRandomizationProvider,
    MicroDuckEnlarged106WalkFlatCfg,
    MicroDuckEnlarged106WalkFlatEnv,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_107" / "scene_flat.xml")
    )


@dataclass
class MicroDuckEnlarged107FullMotorDRConfig:
    """Per-joint, per-episode DR ranges transcribed from the supplied tables."""

    enabled: bool = True
    torque_axis_scale_range: list[float] = field(default_factory=lambda: [0.90, 1.00])
    speed_axis_scale_range: list[float] = field(default_factory=lambda: [0.9773, 1.00])
    peak_curve_blend_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    peak_power_scale_range: list[float] = field(default_factory=lambda: [0.90, 1.00])
    rotor_inertia_nominal_kg_m2: float = 0.03047
    rotor_inertia_scale_range: list[float] = field(default_factory=lambda: [0.737, 1.15])
    coulomb_positive_nominal_nm: float = 0.272
    coulomb_positive_scale_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    coulomb_negative_nominal_nm: float = 0.325
    coulomb_negative_scale_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    viscous_positive_nominal_nm_s_rad: float = 0.02965
    viscous_positive_scale_range: list[float] = field(default_factory=lambda: [0.8286, 1.15])
    viscous_negative_nominal_nm_s_rad: float = 0.03051
    viscous_negative_scale_range: list[float] = field(default_factory=lambda: [0.6891, 1.3109])
    command_delay_steps: int = 2
    command_delay_probability_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    apply_velocity_measurement_fit: bool = True
    velocity_gain_positive: float = 0.9827
    velocity_gain_negative: float = 0.9819

    def validate(self) -> None:
        for name in (
            "torque_axis_scale_range",
            "speed_axis_scale_range",
            "peak_curve_blend_range",
            "peak_power_scale_range",
            "rotor_inertia_scale_range",
            "coulomb_positive_scale_range",
            "coulomb_negative_scale_range",
            "viscous_positive_scale_range",
            "viscous_negative_scale_range",
            "command_delay_probability_range",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all() or values[1] < values[0]:
                raise ValueError(f"full_motor_dr.{name} must be a finite ordered pair")
        if self.torque_axis_scale_range[0] <= 0.0:
            raise ValueError("full_motor_dr torque scale must be positive")
        if self.speed_axis_scale_range[0] <= 0.0:
            raise ValueError("full_motor_dr speed scale must be positive")
        if self.peak_power_scale_range[0] <= 0.0:
            raise ValueError("full_motor_dr peak-power scale must be positive")
        if self.rotor_inertia_nominal_kg_m2 <= 0.0:
            raise ValueError("full_motor_dr rotor inertia nominal must be positive")
        if self.rotor_inertia_scale_range[0] <= 0.0:
            raise ValueError("full_motor_dr rotor-inertia scale must be positive")
        for name in (
            "coulomb_positive_nominal_nm",
            "coulomb_negative_nominal_nm",
            "viscous_positive_nominal_nm_s_rad",
            "viscous_negative_nominal_nm_s_rad",
            "velocity_gain_positive",
            "velocity_gain_negative",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"full_motor_dr.{name} must be finite and positive")
        for name in ("peak_curve_blend_range", "command_delay_probability_range"):
            low, high = getattr(self, name)
            if low < 0.0 or high > 1.0:
                raise ValueError(f"full_motor_dr.{name} must stay within [0, 1]")
        if self.command_delay_steps < 0:
            raise ValueError("full_motor_dr.command_delay_steps must be non-negative")


@dataclass
class MicroDuckEnlarged107DomainRandConfig(MicroDuckEnlarged106WalkDomainRandConfig):
    """Formal mechanical DR combined with the table's controller/sensor DR."""

    randomize_body_mass: bool = True
    body_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.70, 1.30])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.002, 0.002])
    randomize_dof_armature: bool = True
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.90, 1.10])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.90, 1.10])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.90, 1.10])
    randomize_joint_position_bias: bool = True
    joint_position_bias_range_rad: list[float] = field(default_factory=lambda: [-0.005, 0.005])
    randomize_joint_velocity_bias: bool = False
    randomize_gyro_bias: bool = False
    push_robots: bool = False


class MicroDuckEnlarged107DomainRandomizationProvider(
    MicroDuckEnlarged106WalkDomainRandomizationProvider
):
    """Add V1.0.7 full-motor and formal-mechanical reset randomization."""

    @staticmethod
    def _uniform(bounds: list[float], shape: tuple[int, ...]) -> np.ndarray:
        low, high = (float(value) for value in bounds)
        return np.asarray(np.random.uniform(low, high, size=shape), dtype=get_global_dtype())

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = super()._build_extra_info_updates(env, num_reset)
        cfg = env.cfg.full_motor_dr
        cfg.validate()
        shape = (num_reset, env._num_action)
        if not cfg.enabled:
            updates.update(
                {
                    "motor_dr_torque_axis_scale": np.ones(shape, dtype=get_global_dtype()),
                    "motor_dr_speed_axis_scale": np.ones(shape, dtype=get_global_dtype()),
                    "motor_dr_peak_curve_blend": np.zeros(shape, dtype=get_global_dtype()),
                    "motor_dr_peak_power_scale": np.ones(shape, dtype=get_global_dtype()),
                    "motor_dr_rotor_inertia_scale": np.ones(shape, dtype=get_global_dtype()),
                    "motor_dr_command_delay_steps": np.zeros(shape, dtype=np.int32),
                }
            )
            return updates
        delay_probability = self._uniform(cfg.command_delay_probability_range, shape)
        updates.update(
            {
                "motor_dr_torque_axis_scale": self._uniform(cfg.torque_axis_scale_range, shape),
                "motor_dr_speed_axis_scale": self._uniform(cfg.speed_axis_scale_range, shape),
                "motor_dr_peak_curve_blend": self._uniform(cfg.peak_curve_blend_range, shape),
                "motor_dr_peak_power_scale": self._uniform(cfg.peak_power_scale_range, shape),
                "motor_dr_rotor_inertia_scale": self._uniform(cfg.rotor_inertia_scale_range, shape),
                "motor_dr_command_delay_steps": np.where(
                    np.random.uniform(size=shape) < delay_probability,
                    int(cfg.command_delay_steps),
                    0,
                ).astype(np.int32),
            }
        )
        return updates

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, "_v107_base_actuator_gains", None)
        if cached is None:
            kp, kd = env._backend.get_actuator_gains()
            cached = (
                np.asarray(kp, dtype=np.float64).copy(),
                np.asarray(kd, dtype=np.float64).copy(),
            )
            self._v107_base_actuator_gains = cached
        return cached

    def _get_body_inertia_baseline(self, env: Any) -> np.ndarray:
        cached = getattr(self, "_v107_body_inertia_baseline", None)
        if cached is None:
            cached = np.asarray(env._backend.model.body_inertia, dtype=np.float64).copy()
            self._v107_body_inertia_baseline = cached
        return cached

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        payload = plan.randomization
        if payload is None:
            return plan

        backend = env._backend
        num_reset = len(env_ids)
        base_mass, base_friction, _, base_armature = self._get_reset_randomization_baselines(env)
        assert base_mass is not None
        assert base_friction is not None
        assert base_armature is not None
        base_mass = np.asarray(base_mass, dtype=np.float64)
        base_friction = np.asarray(base_friction, dtype=np.float64)
        base_armature = np.asarray(base_armature, dtype=np.float64)
        base_inertia = self._get_body_inertia_baseline(env)
        base_body_id = int(backend.get_body_ids([env.cfg.asset.base_name])[0])

        if payload.body_mass is not None:
            mass_scale = payload.body_mass[:, base_body_id] / base_mass[base_body_id]
            body_mass = np.broadcast_to(base_mass, (num_reset, base_mass.size)).copy()
            body_mass[:, base_body_id] *= mass_scale
            body_inertia = np.broadcast_to(base_inertia, (num_reset, *base_inertia.shape)).copy()
            body_inertia[:, base_body_id, :] *= mass_scale[:, None]
            payload.body_mass = body_mass
            payload.body_inertia = body_inertia
            plan.info_updates["mechanical_dr_trunk_mass_scale"] = np.asarray(
                mass_scale, dtype=get_global_dtype()
            )

        if payload.geom_friction is not None:
            ground_id = int(backend.get_geom_id(env.cfg.asset.ground))
            friction_scale = payload.geom_friction[:, ground_id, 0] / base_friction[ground_id, 0]
            geom_friction = np.broadcast_to(base_friction, (num_reset, *base_friction.shape)).copy()
            for name in ("left_foot_collision", "right_foot_collision"):
                geom_id = int(backend.get_geom_id(name))
                geom_friction[:, geom_id, 0] *= friction_scale
            payload.geom_friction = geom_friction
            plan.info_updates["mechanical_dr_foot_friction_scale"] = np.asarray(
                friction_scale, dtype=get_global_dtype()
            )

        if payload.dof_armature is not None:
            dof_indices = np.asarray(backend.get_joint_dof_indices(env.JOINT_NAMES), dtype=np.intp)
            mechanical_scale = np.divide(
                payload.dof_armature[:, dof_indices],
                base_armature[dof_indices][None, :],
            )
            rotor_scale = np.asarray(
                plan.info_updates["motor_dr_rotor_inertia_scale"], dtype=np.float64
            )
            payload.dof_armature[:, dof_indices] = (
                float(env.cfg.full_motor_dr.rotor_inertia_nominal_kg_m2)
                * mechanical_scale
                * rotor_scale
            )
            plan.info_updates["mechanical_dr_armature_scale"] = np.asarray(
                mechanical_scale, dtype=get_global_dtype()
            )

        if payload.kp is not None:
            base_kp, _ = self._get_base_actuator_gains(env)
            payload.kp = np.asarray(
                base_kp[None, :]
                * self._uniform(
                    env.cfg.domain_rand.kp_multiplier_range,
                    (num_reset, env._num_action),
                ),
                dtype=np.float64,
            )
            plan.info_updates["actuator_kp"] = payload.kp.astype(get_global_dtype())
        if payload.kd is not None:
            _, base_kd = self._get_base_actuator_gains(env)
            payload.kd = np.asarray(
                base_kd[None, :]
                * self._uniform(
                    env.cfg.domain_rand.kd_multiplier_range,
                    (num_reset, env._num_action),
                ),
                dtype=np.float64,
            )
            plan.info_updates["actuator_kd"] = payload.kd.astype(get_global_dtype())
        return plan


class MicroDuckEnlarged107FullDRMixin:
    """Runtime implementation of the V1.0.7 full motor DR surface."""

    JOINT_NAMES = MICRODUCK_ENLARGED_JOINT_NAMES

    def _make_domain_randomization_provider(
        self,
    ) -> MicroDuckEnlarged107DomainRandomizationProvider:
        return MicroDuckEnlarged107DomainRandomizationProvider()

    def _init_motor_constraints(self) -> None:
        self._cfg.full_motor_dr.validate()
        super()._init_motor_constraints()
        history_length = max(int(self._cfg.full_motor_dr.command_delay_steps), 1)
        self._v107_substep_command_history = np.broadcast_to(
            self.default_angles[None, None, :],
            (history_length, self._num_envs, self._num_action),
        ).copy()

    def reset(self, env_indices: np.ndarray):
        result = super().reset(env_indices)
        indices = np.asarray(env_indices, dtype=np.intp)
        self._v107_substep_command_history[:, indices, :] = self.default_angles[None, None, :]
        if self._cfg.full_motor_dr.enabled and self._identified_dynamics_enabled:
            cfg = self._cfg.full_motor_dr
            shape = (len(indices), self._num_action)
            self._identified_coulomb_positive_nm[indices] = (
                cfg.coulomb_positive_nominal_nm
                * np.random.uniform(*cfg.coulomb_positive_scale_range, size=shape)
            )
            self._identified_coulomb_negative_nm[indices] = (
                cfg.coulomb_negative_nominal_nm
                * np.random.uniform(*cfg.coulomb_negative_scale_range, size=shape)
            )
            self._identified_viscous_positive_nm_s_rad[indices] = (
                cfg.viscous_positive_nominal_nm_s_rad
                * np.random.uniform(*cfg.viscous_positive_scale_range, size=shape)
            )
            self._identified_viscous_negative_nm_s_rad[indices] = (
                cfg.viscous_negative_nominal_nm_s_rad
                * np.random.uniform(*cfg.viscous_negative_scale_range, size=shape)
            )
        return result

    def _full_dr_array(self, name: str, default: float) -> np.ndarray:
        if self._state is None or name not in self._state.info:
            return np.full((self._num_envs, self._num_action), default, dtype=get_global_dtype())
        return np.asarray(self._state.info[name], dtype=get_global_dtype())

    def _motor_torque_bounds_for_velocity(
        self, dof_vel: np.ndarray, *, duty: str
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._cfg.full_motor_dr
        if not cfg.enabled or duty != "peak":
            return super()._motor_torque_bounds_for_velocity(dof_vel, duty=duty)
        velocity = np.asarray(dof_vel, dtype=get_global_dtype())
        torque_scale = self._full_dr_array("motor_dr_torque_axis_scale", 1.0)
        speed_scale = self._full_dr_array("motor_dr_speed_axis_scale", 1.0)
        blend = self._full_dr_array("motor_dr_peak_curve_blend", 0.0)
        power_scale = self._full_dr_array("motor_dr_peak_power_scale", 1.0)
        scaled_speed = np.abs(velocity) / speed_scale
        nominal = gf43x40_motoring_torque_limit(scaled_speed) * torque_scale

        peak_torque = self._motor_peak_torque_limits[None, :] * torque_scale
        peak_power = (
            np.asarray([spec.peak_power_w for spec in self._motor_specs], dtype=get_global_dtype())[
                None, :
            ]
            * power_scale
        )
        upper = np.minimum(
            peak_torque,
            np.divide(
                peak_power,
                scaled_speed,
                out=np.full_like(scaled_speed, np.inf),
                where=scaled_speed > 0.0,
            ),
        )
        taper_start = float(GF43X40_10_MOTORING_SPEED_RAD_S[-2])
        taper = np.where(
            scaled_speed <= taper_start,
            1.0,
            np.clip(
                (GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S - scaled_speed)
                / (GF43X40_10_MAX_OBSERVED_OUTPUT_SPEED_RAD_S - taper_start),
                0.0,
                1.0,
            ),
        )
        upper *= taper
        mixed_curve = (1.0 - blend) * nominal + blend * np.maximum(nominal, upper)
        # Peak torque, randomized peak power, and the evidence-speed taper are
        # hard bounds around either curve shape, not optional blend inputs.
        motoring = np.minimum(mixed_curve, upper)
        braking = np.asarray(
            [spec.rated_output_torque_nm for spec in self._motor_specs],
            dtype=get_global_dtype(),
        )[None, :]
        lower = np.where(
            velocity > 0.0,
            -braking,
            np.where(velocity < 0.0, -motoring, -peak_torque),
        )
        upper_bound = np.where(
            velocity > 0.0,
            motoring,
            np.where(velocity < 0.0, braking, peak_torque),
        )
        return lower.astype(get_global_dtype()), upper_bound.astype(get_global_dtype())

    def _motor_pre_step_control(self, backend: Any, ctrl: np.ndarray) -> np.ndarray:
        target = np.asarray(ctrl, dtype=get_global_dtype())
        delay_steps = self._full_dr_array("motor_dr_command_delay_steps", 0.0).astype(np.int32)
        if self._cfg.full_motor_dr.enabled and np.any(delay_steps > 0):
            target = np.where(delay_steps > 0, self._v107_substep_command_history[0], target)
            if self._v107_substep_command_history.shape[0] > 1:
                self._v107_substep_command_history[:-1] = self._v107_substep_command_history[1:]
            self._v107_substep_command_history[-1] = np.asarray(ctrl, dtype=get_global_dtype())
        return super()._motor_pre_step_control(backend, target)

    def _compute_obs(
        self,
        info: dict[str, Any],
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        cfg = self._cfg.full_motor_dr
        measured_velocity = np.asarray(dof_vel, dtype=get_global_dtype())
        if cfg.enabled and cfg.apply_velocity_measurement_fit:
            fits = [spec.empty_load_fit for spec in self._motor_specs]
            if any(fit is None for fit in fits):
                raise RuntimeError("full motor DR requires a velocity fit for every actuator")
            positive_gain = np.full(
                self._num_action,
                float(cfg.velocity_gain_positive),
                dtype=get_global_dtype(),
            )
            negative_gain = np.full(
                self._num_action,
                float(cfg.velocity_gain_negative),
                dtype=get_global_dtype(),
            )
            bias = np.asarray([fit.velocity_bias_rad_s for fit in fits], dtype=get_global_dtype())
            noise_std = np.asarray(
                [fit.velocity_noise_std_upper_bound_rad_s for fit in fits],
                dtype=get_global_dtype(),
            )
            gain = np.where(measured_velocity >= 0.0, positive_gain, negative_gain)
            measured_velocity = (
                measured_velocity * gain
                + bias
                + np.random.normal(size=measured_velocity.shape) * noise_std
            ).astype(get_global_dtype())
        return super()._compute_obs(info, gyro, projected_gravity, dof_pos, measured_velocity)


@registry.envcfg("MicroDuckEnlarged107StandFlat")
@dataclass
class MicroDuckEnlarged107StandFlatCfg(MicroDuckEnlarged106StandFlatCfg):
    """V1.0.7 balance task with formal mechanical and full motor DR."""

    scene: SceneCfg = field(default_factory=_scene)
    domain_rand: MicroDuckEnlarged107DomainRandConfig = field(
        default_factory=MicroDuckEnlarged107DomainRandConfig
    )
    full_motor_dr: MicroDuckEnlarged107FullMotorDRConfig = field(
        default_factory=MicroDuckEnlarged107FullMotorDRConfig
    )


@registry.env("MicroDuckEnlarged107StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged107StandFlatEnv(
    MicroDuckEnlarged107FullDRMixin, MicroDuckEnlarged106StandFlatEnv
):
    """Flat-ground standing on the supplied all-X40 mechanics."""

    _cfg: MicroDuckEnlarged107StandFlatCfg


@registry.envcfg("MicroDuckEnlarged107WalkFlat")
@dataclass
class MicroDuckEnlarged107WalkFlatCfg(MicroDuckEnlarged106WalkFlatCfg):
    """V1.0.7 forward-walking task with formal mechanical and full motor DR."""

    scene: SceneCfg = field(default_factory=_scene)
    domain_rand: MicroDuckEnlarged107DomainRandConfig = field(
        default_factory=MicroDuckEnlarged107DomainRandConfig
    )
    full_motor_dr: MicroDuckEnlarged107FullMotorDRConfig = field(
        default_factory=MicroDuckEnlarged107FullMotorDRConfig
    )


@registry.env("MicroDuckEnlarged107WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged107WalkFlatEnv(
    MicroDuckEnlarged107FullDRMixin, MicroDuckEnlarged106WalkFlatEnv
):
    """Flat-ground locomotion on the supplied all-X40 mechanics."""

    _cfg: MicroDuckEnlarged107WalkFlatCfg


__all__ = [
    "MicroDuckEnlarged107DomainRandConfig",
    "MicroDuckEnlarged107DomainRandomizationProvider",
    "MicroDuckEnlarged107FullMotorDRConfig",
    "MicroDuckEnlarged107StandFlatCfg",
    "MicroDuckEnlarged107StandFlatEnv",
    "MicroDuckEnlarged107WalkFlatCfg",
    "MicroDuckEnlarged107WalkFlatEnv",
]
