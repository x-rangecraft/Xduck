"""Batch GF43X40-10 MIT actuator independent of MuJoCo and host transport.

``torque`` returns the nominal output-side generalized force to pass to a
unit-gain torque actuator.  A physics backend integrates the body and calls
``observe`` with its post-step acceleration.  The default isolated friction
solver reproduces the one-DOF calibration; a coupled robot must explicitly
select and validate its friction treatment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .torque_observer import TorqueResidualObserver


@dataclass(frozen=True)
class GF43X40Parameters:
    physics: dict[str, Any]
    observation: dict[str, Any]
    registers: dict[str, Any]
    envelope_speed_rad_s: tuple[float, ...]
    envelope_torque_nm: tuple[float, ...]
    source_sha256: str
    behavior_reference_sha256: str = ""

    @classmethod
    def from_bundle(cls, path: Path | None = None) -> GF43X40Parameters:
        """Read the versioned snapshot once, during environment construction."""
        source = path or Path(__file__).with_name("calibration.json")
        bundle = json.loads(source.read_text(encoding="utf-8"))
        if bundle.get("schema_version") not in (1, 2) or bundle.get("motor_model") != "GF43X40-10":
            raise ValueError("Expected a GF43X40-10 version-1 or version-2 calibration bundle")
        if bundle["schema_version"] == 2 and not {"mechanical_torque_limit", "envelope_scale"} <= bundle.get("bare_motor_physics", {}).keys():
            raise ValueError("Version-2 calibration requires separate mechanical force limits")
        points: dict[float, float] = {}
        for point in bundle["measured_torque_speed_points"]:
            if point["quality"] == "reference":
                rpm = float(point["rpm"])
                points[rpm] = min(points.get(rpm, float("inf")), float(point["torque"]))
        if len(points) < 2 or not bundle["validation_scope"]["shared_across_all_motors"]:
            raise ValueError("A shared measured speed-torque reference is required")
        if bundle["feedback_observation"].get("model") not in ("kinematic_current_proxy", "mechanical_torque_proxy"):
            raise ValueError("The port supports the validated current-estimate observation model")
        ordered = sorted(points.items())
        timing = bundle["timing_contract"]
        if timing["physics_step_s"] != 0.001:
            raise ValueError("Current calibration requires 1-ms physics steps")
        return cls(
            physics=bundle["bare_motor_physics"],
            observation=bundle["feedback_observation"],
            registers=bundle["registers"],
            envelope_speed_rad_s=tuple(rpm * math.pi / 30 for rpm, _ in ordered),
            envelope_torque_nm=tuple(torque for _, torque in ordered),
            source_sha256=bundle["source_sha256"]["parameters"],
            behavior_reference_sha256=bundle.get(
                "behavior_reference_sha256", bundle["source_sha256"]["parameters"]
            ),
        )


class GF43X40Actuator:
    """One shared parameter set across a ``(num_envs, num_motors)`` batch."""

    def __init__(
        self,
        num_envs: int,
        num_motors: int,
        *,
        dt: float = 0.001,
        parameters: GF43X40Parameters | None = None,
        friction_mode: str = "isolated_1d",
    ) -> None:
        if num_envs <= 0 or num_motors <= 0 or not math.isclose(dt, 0.001, abs_tol=1e-12):
            raise ValueError("Positive batch dimensions and 1-ms physics steps required")
        if friction_mode not in ("isolated_1d", "smooth_coupled_approx", "smooth_stribeck"):
            raise ValueError("Choose isolated_1d or explicitly opt into smooth_coupled_approx")
        self.parameters = parameters or GF43X40Parameters.from_bundle()
        if self.parameters.physics.get("friction_model") == "smooth_stribeck":
            friction_mode = "smooth_stribeck"
        observation = self.parameters.observation
        self._torque_proxy = observation.get("model") == "mechanical_torque_proxy"
        if self._torque_proxy:
            gain = observation.get("torque_feedback_gain", float("nan"))
            if not math.isfinite(gain) or gain < 0:
                raise ValueError("Finite nonnegative torque_feedback_gain required")
            allowed = {"model", "torque_feedback_gain", "velocity_encoding", "velocity_filter_tc",
                       "velocity_feedback_scale", "velocity_feedback_bias", "velocity_filter_kp_slope"}
            allowed.update(("position_feedback_bias", "position_encoding", "torque_encoding",
                            "position_quantum", "velocity_quantum", "torque_quantum"))
            if set(observation) - allowed:
                raise ValueError("Mechanical torque proxy accepts only its gain and velocity observation settings")
        load_model = observation.get("load_feedback_model", "gain_memory")
        if load_model not in ("linear", "gain_memory"):
            raise ValueError("Unknown load feedback model")
        self._linear_load_feedback = load_model == "linear"
        if self._linear_load_feedback and any(observation.get(key, 0.0) != 0.0 for key in (
            "load_memory_gain", "load_velocity_gain", "load_quadratic_gain", "load_bias_gain",
            "load_memory_bias_gain", "load_gain_slope", "load_holding_gain",
            "load_holding_memory_gain", "load_gain_memory_slope",
        )):
            raise ValueError("Linear load feedback accepts only load_feedback_gain")
        for key, default in (("envelope_scale", 1.0), ("mechanical_torque_limit", self.parameters.registers["tmax"])):
            value = self.parameters.physics.get(key, default)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        band_slope = self.parameters.physics.get("kd_gain_band_slope", 0.0)
        if not math.isfinite(band_slope) or abs(band_slope) > 2.0:
            raise ValueError("kd_gain_band_slope must be finite and within [-2, 2]")
        for key in ("load_feedback_gain", "load_memory_gain", "load_velocity_gain", "load_quadratic_gain", "load_bias_gain", "load_memory_bias_gain", "load_gain_slope", "load_holding_gain", "load_holding_memory_gain", "load_gain_memory_slope"):
            if not math.isfinite(self.parameters.observation.get(key, 0.0)):
                raise ValueError(f"{key} must be finite")
        self.shape = (num_envs, num_motors)
        # Per-environment calibration uncertainty, owned by the robot reset.
        self.torque_scale_multiplier = np.ones(self.shape)
        self.friction_scale_multiplier = np.ones(self.shape)
        self.response_time_multiplier = np.ones(self.shape)
        self.dt = dt
        self.friction_mode = friction_mode
        self.effort = np.zeros(self.shape, dtype=np.float64)
        self.feedback_estimate = np.zeros(self.shape, dtype=np.float64)
        self.last_requested = np.zeros(self.shape, dtype=np.float64)
        self.last_torque = np.zeros(self.shape, dtype=np.float64)
        self.internal_velocity = np.zeros(self.shape, dtype=np.float64)
        self.observed_velocity = np.zeros(self.shape, dtype=np.float64)
        self.friction_memory = np.zeros(self.shape, dtype=np.float64)
        self.coast_state = np.ones(self.shape, dtype=np.float64)
        self._initialized = np.zeros(self.shape, dtype=bool)
        self._gain_feature = np.zeros(self.shape, dtype=np.float64)
        self._torque_observer = (
            TorqueResidualObserver(self.parameters.observation, self.shape)
            if "torque_residual_coefficients" in self.parameters.observation else None
        )
        self._observer_command = np.zeros((*self.shape, 5), dtype=np.float64)
        self._speed = np.asarray(self.parameters.envelope_speed_rad_s, dtype=np.float64)
        self._torque = np.asarray(self.parameters.envelope_torque_nm, dtype=np.float64) * self.parameters.physics.get("envelope_scale", 1.0)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        index = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        self.effort[index] = 0.0
        self.feedback_estimate[index] = 0.0
        self.last_requested[index] = 0.0
        self.last_torque[index] = 0.0
        self.internal_velocity[index] = 0.0
        self.observed_velocity[index] = 0.0
        self.friction_memory[index] = 0.0
        self.coast_state[index] = 1.0
        self._initialized[index] = False
        self._gain_feature[index] = 0.0
        if self._torque_observer is not None:
            self._torque_observer.reset(index)
        self._observer_command[index] = 0.0

    def _array(self, value: np.ndarray | float, name: str) -> np.ndarray:
        item = np.asarray(value, dtype=np.float64)
        if item.shape != self.shape or not np.all(np.isfinite(item)):
            raise ValueError(f"{name} must be finite with shape {self.shape}")
        return item

    def compute_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_ff: np.ndarray,
        *,
        effective_mass: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the drive by 1 ms and return nominal output-side torque.

        ``effective_mass`` is the scalar generalized mass for each *isolated*
        axis, including armature.  A diagonal mass from a coupled robot does
        not make the calibrated sticking solver exact for that robot.
        """
        q = self._array(q, "q")
        v = self._array(qd, "qd")
        target = self._array(q_des, "q_des")
        speed_target = self._array(qd_des, "qd_des")
        kp = self._array(kp, "kp")
        kd = self._array(kd, "kd")
        ff = self._array(tau_ff, "tau_ff")
        if np.any(kp < 0) or np.any(kp > 500) or np.any(kd < 0) or np.any(kd > 5):
            raise ValueError("MIT gains must be inside CAN ranges Kp=0..500, Kd=0..5")
        mass = None
        if self.friction_mode == "isolated_1d":
            if effective_mass is None:
                raise ValueError("isolated_1d requires each axis's effective generalized mass")
            mass = self._array(effective_mass, "effective_mass")
            if np.any(mass <= 0):
                raise ValueError("effective_mass must be positive")
        physical = self.parameters.physics
        limits = self.parameters.registers
        self._observer_command[:] = np.stack((kp, kd, target - q, speed_target, ff), axis=-1)
        self.internal_velocity[:] = np.where(self._initialized, self.internal_velocity, v)
        self.observed_velocity[:] = np.where(self._initialized, self.observed_velocity, v)
        self._initialized[:] = True
        itc = physical.get("internal_velocity_tc", 0.0)
        self.internal_velocity += (1.0 if itc <= 0 else -math.expm1(-self.dt / itc)) * (
            v - self.internal_velocity
        )
        gain_feature = kp / (kp + 80.0)
        self._gain_feature[:] = gain_feature
        band_slope = physical.get("kd_gain_band_slope", 0.0)
        def smoothstep(value: np.ndarray) -> np.ndarray:
            bounded = np.clip(value, 0.0, 1.0)
            return bounded * bounded * (3.0 - 2.0 * bounded)
        gain_band = (
            smoothstep((kp - 70.0) / 10.0)
            * (1.0 - smoothstep((kp - 110.0) / 20.0))
            * smoothstep((kd - 2.6) / 0.3)
            * (1.0 - smoothstep((kd - 3.2) / 0.3))
        )
        requested = (
            kp * np.exp(physical.get("kp_gain_slope", 0.0) * gain_feature) * (target - q)
            + kd
            * np.exp((physical.get("kd_gain_slope", 0.0) + band_slope * gain_band) * gain_feature)
            * (speed_target + physical["velocity_target_bias"] - self.internal_velocity)
            + physical["feedforward_gain"] * ff
        )
        requested = np.clip(requested, -limits["tmax"], limits["tmax"])
        time_constant = np.full(self.shape, physical["torque_time_constant"])
        time_constant *= np.where(
            (requested * self.effort < 0) | (np.abs(requested) < np.abs(self.effort)),
            physical.get("release_tc_ratio", 1.0),
            1.0,
        )
        low = physical["braking_transition_low"]
        high = physical["braking_transition_high"]
        blend = np.clip((np.abs(v) - low) / (high - low), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        time_constant += np.where(
            requested * v < 0,
            blend * (physical["braking_time_constant"] - time_constant),
            0.0,
        )
        time_constant *= self.response_time_multiplier
        alpha = -np.expm1(-self.dt / time_constant)
        change = alpha * (requested - self.effort)
        slew_limit = physical.get("effort_slew_limit", math.inf) * self.dt
        self.effort += np.clip(change, -slew_limit, slew_limit)
        self.last_requested[:] = requested

        speed_blend = np.clip((np.abs(v) - 0.5) / 1.5, 0.0, 1.0)
        speed_blend = speed_blend * speed_blend * (3.0 - 2.0 * speed_blend)
        viscous = physical["viscous"] + speed_blend * (
            physical.get("high_speed_viscous", physical["viscous"]) - physical["viscous"]
        )
        coast = np.where(
            (kp == 0) & (kd == 0) & (np.abs(ff) <= 0.007),
            physical["coast_drag_scale"],
            1.0,
        )
        ctc = physical.get("coast_transition_tc", 0.0)
        if self.friction_mode != "smooth_stribeck":
            self.coast_state += (1.0 if ctc <= 0 else -math.expm1(-self.dt / ctc)) * (
                coast - self.coast_state
            )
            coast = self.coast_state
        drive_effort = physical["torque_scale"] * self.torque_scale_multiplier * self.effort - viscous * self.friction_scale_multiplier * coast * v
        if self.friction_mode == "smooth_stribeck":
            fc = np.where(v >= 0, physical["coulomb"], physical["coulomb_negative"]) * coast * self.friction_scale_multiplier
            stribeck = 1 + (physical["static_ratio"] - 1) * np.exp(-np.square(v / physical["friction_velocity"]))
            raw = drive_effort - fc * stribeck * np.tanh(v / physical["friction_velocity"])
        elif self.friction_mode == "isolated_1d":
            assert mass is not None
            required = drive_effort + mass * v / self.dt
            fc = np.where(required >= 0, physical["coulomb"], physical["coulomb_negative"]) * coast * self.friction_scale_multiplier
            distance = physical.get("friction_memory_distance", 0.01)
            if distance > 0:
                self.friction_memory += -np.expm1(-np.abs(v) * self.dt / distance) * (
                    np.copysign(1.0, v) - self.friction_memory
                )
            fc *= 1 + physical.get("friction_memory_gain", 0.0) * 0.5 * (
                1 - np.copysign(1.0, required) * self.friction_memory
            )
            static_ratio = np.where(
                required >= 0,
                physical["static_ratio"],
                physical.get("static_ratio_negative", physical["static_ratio"]),
            )
            moving = fc * (
                1.0 + (static_ratio - 1.0) * np.exp(-np.square(v / physical["friction_velocity"]))
            )
            threshold = np.where(
                np.abs(v) < physical["stiction_velocity"],
                fc * static_ratio,
                moving,
            )
            raw = np.where(
                np.abs(required) <= threshold,
                -mass * v / self.dt,
                drive_effort - np.copysign(moving, required),
            )
        else:
            # Direction history is meaningful for the feedback observer even
            # when coupled dynamics require a smooth friction approximation.
            distance = physical.get("friction_memory_distance", 0.01)
            if distance > 0:
                self.friction_memory += -np.expm1(-np.abs(v) * self.dt / distance) * (
                    np.copysign(1.0, v) - self.friction_memory
                )
            fc = np.where(v >= 0, physical["coulomb"], physical["coulomb_negative"]) * coast * self.friction_scale_multiplier
            fc *= 1 + physical.get("friction_memory_gain", 0.0) * 0.5 * (
                1 - np.copysign(1.0, v) * self.friction_memory
            )
            raw = drive_effort - fc * np.tanh(v / physical["friction_velocity"])

        speed = np.abs(v)
        drive_limit = np.interp(speed.ravel(), self._speed, self._torque).reshape(self.shape)
        drive_limit = np.where(
            speed < self._speed[0],
            min(physical["low_speed_drive_limit"], float(self._torque[0])),
            drive_limit,
        )
        drive_limit = np.where(speed > self._speed[-1], 0.0, drive_limit)
        mechanical_limit = physical.get("mechanical_torque_limit", limits["tmax"])
        brake = min(physical["braking_limit"], mechanical_limit)
        lo = np.where(v > 1e-8, -brake, -drive_limit)
        hi = np.where(v < -1e-8, brake, drive_limit)
        result = np.clip(raw, np.maximum(lo, -mechanical_limit), np.minimum(hi, mechanical_limit))
        self.last_torque[:] = result
        return result

    def observe(self, qd_after: np.ndarray, acceleration_after: np.ndarray) -> np.ndarray:
        """Advance only the current-estimate channel after the physics solve."""
        v = self._array(qd_after, "qd_after")
        acceleration = self._array(acceleration_after, "acceleration_after")
        obs = self.parameters.observation
        if self._torque_proxy:
            self.feedback_estimate[:] = obs["torque_feedback_gain"] * self.last_torque
        else:
            estimate = (
                obs["inertia_gain"] * acceleration
                + obs["coulomb_gain"] * np.tanh(v / obs["friction_velocity"])
                + obs["viscous_gain"] * v
                + obs["feedback_bias"]
                + obs["static_command_gain"]
                * self.last_requested
                * np.exp(-np.square(v / obs["static_velocity_scale"]))
            )
            if obs.get("effort_gain", 0.0):
                estimate += obs["effort_gain"] * self.effort
            estimate += obs.get("effort_error_gain", 0.0) * (self.last_requested - self.effort)
            estimate += (
                obs.get("memory_gain", 0.0)
                * self.friction_memory
                * np.exp(-np.square(v / obs["static_velocity_scale"]))
            )
            load_proxy = self.last_torque - self.parameters.physics["armature"] * acceleration
            active = (self._observer_command[..., 0] > 0) | (self._observer_command[..., 1] > 0) | (np.abs(self._observer_command[..., 4]) > 0.007)
            load_proxy = np.where(active, load_proxy, 0.0)
            estimate += obs.get("load_feedback_gain", 0.0) * load_proxy
            if not self._linear_load_feedback:
                estimate += obs.get("load_memory_gain", 0.0) * np.abs(load_proxy) * self.friction_memory
                estimate += obs.get("load_velocity_gain", 0.0) * np.abs(load_proxy) * np.tanh(v / 0.05)
                estimate += obs.get("load_quadratic_gain", 0.0) * load_proxy * np.abs(load_proxy)
                estimate += obs.get("load_bias_gain", 0.0) * np.tanh(load_proxy / 0.5)
                estimate += obs.get("load_memory_bias_gain", 0.0) * self.friction_memory * np.tanh(np.abs(load_proxy) / 0.25)
                load_hold = np.exp(-np.square(v / 0.05))
                estimate += obs.get("load_gain_slope", 0.0) * load_proxy * self._gain_feature
                estimate += obs.get("load_holding_gain", 0.0) * load_proxy * load_hold
                estimate += obs.get("load_holding_memory_gain", 0.0) * np.abs(load_proxy) * self.friction_memory * load_hold
                estimate += obs.get("load_gain_memory_slope", 0.0) * np.abs(load_proxy) * self.friction_memory * self._gain_feature
            tc = obs["filter_time_constant"]
            alpha = 1.0 if tc <= 0 else -math.expm1(-self.dt / tc)
            self.feedback_estimate += alpha * (estimate - self.feedback_estimate)
        vtc = obs.get("velocity_filter_tc", 0.0)
        vtc = vtc * np.exp(obs.get("velocity_filter_kp_slope", 0.0) * (self._gain_feature - 0.65))
        valpha = np.ones(self.shape)
        np.divide(-self.dt, vtc, out=valpha, where=vtc > 0)
        valpha = np.where(vtc > 0, -np.expm1(valpha), 1.0)
        self.observed_velocity += valpha * (v - self.observed_velocity)
        if self._torque_observer is None:
            return self.feedback_estimate.copy()
        c = self._observer_command
        residual = self._torque_observer.step(
            acceleration,
            v,
            self.last_requested,
            self.effort,
            self.friction_memory,
            c[..., 0],
            c[..., 1],
            c[..., 2],
            c[..., 3],
            c[..., 4],
            self.dt,
        )
        return self.feedback_estimate + residual

    @property
    def velocity_feedback(self) -> np.ndarray:
        """Filtered/scaled velocity before transport bias and CAN encoding."""
        return self.observed_velocity * self.parameters.observation.get(
            "velocity_feedback_scale", 1.0
        )
