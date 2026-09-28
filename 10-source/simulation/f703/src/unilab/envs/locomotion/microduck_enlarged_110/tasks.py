"""Standing and walking owners for the supplied V1.1.0 all-X40 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck_enlarged.walk import (
    MicroDuckEnlargedControlConfig,
)
from unilab.envs.locomotion.microduck_enlarged_107.tasks import (
    MicroDuckEnlarged107StandFlatCfg,
    MicroDuckEnlarged107StandFlatEnv,
    MicroDuckEnlarged107WalkFlatCfg,
    MicroDuckEnlarged107WalkFlatEnv,
)


def _scene() -> SceneCfg:
    return SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "microduck_enlarged_110" / "scene_flat.xml")
    )


@dataclass
class MicroDuckEnlarged110ControlConfig(MicroDuckEnlargedControlConfig):
    """Shared V1.1.0 action contract for cross-policy switching."""

    action_scale: float = 1.0
    Kp: float = 30.0
    Kd: float = 2.0
    simulate_action_latency: bool = False
    joint_limit_margin_deg: float = 5.0
    joint_limit_lookahead_s: float = 0.25


class _V110JointLimitControl:
    """Inset position requests and anticipate approach to the joint boundaries.

    The final MuJoCo position ctrl encodes a bounded torque, not a trajectory
    setpoint. Clipping that ctrl after the response fit would change the torque
    and bypass both the fitted response and the speed-dependent motor bounds.
    This guard instead reserves travel for overshoot at the request boundary.
    Actual joint clearance must still be verified with the selected policy.
    """

    def _init_motor_constraints(self) -> None:
        if (self._cfg.control_config.joint_limit_margin_deg > 0
                or self._cfg.control_config.joint_limit_lookahead_s > 0):
            if not self._cfg.motor_constraints.enabled:
                raise ValueError("V1.1.0 joint-limit governor requires motor_constraints.enabled")
        super()._init_motor_constraints()
        margin = float(self._cfg.control_config.joint_limit_margin_deg)
        limits = np.asarray(self._backend.get_joint_range())
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("joint_limit_margin_deg must be finite and non-negative")
        self._v110_limit_lookahead_s = float(self._cfg.control_config.joint_limit_lookahead_s)
        if not np.isfinite(self._v110_limit_lookahead_s) or self._v110_limit_lookahead_s < 0:
            raise ValueError("joint_limit_lookahead_s must be finite and non-negative")
        inset = limits.dtype.type(np.deg2rad(margin))
        self._v110_safe_joint_low = limits[:, 0] + inset
        self._v110_safe_joint_high = limits[:, 1] - inset
        if np.any(self._v110_safe_joint_low >= self._v110_safe_joint_high):
            raise ValueError("joint_limit_margin_deg leaves no usable joint travel")

    def _motor_pre_step_control(self, backend: Any, ctrl: np.ndarray) -> np.ndarray:
        safe_request = np.clip(ctrl, self._v110_safe_joint_low, self._v110_safe_joint_high)
        # Begin braking before a moving joint reaches the inset boundary.
        # This is a position-request governor, not an inertia/torque model:
        # the original delayed response and motor bounds still execute below.
        q = backend.get_dof_pos()
        v = backend.get_dof_vel()
        predicted = q + self._v110_limit_lookahead_s * v
        upper_risk = np.where(v > 0, np.maximum(predicted - self._v110_safe_joint_high, 0), 0)
        lower_risk = np.where(v < 0, np.maximum(self._v110_safe_joint_low - predicted, 0), 0)
        safe_request = np.clip(safe_request + lower_risk - upper_risk,
                               self._v110_safe_joint_low, self._v110_safe_joint_high)
        if self._state is not None:
            self._state.info["joint_safe_request_rad"] = safe_request.copy()
        return super()._motor_pre_step_control(backend, safe_request)


@registry.envcfg("MicroDuckEnlarged110StandFlat")
@dataclass
class MicroDuckEnlarged110StandFlatCfg(MicroDuckEnlarged107StandFlatCfg):
    """V1.1.0 balance task with the V1.0.7 full GF43X40 constraint surface."""

    scene: SceneCfg = field(default_factory=_scene)
    control_config: MicroDuckEnlarged110ControlConfig = field(
        default_factory=MicroDuckEnlarged110ControlConfig
    )


@registry.env("MicroDuckEnlarged110StandFlat", sim_backend="mujoco")
class MicroDuckEnlarged110StandFlatEnv(_V110JointLimitControl, MicroDuckEnlarged107StandFlatEnv):
    """Flat-ground standing on the supplied V1.1.0 inertial model."""

    _cfg: MicroDuckEnlarged110StandFlatCfg


@registry.envcfg("MicroDuckEnlarged110WalkFlat")
@dataclass
class MicroDuckEnlarged110WalkFlatCfg(MicroDuckEnlarged107WalkFlatCfg):
    """V1.1.0 forward walking with the V1.0.7 full GF43X40 constraint surface."""

    scene: SceneCfg = field(default_factory=_scene)
    control_config: MicroDuckEnlarged110ControlConfig = field(
        default_factory=MicroDuckEnlarged110ControlConfig
    )


@registry.env("MicroDuckEnlarged110WalkFlat", sim_backend="mujoco")
class MicroDuckEnlarged110WalkFlatEnv(_V110JointLimitControl, MicroDuckEnlarged107WalkFlatEnv):
    """Flat-ground locomotion on the supplied V1.1.0 inertial model."""

    _cfg: MicroDuckEnlarged110WalkFlatCfg


__all__ = [
    "MicroDuckEnlarged110ControlConfig",
    "MicroDuckEnlarged110StandFlatCfg",
    "MicroDuckEnlarged110StandFlatEnv",
    "MicroDuckEnlarged110WalkFlatCfg",
    "MicroDuckEnlarged110WalkFlatEnv",
]
