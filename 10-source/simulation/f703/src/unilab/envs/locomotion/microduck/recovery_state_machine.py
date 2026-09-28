"""Official-style fall, soft-landing and policy handoff state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RecoveryState(str, Enum):
    """Mutually exclusive owners of the joint targets."""

    WALK = "walk"
    LIMP = "limp_fall"
    POSE = "limp_pose"
    RECOVERY = "stand_recovery"


@dataclass(frozen=True)
class RecoveryStateMachineConfig:
    """Defaults copied from the released Micro Duck ``robotd`` configuration."""

    fall_tilt_z: float = -0.90
    fall_predict_z: float = -0.50
    fall_lookahead_s: float = 0.300
    fall_debounce_s: float = 0.060
    landing_still_rate_rad_s: float = 1.0
    landing_still_s: float = 0.200
    limp_max_s: float = 1.500
    pose_s: float = 0.600
    limp_gain: int = 50
    pose_gain: int = 160
    # Explicit recovery-policy handoff, required by this project's two-policy
    # contract.  Upstream robotd hands control back immediately after POSE.
    recovered_height_m: float = 0.090
    recovered_tilt_deg: float = 25.0
    recovered_gyro_rad_s: float = 0.30
    recovered_xy_speed_m_s: float = 0.10
    recovered_hold_s: float = 0.500

    def __post_init__(self) -> None:
        positive = {
            "fall_lookahead_s": self.fall_lookahead_s,
            "fall_debounce_s": self.fall_debounce_s,
            "landing_still_rate_rad_s": self.landing_still_rate_rad_s,
            "landing_still_s": self.landing_still_s,
            "limp_max_s": self.limp_max_s,
            "pose_s": self.pose_s,
            "recovered_height_m": self.recovered_height_m,
            "recovered_gyro_rad_s": self.recovered_gyro_rad_s,
            "recovered_xy_speed_m_s": self.recovered_xy_speed_m_s,
            "recovered_hold_s": self.recovered_hold_s,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"state-machine values must be positive: {invalid}")
        if not 0.0 < self.recovered_tilt_deg < 90.0:
            raise ValueError("recovered_tilt_deg must be within (0, 90)")
        if self.limp_gain < 0 or self.pose_gain < 0:
            raise ValueError("gains must be non-negative")


@dataclass(frozen=True)
class RecoveryDecision:
    """One control tick's exclusive owner and target contract."""

    state: RecoveryState
    policy: str | None
    joint_targets: np.ndarray | None
    gain: int | None
    zero_twist: bool
    transition: str | None = None


class OfficialRecoveryStateMachine:
    """Port of robotd's limp-fall phases plus a stable recovery handoff.

    ``WALK`` and ``RECOVERY`` return a policy label; ``LIMP`` and ``POSE``
    return direct joint-position targets.  This makes policy inference and
    scripted safety motion impossible to run on the same tick.
    """

    def __init__(
        self,
        home_position: np.ndarray,
        config: RecoveryStateMachineConfig | None = None,
    ) -> None:
        home = np.asarray(home_position, dtype=np.float64)
        if home.shape != (14,) or not np.isfinite(home).all():
            raise ValueError("home_position must contain 14 finite joint positions")
        self.home_position = home.copy()
        self.config = config or RecoveryStateMachineConfig()
        self.state = RecoveryState.WALK
        self._falling_s = 0.0
        self._limp_s = 0.0
        self._landing_still_s = 0.0
        self._pose_s = 0.0
        self._recovered_s = 0.0
        self._pose_from = home.copy()

    @staticmethod
    def gravity_z_rate(gravity: np.ndarray, gyro: np.ndarray) -> float:
        """Return ``-(omega x gravity).z``; positive means tipping away from upright."""
        gravity = np.asarray(gravity, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        if gravity.shape != (3,) or gyro.shape != (3,):
            raise ValueError("gravity and gyro must each contain three values")
        return float(-(gyro[0] * gravity[1] - gyro[1] * gravity[0]))

    def predicted_gravity_z(self, gravity: np.ndarray, gyro: np.ndarray) -> float:
        return float(
            gravity[2]
            + self.gravity_z_rate(gravity, gyro) * self.config.fall_lookahead_s
        )

    def reset(self) -> None:
        """Abort any in-flight fall sequence and re-arm walking detection."""
        self.state = RecoveryState.WALK
        self._falling_s = 0.0
        self._limp_s = 0.0
        self._landing_still_s = 0.0
        self._pose_s = 0.0
        self._recovered_s = 0.0
        self._pose_from = self.home_position.copy()

    def _walk_decision(self, transition: str | None = None) -> RecoveryDecision:
        return RecoveryDecision(
            state=RecoveryState.WALK,
            policy="walk",
            joint_targets=None,
            gain=None,
            zero_twist=False,
            transition=transition,
        )

    def _recovery_decision(self, transition: str | None = None) -> RecoveryDecision:
        return RecoveryDecision(
            state=RecoveryState.RECOVERY,
            policy="stand_recovery",
            joint_targets=None,
            gain=None,
            zero_twist=True,
            transition=transition,
        )

    def step(
        self,
        *,
        gravity: np.ndarray,
        gyro: np.ndarray,
        joint_position: np.ndarray,
        base_height_m: float,
        base_xy_speed_m_s: float,
        dt_s: float,
        fall_detection_eligible: bool = True,
    ) -> RecoveryDecision:
        gravity = np.asarray(gravity, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        joints = np.asarray(joint_position, dtype=np.float64)
        if gravity.shape != (3,) or gyro.shape != (3,) or joints.shape != (14,):
            raise ValueError("expected gravity[3], gyro[3] and joint_position[14]")
        values = np.concatenate((gravity, gyro, joints, [base_height_m, base_xy_speed_m_s, dt_s]))
        if not np.isfinite(values).all() or dt_s <= 0.0:
            raise ValueError("state-machine inputs must be finite and dt_s positive")

        cfg = self.config
        gyro_rate = float(np.linalg.norm(gyro))

        if self.state is RecoveryState.WALK:
            rate = self.gravity_z_rate(gravity, gyro)
            falling = (
                fall_detection_eligible
                and gravity[2] > cfg.fall_tilt_z
                and rate > 0.0
                and gravity[2] + rate * cfg.fall_lookahead_s > cfg.fall_predict_z
            )
            self._falling_s = self._falling_s + dt_s if falling else 0.0
            if self._falling_s + 1e-12 < cfg.fall_debounce_s:
                return self._walk_decision()
            self.state = RecoveryState.LIMP
            self._falling_s = 0.0
            self._limp_s = 0.0
            self._landing_still_s = 0.0
            return RecoveryDecision(
                state=self.state,
                policy=None,
                joint_targets=joints.copy(),
                gain=cfg.limp_gain,
                zero_twist=True,
                transition="fall_predicted",
            )

        if self.state is RecoveryState.LIMP:
            self._limp_s += dt_s
            self._landing_still_s = (
                self._landing_still_s + dt_s
                if gyro_rate < cfg.landing_still_rate_rad_s
                else 0.0
            )
            landed = self._landing_still_s + 1e-12 >= cfg.landing_still_s
            timed_out = self._limp_s + 1e-12 >= cfg.limp_max_s
            if landed or timed_out:
                self.state = RecoveryState.POSE
                self._pose_s = 0.0
                self._pose_from = joints.copy()
                return RecoveryDecision(
                    state=self.state,
                    policy=None,
                    joint_targets=joints.copy(),
                    gain=cfg.pose_gain,
                    zero_twist=True,
                    transition="landing_still" if landed else "landing_timeout",
                )
            return RecoveryDecision(
                state=self.state,
                policy=None,
                joint_targets=joints.copy(),
                gain=cfg.limp_gain,
                zero_twist=True,
            )

        if self.state is RecoveryState.POSE:
            self._pose_s += dt_s
            alpha = min(self._pose_s / cfg.pose_s, 1.0)
            target = self._pose_from + (self.home_position - self._pose_from) * alpha
            if self._pose_s + 1e-12 >= cfg.pose_s:
                self.state = RecoveryState.RECOVERY
                self._recovered_s = 0.0
                return self._recovery_decision(transition="pose_complete")
            return RecoveryDecision(
                state=self.state,
                policy=None,
                joint_targets=target,
                gain=cfg.pose_gain,
                zero_twist=True,
            )

        if self.state is RecoveryState.RECOVERY:
            recovered_z = -float(np.cos(np.deg2rad(cfg.recovered_tilt_deg)))
            stable = (
                gravity[2] <= recovered_z
                and base_height_m >= cfg.recovered_height_m
                and gyro_rate <= cfg.recovered_gyro_rad_s
                and base_xy_speed_m_s <= cfg.recovered_xy_speed_m_s
            )
            self._recovered_s = self._recovered_s + dt_s if stable else 0.0
            if self._recovered_s + 1e-12 < cfg.recovered_hold_s:
                return self._recovery_decision()
            self.state = RecoveryState.WALK
            self._recovered_s = 0.0
            return self._walk_decision(transition="recovery_stable")

        raise AssertionError(f"unhandled recovery state: {self.state}")
