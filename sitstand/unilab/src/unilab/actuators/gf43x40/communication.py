"""Simulator-time command/feedback transport for GF43X40-10.

This class does not use RK3566 sockets, STM32 firmware or wall-clock sleeps.
It can run under UniLab's physics substep callback.  An unknown fixed network
latency is zero *extra* shift by default, not a measured zero latency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MITCommand:
    q_des: np.ndarray
    qd_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


@dataclass(frozen=True)
class MotorFeedback:
    sample_time_s: float
    available_time_s: float
    q: np.ndarray
    qd: np.ndarray
    tau_est: np.ndarray
    most_recent_command_seq: int


class _DiscreteDelay:
    """Sample integer milliseconds with exact integer observation weights."""

    def __init__(self, delay_ms: list[int], counts: list[int]) -> None:
        values = np.asarray(delay_ms, dtype=np.float64)
        weights = np.asarray(counts, dtype=np.float64)
        if (values.ndim != 1 or not values.size or weights.shape != values.shape
                or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights))
                or np.any(values < 0) or np.any(values != np.floor(values))
                or np.any(weights <= 0) or np.any(weights != np.floor(weights))
                or np.any(np.diff(values) <= 0)):
            raise ValueError("Discrete delays require sorted nonnegative integer ms and positive counts")
        self.delay_ms = values.astype(np.int64)
        self.cumulative_counts = np.cumsum(weights.astype(np.int64))

    def sample_s(self, rng: np.random.Generator) -> float:
        index = np.searchsorted(
            self.cumulative_counts, rng.integers(int(self.cumulative_counts[-1])), side="right"
        )
        return float(self.delay_ms[index]) * 0.001


def _mapped(
    value: np.ndarray, low: float, high: float, bits: int, mode: str = "round"
) -> np.ndarray:
    levels = (1 << bits) - 1
    code = (np.clip(value, low, high) - low) * levels / (high - low)
    if mode not in ("round", "floor"):
        raise ValueError("Unknown CAN encoding mode")
    encoded = np.floor(code + 1e-10) if mode == "floor" else np.rint(code)
    encoded = np.clip(encoded, 0, levels)
    return low + encoded * (high - low) / levels


class GF43X40Communication:
    """Hold 20-ms host frames for a 1-ms motor loop; publish 20-ms feedback.

    ``submit`` accepts observed or synthetic arrival times, including jitter.
    ``advance`` latches due frames before each physics substep. ``sample``
    receives the already-computed motor observation after that substep.
    """

    def __init__(
        self,
        num_envs: int,
        num_motors: int,
        *,
        command_period_s: float = 0.020,
        motor_period_s: float = 0.001,
        feedback_period_s: float = 0.020,
        command_delay_s: float = 0.0,
        feedback_delay_s: float = 0.0,
        quantize_can: bool = True,
        calibration_path: Path | None = None,
        command_jitter_ms: tuple[int, int] = (0, 0),
        feedback_jitter_ms: tuple[int, int] = (0, 0),
        jitter_seed: int = 0,
        jitter_mode: str = "uniform",
    ) -> None:
        if num_envs <= 0 or num_motors <= 0:
            raise ValueError("Positive batch dimensions required")
        if not math.isclose(motor_period_s, 0.001, abs_tol=1e-12):
            raise ValueError("Current GF calibration requires a 1-ms motor loop")
        if (
            not all(math.isfinite(value) for value in (
                command_period_s, feedback_period_s, command_delay_s, feedback_delay_s
            ))
            or command_period_s <= 0
            or feedback_period_s <= 0
            or min(command_delay_s, feedback_delay_s) < 0
        ):
            raise ValueError("Periods must be positive and optional delays nonnegative")
        source = calibration_path or Path(__file__).with_name("calibration.json")
        bundle = json.loads(source.read_text(encoding="utf-8"))
        if bundle.get("motor_model") != "GF43X40-10":
            raise ValueError("Expected GF43X40-10 calibration")
        limits = bundle["registers"]
        self.observation = bundle["feedback_observation"]
        self.position_limit = float(limits["pmax"])
        self.speed_limit = float(limits["vmax"])
        self.torque_map_limit = float(limits["tmax"])
        self.shape = (num_envs, num_motors)
        self.command_period_s = command_period_s
        self.motor_period_s = motor_period_s
        self.feedback_period_s = feedback_period_s
        self.command_delay_s = command_delay_s
        self.feedback_delay_s = feedback_delay_s
        self.quantize_can = quantize_can
        self.command_jitter_ms = self._jitter_bounds(command_jitter_ms)
        self.feedback_jitter_ms = self._jitter_bounds(feedback_jitter_ms)
        self.jitter_seed = jitter_seed
        self.jitter_mode = jitter_mode
        self._command_distribution = None
        self._feedback_distribution = None
        if jitter_mode == "measured_20260918":
            if self.command_jitter_ms != (0, 0) or self.feedback_jitter_ms != (0, 0):
                raise ValueError("Measured jitter cannot be combined with uniform jitter ranges")
            profile = json.loads(Path(__file__).with_name("timing_profile.json").read_text())
            if (profile["name"] != jitter_mode or profile["resolution_ms"] != 1
                    or not math.isclose(command_period_s * 1000, profile["command_period_ms"])
                    or not math.isclose(feedback_period_s * 1000, profile["feedback_period_ms"])):
                raise ValueError("Measured timing profile requires 20-ms command/feedback periods")
            self._command_distribution = _DiscreteDelay(
                profile["command"]["delay_ms"], profile["command"]["counts"]
            )
            self._feedback_distribution = _DiscreteDelay(
                profile["feedback"]["delay_ms"], profile["feedback"]["counts"]
            )
        elif jitter_mode != "uniform":
            raise ValueError("Unknown GF communication jitter mode")
        self.reset()

    @staticmethod
    def _jitter_bounds(bounds: tuple[int, int]) -> tuple[int, int]:
        values = np.asarray(bounds, dtype=np.float64)
        if (values.shape != (2,) or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values != np.floor(values))
                or values[0] > values[1]):
            raise ValueError("Jitter requires ordered nonnegative integer millisecond bounds")
        return int(values[0]), int(values[1])

    @staticmethod
    def _jitter_s(bounds: tuple[int, int], rng: np.random.Generator) -> float:
        # One draw per batched frame, shared by its motors and environments.
        low, high = bounds
        return (low if low == high else int(rng.integers(low, high + 1))) * 0.001

    def _command_jitter_s(self) -> float:
        if self._command_distribution is not None:
            return self._command_distribution.sample_s(self._command_rng)
        return self._jitter_s(self.command_jitter_ms, self._command_rng)

    def _feedback_jitter_s(self) -> float:
        if self._feedback_distribution is not None:
            return self._feedback_distribution.sample_s(self._feedback_rng)
        return self._jitter_s(self.feedback_jitter_ms, self._feedback_rng)

    def _array(self, item: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(item, dtype=np.float64)
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {self.shape}")
        return value.copy()

    def _command(self, command: MITCommand) -> MITCommand:
        fields = {
            key: self._array(getattr(command, key), key)
            for key in ("q_des", "qd_des", "kp", "kd", "tau_ff")
        }
        if np.any(fields["kp"] < 0) or np.any(fields["kd"] < 0):
            raise ValueError("MIT gains must be nonnegative")
        if self.quantize_can:
            fields["q_des"] = _mapped(
                fields["q_des"], -self.position_limit, self.position_limit, 16
            )
            fields["qd_des"] = _mapped(fields["qd_des"], -self.speed_limit, self.speed_limit, 12)
            fields["kp"] = _mapped(fields["kp"], 0.0, 500.0, 12)
            fields["kd"] = _mapped(fields["kd"], 0.0, 5.0, 12)
            fields["tau_ff"] = _mapped(
                fields["tau_ff"], -self.torque_map_limit, self.torque_map_limit, 12
            )
        return MITCommand(**fields)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Clear all transport state, or only the resetting robot environments.

        A partial reset preserves the shared simulator clock and command sequence.
        Queued frames and held feedback for the selected environments are cleared
        so a new episode cannot consume a target from the previous one.
        """
        if env_ids is not None:
            ids = np.asarray(env_ids, dtype=np.intp)
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.shape[0]):
                raise ValueError("env_ids must be a 1-D array of valid environment indices")
            commands = [self.active, *(row[2] for row in self._pending_commands)]
            for command in commands:
                for field in ("q_des", "qd_des", "kp", "kd", "tau_ff"):
                    getattr(command, field)[ids] = 0.0
            frames = [*self._pending_feedback]
            if self._feedback is not None:
                frames.append(self._feedback)
            for frame in frames:
                frame.q[ids] = 0.0
                frame.qd[ids] = 0.0
                frame.tau_est[ids] = 0.0
            return
        zeros = np.zeros(self.shape, dtype=np.float64)
        command_seed, feedback_seed = np.random.SeedSequence(self.jitter_seed).spawn(2)
        self._command_rng = np.random.default_rng(command_seed)
        self._feedback_rng = np.random.default_rng(feedback_seed)
        self.active = MITCommand(*(zeros.copy() for _ in range(5)))
        self.active_seq = 0
        self._next_seq = 1
        self._pending_commands: list[tuple[float, int, MITCommand]] = []
        self._pending_feedback: list[MotorFeedback] = []
        self._feedback: MotorFeedback | None = None
        self._next_feedback_sample_s = self.feedback_period_s
        self._last_submit_s = -math.inf
        self._last_advance_s = -math.inf
        self._last_sample_s = -math.inf

    def submit(self, command: MITCommand, at_s: float, *, extra_delay_s: float = 0.0) -> int:
        """Queue a host frame with fixed, explicit and sampled extra delays."""
        if (not math.isfinite(at_s) or at_s < 0 or at_s < self._last_submit_s
                or not math.isfinite(extra_delay_s) or extra_delay_s < 0):
            raise ValueError("Host arrival times must be monotonic and delays nonnegative")
        encoded = self._command(command)
        jitter_s = self._command_jitter_s()
        seq = self._next_seq
        self._next_seq = (seq + 1) & 0xFFFF
        self._pending_commands.append(
            (at_s + self.command_delay_s + extra_delay_s + jitter_s, seq, encoded)
        )
        self._pending_commands.sort(key=lambda row: row[0])
        self._last_submit_s = at_s
        return seq

    def advance(self, at_s: float) -> MITCommand:
        """Latch latest due target before a 1-ms actuator/physics update."""
        if not math.isfinite(at_s) or at_s < 0 or at_s < self._last_advance_s:
            raise ValueError("Motor clock must be nonnegative and monotonic")
        if self._last_advance_s != -math.inf and not math.isclose(
            at_s - self._last_advance_s, self.motor_period_s, abs_tol=1e-7
        ):
            raise ValueError("GF43X40 actuator must be advanced every 1 ms")
        while self._pending_commands and self._pending_commands[0][0] <= at_s + 1e-12:
            _, seq, command = self._pending_commands.pop(0)
            if 0 < ((seq - self.active_seq) & 0xFFFF) < 0x8000:
                self.active = command
                self.active_seq = seq
        self._last_advance_s = at_s
        return self.active

    def sample(
        self, at_s: float, q: np.ndarray, qd: np.ndarray, tau_est: np.ndarray
    ) -> MotorFeedback | None:
        """Offer post-physics feedback; publication follows the 20-ms cadence."""
        if not math.isfinite(at_s) or at_s < 0 or at_s < self._last_sample_s:
            raise ValueError("Feedback sample times must be monotonic")
        if self._last_sample_s != -math.inf and not math.isclose(
            at_s - self._last_sample_s, self.motor_period_s, abs_tol=1e-7
        ):
            raise ValueError("Motor feedback must be sampled every 1 ms")
        position = self._array(q, "q")
        velocity = self._array(qd, "qd")
        velocity += self.observation.get("velocity_feedback_bias", 0.0)
        torque = self._array(tau_est, "tau_est")
        if self.quantize_can:
            position = _mapped(
                position,
                -self.position_limit,
                self.position_limit,
                16,
                self.observation.get("position_encoding", "round"),
            )
            velocity = _mapped(
                velocity,
                -self.speed_limit,
                self.speed_limit,
                12,
                self.observation.get("velocity_encoding", "round"),
            )
            torque = _mapped(
                torque,
                -self.torque_map_limit,
                self.torque_map_limit,
                12,
                self.observation.get("torque_encoding", "round"),
            )
        self._last_sample_s = at_s
        if at_s + 1e-12 >= self._next_feedback_sample_s:
            self._pending_feedback.append(
                MotorFeedback(
                    sample_time_s=at_s,
                    available_time_s=(at_s + self.feedback_delay_s
                                      + self._feedback_jitter_s()),
                    q=position,
                    qd=velocity,
                    tau_est=torque,
                    most_recent_command_seq=self.active_seq,
                )
            )
            self._pending_feedback.sort(key=lambda frame: frame.available_time_s)
            while self._next_feedback_sample_s <= at_s + 1e-12:
                self._next_feedback_sample_s += self.feedback_period_s
        return self.read_feedback(at_s)

    def read_feedback(self, at_s: float) -> MotorFeedback | None:
        """Return the last available frame, held until the next publication."""
        while self._pending_feedback and self._pending_feedback[0].available_time_s <= at_s + 1e-12:
            frame = self._pending_feedback.pop(0)
            if self._feedback is None or frame.sample_time_s > self._feedback.sample_time_s:
                self._feedback = frame
        return self._feedback
