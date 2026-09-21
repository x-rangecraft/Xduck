"""Communication jitter must alter data availability, not physics time."""

import numpy as np
import pytest

from unilab.actuators.gf43x40 import GF43X40Communication, GF43X40RobotMotor, MITCommand


class Tickets:
    def __init__(self, values):
        self.values = iter(values)

    def integers(self, high):
        ticket = next(self.values)
        assert 0 <= ticket < high
        return ticket


def command(value):
    q = np.full((2, 1), value, dtype=float)
    zero = np.zeros_like(q)
    return MITCommand(q, zero, zero, zero, zero)


def trace(seed):
    link = GF43X40Communication(
        2, 1, quantize_can=False, command_jitter_ms=(1, 5),
        feedback_jitter_ms=(2, 6), jitter_seed=seed,
    )
    commands, feedback = [], []
    last_seq = 0
    last_sample = None
    for tick in range(1000):
        if tick % 20 == 0:
            link.submit(command(tick // 20 + 1), tick * .001)
        active = link.advance(tick * .001)
        if link.active_seq != last_seq:
            delay = tick - (link.active_seq - 1) * 20
            assert 1 <= delay <= 5
            np.testing.assert_array_equal(active.q_des, link.active_seq)
            commands.append((tick, link.active_seq))
            last_seq = link.active_seq
        q = np.full((2, 1), tick + 1.)
        shown = link.sample((tick + 1) * .001, q, q, q)
        if shown is not None:
            sample_tick = round(shown.sample_time_s * 1000)
            np.testing.assert_array_equal(shown.q, sample_tick)
            if sample_tick != last_sample:
                assert 2 <= tick + 1 - sample_tick <= 6
                feedback.append((tick + 1, sample_tick))
                last_sample = sample_tick
    return commands, feedback


def test_jitter_is_bounded_varies_per_frame_and_is_reproducible():
    first = trace(123)
    assert first == trace(123)
    assert first != trace(456)
    commands, feedback = first
    assert len({tick - (seq - 1) * 20 for tick, seq in commands}) > 1
    assert len({arrival - sample for arrival, sample in feedback}) > 1


@pytest.mark.parametrize("bounds", [(-1, 2), (3, 1), (0, .5), (0, float("nan")), (1,)])
@pytest.mark.parametrize("field", ["command_jitter_ms", "feedback_jitter_ms"])
def test_invalid_jitter_bounds_are_rejected(field, bounds):
    with pytest.raises(ValueError, match="integer millisecond"):
        GF43X40Communication(1, 1, **{field: bounds})


def test_late_frames_do_not_roll_back_newer_commands_or_feedback():
    class Delays:
        def __init__(self):
            self.values = iter([30, 0])

        def integers(self, low, high):
            return next(self.values)

    link = GF43X40Communication(
        2, 1, quantize_can=False, command_jitter_ms=(0, 30), feedback_jitter_ms=(0, 30),
    )
    link._command_rng = Delays()
    link._feedback_rng = Delays()
    link.submit(command(1), 0.)
    for tick in range(51):
        if tick == 20:
            link.submit(command(2), .020)
        active = link.advance(tick * .001)
        if tick >= 20:
            np.testing.assert_array_equal(active.q_des, 2.)
        q = np.full((2, 1), tick * .001)
        shown = link.sample(tick * .001, q, q, q)
        if tick < 40:
            assert shown is None
        else:
            assert shown.sample_time_s == pytest.approx(.040)


def test_partial_reset_clears_delayed_frames_without_resetting_other_robot():
    link = GF43X40Communication(
        2, 1, quantize_can=False, command_jitter_ms=(30, 30), feedback_jitter_ms=(10, 10),
    )
    link.submit(command(1), 0.)
    for tick in range(31):
        if tick == 21:
            link.reset(np.array([0]))
        active = link.advance(tick * .001)
        q = np.ones((2, 1))
        shown = link.sample(tick * .001, q, q, q)
    np.testing.assert_array_equal(active.q_des[:, 0], [0, 1])
    np.testing.assert_array_equal(shown.q[:, 0], [0, 1])


def test_robot_holds_initial_observation_until_delayed_feedback_arrives():
    motor = GF43X40RobotMotor(1, 1, kp=60., kd=2., feedback_jitter_ms=(4, 4))
    initial = np.array([[.1]])
    initial_q = motor.quantize_position_feedback(initial)
    initial_v = motor.quantize_velocity_feedback(initial)
    for tick in range(24):
        state = np.array([[.2]])
        motor.begin_substep(state, state, state)
        motor.finish_substep(state, state)
        if tick < 23:
            assert motor.feedback is None
            np.testing.assert_array_equal(motor.quantize_position_feedback(state), initial_q)
            np.testing.assert_array_equal(motor.quantize_velocity_feedback(state), initial_v)
    assert motor.feedback is not None
    np.testing.assert_array_equal(motor.quantize_position_feedback(state), motor.feedback.q)
    assert motor.dt == .001


def test_measured_probability_mass_matches_captured_late_event_counts():
    """Exhaust the equiprobable tickets, including every CDF boundary."""
    link = GF43X40Communication(2, 1, jitter_mode="measured_20260918")
    link._command_rng = Tickets(range(3141))
    commands = np.rint([link._command_jitter_s() * 1000 for _ in range(3141)]).astype(int)
    np.testing.assert_array_equal(np.bincount(commands), [3126, 7, 5, 1, 1, 1])
    link._feedback_rng = Tickets(range(3249))
    feedback = np.rint([link._feedback_jitter_s() * 1000 for _ in range(3249)]).astype(int)
    np.testing.assert_array_equal(np.bincount(feedback), [3247, 2])


def test_measured_late_command_creates_long_then_short_interval_and_stale_feedback():
    link = GF43X40Communication(2, 1, jitter_mode="measured_20260918", quantize_can=False)
    link._command_rng = Tickets([0, 3140, 0])  # normal, 5-ms late, normal
    link._feedback_rng = Tickets([3248, 0])  # 1-ms late, normal
    activations = []
    last_seq = 0
    for tick in range(45):
        if tick % 20 == 0:
            link.submit(command(tick // 20 + 1), tick * .001)
        active = link.advance(tick * .001)
        if link.active_seq != last_seq:
            activations.append(tick)
            last_seq = link.active_seq
        np.testing.assert_array_equal(active.q_des, 1 if tick < 25 else 2 if tick < 40 else 3)
        state = np.full((2, 1), tick + 1.)
        feedback = link.sample((tick + 1) * .001, state, state, state)
        if tick + 1 < 21:
            assert feedback is None
        elif tick + 1 < 40:
            assert feedback.sample_time_s == pytest.approx(.020)
            assert feedback.available_time_s == pytest.approx(.021)
            np.testing.assert_array_equal(feedback.q, 20.)
        else:
            np.testing.assert_array_equal(feedback.q, 40.)
    assert activations == [0, 25, 40]
    assert link.motor_period_s == .001


def test_measured_seed_replays_on_full_reset_and_partial_reset_keeps_rng_state():
    link = GF43X40Communication(2, 1, jitter_mode="measured_20260918", jitter_seed=42)
    def draws():
        return [(link._command_jitter_s(), link._feedback_jitter_s()) for _ in range(10000)]
    first = draws()
    assert any(c > 0 for c, _ in first)
    assert any(f > 0 for _, f in first)
    link.reset(np.array([0]))
    second = draws()
    link.reset()
    assert draws() == first
    assert draws() == second


@pytest.mark.parametrize("settings", [
    {"jitter_mode": "missing"},
    {"jitter_mode": "measured_20260918", "command_jitter_ms": (0, 1)},
    {"jitter_mode": "measured_20260918", "feedback_jitter_ms": (0, 1)},
    {"jitter_mode": "measured_20260918", "command_period_s": .010},
    {"jitter_mode": "measured_20260918", "feedback_period_s": .010},
])
def test_measured_mode_rejects_ambiguous_or_incompatible_settings(settings):
    with pytest.raises(ValueError):
        GF43X40Communication(1, 1, **settings)
