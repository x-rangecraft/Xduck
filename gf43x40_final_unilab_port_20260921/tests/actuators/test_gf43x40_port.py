"""Portable GF motor and transport contracts before whole-robot integration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unilab.actuators.gf43x40 import (
    GF43X40Actuator,
    GF43X40Communication,
    GF43X40Parameters,
    GF43X40RobotMotor,
    MITCommand,
)

FIXTURE = Path(__file__).with_name("fixtures") / "gf43x40_bare_trace.json"


def frame(shape: tuple[int, int], target: float, kp: float = 40.0, kd: float = 2.0) -> MITCommand:
    zeros = np.zeros(shape)
    return MITCommand(
        np.full(shape, target), zeros.copy(), np.full(shape, kp), np.full(shape, kd), zeros.copy()
    )


def test_calibration_is_shared_and_explicit() -> None:
    p = GF43X40Parameters.from_bundle()
    trace = json.loads(FIXTURE.read_text())
    assert len(trace["trace"]) == 1000
    assert trace["source_sha256"] == p.behavior_reference_sha256
    assert p.physics["coast_drag_scale"] == 0.35
    assert len(p.envelope_speed_rad_s) == 27
    assert p.registers["tmax"] == 28.0


@pytest.mark.parametrize("friction_mode", ["isolated_1d", "smooth_coupled_approx"])
@pytest.mark.parametrize("variant", ["exact_compact", "linear", "proxy_current"])
def test_reduced_v8_is_exactly_equal_to_frozen_v8(friction_mode: str, variant: str) -> None:
    """Do not regenerate the pre-reduction bundle or loosen equality tolerances."""
    baseline_path = FIXTURE.parent / "gf43x40_v8_before_reduction.json"
    compact_path = (FIXTURE.parent / "gf43x40_v8_exact_compact.json" if variant=="exact_compact"
                    else FIXTURE.parent / "gf43x40_v8_linear_load.json" if variant=="linear" else None)
    before = GF43X40Parameters.from_bundle(baseline_path)
    after = GF43X40Parameters.from_bundle(compact_path)
    assert after.behavior_reference_sha256 == before.source_sha256
    assert after.physics == before.physics
    assert "feedback_gain" not in after.observation
    assert "memory_gain" not in after.observation
    shape = (2, 14)
    motors = [GF43X40Actuator(*shape, parameters=p, friction_mode=friction_mode)
              for p in (before, after)]
    links = [GF43X40Communication(*shape, calibration_path=path)
             for path in (baseline_path, compact_path)]
    rng = np.random.default_rng(8018)
    mass = np.full(shape, after.physics["armature"] + .1)
    feedback_changed = False
    for tick in range(1000):
        q = rng.uniform(-.5, .5, shape)
        v = rng.uniform(-7., 7., shape)
        if tick % 10 == 0:
            v[:] = 0.
        kp = np.full(shape, (0., 40., 75., 80., 100., 120., 150., 500.)[tick % 8])
        kd = np.full(shape, (0., 2., 2.75, 3., 3.2, 3.35, 4., 5.)[tick % 8])
        command = MITCommand(q*.5, v*.3, kp, kd, np.zeros(shape))
        torque = [m.compute_torque(q, v, command.q_des, command.qd_des,
                  kp, kd, command.tau_ff, effective_mass=mass) for m in motors]
        acceleration = rng.uniform(-30., 30., shape)
        estimates = [m.observe(v, acceleration) for m in motors]
        np.testing.assert_array_equal(torque[0], torque[1])
        if variant == "exact_compact":
            np.testing.assert_array_equal(estimates[0], estimates[1])
        else:
            feedback_changed |= not np.array_equal(estimates[0], estimates[1])
        np.testing.assert_array_equal(motors[0].velocity_feedback, motors[1].velocity_feedback)
        feedback = [link.sample(tick*.001, q, m.velocity_feedback, estimate)
                    for link, m, estimate in zip(links, motors, estimates)]
        assert (feedback[0] is None) == (feedback[1] is None)
        if feedback[0] is not None:
            for key in ("q", "qd", "tau_est"):
                if variant != "exact_compact" and key == "tau_est":
                    continue
                np.testing.assert_array_equal(getattr(feedback[0], key), getattr(feedback[1], key))
        if tick == 499:
            for motor, link in zip(motors, links):
                motor.reset(np.array([0]))
                link.reset(np.array([0]))
    if variant != "exact_compact":
        assert feedback_changed
        assert after.observation["model"] == ("mechanical_torque_proxy" if variant=="proxy_current" else "kinematic_current_proxy")


def test_linear_load_feedback_filter_has_exactly_one_gain() -> None:
    p = GF43X40Parameters.from_bundle(FIXTURE.parent / "gf43x40_v8_linear_load.json")
    zero = replace(p, observation={**p.observation, "load_feedback_gain": 0.0})
    a, b = [GF43X40Actuator(1, 1, parameters=v, friction_mode="smooth_coupled_approx") for v in (p, zero)]
    expected = np.zeros((1, 1))
    alpha = -np.expm1(-.001/p.observation["filter_time_constant"])
    for tick in range(300):
        q = np.array([[.1]])
        v = np.array([[.2 if tick<150 else -.2]])
        c = frame((1, 1), .2, kp=80., kd=3.)
        for motor in (a, b):
            motor.compute_torque(q, v, c.q_des, c.qd_des, c.kp, c.kd, c.tau_ff)
        acceleration = np.array([[.5]])
        load = a.last_torque - p.physics["armature"]*acceleration
        expected += alpha*(p.observation["load_feedback_gain"]*load-expected)
        np.testing.assert_allclose(a.observe(v, acceleration)-b.observe(v, acceleration), expected, rtol=1e-12, atol=1e-12)
    invalid = replace(p, observation={**p.observation, "load_memory_gain": .1})
    with pytest.raises(ValueError, match="only load_feedback_gain"):
        GF43X40Actuator(1, 1, parameters=invalid)


def test_current_torque_proxy_is_direct_and_has_no_hidden_filter() -> None:
    p=GF43X40Parameters.from_bundle()
    assert p.observation["model"] == "mechanical_torque_proxy"
    motor=GF43X40Actuator(2, 3, parameters=p)
    for force in (5., -3., 0., 2.):
        motor.last_torque[:]=force
        actual=motor.observe(np.zeros((2,3)),np.full((2,3),123.))
        np.testing.assert_array_equal(actual,np.full((2,3),force*p.observation["torque_feedback_gain"]))
    motor.reset(np.array([0]))
    np.testing.assert_array_equal(motor.feedback_estimate[0],0.)
    invalid=replace(p,observation={**p.observation,"inertia_gain":.1})
    with pytest.raises(ValueError,match="only its gain"):
        GF43X40Actuator(1,1,parameters=invalid)


def test_single_axis_torque_and_observer_match_frozen_mujoco_trace() -> None:
    motor = GF43X40Actuator(1, 1, parameters=GF43X40Parameters.from_bundle(FIXTURE.parent / "gf43x40_v8_before_reduction.json"))
    trace = json.loads(FIXTURE.read_text())["trace"]

    def scalar(value: float) -> np.ndarray:
        return np.array([[value]], dtype=np.float64)

    for row in trace:
        c = row["command"]
        torque = motor.compute_torque(
            scalar(row["q"]),
            scalar(row["qd"]),
            scalar(c["q_des"]),
            scalar(c["qd_des"]),
            scalar(c["kp"]),
            scalar(c["kd"]),
            scalar(c["tau_ff"]),
            effective_mass=scalar(row["effective_mass"]),
        )
        observed = motor.observe(scalar(row["qd_after"]), scalar(row["acceleration_after"]))
        np.testing.assert_allclose(torque[0, 0], row["torque"], rtol=0, atol=3e-10)
        np.testing.assert_allclose(observed[0, 0], row["feedback_estimate"], rtol=0, atol=3e-10)


def test_fixed_gain_damping_band_matches_frozen_mujoco_trace() -> None:
    trace = json.loads((FIXTURE.parent / "gf43x40_fixed_gain_v8_trace.json").read_text())
    parameters = GF43X40Parameters.from_bundle(FIXTURE.parent / "gf43x40_v8_before_reduction.json")
    assert trace["source_sha256"] == parameters.behavior_reference_sha256
    motor = GF43X40Actuator(1, 1, parameters=parameters)

    def scalar(value: float) -> np.ndarray:
        return np.array([[value]], dtype=np.float64)

    for row in trace["trace"]:
        c = row["command"]
        torque = motor.compute_torque(
            scalar(row["q"]), scalar(row["qd"]), scalar(c["q_des"]),
            scalar(c["qd_des"]), scalar(c["kp"]), scalar(c["kd"]),
            scalar(c["tau_ff"]), effective_mass=scalar(row["effective_mass"]),
        )
        estimate = motor.observe(scalar(row["qd_after"]), scalar(row["acceleration_after"]))
        np.testing.assert_allclose(torque[0, 0], row["torque"], rtol=0, atol=3e-10)
        np.testing.assert_allclose(estimate[0, 0], row["feedback_estimate"], rtol=0, atol=3e-10)


def test_batch_state_reset_and_coupled_mode_is_explicit() -> None:
    motor = GF43X40Actuator(2, 3)
    zeros = np.zeros((2, 3))
    command = frame((2, 3), 0.05)
    with pytest.raises(ValueError, match="effective generalized mass"):
        motor.compute_torque(
            zeros, zeros, command.q_des, command.qd_des, command.kp, command.kd, command.tau_ff
        )
    assert not np.any(motor.effort)
    torque = motor.compute_torque(
        zeros,
        zeros,
        command.q_des,
        command.qd_des,
        command.kp,
        command.kd,
        command.tau_ff,
        effective_mass=np.full((2, 3), 0.047917),
    )
    assert torque.shape == (2, 3)
    before = motor.effort[1].copy()
    motor.reset(np.array([0]))
    assert not np.any(motor.effort[0])
    np.testing.assert_array_equal(motor.effort[1], before)
    approximate = GF43X40Actuator(2, 3, friction_mode="smooth_coupled_approx")
    assert approximate.compute_torque(
        zeros, zeros, command.q_des, zeros, command.kp, command.kd, zeros
    ).shape == (2, 3)
    with pytest.raises(ValueError, match="1-ms"):
        GF43X40Actuator(1, 1, dt=0.005)


def test_coupled_direction_memory_is_causal_and_friction_remains_passive() -> None:
    motor = GF43X40Actuator(1, 1, friction_mode="smooth_coupled_approx")
    zero = np.zeros((1, 1))
    for direction in (1.0, -1.0):
        velocity = np.full((1, 1), direction * 0.25)
        for _ in range(600):
            torque = motor.compute_torque(zero, velocity, zero, zero, zero, zero, zero)
            assert float((torque * velocity)[0, 0]) <= 0
        assert direction * float(motor.friction_memory[0, 0]) > 0.99
    motor.reset()
    assert not np.any(motor.friction_memory)


def test_host_hold_motor_tick_and_feedback_cadence() -> None:
    link = GF43X40Communication(2, 3)
    shape = (2, 3)
    first = frame(shape, 0.05)
    second = frame(shape, -0.05)
    assert link.submit(first, 0.0) == 1
    for step in range(40):
        at = step * 0.001
        if step == 20:
            assert link.submit(second, 0.020) == 2
        active = link.advance(at)
        assert (active.q_des[0, 0] > 0) if step < 20 else (active.q_des[0, 0] < 0)
        shown = link.sample(at + 0.001, np.full(shape, 0.01), np.zeros(shape), np.zeros(shape))
        if step == 18:
            assert shown is None
        if step == 19:
            assert shown is not None and shown.most_recent_command_seq == 1
        if step == 20:
            assert shown is not None and shown.most_recent_command_seq == 1
        if step == 39:
            assert shown is not None and shown.most_recent_command_seq == 2
    assert link.read_feedback(0.041) is shown
    assert shown.q.shape == shape
    assert abs(shown.q[0, 0] - 0.01) < 0.0003815
    first.q_des[:] = 2.0
    assert active.q_des[0, 0] < 0


def test_optional_delay_and_wrong_physics_cadence_fail_closed() -> None:
    link = GF43X40Communication(1, 1, command_delay_s=0.003, feedback_delay_s=0.004)
    link.submit(frame((1, 1), 0.05), 0.0)
    for step in range(4):
        active = link.advance(step * 0.001)
        assert (active.q_des[0, 0] == 0) if step < 3 else (active.q_des[0, 0] > 0)
    with pytest.raises(ValueError, match="every 1 ms"):
        link.advance(0.008)
    link.reset()
    link.submit(frame((1, 1), 0.05), 0.0)
    for step in range(20):
        link.advance(step * 0.001)
        link.sample((step + 1) * 0.001, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    assert link.read_feedback(0.023) is None
    assert link.read_feedback(0.024) is not None


def test_partial_transport_reset_does_not_replay_old_episode_commands() -> None:
    link = GF43X40Communication(2, 1)
    link.submit(frame((2, 1), 0.2), 0.0)
    for step in range(20):
        link.advance(step * 0.001)
        link.sample((step + 1) * 0.001, np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)))
    assert link.read_feedback(0.020) is not None
    link.submit(frame((2, 1), 0.3), 0.020)
    link.reset(np.array([0]))
    active = link.advance(0.020)
    assert active.q_des[0, 0] == 0.0
    assert active.q_des[1, 0] > 0.0
    feedback = link.read_feedback(0.020)
    assert feedback is not None
    assert feedback.q[0, 0] == 0.0
    assert feedback.q[1, 0] > 0.0


def test_robot_motor_uses_20_ms_host_hold_and_partial_reset() -> None:
    motor = GF43X40RobotMotor(2, 14, kp=60.0, kd=2.0)
    position = np.zeros((2, 14))
    velocity = np.zeros((2, 14))
    target = np.full((2, 14), 0.05)
    for step in range(40):
        if step == 20:
            target[:] = -0.05
        torque = motor.begin_substep(target, position, velocity)
        assert torque.shape == (2, 14)
        motor.finish_substep(position, velocity)
    assert motor.feedback is not None
    assert motor.feedback.most_recent_command_seq == 2
    assert motor.communication.active.q_des[0, 0] < 0.0
    motor.reset(np.array([0]))
    assert motor.communication.active.q_des[0, 0] == 0.0
    assert motor.communication.active.q_des[1, 0] < 0.0
    assert motor.actuator.effort[0, 0] == 0.0


def test_robot_motor_accepts_per_joint_mit_gains_and_restores_them() -> None:
    kp = np.linspace(45.0, 75.0, 14)
    kd = np.linspace(1.5, 3.0, 14)
    motor = GF43X40RobotMotor(2, 14, kp=kp, kd=kd)
    np.testing.assert_allclose(motor.kp, np.broadcast_to(kp, (2, 14)))
    np.testing.assert_allclose(motor.kd, np.broadcast_to(kd, (2, 14)))
    motor.kp[0] = 0.0
    motor.kd[0] = 0.0
    motor.reset(np.array([0]))
    np.testing.assert_allclose(motor.kp[0], kp)
    np.testing.assert_allclose(motor.kd[0], kd)
    with pytest.raises(ValueError, match="one value per motor"):
        GF43X40RobotMotor(1, 14, kp=[60.0] * 13, kd=2.0)


def test_actuator_and_communication_run_together_for_two_host_frames() -> None:
    motor = GF43X40Actuator(1, 1)
    link = GF43X40Communication(1, 1)
    mass = np.full((1, 1), motor.parameters.physics["armature"])
    q = np.zeros((1, 1))
    v = np.zeros((1, 1))
    link.submit(frame((1, 1), 0.05), 0.0)
    first_torque = None
    for step in range(40):
        t = step * 0.001
        if step == 20:
            link.submit(frame((1, 1), -0.05), t)
        cmd = link.advance(t)
        torque = motor.compute_torque(
            q,
            v,
            cmd.q_des,
            cmd.qd_des,
            cmd.kp,
            cmd.kd,
            cmd.tau_ff,
            effective_mass=mass,
        )
        if step == 0:
            first_torque = float(torque[0, 0])
        acceleration = torque / mass
        v = v + 0.001 * acceleration
        q = q + 0.001 * v
        estimate = motor.observe(v, acceleration)
        feedback = link.sample(t + 0.001, q, v, estimate)
    assert first_torque is not None and first_torque > 0
    assert motor.last_requested[0, 0] < 0
    assert feedback is not None and feedback.most_recent_command_seq == 2
    assert feedback.q.shape == (1, 1)


def test_optional_dynamic_states_match_reference_and_reset() -> None:
    trace = json.loads(FIXTURE.with_name("gf43x40_dynamic_trace.json").read_text())
    p = replace(
        GF43X40Parameters.from_bundle(), physics=trace["physics"], observation=trace["observation"]
    )
    motor = GF43X40Actuator(1, 1, parameters=p)

    def scalar(value: float) -> np.ndarray:
        return np.array([[value]], dtype=np.float64)

    for row in trace["trace"]:
        c = row["command"]
        torque = motor.compute_torque(
            scalar(row["q"]),
            scalar(row["qd"]),
            scalar(c["q_des"]),
            scalar(c["qd_des"]),
            scalar(c["kp"]),
            scalar(c["kd"]),
            scalar(c["tau_ff"]),
            effective_mass=scalar(row["effective_mass"]),
        )
        estimate = motor.observe(scalar(row["qd_after"]), scalar(row["acceleration_after"]))
        np.testing.assert_allclose(torque[0, 0], row["torque"], rtol=0, atol=3e-10)
        np.testing.assert_allclose(estimate[0, 0], row["feedback_estimate"], rtol=0, atol=3e-10)
        np.testing.assert_allclose(
            motor.velocity_feedback[0, 0], row["velocity_observation"], rtol=0, atol=3e-10
        )
    motor.reset()
    assert not np.any(motor.friction_memory)
    assert not np.any(motor.observed_velocity)
    assert not np.any(motor._initialized)


def test_can_floor_zero_and_saturation() -> None:
    from unilab.actuators.gf43x40.communication import _mapped

    np.testing.assert_allclose(
        _mapped(np.array([0.0, -100.0, 100.0]), -10, 10, 12, "floor"),
        [-10 / 4095, -10, 10],
        rtol=0,
        atol=1e-12,
    )
