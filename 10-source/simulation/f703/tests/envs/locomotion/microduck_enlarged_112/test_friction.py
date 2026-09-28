"""Physical tests of the candidate: loads, breakaway, dissipation and reset."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.envs.locomotion.microduck_enlarged_112.friction import (
    explicit_dry_friction_loss,
    install_dry_friction,
)

FP, FN, BP, BN = 0.272, 0.325, 0.02965, 0.03051


def single_joint(dt=0.002):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
    <worldbody><body><joint name="hinge" type="hinge" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.03 0.03 0.03"/>
    <geom size="0.05" contype="0" conaffinity="0"/></body></worldbody>
    <actuator><motor joint="hinge"/></actuator></mujoco>""")
    install_dry_friction(model, [0], FP, FN)
    return model, mujoco.MjData(model)


def step(model, data, motor, load=0):
    data.ctrl[0] = motor - explicit_dry_friction_loss(data.qvel[0], FP, FN, BP, BN)
    data.qfrc_applied[0] = load
    mujoco.mj_step(model, data)


@pytest.mark.parametrize("motor,load", [(0.2, 0), (-0.25, 0), (0, 0.2), (0.7, -0.5)])
def test_subthreshold_balance_accounts_for_external_load(motor, load):
    model, data = single_joint()
    for _ in range(500):
        step(model, data, motor, load)
    # Soft constraints permit bounded numerical creep, not exact locking.
    assert abs(data.qpos[0]) < 1e-4
    assert abs(data.qvel[0]) < 1e-4


@pytest.mark.parametrize(
    "motor,load,sign", [(0.4, 0, 1), (-0.45, 0, -1), (0, 0.4, 1), (0.2, -0.7, -1)]
)
def test_breakaway_can_be_caused_by_motor_or_external_load(motor, load, sign):
    model, data = single_joint()
    for _ in range(250):
        step(model, data, motor, load)
    assert sign * data.qvel[0] > 0.2


@pytest.mark.parametrize("speed", [0.5, -0.5])
def test_sliding_loss_matches_existing_directional_coefficients(speed):
    model, data = single_joint()
    data.qvel[0] = speed
    data.ctrl[0] = -explicit_dry_friction_loss(speed, FP, FN, BP, BN)
    mujoco.mj_forward(model, data)
    expected = -(FP + BP * speed) if speed > 0 else FN - BN * speed
    total = data.qfrc_actuator[0] + data.qfrc_constraint[0]
    assert total == pytest.approx(expected, abs=1e-7)
    assert total * speed < 0


@pytest.mark.parametrize("dt", [0.002, 0.001])
def test_free_stop_dissipates_and_reversal_needs_threshold(dt):
    model, data = single_joint(dt)
    data.qvel[0] = 0.5
    energies = []
    for _ in range(round(0.5 / dt)):
        step(model, data, 0)
        energies.append(0.5 * 0.03 * data.qvel[0] ** 2)
    assert max(np.diff(energies)) < 1e-9
    assert abs(data.qvel[0]) < 1e-4
    for _ in range(round(0.2 / dt)):
        step(model, data, -0.2)
    assert abs(data.qvel[0]) < 1e-4
    for _ in range(round(0.2 / dt)):
        step(model, data, -0.5)
    assert data.qvel[0] < -0.2


def make_candidate(task, extra=None):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        cfg = compose(
            config_name="config_mlx",
            overrides=[f"task=microduck_enlarged_112_{task}_flat/mujoco_dry_friction"],
        )
    overrides = OmegaConf.to_container(cfg.env, resolve=True)
    overrides["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    overrides.update(extra or {})
    registry.ensure_registries()
    env = registry.make(
        cfg.training.task_name, sim_backend="mujoco", num_envs=2, env_cfg_override=overrides
    )
    env.set_autoreset(False)
    return env


@pytest.mark.parametrize("task", ["stand", "walk"])
def test_full_robot_candidate_step_and_partial_reset(task):
    np.random.seed(211)
    env = make_candidate(task)
    try:
        state = env.init_state()
        for _ in range(100):
            state = env.step(np.zeros((2, 14), dtype=np.float32))
        assert np.isfinite(env.get_dof_pos()).all()
        assert not state.terminated.any()
        np.testing.assert_allclose(state.info["motor_fit_time_constant_s"], 0.01)
        np.testing.assert_allclose(env._identified_coulomb_positive_nm, FP)
        np.testing.assert_allclose(env._identified_coulomb_negative_nm, FN)
        assert "motor_fit_net_torque_nm" not in state.info
        assert "motor_fit_friction_nm" not in state.info
        for i in range(2):
            model = env._backend.get_playback_model(i)
            np.testing.assert_allclose(model.dof_frictionloss[6:], (FP + FN) / 2)
        env.reset(np.array([0], dtype=np.int32))
        state = env.step(np.zeros((2, 14), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        np.testing.assert_allclose(env._identified_coulomb_positive_nm, FP)
    finally:
        env.close()


def test_random_coulomb_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="coulomb_positive_scale_range"):
        make_candidate("stand", {"full_motor_dr": {"coulomb_positive_scale_range": [0.85, 1.15]}})
