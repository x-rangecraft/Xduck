"""Zero-twist stillness shaping without changing the actuator or policy I/O."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.envs.locomotion.microduck_enlarged_112.tasks import MicroDuckEnlarged112WalkFlatEnv


def profile(name):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        return OmegaConf.to_container(compose(
            config_name="config_mlx",
            overrides=[f"task=microduck_enlarged_112_walk_flat/{name}"],
        ), resolve=True)


def test_idle_cost_is_deadbanded_monotonic_and_exactly_zero_gated():
    owner = SimpleNamespace(
        LEG_INDICES=MicroDuckEnlarged112WalkFlatEnv.LEG_INDICES,
        _idle_velocity_deadband=0.15, _idle_velocity_scale=1.0,
    )
    velocity = np.zeros((7, 14), dtype=np.float32)
    velocity[1, owner.LEG_INDICES] = 0.1
    velocity[2, owner.LEG_INDICES] = 1.15
    velocity[3, owner.LEG_INDICES] = -2.15
    velocity[4:, owner.LEG_INDICES] = 3.0
    velocity[0, 5:9] = 100.0  # Head movement is deliberately independent.
    command = np.zeros((7, 3), dtype=np.float32)
    command[4, 0] = 0.001
    command[5, 1] = -0.001
    command[6, 2] = 0.001
    before = velocity.copy(), command.copy()
    result = MicroDuckEnlarged112WalkFlatEnv._idle_leg_motion(
        owner, SimpleNamespace(dof_vel=velocity, info={"commands": command})
    )
    np.testing.assert_allclose(result, [0, 0, 1, 4, 0, 0, 0], atol=1e-6)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(velocity, before[0])
    np.testing.assert_array_equal(command, before[1])


def test_refinement_profile_preserves_plant_commands_and_mature_rewards():
    old = profile("mujoco_remote_head_bias")
    new = profile("mujoco_remote_idle_refinement")
    assert new["reward"]["scales"].pop("idle_leg_motion") == -2.0
    assert new["env"].pop("idle_leg_velocity_deadband_rad_s") == 0.15
    assert new["env"].pop("idle_leg_velocity_scale_rad_s") == 1.0
    assert new["reward"]["head_pose_bias_schedule_scales"] == [3.0]
    old["reward"].update(head_pose_bias_schedule_steps=[0], head_pose_bias_schedule_scales=[3.0])
    old["reward"]["scales"]["head_pose_bias"] = 3.0
    for cfg in (old, new):
        for section in ("training", "algo"):
            for field in ("seed", "log_root", "load_run", "max_iterations"):
                cfg[section].pop(field, None)
    assert new == old


def test_idle_reward_dispatch_and_unchanged_io():
    cfg = profile("mujoco_remote_idle_refinement")
    override = cfg["env"]
    override["reward_config"] = cfg["reward"]
    registry.ensure_registries()
    env = registry.make(cfg["training"]["task_name"], sim_backend="mujoco", num_envs=2,
                        env_cfg_override=override)
    try:
        state = env.init_state()
        q = env.get_dof_pos().copy()
        ctx = SimpleNamespace(dof_vel=np.ones((2, 14), np.float32),
                              info={"commands": np.zeros((2, 3), np.float32)})
        np.testing.assert_allclose(env._reward_fns["idle_leg_motion"](ctx), 0.85**2)
        np.testing.assert_array_equal(env.get_dof_pos(), q)
        state = env.step(np.zeros((2, 14), np.float32))
        assert state.obs["obs"].shape == (2, 61)
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


@pytest.mark.parametrize("field,value", [
    ("idle_leg_velocity_deadband_rad_s", -0.1),
    ("idle_leg_velocity_deadband_rad_s", np.nan),
    ("idle_leg_velocity_scale_rad_s", 0.0),
    ("idle_leg_velocity_scale_rad_s", np.inf),
    ("idle_home_target_scale_rad", 0.0),
    ("idle_home_target_scale_rad", np.nan),
])
def test_invalid_idle_scales_fail_before_simulation(field, value):
    cfg = profile("mujoco_remote_idle_refinement")
    override = cfg["env"]
    override["reward_config"] = cfg["reward"]
    override[field] = value
    registry.ensure_registries()
    with pytest.raises(ValueError, match=field):
        registry.make(cfg["training"]["task_name"], sim_backend="mujoco", num_envs=1,
                      env_cfg_override=override)


def test_idle_home_cost_uses_raw_scaled_legs_and_exact_zero_commands():
    owner = SimpleNamespace(
        LEG_INDICES=MicroDuckEnlarged112WalkFlatEnv.LEG_INDICES,
        _idle_home_scale=0.25,
        _cfg=SimpleNamespace(control_config=SimpleNamespace(action_scale=0.5)),
    )
    action = np.zeros((6, 14), np.float32)
    action[0, 5:9] = 100
    action[1, owner.LEG_INDICES] = 0.5
    action[2, owner.LEG_INDICES] = -1
    action[3:, owner.LEG_INDICES] = 1
    command = np.zeros((6, 3), np.float32)
    command[3:] = np.eye(3) * 0.001
    before = action.copy(), command.copy()
    result = MicroDuckEnlarged112WalkFlatEnv._idle_home_target(owner, SimpleNamespace(
        info={"commands": command, "v112_reward_raw_action": action}))
    np.testing.assert_allclose(result, [0, 1, 4, 0, 0, 0])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(action, before[0])
    np.testing.assert_array_equal(command, before[1])


def test_idle_home_profile_only_replaces_idle_objective():
    old = profile("mujoco_remote_idle_refinement")
    new = profile("mujoco_remote_idle_home")
    assert new["env"].pop("idle_home_target_scale_rad") == 0.25
    assert new["reward"]["scales"].pop("idle_home_target") == -2
    assert new["reward"]["scales"]["idle_leg_motion"] == 0
    old["reward"]["scales"]["idle_leg_motion"] = 0
    for cfg in (old, new):
        cfg["algo"].pop("max_iterations")
        cfg["training"].pop("log_root")
    assert new == old


def test_idle_home_dispatch_rollout():
    cfg = profile("mujoco_remote_idle_home")
    override = cfg["env"]
    override["reward_config"] = cfg["reward"]
    registry.ensure_registries()
    env = registry.make(cfg["training"]["task_name"], sim_backend="mujoco", num_envs=2,
                        env_cfg_override=override)
    try:
        state = env.init_state()
        state.info["commands"][:] = 0
        state = env.step(np.full((2, 14), 0.25, np.float32))
        assert state.obs["obs"].shape == (2, 61)
        assert np.isfinite(state.reward).all()
        np.testing.assert_allclose(env._reward_fns["idle_home_target"](
            SimpleNamespace(info=state.info)), 1)
    finally:
        env.close()
