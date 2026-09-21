"""External XL330 Manager-Based owner and public policy I/O contract."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.cli import package_root
from unilab.envs import ManagerBasedRlEnvCfg

from microduck_rl_unilab.tasks.microduck.bam_action import BamVoltageActionCfg
from microduck_rl_unilab.tasks.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = "MicroduckXl330OfficialVelocityFlat"
OWNER = "task=microduck_xl330_velocity_flat/mujoco"


def owner():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[OWNER])
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(TASK)
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg, override


def test_owner_contract():
    cfg, env_cfg, _ = owner()
    assert cfg.training.task_name == TASK
    assert cfg.training.sim_backend == "mujoco"
    assert registry.list_registered_envs()[TASK]["available_backends"] == ["mujoco"]
    assert env_cfg.sim_dt == pytest.approx(0.005)
    assert env_cfg.ctrl_dt == pytest.approx(0.02)
    assert isinstance(env_cfg.actions["joint_pos"], BamVoltageActionCfg)
    assert list(env_cfg.observations["policy"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
        "twist_command", "head_pose_command", "body_pose_command",
    ]
    assert MICRODUCK_NUM_ACTION == 14
    assert MICRODUCK_ACTOR_OBS_DIM == 61
    assert MICRODUCK_CRITIC_OBS_DIM == 76
    assert "Euler" in (ROOT / "assets/robots/microduck/scene_flat_bam.xml").read_text()


@pytest.mark.slow
def test_mujoco_reset_and_step():
    pytest.importorskip("unisim.backend.mujoco.backend")
    _, _, override = owner()
    env = registry.make(TASK, sim_backend="mujoco", num_envs=2, env_cfg_override=override)
    try:
        obs, _ = env.reset(seed=7)
        assert obs["obs"].shape == (2, 61)
        assert obs["critic"].shape == (2, 76)
        assert env.action_space.shape == (14,)
        assert env._uses_pre_step_control
        # The "home" keyframe is the old scene_walk.xml STAND pose.
        expected_stand = np.array([
            0.0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
            0.3490658503988659, 0.3490658503988659, 0.0, 0.0,
            0.0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
        ])
        np.testing.assert_allclose(env.scene["robot"].data.default_joint_pos[0], expected_stand, atol=2e-7)
        root_z = env.scene["robot"].data.root_link_pos_w[:, 2]
        assert np.all((root_z >= 0.12) & (root_z <= 0.13))
        state = env.step(np.zeros((2, 14), dtype=np.float32))
        assert all(np.isfinite(value).all() for value in (*state.obs.values(), state.reward))
    finally:
        env.close()


def test_xduck_critic_order_rewards_and_reset_contract():
    """Freeze the pre-migration Xduck XL330 owner, rather than only dimensions."""
    _, env_cfg, _ = owner()
    assert list(env_cfg.observations["critic"].terms) == [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "twist_command", "foot_height", "foot_air_time", "foot_contact",
        "foot_contact_forces", "head_pose_command", "body_pose_command",
    ]
    expected = {
        "leg_pose": 1.0,
        "upright": 2.0,
        "foot_slip": -0.1,
        "self_collisions": -1.0,
        "air_time": 3.0,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 2.0,
        "action_rate": -0.1,
        "foot_clearance": -2.0,
        "foot_swing_height": -0.25,
        "head_pose_tracking": 2.0,
        "body_pose_tracking": 0.0,
        "head_pose_bias": 0.0,
        "dof_pos_limits": -1.0,
    }
    assert {name: term.weight for name, term in env_cfg.rewards.items()} == expected
    assert env_cfg.scale_rewards_by_dt
    assert env_cfg.rewards["angular_momentum"].params["reference"] == pytest.approx(0.25)
    reset = env_cfg.events["reset_base"].params
    assert reset["pose_range"]["z"] == [0.0, 0.01]
    assert reset["pose_range"]["yaw"] == [-np.pi, np.pi]
    assert reset["velocity_range"] == {}
    assert env_cfg.scene.default_keyframe_name == "home"
    twist = env_cfg.commands["twist"]
    assert twist.rel_forward_envs == pytest.approx(0.2)
    assert twist.rel_standing_envs == pytest.approx(0.02)
    assert twist.turn_in_place_fraction == pytest.approx(0.15)
    assert twist.resampling_time_range == [3.0, 8.0]
    assert list(twist.ranges.lin_vel_x) == [-0.4, 0.4]
    assert list(twist.ranges.lin_vel_y) == [-0.3, 0.3]
    assert list(twist.ranges.ang_vel_z) == [-1.0, 1.0]
    assert env_cfg.observations["critic"].terms["foot_contact_forces"].params["sensor_names"] == [
        "left_foot_netforce", "right_foot_netforce",
    ]


@pytest.mark.slow
def test_turn_bucket_overrides_standing_and_critic_keeps_raw_netforce():
    pytest.importorskip("unisim.backend.mujoco.backend")
    _, _, override = owner()
    override["commands"]["twist"]["rel_standing_envs"] = 1.0
    override["commands"]["twist"]["turn_in_place_fraction"] = 1.0
    env = registry.make(TASK, sim_backend="mujoco", num_envs=4, env_cfg_override=override)
    try:
        obs, _ = env.reset(seed=19)
        command = env.command_manager.get_command("twist")
        np.testing.assert_allclose(command[:, :2], 0.0)
        assert np.all(np.abs(command[:, 2]) >= 0.4)
        assert np.all(np.abs(command[:, 2]) <= 1.0)
        assert not np.any(env.command_manager.get_term("twist").is_standing_env)
        raw = np.concatenate(
            [
                env.scene.bind_sensor_data(("left_foot_netforce",)).read(),
                env.scene.bind_sensor_data(("right_foot_netforce",)).read(),
            ],
            axis=1,
        )
        np.testing.assert_allclose(obs["critic"][:, 60:66], raw)
        angular_momentum = env.scene.bind_sensor_data(("root_angmom",)).read()
        term = env.reward_manager.get_term_cfg("angular_momentum").func
        np.testing.assert_allclose(
            term(env), np.sum(np.square(angular_momentum / 0.25), axis=1)
        )
    finally:
        env.close()
