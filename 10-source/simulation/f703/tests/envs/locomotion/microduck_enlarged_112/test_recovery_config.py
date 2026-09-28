"""Protect the V112 recovery objective from accidental profile drift."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_recovery_preserves_behavior_and_requires_travel_protection():
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/ppo"), version_base="1.3"):
        configs = [OmegaConf.to_container(compose(
            config_name="config_mlx",
            overrides=[f"task=microduck_enlarged_112_walk_flat/{name}"],
        ), resolve=True) for name in ("mujoco_remote_head_level", "mujoco_remote_recovery")]
    baseline, recovery = configs
    expected_reward = baseline["reward"]
    expected_reward["scales"]["head_level_bias"] = 6.0
    assert recovery["reward"] == expected_reward
    assert "idle_leg_motion" not in recovery["reward"]["scales"]
    env = recovery["env"]
    assert env["head_level_moving_multiplier"] == 1.0
    assert env["head_level_bias_schedule_steps"] == [0]
    assert env["head_level_bias_schedule_scales"] == [6.0]
    assert env["joint_limit_barrier_scale"] == 25.0
    assert env["joint_limit_barrier_margin_rad"] == 0.01
    assert env["joint_limit_termination_margin_rad"] == 0.005
    for key in ("control_config", "motor_constraints", "identified_dynamics",
                "full_motor_dr", "domain_rand", "commands", "noise_config"):
        assert env[key] == baseline["env"][key]
    assert recovery["algo"]["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert recovery["algo"]["algorithm"]["disable_finite_checks"] is False
