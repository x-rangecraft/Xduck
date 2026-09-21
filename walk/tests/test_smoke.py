"""Run with the pinned company UniLab runtime and MuJoCo extra."""

import numpy as np

from xduck_walk_latest.task import make_walk_env, walk_cfg


def test_gf_walk_manager_smoke():
    cfg = walk_cfg()
    cfg.validate()
    env = make_walk_env(cfg, num_envs=2)
    try:
        assert env.obs_groups_spec == {"obs": 61, "critic": 76}
        obs, _ = env.reset()
        assert obs["obs"].shape == (2, 61)
        assert obs["critic"].shape == (2, 76)
        for _ in range(3):
            state = env.step(np.zeros((2, 14), dtype=np.float32))
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.obs["critic"]).all()
            assert np.isfinite(state.reward).all()
        term = env.action_manager.get_term("joint_pos")
        assert term.motor.actuator.shape == (2, 14)
        assert term.motor._tick == 60
    finally:
        env.close()
