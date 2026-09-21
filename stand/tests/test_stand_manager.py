"""Real backend checks for the XDuck stand Manager-Based migration."""

from __future__ import annotations

import numpy as np

from xduck_stand_latest.task import make_stand_env


def test_handoff_preserves_previous_action_and_zero_command():
    env = make_stand_env(num_envs=4)
    try:
        obs, _ = env.reset(seed=42)
        event = env.event_manager.get_term_cfg("handoff_reset").func
        assert obs["obs"].shape == (4, 61)
        assert obs["critic"].shape == (4, 76)
        np.testing.assert_allclose(obs["obs"][:, -13:], 0.0)
        np.testing.assert_allclose(env.action_manager.action, event.last_actions)
        np.testing.assert_allclose(event.last_actions, event.actions[event.selected_indices])
    finally:
        env.close()


def test_stand_step_finite_and_settled_metric_available():
    env = make_stand_env(num_envs=2)
    try:
        env.reset(seed=7)
        state = env.step(np.zeros((2, 14), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert state.obs["obs"].shape == (2, 61)
        assert state.obs["critic"].shape == (2, 76)
        assert "stop_settled" in env.metrics_manager.active_terms
    finally:
        env.close()


def test_selected_599_checkpoint_loads_with_latest_ppo_runner():
    """Check ABI/loadability; this does not assert physical trajectory parity."""
    import json
    from pathlib import Path

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "checkpoints/run_config.json").read_text())["config"]["algo"]
    saved["algorithm"]["class_name"] = "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO"
    train_cfg = normalize_ppo_train_cfg(saved)
    train_cfg.setdefault("runner", {})["logger"] = "none"
    env = make_stand_env(num_envs=1)
    try:
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
        runner.load(str(root / "checkpoints/model_599_stand_push02.pt"), map_location="cpu")
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            action = runner.get_inference_policy(device="cpu")(obs)
            wrapped.step(action)
        assert action.shape == (1, 14)
        assert torch.isfinite(action).all()
    finally:
        env.close()
