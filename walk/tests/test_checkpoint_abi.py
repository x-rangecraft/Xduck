"""The selected 1975 walking checkpoint still matches the latest RSL model ABI."""

import json
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict


def _checkpoint() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "checkpoints/model_1975_strong_bounded.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("selected walking checkpoint")


def test_selected_walk_checkpoint_loads_strictly():
    checkpoint = torch.load(_checkpoint(), map_location="cpu", weights_only=False)
    recipe = json.loads((_checkpoint().parent / "run_config.json").read_text())["config"]["algo"]
    policy = recipe["policy"]
    obs = TensorDict({
        "actor": torch.zeros((1, 61)),
        "critic": torch.zeros((1, 76)),
    }, batch_size=[1])
    actor = MLPModel(
        obs, {"actor": ["actor"]}, "actor", 14,
        hidden_dims=policy["actor_hidden_dims"], activation=policy["activation"],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "xduck_walk_latest.bounded_gaussian.UnitSlopeBoundedGaussianDistribution",
            "init_std": policy["init_noise_std"], "min_std": policy["min_std"],
            "max_std": policy["max_std"], "action_low": policy["action_low"],
            "action_high": policy["action_high"],
        },
    )
    critic = MLPModel(
        obs, {"critic": ["critic"]}, "critic", 1,
        hidden_dims=policy["critic_hidden_dims"], activation=policy["activation"],
        obs_normalization=True,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    with torch.inference_mode():
        action = actor(obs)
        value = critic(obs)
    assert action.shape == (1, 14) and torch.isfinite(action).all()
    assert value.shape == (1, 1) and torch.isfinite(value).all()
    assert torch.all(action >= torch.tensor(policy["action_low"]))
    assert torch.all(action <= torch.tensor(policy["action_high"]))
