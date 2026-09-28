"""Physical-unit training regularizer; deployment actor remains unchanged."""
import sys

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("MLX requires macOS", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten

from unilab.algos.mlx.common import RolloutBuffer
from unilab.algos.mlx.ppo import MLPActorCritic, PPOConfig, PPOTrainer


def model():
    return MLPActorCritic(3, 2, [], [], activation="elu")


def trainer(net, scale=1.0, weight=100.0):
    return PPOTrainer(net, PPOConfig(policy_consistency={
        "weight": weight, "obs_noise_half_width": [.03, 0., 0.],
        "action_scale": [scale, scale],
    }))


def test_units_mask_and_actor_gradient():
    net = model()
    net.actor.layers[0].weight = mx.array([[2., 0., 0.], [0., 3., 0.]])
    net.actor.layers[0].bias = mx.zeros((2,))
    batch = {"obs": mx.zeros((4, 3)),
             "consistency_delta": mx.array([[.03, 0., 0.]] * 4)}
    learner = trainer(net)
    loss, gradients = nn.value_and_grad(net, learner._consistency_loss)(net, batch)
    assert float(loss.item()) == pytest.approx(.06**2 / 2, rel=1e-5)
    assert float(trainer(net, scale=.12)._consistency_loss(net, batch).item()) == pytest.approx(
        float(loss.item()) * .12**2, rel=1e-5)
    assert any(np.any(np.array(v)) for _, v in tree_flatten(gradients["actor"]))
    assert all(not np.any(np.array(v)) for _, v in tree_flatten(gradients["critic"]))
    unchanged = np.array(net.policy(batch["obs"]))
    learner._consistency_loss(net, batch)
    np.testing.assert_array_equal(unchanged, np.array(net.policy(batch["obs"])))
    batch["consistency_delta"] = mx.zeros((4, 3))
    assert float(learner._consistency_loss(net, batch).item()) == 0


def test_disabled_parity_and_finite_update():
    net = model()
    buf = RolloutBuffer(4, 2, 3, 2, gamma=.99, lam=.95)
    for _ in range(4):
        obs = mx.random.normal((2, 3))
        act, logp, val, mean, std = net.act(obs)
        buf.add(obs=obs, actions=act, log_probs=logp, action_mean=mean,
                action_std=std, rewards=mx.ones((2,)), dones=mx.zeros((2,)), values=val)
    buf.compute_returns_and_advantages(mx.zeros((2,)))
    batch = next(buf.mini_batch_generator(1, 1))
    base = PPOTrainer(net, PPOConfig())
    disabled = trainer(net, weight=0.)
    np.testing.assert_array_equal(np.array(base._loss_fn(net, batch)),
                                  np.array(disabled._loss_fn(net, batch)))
    mx.random.seed(71)
    disabled._loss_fn(net, batch)
    observed = np.array(mx.random.normal((4,)))
    mx.random.seed(71)
    np.testing.assert_array_equal(observed, np.array(mx.random.normal((4,))))
    enabled = trainer(net)
    enabled.cfg.num_mini_batches = 1
    enabled.cfg.num_learning_epochs = 1
    enabled.cfg.finite_check_interval = 1
    metrics = enabled.update(buf)
    assert metrics["updates_applied"] == 1
    assert metrics["skipped_nonfinite_loss"] == metrics["skipped_nonfinite_grads"] == 0
    assert metrics["policy_consistency_rad2"] >= 0
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("settings", [
    {"weight": -1}, {"weight": float("nan")}, {"wieght": 1},
    {"weight": 1, "obs_noise_half_width": [0, 0, 0], "action_scale": [1, 1]},
    {"weight": 1, "obs_noise_half_width": [.1, 0], "action_scale": [1, 1]},
    {"weight": 1, "obs_noise_half_width": [.1, 0, 0], "action_scale": [0, 1]},
])
def test_invalid_config_fails(settings):
    with pytest.raises(ValueError):
        PPOTrainer(model(), PPOConfig(policy_consistency=settings))
