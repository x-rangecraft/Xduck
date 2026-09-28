"""Unit tests for MLX PPO — MLPActorCritic, RolloutBuffer, PPOTrainer.

MLX is macOS-only (Apple Silicon Metal backend). These tests are skipped
automatically on Linux/Windows or when mlx is not installed.

Run:
    uv run pytest tests/algos/test_mlx_ppo.py -v
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _mlx_runtime_usable() -> bool:
    """Probe whether importing mlx.core is safe in a subprocess on this host."""
    if sys.platform != "darwin":
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0


if sys.platform != "darwin":
    pytest.skip("MLX is macOS-only", allow_module_level=True)

if not _mlx_runtime_usable():
    pytest.skip("mlx runtime aborts in subprocess on this host", allow_module_level=True)

mlx = pytest.importorskip("mlx.core", reason="mlx not installed")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from unilab.algos.mlx.common import RolloutBuffer
from unilab.algos.mlx.ppo import MLPActorCritic, PPOConfig, PPOTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 12
_ACT_DIM = 4
_NUM_ENVS = 8
_NUM_STEPS = 6
_HIDDEN = [32, 32]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**kwargs) -> MLPActorCritic:
    return MLPActorCritic(
        obs_dim=_OBS_DIM,
        action_dim=_ACT_DIM,
        actor_hidden_dims=_HIDDEN,
        critic_hidden_dims=_HIDDEN,
        **kwargs,
    )


def _make_buffer() -> RolloutBuffer:
    return RolloutBuffer(
        num_steps=_NUM_STEPS,
        num_envs=_NUM_ENVS,
        obs_dim=_OBS_DIM,
        action_dim=_ACT_DIM,
        gamma=0.99,
        lam=0.95,
    )


def _fill_buffer(buf: RolloutBuffer, model: MLPActorCritic) -> None:
    for _ in range(_NUM_STEPS):
        obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
        actions, log_probs, values, mean, std = model.act(obs)
        rewards = mx.random.normal((_NUM_ENVS,))
        dones = mx.zeros((_NUM_ENVS,))
        buf.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=rewards,
            dones=dones,
            values=values,
        )


# ---------------------------------------------------------------------------
# PPOConfig defaults
# ---------------------------------------------------------------------------


def test_ppo_config_defaults():
    cfg = PPOConfig()
    assert cfg.num_learning_epochs == 4
    assert cfg.num_mini_batches == 4
    assert 0.0 < cfg.clip_param < 1.0
    assert cfg.learning_rate > 0


def test_ppo_config_custom():
    cfg = PPOConfig(learning_rate=1e-3, clip_param=0.1, num_mini_batches=2)
    assert cfg.learning_rate == 1e-3
    assert cfg.clip_param == 0.1
    assert cfg.num_mini_batches == 2


def test_asymmetric_critic_storage_update_and_normalizer_checkpoint(tmp_path):
    mx.random.seed(51)
    model = _make_model(critic_obs_dim=17, obs_normalization=True)
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM)) + 5.0
    critic = mx.random.normal((_NUM_ENVS, 17)) - 10.0
    model.update_normalization(obs, critic)
    with pytest.raises(ValueError, match="explicit critic"):
        model.act(obs)
    buf = RolloutBuffer(_NUM_STEPS, _NUM_ENVS, _OBS_DIM, _ACT_DIM, 0.99, 0.95, critic_obs_dim=17)
    for _ in range(_NUM_STEPS):
        actions, log_probs, values, mean, std = model.act(obs, critic)
        buf.add(
            obs,
            actions,
            log_probs,
            mean,
            std,
            mx.ones((_NUM_ENVS,)),
            mx.zeros((_NUM_ENVS,)),
            values,
            critic_obs=critic,
        )
    buf.compute_returns_and_advantages(model.value(critic))
    batch = next(buf.mini_batch_generator(2, 1))
    assert batch["obs"].shape[-1] == _OBS_DIM
    assert batch["critic"].shape[-1] == 17
    np.testing.assert_allclose(
        np.asarray(batch["critic"][:, 0]).mean(), np.asarray(critic[:, 0]).mean(), atol=1.0
    )
    trainer = PPOTrainer(model, PPOConfig(num_learning_epochs=1, num_mini_batches=2))
    stats_before = np.asarray(model.obs_normalizer.mean).copy()
    metrics = trainer.update(buf)
    assert metrics["updates_applied"] == 2
    np.testing.assert_array_equal(np.asarray(model.obs_normalizer.mean), stats_before)
    assert not any("normalizer" in k for k, _ in tree_flatten(model.trainable_parameters()))
    checkpoint = tmp_path / "model.safetensors"
    model.save_weights(str(checkpoint))
    restored = _make_model(critic_obs_dim=17, obs_normalization=True)
    restored.load_weights(str(checkpoint), strict=True)
    np.testing.assert_array_equal(np.asarray(model.policy(obs)), np.asarray(restored.policy(obs)))
    np.testing.assert_array_equal(
        np.asarray(model.value(critic)), np.asarray(restored.value(critic))
    )
    assert float(restored.obs_normalizer.count.item()) == _NUM_ENVS
    buf.clear()
    assert buf.critic_observations == []


def test_microduck_mirror_loss_matches_torch_value_and_gradient():
    import torch

    from unilab.envs.locomotion.microduck.symmetry import MicroDuckSymmetryAugmentation

    mx.random.seed(52)
    model = MLPActorCritic(61, 14, [], [], critic_obs_dim=74)
    sym = MicroDuckSymmetryAugmentation(device="mlx")
    trainer = PPOTrainer(
        model,
        PPOConfig(symmetry_cfg={"use_mirror_loss": True, "mirror_loss_coeff": 0.5}),
        symmetry=sym,
    )
    obs = mx.random.normal((8, 61))
    np.testing.assert_array_equal(np.asarray(sym.mirror_obs(sym.mirror_obs(obs))), np.asarray(obs))
    action = mx.random.normal((8, 14))
    np.testing.assert_array_equal(
        np.asarray(sym.mirror_action(sym.mirror_action(action))), np.asarray(action)
    )
    # An independent Torch autograd calculation checks detach direction as well as value.
    torch_sym = MicroDuckSymmetryAugmentation(device="cpu")
    linear = torch.nn.Linear(61, 14)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(np.array(model.actor.layers[0].weight)))
        linear.bias.copy_(torch.from_numpy(np.array(model.actor.layers[0].bias)))
    x = torch.from_numpy(np.array(obs))
    target = torch_sym.mirror_action(linear(x)).detach()
    expected = ((linear(torch_sym.mirror_obs(x)) - target) ** 2).mean()
    expected.backward()
    loss, grads = nn.value_and_grad(model, trainer._mirror_loss)(model, obs)
    np.testing.assert_allclose(loss.item(), expected.item(), rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(grads["actor"]["layers"][0]["weight"]),
        linear.weight.grad.numpy(),
        rtol=2e-5,
        atol=1e-8,
    )
    with pytest.raises(ValueError, match="env-owned"):
        PPOTrainer(model, trainer.cfg)


def test_normalized_actor_export_matches_live_mlx_policy(tmp_path):
    import onnxruntime as ort

    from unilab.algos.mlx.ppo.export import export_actor_onnx

    model = _make_model(critic_obs_dim=17, obs_normalization=True)
    model.update_normalization(
        mx.random.normal((64, _OBS_DIM)) + 4, mx.random.normal((64, 17)) - 20
    )
    path = tmp_path / "policy.onnx"
    export_actor_onnx(
        dict(tree_flatten(model.parameters())), path, activation="tanh", action_dim=_ACT_DIM
    )
    sample = mx.random.normal((5, _OBS_DIM)) * 2 + 3
    actual = ort.InferenceSession(str(path)).run(None, {"obs": np.asarray(sample)})[0]
    np.testing.assert_allclose(actual, np.asarray(model.policy(sample)), atol=1e-5, rtol=1e-4)


def test_sequential_rollouts_preserve_gae_segment_boundaries():
    # Two two-step segments. The huge reward in segment 2 must not affect
    # segment 1's GAE; the first segment bootstraps from its own final value.
    buffer = RolloutBuffer(4, 1, 1, 1, gamma=1.0, lam=1.0)
    zero = mx.zeros((1,))
    for i, reward in enumerate([1.0, 2.0, 100.0, 200.0]):
        buffer.add(
            mx.zeros((1, 1)),
            mx.zeros((1, 1)),
            zero,
            mx.zeros((1, 1)),
            mx.ones((1, 1)),
            mx.array([reward]),
            zero,
            zero,
        )
        if i == 1:
            buffer.end_segment(mx.array([10.0]))
    buffer.compute_returns_and_advantages(mx.array([20.0]))
    np.testing.assert_allclose(np.asarray(buffer.returns).ravel(), [13.0, 12.0, 320.0, 220.0])


# ---------------------------------------------------------------------------
# MLPActorCritic — construction
# ---------------------------------------------------------------------------


def test_model_init_log_std():
    model = _make_model(noise_std_type="log")
    assert hasattr(model, "log_std")
    assert model.log_std.shape == (_ACT_DIM,)


def test_model_init_scalar_std():
    model = _make_model(noise_std_type="scalar")
    assert hasattr(model, "std")
    assert model.std.shape == (_ACT_DIM,)


def test_model_init_state_dependent_std():
    model = _make_model(state_dependent_std=True, noise_std_type="log")
    # state-dependent std: no top-level log_std attribute
    assert not hasattr(model, "log_std")


def test_model_init_unknown_std_type_raises():
    with pytest.raises(ValueError, match="Unknown noise_std_type"):
        _make_model(noise_std_type="invalid")


# ---------------------------------------------------------------------------
# MLPActorCritic — forward pass shapes
# ---------------------------------------------------------------------------


def test_model_policy_output_shape():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    out = model.policy(obs)
    mx.eval(out)
    assert out.shape == (_NUM_ENVS, _ACT_DIM)


def test_model_value_output_shape():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    val = model.value(obs)
    mx.eval(val)
    assert val.shape == (_NUM_ENVS,)


def test_model_act_output_shapes():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    mx.eval(actions, log_probs, values, mean, std)
    assert actions.shape == (_NUM_ENVS, _ACT_DIM)
    assert log_probs.shape == (_NUM_ENVS,)
    assert values.shape == (_NUM_ENVS,)
    assert mean.shape == (_NUM_ENVS, _ACT_DIM)
    assert std.shape == (_NUM_ENVS, _ACT_DIM)


def test_elu_model_has_finite_gradients_for_large_positive_activations():
    """ELU must not overflow its inactive exponential branch."""
    model = _make_model(activation="elu")
    obs = mx.full((_NUM_ENVS, _OBS_DIM), 1_000.0)

    def loss_fn(current_model, x):
        return mx.sum(current_model.policy(x))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, obs)
    mx.eval(loss, grads)
    assert np.isfinite(float(loss.item()))
    assert all(np.isfinite(np.asarray(gradient)).all() for _, gradient in tree_flatten(grads))


def test_model_clipped_log_std_within_bounds():
    model = _make_model(noise_std_type="log", min_log_std=-5.0, max_log_std=2.0)
    log_std = model.clipped_log_std()
    mx.eval(log_std)
    arr = np.array(log_std.tolist())
    assert (arr >= -5.0).all()
    assert (arr <= 2.0).all()


def test_model_with_obs_normalization():
    """Obs normalizer should not crash on forward pass."""
    model = _make_model(obs_normalization=True)
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    out = model.policy(obs)
    mx.eval(out)
    assert out.shape == (_NUM_ENVS, _ACT_DIM)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


def test_buffer_step_increments():
    buf = _make_buffer()
    model = _make_model()
    assert buf.step == 0
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    buf.add(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        action_mean=mean,
        action_std=std,
        rewards=mx.zeros((_NUM_ENVS,)),
        dones=mx.zeros((_NUM_ENVS,)),
        values=values,
    )
    assert buf.step == 1


def test_buffer_overflow_raises():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    with pytest.raises(OverflowError):
        buf.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=mx.zeros((_NUM_ENVS,)),
            dones=mx.zeros((_NUM_ENVS,)),
            values=values,
        )


def test_buffer_compute_returns_stacks_arrays():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)
    mx.eval(buf.advantages, buf.returns)
    # After compute, arrays are stacked
    assert buf.advantages.shape == (_NUM_STEPS, _NUM_ENVS)
    assert buf.returns.shape == (_NUM_STEPS, _NUM_ENVS)


def test_buffer_clear_resets_step():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    assert buf.step == _NUM_STEPS
    buf.clear()
    assert buf.step == 0
    assert buf.observations == []


def test_buffer_mini_batch_generator_shapes():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    num_mini_batches = 2
    batch_size = _NUM_STEPS * _NUM_ENVS
    mini_batch_size = batch_size // num_mini_batches

    batches = list(buf.mini_batch_generator(num_mini_batches=num_mini_batches, num_epochs=1))
    assert len(batches) == num_mini_batches
    b = batches[0]
    mx.eval(b["obs"])
    assert b["obs"].shape == (mini_batch_size, _OBS_DIM)
    assert b["actions"].shape == (mini_batch_size, _ACT_DIM)
    assert b["old_log_probs"].shape == (mini_batch_size,)
    assert b["returns"].shape == (mini_batch_size,)
    assert b["advantages"].shape == (mini_batch_size,)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


def test_ppo_trainer_update_returns_metrics():
    model = _make_model()
    cfg = PPOConfig(num_learning_epochs=1, num_mini_batches=2)
    trainer = PPOTrainer(model, cfg)

    buf = _make_buffer()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    metrics = trainer.update(buf, iteration=0)
    assert isinstance(metrics, dict)
    for key in ("surrogate", "value", "entropy", "approx_kl"):
        assert key in metrics, f"missing key: {key}"


def test_ppo_trainer_update_decreases_loss():
    """Loss should be finite after one update step."""
    model = _make_model()
    cfg = PPOConfig(num_learning_epochs=2, num_mini_batches=2)
    trainer = PPOTrainer(model, cfg)

    buf = _make_buffer()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    metrics = trainer.update(buf, iteration=0)
    assert np.isfinite(metrics["surrogate"])
    assert np.isfinite(metrics["value"])


def test_ppo_actor_input_column_adaptation_freezes_the_rest_of_actor():
    model = _make_model()
    trainer = PPOTrainer(
        model,
        PPOConfig(
            num_learning_epochs=1,
            num_mini_batches=2,
            actor_train_input_columns=[2],
        ),
    )
    before = {key: np.asarray(value).copy() for key, value in tree_flatten(model.parameters())}
    buf = _make_buffer()
    _fill_buffer(buf, model)
    buf.compute_returns_and_advantages(model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM))))
    trainer.update(buf, iteration=0)
    after = dict(tree_flatten(model.parameters()))

    first_before = before["actor.layers.0.weight"]
    first_after = np.asarray(after["actor.layers.0.weight"])
    np.testing.assert_array_equal(first_after[:, :2], first_before[:, :2])
    np.testing.assert_array_equal(first_after[:, 3:], first_before[:, 3:])
    assert np.any(first_after[:, 2] != first_before[:, 2])
    for key, value in before.items():
        if key.startswith("actor.") and key != "actor.layers.0.weight":
            np.testing.assert_array_equal(np.asarray(after[key]), value)
        if key in ("std", "log_std"):
            np.testing.assert_array_equal(np.asarray(after[key]), value)


# ---------------------------------------------------------------------------
# Full training iteration on real env (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mlx_ppo_one_iteration_real_env(default_go2_reward_config):
    """Run 1 full MLX PPO iteration (collect rollout + update) on a real env."""
    _mujoco = pytest.importorskip("mujoco")

    from unilab.base import registry
    from unilab.base.observations import flatten_obs_dict
    from unilab.base.registry import ensure_registries
    from unilab.structured_configs import PPOConfig as PPOStructuredConfig

    ensure_registries()

    env_name = "Go2JoystickFlat"
    num_envs = 4
    num_steps = 8

    cfg = PPOStructuredConfig()
    algo_cfg = cfg.algorithm

    env = registry.make(
        env_name,
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": default_go2_reward_config},
    )
    obs_dim = sum(env.obs_groups_spec.values())
    action_dim = env.action_space.shape[0]

    model = MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
    )
    ppo_cfg = PPOConfig(
        num_learning_epochs=1,
        num_mini_batches=2,
        clip_param=float(algo_cfg.clip_param),
        gamma=float(algo_cfg.gamma),
        lam=float(algo_cfg.lam),
    )
    trainer = PPOTrainer(model, ppo_cfg)

    # Init env and get first obs
    if env.state is None:
        env.init_state()
    reset_indices = np.arange(num_envs, dtype=np.int32)
    obs_dict, _ = env.reset(reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict))

    # Collect rollout
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
    )

    for _ in range(num_steps):
        actions, log_probs, values, mean, std = model.act(obs)
        mx.eval(actions, log_probs, values, mean, std)

        env_actions = np.asarray(actions)
        state = env.step(env_actions)
        raw_obs = flatten_obs_dict(state.obs)
        rewards = mx.array(state.reward)
        dones = mx.array((state.terminated | state.truncated).astype(np.float32))

        buffer.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=rewards,
            dones=dones,
            values=values,
        )
        obs = mx.array(raw_obs)

    last_values = model.value(obs)
    buffer.compute_returns_and_advantages(last_values)

    # PPO update
    metrics = trainer.update(buffer, iteration=0)
    assert isinstance(metrics, dict)
    assert np.isfinite(metrics["surrogate"]), (
        f"surrogate loss is not finite: {metrics['surrogate']}"
    )
    assert np.isfinite(metrics["value"]), f"value loss is not finite: {metrics['value']}"

    env.close()
