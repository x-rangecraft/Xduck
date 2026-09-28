"""Export MLX PPO actors with their saved observation normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_actor_onnx(
    weights: dict,
    output: Path,
    *,
    activation: str,
    action_dim: int,
) -> float:
    """Export and compare ONNX directly with the MLX actor on nonzero inputs.

    Accepts flattened checkpoint weights, including frozen running statistics.
    Critic weights and critic normalization never enter the exported graph.
    """
    import mlx.core as mx
    import onnxruntime as ort
    import torch
    import torch.nn as nn

    from unilab.algos.mlx.common.activations import get_activation

    activations = {
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,
        "identity": nn.Identity,
    }
    if activation not in activations:
        raise ValueError(f"Unsupported ONNX actor activation: {activation}")
    indices = sorted(
        int(k.split(".")[2])
        for k in weights
        if k.startswith("actor.layers.") and k.endswith(".weight")
    )
    if not indices or indices != list(range(len(indices))):
        raise ValueError("Checkpoint must contain contiguous actor layers")
    obs_dim = weights["actor.layers.0.weight"].shape[1]
    norm_keys = ("obs_normalizer.mean", "obs_normalizer.std")
    has_norm = any(k.startswith("obs_normalizer.") for k in weights)
    if has_norm and not all(k in weights for k in norm_keys):
        raise ValueError("Actor normalization statistics are incomplete")
    mean = (
        np.array(weights[norm_keys[0]], dtype=np.float32)
        if has_norm
        else np.zeros((1, obs_dim), np.float32)
    )
    scale = (
        np.array(weights[norm_keys[1]], dtype=np.float32) + 1e-2
        if has_norm
        else np.ones((1, obs_dim), np.float32)
    )

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean.copy()))
            self.register_buffer("scale", torch.from_numpy(scale.copy()))
            self.layers = nn.ModuleList()
            self.activation = activations[activation]()
            for i in indices:
                w = np.array(weights[f"actor.layers.{i}.weight"], dtype=np.float32)
                b = np.array(weights[f"actor.layers.{i}.bias"], dtype=np.float32)
                layer = nn.Linear(w.shape[1], w.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(w.copy()))
                    layer.bias.copy_(torch.from_numpy(b.copy()))
                self.layers.append(layer)

        def forward(self, obs):
            value = (obs - self.mean) / self.scale
            for layer in self.layers[:-1]:
                value = self.activation(layer(value))
            return self.layers[-1](value)[..., :action_dim]

    actor = Actor().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            actor,
            (torch.zeros((1, obs_dim)),),
            str(output),
            input_names=["obs"],
            output_names=["action"],
            opset_version=17,
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        )
    rng = np.random.default_rng(1729)
    samples = mean + rng.normal(size=(32, obs_dim)).astype(np.float32) * scale
    value = (mx.array(samples) - mx.array(mean)) / mx.array(scale)
    act_fn = get_activation(activation)
    for i in indices:
        value = value @ mx.array(weights[f"actor.layers.{i}.weight"]).T + mx.array(
            weights[f"actor.layers.{i}.bias"]
        )
        if i != indices[-1]:
            value = act_fn(value)
    expected = np.asarray(value[:, :action_dim])
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        None,
        {"obs": samples},
    )[0]
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        raise RuntimeError("ONNX actor does not match MLX actor including normalization")
    return float(np.max(np.abs(actual - expected)))
