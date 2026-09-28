"""Activation helpers for MLX models."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx


def get_activation(name: str | None) -> Callable[[mx.array], mx.array]:
    """Resolve a string activation name to a callable."""
    if name is None:
        return lambda x: x

    name = name.lower()
    if name == "relu":
        return lambda x: mx.maximum(x, 0.0)
    if name == "elu":
        # Clamp the exponential branch before evaluating it.  Both a direct
        # ``where`` expression and MLX's current fused ELU can evaluate
        # ``exp(x)`` for large positive hidden activations and leak NaNs from
        # that inactive branch into the gradient.
        return lambda x: mx.maximum(x, 0.0) + mx.expm1(mx.minimum(x, 0.0))
    if name == "tanh":
        return lambda x: mx.tanh(x)
    if name == "sigmoid":
        return lambda x: mx.sigmoid(x)
    if name == "swish":
        return lambda x: x * mx.sigmoid(x)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name}")
