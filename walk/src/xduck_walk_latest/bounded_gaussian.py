"""Bounded Gaussian action distribution for safety-critical position targets."""

from __future__ import annotations

import copy

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal


class _BoundedMean(nn.Module):
    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor) -> None:
        super().__init__()
        if torch.any(action_low >= 0.0) or torch.any(action_high <= 0.0):
            raise ValueError("bounded action intervals must contain zero")
        self.register_buffer("negative_scale", -action_low)
        self.register_buffer("positive_scale", action_high)
        self.register_buffer("reference_scale", torch.maximum(-action_low, action_high))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.where(value >= 0.0, self.positive_scale, self.negative_scale)
        return scale * torch.tanh(value)


class BoundedGaussianDistribution(Distribution):
    """Tanh-squashed Gaussian with asymmetric action bounds and bounded base std.

    PPO's KL remains the exact KL between the two base Gaussians. The action
    log-probability includes the affine-tanh Jacobian. The entropy returned to
    PPO is the base-Gaussian entropy; ``max_std`` therefore provides the hard
    exploration budget even when an entropy bonus pushes upward.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float,
        min_std: float,
        max_std: float,
        action_low: list[float],
        action_high: list[float],
    ) -> None:
        super().__init__(output_dim)
        if not 0.0 < min_std < init_std < max_std:
            raise ValueError("expected 0 < min_std < init_std < max_std")
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.shape != (output_dim,) or high.shape != (output_dim,):
            raise ValueError(f"action bounds must each contain {output_dim} values")
        if torch.any(high <= low):
            raise ValueError("each action_high entry must exceed action_low")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("min_std", torch.tensor(float(min_std)))
        self.register_buffer("max_std", torch.tensor(float(max_std)))
        fraction = (init_std - min_std) / (max_std - min_std)
        raw = torch.logit(torch.tensor(fraction, dtype=torch.float32))
        self.raw_std_param = nn.Parameter(raw.repeat(output_dim))
        self._base_distribution: Normal | None = None
        self._mean_transform = _BoundedMean(low, high)

    @property
    def _action_std(self) -> torch.Tensor:
        return self.min_std + (self.max_std - self.min_std) * torch.sigmoid(self.raw_std_param)

    @property
    def _base_std(self) -> torch.Tensor:
        # Configuration and logging use local action-space radians. The base
        # Gaussian lives before the asymmetric affine-tanh transform.
        return self._action_std / self._mean_transform.reference_scale

    def _transform(self, value: torch.Tensor) -> torch.Tensor:
        return self._mean_transform(value)

    def update(self, mlp_output: torch.Tensor) -> None:
        self._base_distribution = Normal(mlp_output, self._base_std.expand_as(mlp_output))

    def sample(self) -> torch.Tensor:
        return self._transform(self._base_distribution.sample())  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self._transform(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return copy.deepcopy(self._mean_transform)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        linear_layers = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("bounded action distribution requires an MLP with a Linear output")
        output = linear_layers[-1]
        nn.init.uniform_(output.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(output.bias)

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._transform(self._base_distribution.mean)  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        return self._action_std.expand_as(self._base_distribution.mean)  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        # Local action-space entropy. The bounded std is the safety contract;
        # omitting the state-dependent tanh contraction avoids a biased Monte
        # Carlo entropy estimate while preserving the intended exploration push.
        return (
            self._base_distribution.entropy() + torch.log(self._mean_transform.reference_scale)
        ).sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (
            self._base_distribution.mean,
            self._base_std.expand_as(self._base_distribution.mean),
        )  # type: ignore[union-attr]

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        scale = torch.where(outputs >= 0.0, self.action_high, -self.action_low)
        normalized = torch.clamp(outputs / scale, -1.0 + 1e-6, 1.0 - 1e-6)
        base_value = torch.atanh(normalized)
        log_jacobian = torch.log(scale) + torch.log1p(-normalized.square() + 1e-6)
        return (self._base_distribution.log_prob(base_value) - log_jacobian).sum(dim=-1)  # type: ignore[union-attr]

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        ).sum(dim=-1)


class _UnitSlopeBoundedMean(_BoundedMean):
    """Keep unit slope around zero despite strongly asymmetric physical ranges."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.where(value >= 0.0, self.positive_scale, self.negative_scale)
        return scale * torch.tanh(value / scale)


class UnitSlopeBoundedGaussianDistribution(BoundedGaussianDistribution):
    """Smooth bounded policy outputs in action units; no actuator clipping change."""

    def __init__(self, output_dim, init_std, min_std, max_std, action_low, action_high):
        super().__init__(output_dim, init_std, min_std, max_std, action_low, action_high)
        self._mean_transform = _UnitSlopeBoundedMean(self.action_low, self.action_high)

    @property
    def _base_std(self):
        return self._action_std

    @property
    def entropy(self):
        return self._base_distribution.entropy().sum(dim=-1)

    def log_prob(self, outputs):
        scale = torch.where(outputs >= 0.0, self.action_high, -self.action_low)
        normalized = torch.clamp(outputs / scale, -1.0 + 1e-6, 1.0 - 1e-6)
        base_value = scale * torch.atanh(normalized)
        log_jacobian = torch.log1p(-normalized.square() + 1e-6)
        return (self._base_distribution.log_prob(base_value) - log_jacobian).sum(dim=-1)
