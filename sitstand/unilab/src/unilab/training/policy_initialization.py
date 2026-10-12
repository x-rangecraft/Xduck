"""Training-only initialization for an ordinary Gaussian MLP policy."""
import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class SmallMeanGaussianDistribution(GaussianDistribution):
    """Keep Gaussian behavior; initialize only the MLP's last linear layer."""

    def __init__(self, output_dim, init_std=0.03, std_type="log", init_mean_gain=0.01):
        if not math.isfinite(init_mean_gain) or init_mean_gain <= 0:
            raise ValueError("init_mean_gain must be finite and positive")
        super().__init__(output_dim, init_std, std_type)
        self.init_mean_gain = float(init_mean_gain)

    def init_mlp_weights(self, mlp):
        layer = mlp[-1]
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError("Small-mean initialization requires a final linear layer")
        torch.nn.init.orthogonal_(layer.weight, gain=self.init_mean_gain)
        torch.nn.init.zeros_(layer.bias)
