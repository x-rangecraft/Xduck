"""Official Gaussian action semantics with a motor-specific exploration budget."""

import torch
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from torch import nn
from torch.distributions import Normal


class BudgetedGaussianDistribution(GaussianDistribution):
    """Unsquashed Gaussian; the environment bounds executed position targets.

    Inherit upstream sampling, log-probability, entropy, KL, deterministic
    export and default MLP initialization. Only the learnable std range differs.
    Raw actions must remain unchanged in PPO storage and action observations.
    Bound buffers document the execution contract, not distribution support.
    This distinct state schema deliberately rejects old squashed checkpoints.
    """

    def __init__(self, output_dim, init_std, min_std, max_std, action_low, action_high):
        Distribution.__init__(self, output_dim)
        if not 0 < min_std < init_std < max_std:
            raise ValueError("expected 0 < min_std < init_std < max_std")
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if (
            low.shape != (output_dim,)
            or high.shape != (output_dim,)
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or torch.any(low >= high)
        ):
            raise ValueError("expected finite ordered action bounds matching output_dim")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("min_std", torch.tensor(float(min_std)))
        self.register_buffer("max_std", torch.tensor(float(max_std)))
        fraction = torch.tensor((init_std - min_std) / (max_std - min_std))
        self.gaussian_std_logit = nn.Parameter(torch.logit(fraction).repeat(output_dim))
        self._distribution = None

    def update(self, mlp_output):
        std = self.min_std + (self.max_std - self.min_std) * self.gaussian_std_logit.sigmoid()
        self._distribution = Normal(mlp_output, std.expand_as(mlp_output))
