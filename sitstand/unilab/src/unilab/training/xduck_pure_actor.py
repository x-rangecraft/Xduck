"""Fully trainable policy initialized to preserve its previous motor target."""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories


class PureActor(MLPModel):
    expected_obs_dim = 62
    residual_scale = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last = [layer for layer in self.mlp if isinstance(layer, torch.nn.Linear)][-1]
        if self.obs_dim != self.expected_obs_dim or last.out_features != 14:
            raise ValueError(f"Actor requires {self.expected_obs_dim}D observations / 14D means")
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.history = torch.nn.Linear(14, 14, bias=False)
        with torch.no_grad():
            self.history.weight.copy_(torch.eye(14))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        mean = self.residual_scale * self.mlp(self.obs_normalizer(raw)) + self.history(
            raw[:, 34:48]
        )
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self):
        raise NotImplementedError(
            "PureActor needs an export that preserves its trainable history branch"
        )

    def as_onnx(self, verbose=False):
        raise NotImplementedError(
            "PureActor needs an export that preserves its trainable history branch"
        )


class ContinuousTargetActor(PureActor):
    """Revision 4: trainable target-memory initialization, no reference motion.

    History contains the executed target offset, including limits, in radians.
    Both the identity-initialized memory and zero-initialized correction learn.
    """

    expected_obs_dim = 61

    def __init__(self, *args, residual_scale=0.05, **kwargs):
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        super().__init__(*args, **kwargs)
