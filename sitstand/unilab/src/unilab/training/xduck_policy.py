"""Neural direct actor with generic cosine time features and learned state feedback.

The timing weights are fitted from demonstrations. Inference reads no reference
file, joint table, or sampled trajectory. PPO optimizes the state feedback while retaining the learned nominal timing.
"""

import torch
from rsl_rl.models import MLPModel


class TimingActor(MLPModel):
    def __init__(self, *args, timing_features=96, **kwargs):
        if kwargs.get("obs_normalization", False):
            raise ValueError("TimingActor requires raw command and clock inputs")
        super().__init__(*args, **kwargs)
        self.register_buffer("frequencies", torch.arange(timing_features, dtype=torch.float32))
        self.timing_head = torch.nn.Linear(2 * timing_features, 14, bias=False)
        torch.nn.init.zeros_(self.timing_head.weight)
        self.timing_head.requires_grad_(False)
        # Preserve the supervised prior on the first PPO rollout. The feedback
        # branch remains trainable and learns state-dependent corrections.
        output = [m for m in self.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)

    def time_features(self, raw):
        phase = raw[:, 61:62].clamp(0, 1)
        command = raw[:, 48:49].clamp(0, 1)
        basis = torch.cos(torch.pi * phase * self.frequencies[None, :])
        return torch.cat((basis * (1 - command), basis * command), dim=1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        raw = torch.cat([obs[key] for key in self.obs_groups], dim=-1)
        output = self.mlp(self.get_latent(obs, masks, hidden_state)) + self.timing_head(
            self.time_features(raw)
        )
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output


class ProprioceptiveFeedback(torch.nn.Module):
    """Learned early ankle correction using only the existing actor observations."""

    def __init__(self, residual, timing_head, frequencies):
        super().__init__()
        self.residual = residual
        self.register_buffer("prior_weights", timing_head.weight.detach().clone())
        self.register_buffer("prior_frequencies", frequencies.detach().clone())
        self.register_buffer("feature_mean", torch.zeros(2, 35))
        self.register_buffer("feature_scale", torch.ones(2, 35))
        self.estimator = torch.nn.Parameter(torch.zeros(2, 36, 2))
        projection = torch.zeros(2, 14)
        projection[0, 4] = 1
        projection[1, 13] = 1
        self.register_buffer("ankle_projection", projection)

    @staticmethod
    def features(raw):
        return torch.cat(
            (raw[:, 6:20] - raw[:, 34:48], raw[:, 20:34], raw[:, :6], raw[:, 61:62]), dim=1
        )

    def forward(self, raw):
        residual = self.residual(raw)
        command = (raw[:, 48] > 0.5).long()
        features = (self.features(raw) - self.feature_mean[command]) / self.feature_scale[command]
        features = torch.cat((features, torch.ones_like(raw[:, :1])), dim=1)
        estimate = torch.bmm(features.unsqueeze(1), self.estimator[command]).squeeze(1)
        # Bias is unobservable at the first instant. Ramp in over 0.2 s, and
        # suppress stale/zero previous-action history immediately after reset.
        gate = (raw[:, 61:62] * 4 / 0.2).clamp(0, 1)
        consistent = (raw[:, 6:20] - raw[:, 34:48]).abs().amax(dim=1, keepdim=True) < 0.2
        previous_phase = (raw[:, 61:62] - 0.005).clamp(0, 1)
        previous_phase = torch.where(raw[:, 61:62] >= 1, raw[:, 61:62], previous_phase)
        basis = torch.cos(torch.pi * previous_phase * self.prior_frequencies[None, :])
        command_float = raw[:, 48:49].clamp(0, 1)
        previous_nominal = torch.nn.functional.linear(
            torch.cat((basis * (1 - command_float), basis * command_float), dim=1),
            self.prior_weights,
        )
        # Previous action also contains the learned residual. Remove its current
        # estimate before reconstructing filter memory; otherwise a constant PPO
        # ankle residual is fed back with gain 1/(1-0.9) = 10.
        previous_correction = (raw[:, 34:48] - previous_nominal - residual)[:, [4, 13]].clamp(
            -0.06, 0.06
        )
        correction = (
            (0.9 * previous_correction - 0.1 * estimate.clamp(-0.06, 0.06)) * gate * consistent
        )
        return residual + correction @ self.ankle_projection


class CorrectiveTimingActor(TimingActor):
    """Timing prior plus trainable proprioceptive correction and residual MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = ProprioceptiveFeedback(self.mlp, self.timing_head, self.frequencies)
