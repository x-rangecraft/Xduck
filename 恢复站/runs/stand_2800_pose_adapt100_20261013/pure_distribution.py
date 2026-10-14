"""Reward useful tanh increment entropy, not unbounded latent Gaussian entropy."""
import math
import torch
from rsl_rl.modules.distribution import GaussianDistribution
class PureGaussian(GaussianDistribution):
 def init_mlp_weights(self,mlp):
  torch.nn.init.orthogonal_(mlp[-1].weight,gain=.01);torch.nn.init.zeros_(mlp[-1].bias)
 @property
 def entropy(self):
  # H[tanh(X)] = H[X] + E log(1-tanh(X)^2). Antithetic reparameterized
  # Monte Carlo is differentiable in mean and std. No hard std bound or reset.
  eps=torch.randn_like(self.mean);x=self.mean[None,...]+self.std[None,...]*torch.stack([eps,-eps])
  log_jacobian=2*(math.log(2)-x-torch.nn.functional.softplus(-2*x))
  return super().entropy+log_jacobian.mean(0).sum(-1)
 # log_prob and KL remain those of the sampled raw Gaussian actions; PPO stores
 # those raw actions and uses the same deterministic tanh/integrator decoder.
