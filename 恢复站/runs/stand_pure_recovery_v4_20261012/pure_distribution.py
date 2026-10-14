"""Fresh neutral policy mean with independent, full-sized Gaussian exploration."""
import torch
from rsl_rl.modules.distribution import GaussianDistribution
class PureGaussian(GaussianDistribution):
 def init_mlp_weights(self,mlp):
  torch.nn.init.orthogonal_(mlp[-1].weight,gain=.01)
  torch.nn.init.zeros_(mlp[-1].bias)
