"""PPO with stage-bounded exploration; projection happens between rollout batches."""
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from .masked_stop_ppo import MaskedStopPPO
class SmallMeanGaussian(GaussianDistribution):
 def __init__(self,*args,initial_mean_gain=.1,**kwargs):
  self.initial_mean_gain=initial_mean_gain;super().__init__(*args,**kwargs)
 def init_mlp_weights(self,mlp):
  with torch.no_grad():mlp[-1].weight.mul_(self.initial_mean_gain);mlp[-1].bias.mul_(self.initial_mean_gain)
class BoundedStopPPO(MaskedStopPPO):
 def __init__(self,*args,exploration_std_min=.02,exploration_std_max=.18,**kwargs):
  super().__init__(*args,**kwargs)
  if not 0<exploration_std_min<exploration_std_max:raise ValueError('Invalid exploration bounds')
  self.exploration_std_min=exploration_std_min;self.exploration_std_max=exploration_std_max;self.project_std()
  self._std_hook=self.optimizer.register_step_post_hook(lambda optimizer,args,kwargs:self.project_std())
 def project_std(self):
  with torch.no_grad():self.actor.distribution.std_param.clamp_(self.exploration_std_min,self.exploration_std_max)
 def update(self):
  result=super().update();self.project_std();result['exploration_std_max']=float(self.actor.distribution.std_param.max());return result
