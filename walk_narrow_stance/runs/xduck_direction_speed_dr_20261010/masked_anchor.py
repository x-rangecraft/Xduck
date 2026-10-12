"""Preserve accepted motion and clean directional knowledge with per-output masks."""
import numpy as np,torch
from .rsl_rl_ppo import FinalObservationAwarePPO
class DirectionAnchoredPPO(FinalObservationAwarePPO):
 def __init__(self,*args,anchor_dataset=None,anchor_weight=.03,anchor_batch_size=384,fixed_learning_rate=5e-5,**kwargs):
  super().__init__(*args,**kwargs)
  self.anchor_weight=anchor_weight;self.anchor_batch_size=anchor_batch_size;self.fixed_learning_rate=fixed_learning_rate
  if self.is_multi_gpu or self.enable_compile:raise ValueError('Single-GPU eager training only')
  with np.load(anchor_dataset) as z:
   self.anchor_obs=torch.tensor(z['obs'],device=self.device,dtype=torch.float32);self.anchor_mean=torch.tensor(z['mean'],device=self.device,dtype=torch.float32);self.anchor_mask=torch.tensor(z['mask'],device=self.device,dtype=torch.float32);self.anchor_phase=torch.tensor(z['phase'],device=self.device,dtype=torch.long)
  # Let PPO correct yaw on straight/translation anchors without fixing old yaw bias.
  straight_translation=(abs(self.anchor_obs[:,48])>.05)&(abs(self.anchor_obs[:,50])<.05)
  for row in [0,9]:self.anchor_mask[straight_translation,row]=0
  self.pools=[torch.nonzero(self.anchor_phase==i).flatten() for i in range(int(self.anchor_phase.max())+1)]
  assert all(len(x)>0 for x in self.pools)
  self.generator=torch.Generator(device=self.device);self.generator.manual_seed(1490091)
  self.optimizer.register_step_pre_hook(self._before_step)
 def train_mode(self):super().train_mode();self.actor.obs_normalizer.eval()
 @torch.enable_grad()
 def _before_step(self,optimizer,args,kwargs):
  losses=[]
  for i,pool in enumerate(self.pools):
   ids=pool[torch.randint(len(pool),(self.anchor_batch_size//len(self.pools),),device=self.device,generator=self.generator)]
   prediction=self.actor.mlp(self.actor.obs_normalizer(self.anchor_obs[ids]));mask=self.anchor_mask[ids]
   error=((prediction-self.anchor_mean[ids])/.35).square()*mask
   losses.append(error.sum()/mask.sum().clamp_min(1))
  weights=[2.,1.,.25,.25,1.5]
  loss=self.anchor_weight*sum(weights[i]*v for i,v in enumerate(losses))
  loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
  self.anchor_sum+=float(loss.detach());self.anchor_count+=1
 def update(self):
  self.actor.obs_normalizer.eval();self.learning_rate=self.fixed_learning_rate
  for g in self.optimizer.param_groups:g['lr']=self.fixed_learning_rate
  self.anchor_sum=0.;self.anchor_count=0;result=super().update();result['direction_anchor']=self.anchor_sum/max(self.anchor_count,1);return result
