"""PPO learning only from executed Stand actions in continuous mixed rollouts."""
import torch
from rsl_rl.storage import RolloutStorage
from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO
class MaskedStopPPO(FinalObservationAwarePPO):
 def __init__(self,*args,continuity_weight=.05,**kwargs):
  super().__init__(*args,**kwargs);self.continuity_weight=continuity_weight
 def compute_returns(self,obs):
  super().compute_returns(obs);s=self.storage;valid=s.observations['stop_valid'].squeeze(-1)>.5;raw=s.returns-s.values;chosen=raw[valid]
  if chosen.numel()<2:raise RuntimeError('No sufficient executed Stand transitions')
  s.advantages=(raw-chosen.mean())/(chosen.std()+1e-8);self.valid_fraction=float(valid.float().mean())
 def update(self):
  storage=self.storage;original=storage.mini_batch_generator
  def masked(batches,epochs):
   for b in original(batches,epochs):
    keep=b.observations['stop_valid'].squeeze(-1)>.5
    if not keep.any():continue
    yield RolloutStorage.Batch(observations=b.observations[keep],actions=b.actions[keep],values=b.values[keep],advantages=b.advantages[keep],returns=b.returns[keep],old_actions_log_prob=b.old_actions_log_prob[keep],old_distribution_params=tuple(x[keep] for x in b.old_distribution_params))
  storage.mini_batch_generator=masked
  try:
   out=super().update();out['executed_stand_sample_fraction']=self.valid_fraction
   mask=(storage.observations['stop_valid'].squeeze(-1)>.5)&(storage.observations['stop_age'].squeeze(-1)<=.12)
   if mask.any() and self.continuity_weight>0:
    x=storage.observations['actor'][mask];pred=self.actor.mlp(self.actor.obs_normalizer(x));loss=self.continuity_weight*(pred-x[:,34:48]).square().mean();self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm);self.optimizer.step();out['handoff_continuity']=float(loss.detach())
   return out
  finally:storage.mini_batch_generator=original
