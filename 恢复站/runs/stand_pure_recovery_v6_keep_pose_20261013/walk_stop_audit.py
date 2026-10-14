"""Evaluation-only absolute walk targets followed by the native incremental stand decoder."""
import numpy as np
from unilab.base import registry
from .pure_recovery import PureRecoveryCfg,PureRecoveryEnv

@registry.envcfg('XDuckWalkStopAudit')
class WalkStopAuditCfg(PureRecoveryCfg):
 pass

@registry.env('XDuckWalkStopAudit',sim_backend='mjwarp')
class WalkStopAuditEnv(PureRecoveryEnv):
 def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
  self.audit_walking=np.zeros(num_envs,bool)
  self.audit_walk_target=np.zeros((num_envs,14))
  super().__init__(cfg,num_envs,backend_type)

 def apply_action(self,actions,state):
  target=super().apply_action(actions,state)
  mask=self.audit_walking
  if mask.any():
   desired=np.clip(self.audit_walk_target[mask],self._target_low,self._target_high)
   target[mask]=desired;self._current_action_target[mask]=desired
   state.info['current_actions'][mask]=(desired-self.default_angles)/.12
   self._raw_overflow[mask]=0.
  return target
