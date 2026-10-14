"""Fresh XDuck balance actor: performance-gated perturbation curriculum on rigid mechanics."""
from dataclasses import dataclass,field
from collections import deque
import numpy as np
from unilab.base import registry
from unilab.utils.rotation import np_quat_apply,np_quat_mul
from .rigid_recovery import RigidRecoveryCfg,RigidRecoveryEnv
from .stand_flex import StandFlexReset
from .balance_reward import BalanceRewardCfg,score_balance

@registry.envcfg('XDuckScratchBalance')
@dataclass
class ScratchBalanceCfg(RigidRecoveryCfg):
 balance_reward: BalanceRewardCfg=field(default_factory=BalanceRewardCfg)
 no_push_episode_fraction: float=.3
 level_bank: str=''
 curriculum_min_updates: int=25
 curriculum_window_episodes: int=128
 curriculum_pass_fraction: float=.9
 curriculum_max_level: int=5
 static_fractions: list=field(default_factory=lambda:[1.,1.,.9,.8,.65,.5])
 short_push_seconds: list=field(default_factory=lambda:[.08,.25])
 sustained_push_seconds: list=field(default_factory=lambda:[.4,1.5])
 push_body_names: list=field(default_factory=lambda:['trunk_base','yaw_roll_motion'])

class ScratchReset(StandFlexReset):
 def build_reset_plan(self,env,ids):
  plan=super().build_reset_plan(env,ids)
  ids=np.asarray(ids);n=len(ids);level=env._level
  bank=env._level_banks[level];choose=env._handoff_rng.integers(len(bank['qpos']),size=n)
  q=bank['qpos'][choose].copy();v=bank['qvel'][choose].copy();actions=bank['actions'][choose].copy()
  # Always preserve some easy states and start the frozen walker from the new stand.
  easy=(env._handoff_rng.random(n)<.3)|(env._walk_frames[ids]>0)
  base=env._level_banks[0];q[easy]=base['qpos'][0];v[easy]=0;actions[easy]=0
  yaw=plan.qpos[:,3:7].copy();origin=plan.qpos[:,:2].copy();q[:,:2]+=origin;q[:,3:7]=np_quat_mul(yaw,q[:,3:7]);v[:,:3]=np_quat_apply(yaw,v[:,:3]);v[:,3:6]=np_quat_apply(yaw,v[:,3:6])
  plan.qpos[:]=q;plan.qvel[:]=v;plan.info_updates['last_actions']=actions.copy();plan.info_updates['current_actions']=actions.copy()
  target=np.clip(env.default_angles+actions*.12,env._target_low,env._target_high);env._current_action_target[ids]=target;env._previous_action_target[ids]=target
  env._last_up[ids]=1-2*(q[:,4]**2+q[:,5]**2)
  plan.info_updates['balance_curriculum_level']=np.full(n,level)
  return plan

@registry.env('XDuckScratchBalance',sim_backend='mjwarp')
class ScratchBalanceEnv(RigidRecoveryEnv):
 def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
  cfg.balance_reward.validate()
  self._failure_paid=np.zeros(num_envs,bool);self._level=0;self._level_started_update=0;self._held=np.zeros(num_envs);self._force_body=np.zeros(num_envs,int);self._zero_push=np.zeros(num_envs,bool);self._last_force=np.zeros((num_envs,len(cfg.push_body_names),3),np.float32);self._end_scores=deque(maxlen=cfg.curriculum_window_episodes);self._promotions=[];self._next_check_update=cfg.curriculum_min_updates
  with np.load(cfg.level_bank) as z:self._level_banks=[{key:z[f'l{i}_{key}'].copy() for key in ['qpos','qvel','actions']} for i in range(6)]
  super().__init__(cfg,num_envs,backend_type);self._push_ids=np.array([self._backend.get_body_id(name) for name in cfg.push_body_names],np.int32)
 def _make_dr_provider(self):return ScratchReset()
 def _stage(self,stages):return stages[min(self._level,len(stages)-1)][1:]
 def _reset_episode_history(self,ids):
  self.cfg.static_episode_fraction=self.cfg.static_fractions[self._level]
  super()._reset_episode_history(ids);ids=np.asarray(ids);static=self._walk_frames[ids]==0
  # Static initial states enter learner control immediately, without a teacher.
  self._phase_age[ids[static]]=round(self.cfg.prepare_seconds/.02)
  self._held[ids]=0;self._failure_paid[ids]=False;self._zero_push[ids]=self._handoff_rng.random(len(ids))<self.cfg.no_push_episode_fraction
 def apply_action(self,actions,state):
  # Use the continuous mixed-policy action owner, but own physical force sampling here.
  from .live_handoff import LiveHandoffEnv
  target=LiveHandoffEnv.apply_action(self,actions,state);valid=self.stop_valid
  state.info['current_actions'][valid]=(target[valid]-self.default_angles)/.12
  self._force_next[valid]-=1;self._force_left[valid]=np.maximum(0,self._force_left[valid]-1)
  launch_before=self.cfg.stop_seconds-2.-max(self.cfg.short_push_seconds[1],self.cfg.sustained_push_seconds[1])
  hit=valid&(self.stop_age<launch_before)&(self._force_next<=0)&(self._force_left==0)&~self._zero_push
  if hit.any():
   n=int(hit.sum());lo,hi=self._stage(self.cfg.force_weight_stages);theta=self._handoff_rng.uniform(-np.pi,np.pi,n);mag=self._handoff_rng.uniform(lo,hi,n)*self._weight_n
   self._force_xy[hit]=np.c_[np.cos(theta),np.sin(theta)]*mag[:,None];self._force_body[hit]=self._handoff_rng.integers(len(self._push_ids),size=n)
   short=self._handoff_rng.random(n)<.5;duration=np.where(short,self._handoff_rng.uniform(*self.cfg.short_push_seconds,n),self._handoff_rng.uniform(*self.cfg.sustained_push_seconds,n))
   self._force_left[hit]=np.maximum(1,np.rint(duration/.02).astype(int));self._force_next[hit]=self._force_left[hit]+self._handoff_rng.integers(100,200,n);self._force_events+=int(np.count_nonzero(mag))
  force=np.zeros((self.num_envs,len(self._push_ids),3),np.float32);active=np.flatnonzero(valid&(self._force_left>0));force[active,self._force_body[active],:2]=self._force_xy[active]
  self._last_force[:]=force;state.info['applied_balance_force_w']=force.copy();self._backend.apply_body_force(self._push_ids,force);self._force_control_frames+=int(np.count_nonzero(np.linalg.norm(force,axis=2)>0));return target
 def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
  cfg=self.cfg.balance_reward
  height_ratio=self._backend.get_base_pos()[:,2]/self._init_qpos[2]
  joint_error=dof_pos-self.default_angles
  action_delta=(self._current_action_target-self._previous_action_target)/.12
  limit_error=np.sum(np.maximum(self._soft_joint_limits[:,0]-dof_pos,0)+np.maximum(dof_pos-self._soft_joint_limits[:,1],0),axis=1)
  terms,phase=score_balance(cfg,upvector,gyro,linvel,height_ratio,self._contact,joint_error,self.get_dof_vel(),action_delta,limit_error)
  up=np.clip(upvector[:,2],-1,1)
  bad=(up<np.cos(np.deg2rad(self.cfg.failure_tilt_deg)))|(height_ratio<self.cfg.minimum_root_height_ratio)
  found=np.concatenate([self._backend.get_sensor_data(name) for name in self.cfg.sensor.self_contacts],axis=1)
  terms['self_collisions']=-cfg.self_collision_cost_weight*np.sum(found>0,axis=1)
  terms['failure']=-cfg.failure_cost_weight*bad*self.stop_valid
  ended=self.stop_valid&(bad|(self.stop_age>=self.cfg.stop_seconds-self.cfg.ctrl_dt))
  next_potential=np.where(ended,0.,up)
  progress=cfg.upright_progress_weight*(cfg.progress_discount*next_potential-self._last_up)*self.stop_valid
  self._last_up[:]=up
  first_failure=bad&self.stop_valid&~self._failure_paid
  self._failure_paid|=first_failure
  terminal_cost=cfg.terminal_failure_cost*first_failure
  reward=sum(terms.values())*self.cfg.ctrl_dt+progress-terminal_cost
  state.info['balance_phase_weights']=np.stack([phase['quiet'],phase['braking'],phase['recovery']],axis=1)
  log=state.info.setdefault('log',{})
  for name,value in terms.items():log['Episode_Reward/'+name]=float(np.mean(value))
  for name in ['quiet','braking','recovery']:log['balance/phase_'+name]=float(np.mean(phase[name]))
  log['audit/progress_reward_per_step']=float(progress.mean())
  log['audit/terminal_failure_cost']=float(terminal_cost.mean())
  log['balance/predicted_tilt_deg']=float(np.mean(phase['predicted_tilt_deg']))
  log['balance/linear_speed_m_s']=float(np.linalg.norm(linvel[:,:2],axis=1).mean())
  log['balance/joint_speed_rms_rad_s']=float(np.sqrt(np.mean(self.get_dof_vel()**2)))
  log['recovery/force_events']=float(self._force_events)
  log['recovery/force_active_fraction']=float(np.mean(np.linalg.norm(self._last_force,axis=2).sum(1)>0))
  log['recovery/flexible_dofs']=0.
  return reward.astype(np.float32)

 def update_state(self,state):
  before=self.stop_valid.copy();out=super().update_state(state)
  q=self._backend.get_base_quat();up=1-2*(q[:,1]**2+q[:,2]**2);height=self._backend.get_base_pos()[:,2];speed=np.linalg.norm(self.get_local_linvel()[:,:2],axis=1);omega=np.linalg.norm(self.get_gyro(),axis=1)
  stable=before&(up>np.cos(np.deg2rad(12)))&(height>.8*self._init_qpos[2])&(speed<.08)&(omega<.4)&self._contact.all(1)
  self._held=np.where(stable,self._held+.02,0.)
  end=before&out.terminated
  if end.any():self._end_scores.extend((self._held[end]>=1.).tolist())
  update=self.step_counter//self.cfg.rollout_steps
  if update>=self._next_check_update:
   score=float(np.mean(self._end_scores)) if self._end_scores else 0.;self._next_check_update=update+25
   if self._level<self.cfg.curriculum_max_level and len(self._end_scores)>=self.cfg.curriculum_window_episodes and score>=self.cfg.curriculum_pass_fraction and update-self._level_started_update>=self.cfg.curriculum_min_updates:
    self._level+=1;self._level_started_update=update;self._promotions.append(dict(update=int(update),level=int(self._level),preceding_success_fraction=score));self._end_scores.clear()
  log=out.info.setdefault('log',{});log['balance/curriculum_level']=float(self._level);log['balance/window_success_fraction']=float(np.mean(self._end_scores)) if self._end_scores else 0.;log['balance/window_episodes']=float(len(self._end_scores));return out
