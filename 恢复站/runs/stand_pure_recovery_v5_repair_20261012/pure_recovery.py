"""Pure on-policy XDuck rescue: no expert, reference path, or warmup actor."""
from dataclasses import dataclass,field
from collections import deque
import numpy as np
from unilab.base import registry
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.utils.rotation import np_quat_mul,np_quat_apply
from .stand_flex import StandFlexCfg,StandFlexEnv
from .rigid_gpu import SoleTiltReset
from .pure_reward import potential,reward_rates,stand_pose_mask

@registry.envcfg('XDuckPureRecovery')
@dataclass
class PureRecoveryCfg(StandFlexCfg):
 reset_bank: str=''
 contact_sensor_names: list[str]=field(default_factory=list)
 decoder: str='incremental'
 target_increment_rad: list[float]=field(default_factory=lambda:[.015,.010,.080,.060,.075,.110,.025,.025,.010,.015,.010,.080,.060,.075])
 discovery_gamma: float=.995
 discovery_terminal_cost: float=10.
 discovery_level: int=0
 discovery_phase: str='discovery'
 discovery_seed: int=41021
 episode_seconds: float=8.
 reset_mode: str='training'
 goal_height_m: float=.319637239
 head_reference_axis: list[float]=field(default_factory=lambda:[0.,0.,1.])
 push_body_names: list[str]=field(default_factory=lambda:['trunk_base','yaw_roll_motion'])
 force_ranges_N: list=field(default_factory=lambda:[[3.,18.],[5.,24.],[8.,30.],[10.,40.]])
 force_durations_s: list=field(default_factory=lambda:[[.08,.4],[.08,.6],[.1,1.],[.1,1.5]])

class PureReset(SoleTiltReset):
 def _get_base_actuator_gains(self,env):
  c=env.cfg.control_config;g=c.position_gains()
  return np.asarray(c.joint_kp if c.joint_kp is not None else np.full(14,g['kp'])),np.asarray(c.joint_kd if c.joint_kd is not None else np.full(14,g['kd']))
 def build_reset_plan(self,env,env_ids):
  plan=super().build_reset_plan(env,env_ids);ids=np.asarray(env_ids);n=len(ids)
  if env.cfg.reset_mode=='paired':
   bank=env._paired;i=ids%len(bank['qpos'])
  else:
   bank=env._banks[env._level];i=env._rng.integers(len(bank['qpos']),size=n)
  q=bank['qpos'][i].copy();v=bank['qvel'][i].copy()
  if env.cfg.reset_mode!='paired':
   yaw=plan.qpos[:,3:7].copy();q[:,:2]+=plan.qpos[:,:2];q[:,3:7]=np_quat_mul(yaw,q[:,3:7]);v[:,:3]=np_quat_apply(yaw,v[:,:3])
   # Free-joint angular qvel is body-local; yaw rotates translation only.
  plan.qpos[:]=q;plan.qvel[:]=v;target=q[:,env._policy_qpos_indices].copy()
  env._current_action_target[ids]=target;env._previous_action_target[ids]=target
  a=(target-env.default_angles)/.12;plan.info_updates['last_actions']=a.copy();plan.info_updates['current_actions']=a.copy()
  env._kind[ids]=bank['kind'][i];env._fixture_force[ids]=bank['fixture_force'][i];env._fixture_duration[ids]=bank['fixture_duration'][i]
  w,x,y,z=q[:,3:7].T;u=np.c_[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]
  env._prev_phi[ids]=potential(u,q[:,2]/env._init_qpos[2],v[:,3:6],env._height_goal_ratio);return plan

@registry.env('XDuckPureRecovery',sim_backend='mjwarp')
class PureRecoveryEnv(StandFlexEnv):
 def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
  self._rng=np.random.default_rng(cfg.discovery_seed);self._level=cfg.discovery_level;self._phase=cfg.discovery_phase
  self._age=np.zeros(num_envs,int);self._kind=np.zeros(num_envs,int);self._fixture_force=np.zeros((num_envs,2));self._fixture_duration=np.zeros(num_envs)
  self._prev_phi=np.zeros(num_envs);self._failure_paid=np.zeros(num_envs,bool);self._bad=np.zeros(num_envs,bool);self._held=np.zeros(num_envs)
  self._episode_recovered=np.zeros(num_envs,bool);self._ever_risk=np.zeros(num_envs,bool);self._scores=deque(maxlen=512)
  self._force_left=np.zeros(num_envs,int);self._force_next=np.zeros(num_envs,int);self._force_xy=np.zeros((num_envs,2));self._force_body=np.zeros(num_envs,int)
  self._last_force=np.zeros((num_envs,len(cfg.push_body_names),3),np.float32);self._force_events=0;self._raw_overflow=np.zeros((num_envs,14))
  with np.load(cfg.reset_bank) as b:
   self._banks=[{k:b[f'l{i}_{k}'].copy() for k in ['qpos','qvel','kind','fixture_force','fixture_duration']} for i in range(4)]
   self._paired={k:b[f'paired_{k}'].copy() for k in ['qpos','qvel','kind','fixture_force','fixture_duration']}
  super().__init__(cfg,num_envs,backend_type)
  self._height_goal_ratio=cfg.goal_height_m/self._init_qpos[2]
  self._head_pose_id=np.asarray([self._backend.get_body_id('yaw_roll_motion')],np.int32)
  self._head_axis=np.asarray(cfg.head_reference_axis)
  self._push_ids=np.array([self._backend.get_body_id(n) for n in cfg.push_body_names],np.int32)
  self._increments=np.asarray(cfg.target_increment_rad)
  assert self._increments.shape==(14,) and np.all(self._increments>0)
  assert not cfg.flexible_dr_enabled and cfg.physics_microsteps==1 and cfg.decoder in ['incremental','absolute']
 def _make_dr_provider(self):return PureReset()
 def _reset_episode_history(self,ids):
  super()._reset_episode_history(ids);self._age[ids]=0;self._bad[ids]=False;self._failure_paid[ids]=False;self._held[ids]=0
  self._episode_recovered[ids]=False;self._ever_risk[ids]=False;self._force_left[ids]=0;self._force_xy[ids]=0;self._force_next[ids]=25;self._last_force[ids]=0
 def apply_action(self,actions,state):
  LocomotionBaseEnv.apply_action(self,actions,state)
  raw=np.asarray(state.info['current_actions']);self._previous_action_target[:]=self._current_action_target
  target=self._current_action_target+self._increments*np.tanh(raw) if self.cfg.decoder=='incremental' else self.default_angles+.12*raw
  clipped=np.clip(target,self._target_low,self._target_high);self._raw_overflow[:]=target-clipped;self._current_action_target[:]=clipped
  state.info['current_actions'][:]=(clipped-self.default_angles)/.12
  self._force_left[:]=np.maximum(0,self._force_left-1);self._force_next-=1
  launch=(self._kind==2)&(self._force_next<=0)&(self._force_left==0)&(self._age*.02<self.cfg.episode_seconds-2-max(self.cfg.force_durations_s[self._level]))
  ids=np.flatnonzero(launch)
  if len(ids):
   if self.cfg.reset_mode=='paired':
    self._force_xy[ids]=self._fixture_force[ids];duration=self._fixture_duration[ids];self._force_body[ids]=0
   else:
    angle=self._rng.uniform(-np.pi,np.pi,len(ids));mag=self._rng.uniform(*self.cfg.force_ranges_N[self._level],len(ids));self._force_xy[ids]=np.c_[np.cos(angle),np.sin(angle)]*mag[:,None]
    duration=self._rng.uniform(*self.cfg.force_durations_s[self._level],len(ids));self._force_body[ids]=self._rng.integers(len(self._push_ids),size=len(ids))
   self._force_left[ids]=np.maximum(1,np.rint(duration/.02).astype(int));self._force_next[ids]=self._force_left[ids]+self._rng.integers(100,200,len(ids));self._force_events+=len(ids)
  force=np.zeros_like(self._last_force);ids=np.flatnonzero(self._force_left>0);force[ids,self._force_body[ids],:2]=self._force_xy[ids]
  self._backend.apply_body_force(self._push_ids,force);self._last_force[:]=force
  return clipped
 def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
  h=self._backend.get_base_pos()[:,2]/self._init_qpos[2];up=upvector[:,2]
  floor=np.stack([self._backend.get_sensor_data(n).reshape(self.num_envs,-1).max(1) for n in self.cfg.contact_sensor_names],axis=1).max(1)>0
  self._bad[:]=floor|(up<np.cos(np.deg2rad(70)))|(h<.45)
  _,head_quat=self._backend.get_body_pose_w(self._head_pose_id)
  head_axis_w=np_quat_apply(head_quat[:,0],np.broadcast_to(self._head_axis,(self.num_envs,3)));head_up=np.clip(head_axis_w[:,2],-1,1)
  rates=reward_rates(up,h,linvel,gyro,self.get_dof_vel(),(self._current_action_target-self._previous_action_target)/.12,self._phase,dof_pos-self.default_angles,head_up,self._height_goal_ratio,self._init_qpos[2])
  for name in ['alive','upright','upright_sharp','height_broad','height_sharp','head_pose','head_upright','leg_pose','quiet_bonus']:rates[name]=np.where(self._bad,0.,rates[name])
  rates['target_overflow']=-.05*np.sum((self._raw_overflow/.08)**2,axis=1)
  phi=potential(upvector,h,gyro,self._height_goal_ratio);next_phi=np.where(self._bad,0.,phi);progress=self.cfg.discovery_gamma*next_phi-self._prev_phi;self._prev_phi[:]=phi
  first=self._bad&~self._failure_paid;self._failure_paid|=first
  reward=.02*sum(rates.values())+progress-self.cfg.discovery_terminal_cost*first
  tilt=np.degrees(np.arccos(np.clip(up,-1,1)));speed=np.linalg.norm(linvel[:,:2],axis=1);omega=np.linalg.norm(gyro,axis=1)
  self._ever_risk|=(tilt>8)|(speed>.12)|(omega>.5)
  joint_error=dof_pos-self.default_angles
  self._stand_pose_mask=stand_pose_mask(h*self._init_qpos[2],self.cfg.goal_height_m,up,head_up,joint_error)
  state.info['stand_pose_valid']=self._stand_pose_mask.copy()
  stable=self._stand_pose_mask&(speed<.08)&(omega<.4)&~self._bad
  self._held[:]=np.where(stable,self._held+.02,0);self._episode_recovered|=(self._held>=.3)&self._ever_risk
  log=state.info.setdefault('log',{})
  for name,value in rates.items():log['PureReward/'+name]=float(value.mean())
  log.update({'pure/force_events':float(self._force_events),'pure/recovery_window_rate':float(np.mean(self._scores)) if self._scores else 0.,'pure/risky_fraction':float(self._ever_risk.mean()),'pure/target_overflow_fraction':float(np.mean(np.abs(self._raw_overflow)>1e-6)),'pure/terminal_failure_fraction':float(first.mean()),'pure/joint_speed_rms':float(np.sqrt(np.mean(self.get_dof_vel()**2)))})
  log.update({'pose/root_height_m':float((h*self._init_qpos[2]).mean()),'pose/height_goal_m':self.cfg.goal_height_m,'pose/head_tilt_deg':float(np.degrees(np.arccos(head_up)).mean()),'pose/head_error_deg':float(np.degrees(np.abs(joint_error[:,5:9])).mean()),'pose/hip_error_deg':float(np.degrees(np.abs(joint_error[:,[0,1,9,10]])).mean())})
  return reward.astype(np.float32)
 def _terminated(self,upvector):return self._bad.copy()
 def update_state(self,state):
  out=super().update_state(state);self._age+=1;timeout=(self._age>=round(self.cfg.episode_seconds/.02))&~self._bad
  end=self._bad|timeout
  if end.any():self._scores.extend(self._episode_recovered[end].tolist())
  return out.replace(truncated=timeout)
