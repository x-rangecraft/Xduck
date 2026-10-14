"""XDuck rigid recovery curriculum; all motor/transport physics inherited from GF owner."""
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np,mujoco
from unilab.base import registry
from .live_handoff import LiveHandoffCfg,LiveHandoffEnv,leg_pose_reward

@registry.envcfg('XDuckRigidRecovery')
@dataclass
class RigidRecoveryCfg(LiveHandoffCfg):
 force_weight_stages: list=field(default_factory=lambda:[[0,.02,.06],[50,.04,.12],[150,.06,.18]])
 force_duration_range: list=field(default_factory=lambda:[.12,1.2])
 force_interval_range: list=field(default_factory=lambda:[2.,4.])
 pose_weight_stages: list=field(default_factory=lambda:[[0,.1],[150,.25],[250,.5]])
 smooth_weight_stages: list=field(default_factory=lambda:[[0,.01],[150,.025],[250,.05]])
 static_episode_fraction: float=.35

@registry.env('XDuckRigidRecovery',sim_backend='mjwarp')
class RigidRecoveryEnv(LiveHandoffEnv):
 def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
  assert not cfg.flexible_dr_enabled and cfg.stand_slew_rad==0
  self._force_left=np.zeros(num_envs,int);self._force_next=np.zeros(num_envs,int);self._force_xy=np.zeros((num_envs,2));self._force_events=0;self._force_control_frames=0;self._last_up=np.ones(num_envs)
  m=mujoco.MjModel.from_binary_path(cfg.warp_compiled_model_file);assert (m.nq,m.nv,m.nu)==(21,20,14)
  self._weight_n=float(m.body_mass.sum()*9.81)
  super().__init__(cfg,num_envs,backend_type)
  self._push_ids=np.array([self._backend.get_body_id('trunk_base')],np.int32)
 def _reset_episode_history(self,ids):
  super()._reset_episode_history(ids)
  static=self._handoff_rng.random(len(ids))<self.cfg.static_episode_fraction;self._walk_frames[np.asarray(ids)[static]]=0
  self._force_left[ids]=0;self._force_xy[ids]=0;self._force_next[ids]=self._handoff_rng.integers(15,65,len(ids));self._last_up[ids]=1.
 def _stage(self,stages):
  update=self.step_counter//self.cfg.rollout_steps
  return next(x[1:] for x in reversed(stages) if update>=x[0])
 def apply_action(self,actions,state):
  target=super().apply_action(actions,state);valid=self.stop_valid
  state.info['current_actions'][valid]=(target[valid]-self.default_angles)/.12
  self._force_next[valid]-=1;self._force_left[valid]=np.maximum(0,self._force_left[valid]-1)
  hit=valid&(self._force_next<=0)&(self._force_left==0)
  if hit.any():
   n=int(hit.sum());lo,hi=self._stage(self.cfg.force_weight_stages);theta=self._handoff_rng.uniform(-np.pi,np.pi,n);mag=self._handoff_rng.uniform(lo,hi,n)*self._weight_n
   self._force_xy[hit]=np.c_[np.cos(theta),np.sin(theta)]*mag[:,None];self._force_left[hit]=self._handoff_rng.integers(round(self.cfg.force_duration_range[0]/.02),round(self.cfg.force_duration_range[1]/.02)+1,n);self._force_next[hit]=self._force_left[hit]+self._handoff_rng.integers(round(self.cfg.force_interval_range[0]/.02),round(self.cfg.force_interval_range[1]/.02)+1,n);self._force_events+=n
  force=np.zeros((self.num_envs,1,3),np.float32);active=valid&(self._force_left>0);force[active,0,:2]=self._force_xy[active]
  self._backend.apply_body_force(self._push_ids,force);self._force_control_frames+=int(active.sum())
  return target
 def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
  # No requirement for both feet on the ground during recovery; terminal target is a quiet stand.
  dt=self.cfg.ctrl_dt;up=np.clip(upvector[:,2],-1,1);height=self._backend.get_base_pos()[:,2];speed=np.linalg.norm(linvel[:,:2],axis=1);omega=np.linalg.norm(gyro,axis=1)
  stable=(np.clip((up-np.cos(np.deg2rad(25)))/(1-np.cos(np.deg2rad(25))),0,1)*np.exp(-(speed/.12)**2-(omega/.6)**2));stable*=self.stop_valid
  leg=[0,1,2,3,4,9,10,11,12,13];pose=leg_pose_reward((dof_pos-self.default_angles)[:,leg],np.array([.3,.15,.4,.4,.3]*2))
  pose_w=self._stage(self.cfg.pose_weight_stages)[0];smooth_w=self._stage(self.cfg.smooth_weight_stages)[0];raw=state.info['current_actions'];last=state.info['last_actions'];motion=np.mean((raw-last)**2,axis=1)
  height_score=np.exp(-((height/self._init_qpos[2]-1)/.2)**2);near=np.exp(-(speed/.05)**2-(omega/.25)**2)*self._contact.all(1)
  r=4.*up+1.*height_score+stable*(pose_w*pose+2.*near)-smooth_w*(.1+.9*stable)*motion
  # Dense upright-progress teaches corrections without taxing necessary movement.
  progress=2.*(up-self._last_up)*self.stop_valid;self._last_up[:]=up
  below=np.maximum(self._soft_joint_limits[:,0]-dof_pos,0);above=np.maximum(dof_pos-self._soft_joint_limits[:,1],0);r-=np.sum(below+above,axis=1)
  r-=np.mean((dof_pos[:,5:9]-self.default_angles[5:9])**2,axis=1)*.5
  bad=(up<np.cos(np.deg2rad(self.cfg.failure_tilt_deg)))|(height<self._init_qpos[2]*self.cfg.minimum_root_height_ratio);r-=bad*5.
  log=state.info.setdefault('log',{});log.update({'recovery/stability_gate':float(stable.mean()),'recovery/pose_weight':float(pose_w),'recovery/smooth_weight':float(smooth_w),'recovery/force_events':float(self._force_events),'recovery/force_peak_n':float(np.linalg.norm(self._force_xy,axis=1).max()),'recovery/force_active_fraction':float(np.mean(self._force_left>0)),'recovery/flexible_dofs':0.})
  return (r*dt+progress).astype(np.float32)
