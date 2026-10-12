from pathlib import Path
import sys,json,numpy as np,torch,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/scripts'));sys.path.insert(0,str(R/'project/src'))
import train_rsl_rl as E
import xc_walking_v3_policy as D
FIELDS=['geom_pos','geom_quat','jnt_stiffness','dof_damping','body_mass','body_inertia','body_ipos','geom_friction','dof_armature']
def env(n,config='COLLECT_CONFIG.yaml',dr_seed=None):
 E.ensure_registries();c=OmegaConf.load(R/config);c.env.max_episode_seconds=30.
 if c.training.task_name=='XDuckStartupFlex':
  from dataclasses import fields
  from unilab.envs.locomotion.xduck_gf43x40.startup_flex import StartupFlexCfg
  valid={f.name for f in fields(StartupFlexCfg)}
  for k in list(c.env):
   if k not in valid:del c.env[k]
 e=E.create_env(c,num_envs=n,env_cfg_override=E.build_ppo_env_cfg_override(c))
 if dr_seed is not None:
  e._flex_rng=np.random.default_rng(dr_seed+1);e._sole_rng=np.random.default_rng(dr_seed+2)
 return e
def metadata():
 m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));ji=m.actuator_trnid[:,0];qi=m.jnt_qposadr[ji];vi=m.jnt_dofadr[ji]
 assert [m.joint(int(j)).name for j in ji]==D.TRAINING_JOINTS
 sites=[m.site(s+'_foot').id for s in ['left','right']];names=[m.body(int(m.site_bodyid[i])).name for i in sites];local=np.array([m.site_pos[i] for i in sites]);return m,qi,vi,names,local
def state(e,m):
 p=e._backend.get_physics_state();return p[:,1:1+m.nq].copy(),p[:,1+m.nq:1+m.nq+m.nv].copy()
def rotate(q,v):return v+2*np.cross(q[...,1:],np.cross(q[...,1:],v)+q[...,:1]*v)
def feet(e,q,bodyids,local):
 pos,quat=e._backend.get_body_pose_w(bodyids);world=pos+rotate(quat,np.broadcast_to(local,pos.shape));inverse=q[:,3:7].copy();inverse[:,1:]*=-1;return rotate(inverse[:,None,:],world-q[:,None,:3])
