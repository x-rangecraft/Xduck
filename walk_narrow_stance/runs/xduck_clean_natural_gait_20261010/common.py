from pathlib import Path
import sys,copy,pickle,importlib.util
import numpy as np,onnxruntime as ort
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'scripts'));import train_rsl_rl as E
spec=importlib.util.spec_from_file_location('consumer',R/'xc_walking_v3_policy.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
FIELDS=['geom_pos','geom_quat','jnt_stiffness','dof_damping','body_mass','body_inertia','body_ipos','geom_friction','dof_armature']
def actor(name="source.onnx"):
 opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
 return ort.InferenceSession(str(R/name),sess_options=opt,providers=['CPUExecutionProvider'])
def env(n,step=30720):
 E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.start_course_enabled=False;c.env.startup_training=False;c.env.startup_reward=False;c.env.toe_heading_reward=False;c.env.startup_initial_step=step;c.env.yaw_only_tracking=False;c.env.moving_action_rate_weight=None;c.env.domain_rand.velocity_pushes=False;c.env.max_episode_seconds=30.;c.env.audit_twist=[.3,0,0];c.env.curriculum.standing_stages=[{'step':0,'value':0.}]
 for k in ['resampling_time_range','head_resampling_time_range','body_resampling_time_range']:c.env.commands[k]=[1000,1000]
 return E.create_env(c,num_envs=n,env_cfg_override=E.build_ppo_env_cfg_override(c))
def motor_rows(x,rows):
 old=x.shape[0];new=len(rows);memo={}
 def f(v):
  if id(v) in memo:return memo[id(v)]
  if isinstance(v,np.ndarray):out=v[rows].copy() if v.ndim and v.shape[0]==old else v.copy();memo[id(v)]=out;return out
  if isinstance(v,(str,int,float,bool,type(None),np.generic)):return v
  if isinstance(v,tuple):return (new,14) if v==(old,14) else tuple(f(a) for a in v)
  if isinstance(v,list):out=[];memo[id(v)]=out;out.extend(f(a) for a in v);return out
  if isinstance(v,dict):out={};memo[id(v)]=out;out.update({k:f(a) for k,a in v.items()});return out
  if hasattr(v,'__dict__'):
   out=copy.copy(v);memo[id(v)]=out
   for k,a in vars(v).items():object.__setattr__(out,k,f(a))
   return out
  out=copy.deepcopy(v);memo[id(v)]=out;return out
 return f(x)
def frame_list(e,commands):
 q=e._motor.quantize_position_feedback(e.get_dof_pos())+e._encoder_bias;dq=e._motor.quantize_velocity_feedback(e.get_dof_vel());gy=e._rotate_imu(e.get_gyro());g=e._rotate_imu(e._projected_gravity());frames=[]
 for j in range(len(commands)):
  qp=np.zeros(15);qv=np.zeros(15);qp[D.TRAINING_INDEX]=q[j];qv[D.TRAINING_INDEX]=dq[j]
  frames.append({'positions':qp,'velocities':qv,'imu':{'gyro':gy[j],'gravity':g[j]},'command':{'twist':commands[j].copy(),'head':[0.]*4,'body':[0.]*3},'policy_context':{'action':'walk','body_active':False}})
 return frames
def snapshot(e,policies,tag):
 phi=e._backend.get_physics_state();p={k:getattr(e._backend._device_model,k).numpy().copy() for k in FIELDS};p.update(qpos=phi[:,1:22].copy(),qvel=phi[:,22:42].copy(),encoder_bias=e._encoder_bias.copy(),imu_rotation=e._imu_rotation.copy(),sole_angles=e._sole_angles.copy(),sole_delta_quat=e._sole_delta_quat.copy(),last_offset=np.array([x.last_physical_offset for x in policies]),previous_velocity=np.array([x.previous_velocity for x in policies]),target=e._current_action_target.copy(),previous_target=e._previous_action_target.copy(),step=np.array(e.step_counter))
 np.savez_compressed(R/(tag+'.npz'),**p)
 with (R/(tag+'_motor.pkl')).open('wb') as f:pickle.dump(e._motor,f)
def restore(e,policies,tag,ids):
 from unilab.dr.types import ResetRandomizationPayload
 z=np.load(R/(tag+'.npz'));n=len(ids);e._backend.set_state(np.arange(n,dtype=np.int32),z['qpos'][ids],z['qvel'][ids],ResetRandomizationPayload(**{k:z[k][ids] for k in FIELDS}))
 with (R/(tag+'_motor.pkl')).open('rb') as f:e._motor=motor_rows(pickle.load(f),np.asarray(ids))
 for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:getattr(e,'_'+k)[:]=z[k][ids]
 e._current_action_target[:]=z['target'][ids];e._previous_action_target[:]=z['previous_target'][ids]
 for p,j in zip(policies,ids):p.last_physical_offset=z['last_offset'][j].copy();p.previous_velocity=z['previous_velocity'][j].copy()
 return z
