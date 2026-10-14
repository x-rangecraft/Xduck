"""Paused-job diagnosis using native actor inference; no training or gradient updates."""
from pathlib import Path
import sys,json,copy
import numpy as np,torch
from tensordict import TensorDict
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from rsl_rl.models import MLPModel
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
E.ensure_registries();torch.set_num_threads(2)
results=[];parity=0.;N=8
modes=[('reference_target','block_02/final.pt',None),('original650','stand_parent.pt',0.),('initial_random_mean','block_00_start_actor.pt',0.),('trained300_mean','block_02/final.pt',0.),('trained300_sample','block_02/final.pt',1.),('trained300_sample_01','block_02/final.pt',.1)]
for name,checkpoint,noise in modes:
 np.random.seed(31301);torch.manual_seed(31301);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.curriculum_max_level=0;c.env.static_fractions=[1.]*6;c.env.stop_seconds=30.;c.env.max_episode_seconds=40.
 e=E.create_env(c,num_envs=N,env_cfg_override=E.build_ppo_env_cfg_override(c))
 try:
  e.init_state();e.set_autoreset(False);e._zero_push[:]=True
  cfg=E.normalize_ppo_train_cfg(OmegaConf.to_container(c.algo,resolve=True));ac=copy.deepcopy(cfg['actor']);ac.pop('class_name');td=TensorDict({'actor':torch.zeros((N,61)),'critic':torch.zeros((N,76))},batch_size=[N]);native=MLPModel(td,cfg['obs_groups'],'actor',14,**ac);state=torch.load(R/checkpoint,weights_only=True,map_location='cpu')['actor_state_dict'];native.load_state_dict(state,strict=True);native.eval();frozen=FrozenActor(R/checkpoint)
  std=state['distribution.std_param'];held=np.zeros(N);best=np.zeros(N);fall=np.zeros(N,bool);path=np.zeros(N);previous=e._backend.get_base_pos().copy();trace=[]
  for t in range(400):
   obs=torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)
   with torch.inference_mode():mean=native(TensorDict({'actor':obs},batch_size=[N]));manual=frozen(obs)
   parity=max(parity,float((mean-manual).abs().max()))
   if noise is None:raw=np.zeros((N,14),np.float32)
   else:raw=(mean+torch.randn_like(mean)*std*noise).numpy() if noise else mean.numpy()
   st=e.step(raw);q=e._backend.get_physics_state()[:,1:22];tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));vel=e.get_local_linvel();gyro=e.get_gyro();jv=e.get_dof_vel();jr=np.sqrt(np.mean(jv*jv,axis=1));speed=np.linalg.norm(vel[:,:2],axis=1);omega=np.linalg.norm(gyro,axis=1);contact=e._contact.copy()
   fall|=(tilt>55)|(q[:,2]<.6*e._init_qpos[2]);conditions=np.c_[speed<.03,omega<.2,jr<.15,contact.all(1),tilt<10];stable=conditions.all(1)&~fall;held=np.where(stable,held+.02,0);best=np.maximum(best,held);path+=np.linalg.norm(q[:,:2]-previous[:,:2],axis=1);previous=q[:,:3].copy()
   logs=st.info['log'];trace.append(dict(speed=speed.copy(),gyro_norm=omega.copy(),joint_speed=jr.copy(),tilt=tilt.copy(),phase=st.info['balance_phase_weights'].copy(),conditions=conditions.copy(),reward=st.reward.copy(),zero_angular=logs.get('Episode_Reward/zero_angular_velocity'),action_rate=logs.get('Episode_Reward/action_rate'),joint_cost=logs.get('Episode_Reward/joint_velocity')))
  z={k:np.array([x[k] for x in trace]) for k in trace[0]};tail=slice(100,None)
  result=dict(mode=name,checkpoint=checkpoint,mean_distribution_std=float(std.mean()),sampling_multiplier=noise,falls=int(fall.sum()),settled=int((best>=1).sum()),cases=N,mean_path_m=float(path.mean()),mean_speed_m_s=float(z['speed'][tail].mean()),mean_gyro_rad_s=float(z['gyro_norm'][tail].mean()),mean_joint_speed_rad_s=float(z['joint_speed'][tail].mean()),mean_tilt_deg=float(z['tilt'][tail].mean()),mean_phase_weights=z['phase'][tail].mean((0,1)).tolist(),condition_pass_fraction=z['conditions'][tail].mean((0,1)).tolist(),best_continuous_stable_s=best.tolist(),mean_reward=float(z['reward'].mean()))
  results.append(result);np.savez_compressed(R/(name+'_diagnostic.npz'),**z);print(json.dumps(result),flush=True)
 finally:e.close()
assert parity<1e-5
(R/'ROOT_CAUSE_DIAGNOSIS.json').write_text(json.dumps(dict(ppo_updates=0,training_paused=True,native_vs_manual_max_error=parity,condition_order=['horizontal speed<.03','gyro norm<.2','joint speed RMS<.15','both feet contact','tilt<10deg'],independent_same_seed_replays=True,results=results),indent=2)+'\n');print('PARITY',parity,flush=True)
