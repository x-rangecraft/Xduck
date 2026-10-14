"""Independent finite-horizon verification with recorded force and foot trajectories."""
from pathlib import Path
import argparse,sys,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor,COMMANDS
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--level',type=int,default=0);ap.add_argument('--mode',choices=['none','short','sustained'],default='none');ap.add_argument('--walk-seconds',type=float,default=0.);ap.add_argument('--seed',type=int,default=30101);ap.add_argument('--tag',required=True);a=ap.parse_args()
E.ensure_registries();np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.static_fractions=[0. if a.walk_seconds else 1.]*6;c.env.curriculum_max_level=0;c.env.stop_seconds=30.;c.env.max_episode_seconds=40.
if a.mode=='short':c.env.short_push_seconds=c.env.sustained_push_seconds=[.12,.2]
if a.mode=='sustained':c.env.short_push_seconds=c.env.sustained_push_seconds=[.8,1.5]
N=24;e=E.create_env(c,num_envs=N,env_cfg_override=E.build_ppo_env_cfg_override(c));actor=FrozenActor(R/a.checkpoint)
try:
 e._level=a.level;e.init_state();e.set_autoreset(False)
 if a.walk_seconds:e._walk_frames[:]=round(a.walk_seconds/.02);e._walk_commands[:]=np.repeat(COMMANDS,2,axis=0);e._phase_age[:]=0
 e._zero_push[:]=a.mode=='none';e._force_next[:]=25
 starts=np.full(N,-1,int);prebad=np.zeros(N,bool);falls=np.zeros(N,bool);held=np.zeros(N);settle=np.full(N,np.nan);max_tilt=np.zeros(N);trace=[];distance=np.zeros(N);previous=e._backend.get_base_pos().copy()
 for t in range(round((1+a.walk_seconds+8)/.02)):
  valid=e.stop_valid.copy();hit=valid&(starts<0);starts[hit]=t
  with torch.inference_mode():raw=actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  st=e.step(raw);q=e._backend.get_physics_state()[:,1:22].copy();tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));bad=(tilt>55)|(q[:,2]<.6*e._init_qpos[2]);prebad|=bad&~valid;age=t-starts;active=valid&(age<400);falls|=bad&active;max_tilt[active]=np.maximum(max_tilt[active],tilt[active]);vel=e.get_local_linvel();gyro=e.get_gyro();distance[active]+=np.linalg.norm((q[:,:2]-previous[:,:2])[active],axis=1);previous=q[:,:3].copy()
  stable=(np.linalg.norm(vel[:,:2],axis=1)<.03)&(np.linalg.norm(gyro,axis=1)<.2)&(np.sqrt(np.mean(e.get_dof_vel()**2,axis=1))<.15)&e._contact.all(1)&(tilt<10)&~falls
  held=np.where(stable&active,held+.02,0);new=(held>=1)&np.isnan(settle)&active;settle[new]=age[new]*.02
  footpos,_=e._backend.get_body_pose_w(e._sole_body_ids)
  trace.append(dict(gyro=gyro.copy(),dof_velocity=e.get_dof_vel().copy(),phase=st.info['balance_phase_weights'].copy(),reward=st.reward.copy(),action_raw=raw.copy(),action_target=e._current_action_target.copy(),qpos=q,contact=e._contact.copy(),foot_pos=footpos.copy(),force=e._last_force.copy(),tilt=tilt.copy(),velocity=vel.copy(),stop_active=active.copy()))
 rows=[dict(case=j,walk_failed=bool(prebad[j]),stop_failed=bool(falls[j]),settled_within_8s=bool(np.isfinite(settle[j])),settle_time_s=None if np.isnan(settle[j]) else float(settle[j]),max_tilt_deg=float(max_tilt[j]),stop_path_length_m=float(distance[j])) for j in range(N)];eligible=[x for x in rows if not x['walk_failed']]
 summary=dict(eligible=len(eligible),falls=sum(x['stop_failed'] for x in eligible),settled=sum(x['settled_within_8s'] for x in eligible))
 report=dict(checkpoint=a.checkpoint,level=a.level,mode=a.mode,seed=a.seed,walk_seconds=a.walk_seconds,summary=summary,rows=rows,qualified=False,fall_definition='tilt>55deg or root height<60% nominal',no_exact_paired_trajectory_claim=True,force_body_names=list(c.env.push_body_names),sole_installation_angle_deg=np.rad2deg(e._sole_angles).tolist())
 (R/(a.tag+'_results.json')).write_text(json.dumps(report,indent=2)+'\n');z={k:np.array([v[k] for v in trace]) for k in trace[0]};np.savez_compressed(R/(a.tag+'_trace.npz'),**z,sole_delta_quat=e._sole_delta_quat.copy(),sole_axes=np.array(e._sole_axes),sole_angles=e._sole_angles.copy());print(json.dumps(summary),flush=True)
finally:e.close()
