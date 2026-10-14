"""Same-seed full initialization, then uninterrupted frozen Walk -> Stand."""
from pathlib import Path
import sys,argparse,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor,COMMANDS
ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='stand_parent.pt');ap.add_argument('--tag',default='baseline');ap.add_argument('--walk-seconds',type=float,default=1.2);ap.add_argument('--seed',type=int,default=23101);ap.add_argument('--base-tag',default='baseline');ap.add_argument('--slew',type=float,default=0.);ap.add_argument('--force-mode',choices=['none','short','sustained'],default='none');ap.add_argument('--force-weight',type=float,default=.06);a=ap.parse_args();E.ensure_registries();np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.stand_slew_rad=a.slew if a.slew is not None else (0. if a.candidate=='stand_parent.pt' else .05);c.env.force_weight_stages=[[0,0.,0.]] if a.force_mode=='none' else [[0,a.force_weight,a.force_weight]];c.env.force_duration_range=[.12,.2] if a.force_mode=='short' else [.8,1.2];c.env.static_episode_fraction=0.;c.env.stop_seconds=20.;c.env.max_episode_seconds=40.;N=24;e=E.create_env(c,num_envs=N,env_cfg_override=E.build_ppo_env_cfg_override(c));model=FrozenActor(R/a.candidate)
try:
 e.init_state();e.set_autoreset(False);b=e._backend;e._walk_commands[:]=np.repeat(COMMANDS,2,axis=0);times=np.full(N,round(a.walk_seconds/.02),int);e._walk_frames[:]=times;e._phase_age[:]=0
 fields=['geom_pos','geom_quat','jnt_stiffness','dof_damping','body_mass','body_inertia','body_ipos','geom_friction','dof_armature'];params={k:getattr(b._device_model,k).numpy().copy() for k in fields}
 params.update(kp=e._motor.kp.copy(),kd=e._motor.kd.copy(),torque_multiplier=e._motor.actuator.torque_scale_multiplier.copy(),friction_multiplier=e._motor.actuator.friction_scale_multiplier.copy(),response_multiplier=e._motor.actuator.response_time_multiplier.copy(),imu_rotation=e._imu_rotation.copy(),imu_delay=e._imu_delay_lag.copy())
 starts=np.full(N,-1,int);walk_bad=np.zeros(N,bool);stop_bad=np.zeros(N,bool);eligible=np.zeros(N,bool);held=np.zeros(N);first_settle=np.full(N,np.nan);max_tilt=np.zeros(N);first_jump=np.zeros(N);trace=[];incoming=np.zeros((N,8));stop_frames=300
 for t in range(50+int(times.max())+stop_frames):
  valid=e.stop_valid.copy();hit=valid&(starts<0)
  if hit.any():
   starts[hit]=t;lv=e.get_local_linvel();gy=e.get_gyro();p=b.get_physics_state()[:,1:1+len(e._init_qpos)];tilt=np.degrees(np.arccos(np.clip(1-2*(p[:,4]**2+p[:,5]**2),-1,1)));eligible[hit]=(~walk_bad[hit])&(tilt[hit]<25)&(p[hit,2]>.25)&((np.linalg.norm(lv[hit,:2],axis=1)>.03)|(np.abs(gy[hit,2])>.15));incoming[hit]=np.c_[lv[hit],gy[hit],e._contact[hit]]
  previous=e._current_action_target.copy()
  with torch.inference_mode():raw=model(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  st=e.step(raw);full=b.get_physics_state()[:,1:1+len(e._init_qpos)].copy();vel=e.get_local_linvel().copy();gy=e.get_gyro();tilt=np.degrees(np.arccos(np.clip(1-2*(full[:,4]**2+full[:,5]**2),-1,1)));bad=(tilt>55)|(full[:,2]<e._init_qpos[2]*.6)|st.terminated
  walk_bad|=bad&~valid;age=t-starts;active=valid&(age<stop_frames);stop_bad|=bad&active;max_tilt[active]=np.maximum(max_tilt[active],tilt[active]);first_jump[hit]=np.max(abs(e._current_action_target[hit]-previous[hit]),axis=1)
  settled=(np.linalg.norm(vel[:,:2],axis=1)<.03)&(np.linalg.norm(gy,axis=1)<.2)&(np.sqrt(np.mean(e.get_dof_vel()**2,axis=1))<.15)&e._contact.all(1)&(tilt<10)&~stop_bad
  held=np.where(settled&active,held+.02,0);just=(held>=1)&np.isnan(first_settle)&active;first_settle[just]=age[just]*.02
  trace.append(dict(qpos=full,motor=e.get_dof_pos().copy(),velocity=vel,target=e._current_action_target.copy(),tilt=tilt.copy(),stop_active=active.copy(),contact=e._contact.copy()))
  if (t+1)%100==0:print('HANDOFF_EVAL',t+1,'eligible',int(eligible.sum()),'stop_falls',int(stop_bad.sum()),flush=True)
 z={k:np.array([x[k] for x in trace]) for k in trace[0]};rows=[]
 for j in range(N):
  start=starts[j];end=min(len(trace),start+300);tail=z['motor'][max(start,end-50):end,j];pose=float(abs(tail.mean(0)-e.default_angles).max()) if len(tail) else None
  rows.append(dict(case=j,command=e._walk_commands[j].tolist(),walk_seconds=float(e._walk_frames[j]*.02),walk_failed=bool(walk_bad[j]),eligible=bool(eligible[j]),incoming=incoming[j].tolist(),stop_failed=bool(stop_bad[j]),max_stop_tilt_deg=float(max_tilt[j]),first_target_jump_rad=float(first_jump[j]),settled_within_6s=bool(np.isfinite(first_settle[j])),settle_time_s=None if np.isnan(first_settle[j]) else float(first_settle[j]),final_mean_pose_error_rad=pose))
 rr=[x for x in rows if x['eligible']];summary=dict(eligible=len(rr),falls=sum(x['stop_failed'] for x in rr),settled=sum(x['settled_within_6s'] for x in rr),wobble_under20=sum(x['max_stop_tilt_deg']<20 for x in rr),median_settle_s=float(np.median([x['settle_time_s'] for x in rr if x['settle_time_s'] is not None])) if any(x['settle_time_s'] is not None for x in rr) else None,mean_first_target_jump_rad=float(np.mean([x['first_target_jump_rad'] for x in rr])))
 pair_error=0.
 np.savez_compressed(R/(a.tag+'_diagnostic.npz'),**z,**params)
 if a.candidate!='stand_parent.pt' and (R/(a.base_tag+'_trace.npz')).exists():
  base=np.load(R/(a.base_tag+'_trace.npz'));br=json.loads((R/(a.base_tag+'_results.json')).read_text());print('ELIGIBILITY_MATCH',bool(np.array_equal(eligible,np.array([x['eligible'] for x in br['rows']]))),flush=True)
  for k,v in params.items():
   print('PARAM_DIFF',k,float(abs(v-base[k]).max()),flush=True);np.testing.assert_array_equal(v,base[k])
  for j in range(N):pair_error=max(pair_error,float(abs(z['qpos'][:starts[j],j]-base['qpos'][:starts[j],j]).max()))
  print('NUMERICAL_PREFIX_DIVERGENCE',pair_error,flush=True)
 report=dict(candidate=a.candidate,rows=rows,summary=summary,before_switch_max_physics_diff=pair_error,same_seed_full_initialization=True,exact_incoming_state_match=pair_error<1e-5,force_mode=a.force_mode,force_weight_fraction=a.force_weight,flexible_dofs=0,continuous_unreset_physics_and_motor=True,stand_slew_rad=c.env.stand_slew_rad,stop_duration_s=6.,seed=a.seed,walk_seconds=a.walk_seconds)
 (R/(a.tag+'_results.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(R/(a.tag+'_trace.npz'),**z,**params);print(json.dumps(summary),flush=True)
finally:e.close()
