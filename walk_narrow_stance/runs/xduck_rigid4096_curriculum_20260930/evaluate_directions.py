from pathlib import Path
import sys,json,argparse
import numpy as np,torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'scripts'));import train_rsl_rl as E
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--seconds',type=float,default=12);p.add_argument('--pushes',action='store_true');a=p.parse_args()
commands=[('forward010',(.10,0,0)),('forward0125',(.125,0,0)),('forward015',(.15,0,0)),('forward0175',(.175,0,0)),('forward020',(.20,0,0)),('forward0225',(.225,0,0)),('forward030',(.30,0,0)),('backward',(-.20,0,0)),('left',(0,.20,0)),('right',(0,-.20,0)),('forward_left',(.15,.15,0)),('forward_right',(.15,-.15,0)),('backward_left',(-.15,.15,0)),('backward_right',(-.15,-.15,0)),('turn_left',(0,0,.5)),('turn_right',(0,0,-.5)),('walk_turn_left',(.20,0,.4)),('walk_turn_right',(.20,0,-.4))]
N=4*len(commands);E.ensure_registries();E.patch_rsl_rl_resume_state();np.random.seed(50021);torch.manual_seed(50021);torch.set_num_threads(2)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);step=int(saved['infos']['unilab_progress']['env_steps_per_env'])
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
for key,stages in list(c.env.curriculum.items()):
 if key.endswith('_stages') and OmegaConf.is_list(stages):
  selected=OmegaConf.to_container([s for s in stages if int(s.step)<=step][-1]);selected['step']=0;c.env.curriculum[key]=[selected]
c.env.curriculum.standing_stages=[{'step':0,'value':0.}];c.env.curriculum.step_cap=step;c.env.audit_twist=commands[0][1];c.env.audit_velocity=None;c.env.max_episode_seconds=max(30.,a.seconds+5.)
c.env.commands.resampling_time_range=[1000,1000];c.env.commands.head_resampling_time_range=[1000,1000];c.env.commands.body_resampling_time_range=[1000,1000];c.env.domain_rand.velocity_pushes=a.pushes
OmegaConf.save(c,R/(a.tag+'_config.yaml'));e=E.create_env(c,num_envs=N,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();e.step_counter=step;e._update_curriculum()
 algo=E._algo_config_dict(c);wrapped=E._resolve_ppo_wrapper_cls(algo)(e,device='cpu');train=E.normalize_ppo_train_cfg(algo);train['logger']='none';runner=OnPolicyRunner(wrapped,train,log_dir=None,device='cpu');runner.load(str(a.checkpoint),map_location='cpu');actor=runner.get_inference_policy(device='cpu');e.set_autoreset(False)
 # Reuse the same four mass/friction draws in every direction; geometry and COM reset seeds also match.
 for v in [e._mass_inertia_scale,e._foot_friction]:v[:]=np.tile(v[:4],len(commands))
 for k,(name,command) in enumerate(commands):
  e._cfg.audit_twist=command;np.random.seed(50123);e._sole_rng=np.random.default_rng(20260930);e.reset(np.arange(k*4,k*4+4,dtype=np.int32))
 e._next_push_step[:]=step+np.ceil(np.random.default_rng(529).uniform(*e._cfg.domain_rand.velocity_push_interval_s,N)/e._cfg.ctrl_dt).astype(int)
 alive=np.ones(N,dtype=bool);failed=[None]*N;rows=[];target=np.repeat(np.array([x[1] for x in commands]),4,axis=0)
 # Fixed commanded inputs are evaluation instrumentation. All physics and policy actions remain live.
 def hold_commands(state):
  state.info['commands'][:,:3]=target
  state.info['commands'][:,3:]=0
 e._maybe_resample_commands=hold_commands
 hold_commands(e.state)
 e._state=e.update_state(e.state)
 np.testing.assert_allclose(e.state.info['commands'][:,:3],target,atol=1e-6)

 for i in range(round(a.seconds/.02)):
  prev_push=e._next_push_step.copy()
  with torch.inference_mode():action=actor(wrapped.get_observations(),stochastic_output=False).numpy()
  st=e.step(action)
  np.testing.assert_allclose(st.info['commands'][:,:3],target,atol=1e-6)
  for j in range(N):
   if st.terminated[j] and failed[j] is None:failed[j]=(i+1)*.02
  alive&=~np.asarray(st.terminated,dtype=bool)
  measured=np.c_[e.get_local_linvel()[:,:2],e.get_gyro()[:,2]]
  force=np.stack([np.linalg.norm(e._backend.get_sensor_data(n),axis=1) for n in e._cfg.sensor.feet_force],1)
  q=e._backend.get_base_quat();w,x,y,z=q.T;tilt=np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))
  rows.append(dict(measured=measured.copy(),contact=e._contact.copy(),force=force,alive=alive.copy(),push=(e._next_push_step!=prev_push),tilt=tilt,foot_height=e._foot_height.copy(),qpos=e._backend.get_physics_state()[:,1:22].copy()))
  if not all(np.isfinite(v).all() for v in rows[-1].values()):raise RuntimeError(f'Nonfinite direction evaluation at{i}')
 z={k:np.array([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(R/(a.tag+'_trace.npz'),**z,commands=target,command_names=np.array([x[0] for x in commands]),sole_tilt_deg=np.degrees(e._sole_angles),geom_pos=e._backend._device_model.geom_pos.numpy(),geom_quat=e._backend._device_model.geom_quat.numpy())
 results=[];bw=e._backend._device_model.body_mass.numpy().sum(1)*9.81
 for k,(name,command) in enumerate(commands):
  cases=[]
  for j in range(k*4,k*4+4):
   mask=z['alive'][:,j]&(np.arange(len(rows))>=100);steady=mask.copy()
   for push_idx in np.flatnonzero(z['push'][:,j]):steady[push_idx:push_idx+50]=False
   v=z['measured'][:,j];moving=np.convolve(v[:,0],np.ones(25)/25,'valid');m_mask=mask[24:]
   loaded=z['force'][:,j]>.05*bw[j];periods=[];events=[];swing_ms=[]
   for side in range(2):
    swing=mask&~z['contact'][:,j,side]&loaded[:,1-side]
    edge=np.diff(np.r_[False,swing,False].astype(int));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    completed=[(s,t) for s,t in zip(starts,ends) if s>100 and t<len(rows) and mask[t] and t-s>=5]
    touching=np.array([t*.02 for s,t in completed]);periods.extend(np.diff(touching).tolist());events.extend((t,side) for s,t in completed);swing_ms.extend([(t-s)*20 for s,t in completed])
   periods=[t for t in periods if .25<=t<=1.2];cadence=2/float(np.median(periods)) if periods else None;events.sort()
   row=dict(case=j-k*4,failure_s=failed[j],valid_seconds_after2s=float(mask.sum()*.02),mean_velocity=v[mask].mean(0).tolist() if mask.any() else None,raw_velocity_std=v[mask].std(0).tolist() if mask.any() else None,steady_velocity_std_excluding1s_after_push=v[steady].std(0).tolist() if steady.any() else None,moving_mean05_vx_std=float(moving[m_mask].std()) if m_mask.any() else None,moving_mean05_vx_p05_p95=np.percentile(moving[m_mask],[5,95]).tolist() if m_mask.any() else None,moving_mean05_fraction_in015_020=float(((moving[m_mask]>=.15)&(moving[m_mask]<=.20)).mean()) if m_mask.any() else None,velocity_tracking_rmse=np.sqrt(np.mean((v[mask]-np.array(command))**2,axis=0)).tolist() if mask.any() else None,cadence_steps_per_s=cadence,alternation_fraction=float(np.mean([aa[1]!=bb[1] for aa,bb in zip(events,events[1:])])) if len(events)>1 else None,swing_median_ms=float(np.median(swing_ms)) if swing_ms else None,single_support_fraction=float((loaded[mask].sum(1)==1).mean()) if mask.any() else None,tilt_p95_deg=float(np.percentile(z['tilt'][mask,j],95)) if mask.any() else None,push_count=int(z['push'][:,j].sum()))
   cases.append(row)
  results.append(dict(name=name,command=list(command),cases=cases))
 report=dict(checkpoint=str(a.checkpoint),completed_updates=int(saved['iter'])+1,envs=N,seconds=a.seconds,control_sample_dt=.02,DR_multiplier=e._dr_multiplier,air_time_threshold_ms=e._cfg.reward_config.air_time_threshold_min*1000,external_velocity_pushes=a.pushes,reference_target_speed=[.15,.20],reference_cadence_steps_per_s=3.55059599289891,reference_nominal_speed_statistics={'raw_std':.029172702025688425,'moving05_std':.005181630445264303},directions=results)
 (R/(a.tag+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps({'result':str(R/(a.tag+'.json')),'surviving':int(alive.sum()),'total':N},indent=2))
finally:e.close()
