from engine import *
from startup_walk_policy import NeuralStartupWalk
ap=argparse.ArgumentParser();ap.add_argument('--parameter-bank',default='final_125_params.npz');ap.add_argument('--seconds',type=float,default=12.);ap.add_argument('--prefix',default='direction_tracking');args=ap.parse_args()
scenarios=[('forward',[.3,0,0]),('backward',[-.3,0,0]),('turn_left',[0,0,.4]),('turn_right',[0,0,-.4]),('forward_left',[.3,0,.4]),('forward_right',[.3,0,-.4])]
B=8;N=B*len(scenarios);ids=np.tile(np.arange(B),len(scenarios));source=np.load(R/args.parameter_bank)
e=env(N);ps=[D.Policy() for _ in range(N)];prototype=NeuralStartupWalk(R);controllers=[copy.copy(prototype) for _ in range(N)]
for p in controllers:p.reset()
cmd=np.zeros((N,3));wanted_command=np.repeat(np.array([v for name,v in scenarios]),B,axis=0);rows=[];pre=[]
goal=np.asarray(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad'])
try:
 e.init_state();e.set_autoreset(False)
 base={k:source[k].copy() for k in FIELDS+['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']}
 from unilab.dr.types import ResetRandomizationPayload
 e._backend.set_state(np.arange(N,dtype=np.int32),source['qpos'][ids],source['qvel'][ids],ResetRandomizationPayload(**{k:base[k][ids] for k in FIELDS}))
 for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:getattr(e,'_'+k)[:]=base[k][ids]
 for p,f in zip(ps,frame_list(e,cmd)):p.reset({'joint_names':D.JOINTS},f)
 def hold(st):st.info['commands'][:,:3]=cmd;st.info['commands'][:,3:]=0
 e._maybe_resample_commands=hold;start_target=e._current_action_target.copy();measure=feet(e)
 for tick in range(200):
  hold(e.state);frames=frame_list(e,cmd);action=np.zeros((N,14))
  u=np.clip(tick/75,0,1);u=u*u*u*(10+u*(-15+6*u));targets=(1-u)*start_target+u*goal
  for j,(p,f) in enumerate(zip(ps,frames)):
   f['policy_context']['action']='stand';p.preprocess(f,None);p.postprocess({'actions':(targets[j]-D.REFERENCE)/D.STAND_ACTION_SCALE},f);action[j]=p.last_physical_offset/e._physical_scales
  e.step(action);np.testing.assert_allclose(e._current_action_target,targets,atol=3e-7)
  pre.append(e._backend.get_physics_state()[:,1:22].copy())
 initial=e._backend.get_physics_state()[:,1:22].copy()
 for tick in range(round(args.seconds/.02)):
  cmd[:]=wanted_command;hold(e.state);frames=frame_list(e,cmd);raw=np.zeros((N,14),np.float32);obs=np.zeros((N,61),np.float32);target=np.zeros_like(raw);action=np.zeros_like(raw)
  for j,(consumer,policy,f) in enumerate(zip(ps,controllers,frames)):
   ob=consumer.preprocess(f,None);obs[j]=ob['obs'][0];raw[j]=policy.infer(ob['obs'])[0]
   np.testing.assert_allclose(ob['obs'][0,48:51],wanted_command[j],atol=1e-7)
   out=consumer.postprocess({'actions':raw[j]},f);target[j]=[out[n]['position'] for n in D.TRAINING_JOINTS];action[j]=consumer.last_physical_offset/e._physical_scales
  st=e.step(action);np.testing.assert_allclose(e._current_action_target,target,atol=3e-7)
  q=e._backend.get_physics_state()[:,1:22].copy()
  rows.append(dict(qpos=q,obs=obs,raw=raw,target=target,velocity=e.get_local_linvel().copy(),clearance=e._foot_height.copy(),feet=measure(q),terminated=st.terminated.copy()))
  if (tick+1)%200==0:print(args.prefix,tick+1,flush=True)
 z={k:np.array([v[k] for v in rows]) for k in rows[0]};names=np.array([n for n,v in scenarios]);fields={k:base[k][ids] for k in FIELDS}
 np.savez_compressed(R/(args.prefix+'_trace.npz'),**z,**fields,names=names,source_indices=ids,commands=wanted_command)
 np.savez_compressed(R/(args.prefix+'_stand_trace.npz'),qpos=np.array(pre),**fields,names=names,source_indices=ids)
 records=[]
 for j in range(N):
  q=z['qpos'][:,j];v=z['velocity'][150:,j];command=wanted_command[j];name=scenarios[j//B][0]
  qw,qx,qy,qz=q[:,3:7].T;yaw=np.unwrap(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
  angular=np.diff(yaw)/.02;late_angular=angular[150:];smooth_angular=np.convolve(late_angular,np.ones(50)/50,'valid')
  init=initial[j];w,x,y,zz=init[3:7];heading=np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz));delta=q[-1,:2]-init[:2]
  axis=np.array([np.cos(heading),np.sin(heading)]);left=np.array([-axis[1],axis[0]])
  along=float(delta@axis);lateral=float(delta@left)
  tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));stable=bool(tilt.max()<20 and q[:,2].min()>.25)
  translation_ok=True;turn_ok=True;straight_ok=True
  if command[0]:
   sign=np.sign(command[0]);windows=np.convolve(v[:,0],np.ones(50)/50,'valid')
   translation_ok=bool(sign*v[:,0].mean()>.05 and sign*windows.min()>.03) if sign>0 else bool(sign*v[:,0].mean()>.05 and sign*windows.max()>.03)
  if command[2]:
   sign=np.sign(command[2]);turn_ok=bool(sign*late_angular.mean()>.10 and np.mean(sign*smooth_angular>.05)>.8)
  else:
   direction_angle=np.degrees(np.arctan2(abs(lateral),np.sign(command[0])*along))
   straight_ok=bool(np.sign(command[0])*along>.2 and direction_angle<45 and np.degrees(np.ptp(yaw))<45)
  record=dict(scenario=name,case=j%B,trace_index=j,command=command.tolist(),stable=stable,tilt_max_deg=float(tilt.max()),minimum_height_m=float(q[:,2].min()),vx_mean_mps=float(v[:,0].mean()),vy_mean_mps=float(v[:,1].mean()),yaw_rate_mean_rad_s=float(late_angular.mean()),yaw_rate_signed_fraction=float(np.mean(np.sign(command[2])*smooth_angular>.05)) if command[2] else None,yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),yaw_span_deg=float(np.degrees(np.ptp(yaw))),initial_forward_displacement_m=along,initial_left_displacement_m=lateral,translation_direction_ok=translation_ok,turn_direction_ok=turn_ok,straight_direction_ok=straight_ok,rough_tracking_pass=bool(stable and translation_ok and turn_ok and straight_ok))
  records.append(record)
 (R/(args.prefix+'_results.json')).write_text(json.dumps(dict(parameter_bank=args.parameter_bank,stand599_inference_calls=0,current_delivered_policy=True,runtime_profile='0.6-1.2s trained startup -> accepted Walk; actual consumer history',seconds=args.seconds,records=records),indent=2))
 print('SUMMARY',[(n,sum(r['rough_tracking_pass'] for r in records if r['scenario']==n),sum(r['stable'] for r in records if r['scenario']==n)) for n,v in scenarios],flush=True)
finally:e.close()
