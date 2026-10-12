from flex_common import *
import argparse,hashlib,onnxruntime as ort
ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=8);ap.add_argument('--seconds',type=float,default=12);args=ap.parse_args();B=args.n
bank=np.load(R/'validation_flex_bank_params.npz');scenarios=[('forward',[.3,0,0]),('backward',[-.3,0,0]),('left',[0,0,.4]),('right',[0,0,-.4]),('forward_left',[.3,0,.4]),('forward_right',[.3,0,-.4])];model_names=['parent_r2.pt','ppo_flex100/model_50.pt'];ids=np.tile(np.arange(B),12);N=len(ids)
class Actor(torch.nn.Module):
 def __init__(self,ck):
  super().__init__();sd=ck['actor_state_dict'];self.register_buffer('mean',sd['obs_normalizer._mean']);self.register_buffer('std',sd['obs_normalizer._std']);self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14));self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
sessions=[ort.InferenceSession(str(R/n),sess_options=options,providers=['CPUExecutionProvider']) for n in ['parent_r2.onnx','flex_adapt_50.onnx']];max_parity_error=0.
torch.set_num_threads(2);models=[Actor(torch.load(R/n,map_location='cpu',weights_only=False)).eval() for n in model_names];e=env(N);m,qi,vi,names,local=metadata();goal=np.array(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad']);cmd=np.zeros((N,3));wanted=np.tile(np.repeat(np.array([c for n,c in scenarios]),B,axis=0),(2,1));rows=[]
try:
 e.init_state();e.set_autoreset(False);bodyids=np.array([e._backend.get_body_id(n) for n in names],np.int32)
 from unilab.dr.types import ResetRandomizationPayload
 e._backend.set_state(np.arange(N,dtype=np.int32),bank['qpos'][ids],bank['qvel'][ids],ResetRandomizationPayload(**{k:bank[k][ids] for k in FIELDS}))
 for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:getattr(e,'_'+k)[:]=bank[k][ids]
 sensor_rng=np.random.default_rng(1860091);delay_rng=np.random.default_rng(1860092)
 def paired_noise(data,scale):
  noise=sensor_rng.uniform(-1,1,(N//2,data.shape[1])).astype(data.dtype)
  return data+np.concatenate([noise,noise])*float(e._cfg.noise_config.level)*scale
 def paired_imu_delay(env_ids=None):
  lo,hi=e._imu_delay_bounds
  if env_ids is None:
   lag=delay_rng.integers(lo,hi+1,N//2);e._imu_delay_lag[:]=np.concatenate([lag,lag])
  else:e._imu_delay_lag[env_ids]=delay_rng.integers(lo,hi+1,len(env_ids))
 e._obs_noise=paired_noise;e._sample_imu_delay=paired_imu_delay
 def hold(st):st.info['commands'][:,:3]=cmd;st.info['commands'][:,3:]=0
 e._maybe_resample_commands=hold;first=e._current_action_target.copy()
 for tick in range(200):
  hold(e.state);u=np.clip(tick/75,0,1);u=u*u*u*(10+u*(-15+6*u));target=(1-u)*first+u*goal;raw=(target-e.default_angles)/e._physical_scales;e.step(raw);np.testing.assert_allclose(e._current_action_target,target,atol=4e-7)
 initial,_=state(e,m)
 for tick in range(round(args.seconds/.02)):
  cmd[:]=wanted;hold(e.state);obs=e.state.obs['obs'].copy();obs[:,48:51]=cmd
  with torch.inference_mode():raw=np.concatenate([model(torch.tensor(x,dtype=torch.float32)).numpy() for model,x in zip(models,np.split(obs,2))])
  assert np.isfinite(raw).all()
  if tick in [0,150,round(args.seconds/.02)-1]:
   for mi,session in enumerate(sessions):
    for j in [mi*6*B,mi*6*B+6*B-1]:
     expected=session.run(None,{'obs':obs[j:j+1].astype(np.float32)})[0][0];err=float(abs(expected-raw[j]).max());assert err<2e-5;max_parity_error=max(max_parity_error,err)
  target=np.clip(D.REFERENCE+raw*D.WALK_ACTION_SCALE,D.TARGET_LOW,D.TARGET_HIGH);st=e.step(raw);np.testing.assert_allclose(e._current_action_target,target,atol=4e-7)
  q,v=state(e,m);rows.append(dict(qpos=q,velocity=e.get_local_linvel().copy(),feet=feet(e,q,bodyids,local),clearance=e._foot_height.copy(),target=target.copy()))
  if (tick+1)%100==0:print('FLEX_EVAL',tick+1,flush=True)
 z={k:np.array([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(R/'adapt50_validation_trace.npz',**z,**{k:bank[k][ids] for k in FIELDS},initial=initial,motor_qpos_indices=qi,source_indices=ids)
 records=[]
 for j in range(N):
  q=z['qpos'][:,j];v=z['velocity'][:,j];w,x,y,zz=q[:,3:7].T;tilt=np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)));yaw=np.unwrap(np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz)));turnrate=np.r_[0,np.diff(yaw)/.02];c=wanted[j];trans=bool(c[0]);signed=np.sign(c[0])*v[:,0] if trans else np.sign(c[2])*turnrate;windows=np.convolve(signed,np.ones(50)/50,'valid');stable=bool(tilt.max()<20 and q[:,2].min()>.25)
  h=z['clearance'][:,j];air=np.zeros(2,bool);peak=np.zeros(2);ev=[]
  for t in range(len(h)):
   leave=(~air)&(h[t]>.0015);peak=np.maximum(peak,h[t]);land=air&(h[t]<.0007)
   for side in np.flatnonzero(land):
    if peak[side]>=(.012 if trans else .008):ev.append((t*.02,int(side),float(peak[side]*1000)))
   peak[land]=0;air=(air|leave)&(~land)
  early=[r for r in ev if r[0]<=2.5];late=[r for r in ev if r[0]>=3];start=bool(len({r[1] for r in early})==2 and windows[:76].max()>(.05 if trans else .1) and tilt[:125].max()<20)
  active=bool(stable and start and min(sum(r[1]==s for r in late) for s in range(2))>=6 and signed[150:].mean()>(.05 if trans else .1) and windows[150:].min()>(.03 if trans else .05))
  w0,x0,y0,z0=initial[j,3:7];heading0=np.arctan2(2*(w0*z0+x0*y0),1-2*(y0*y0+z0*z0));delta=q[-1,:2]-initial[j,:2];along=float(delta@np.array([np.cos(heading0),np.sin(heading0)]));lateral=float(delta@np.array([-np.sin(heading0),np.cos(heading0)]));angle=float(np.degrees(np.arctan2(abs(lateral),np.sign(c[0])*along))) if trans else None
  translation_heading=bool(np.sign(c[0])*along>.2 and angle<60 and abs(np.degrees(yaw[-1]-yaw[0]))<90) if trans and not c[2] else True
  turns_right_way=bool(np.sign(c[2])*turnrate[150:].mean()>.10 and np.mean(np.convolve(np.sign(c[2])*turnrate[150:],np.ones(50)/50,'valid')>.05)>.8) if c[2] else True
  rough_tracking=bool(active and translation_heading and turns_right_way)
  periods=[float(v) for side in range(2) for v in np.diff([r[0] for r in late if r[1]==side and r[2]>=20])]
  roll=np.degrees(np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y)))
  width=(z['feet'][150:,j,0,1]-z['feet'][150:,j,1,1])*1000;peaks=[[r[2] for r in late if r[1]==s] for s in range(2)];records.append(dict(model='before_r2' if j<6*B else 'adapt50',scenario=scenarios[(j//B)%6][0],case=j%B,trace_index=j,started=start,sustained=active,stable=stable,vx_mean=float(v[150:,0].mean()),signed_progress_mean=float(signed[150:].mean()),yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),width_mm=float(width.mean()),foot_peak_median_mm=[float(np.median(p)) if p else 0. for p in peaks],tilt_max_deg=float(tilt.max()),rough_tracking=rough_tracking,translation_heading_ok=translation_heading,turn_direction_ok=turns_right_way,net_forward_displacement_m=along,net_left_displacement_m=lateral,net_translation_angle_deg=angle,yaw_rate_mean=float(turnrate[150:].mean()),same_foot_period_s=float(np.median(periods)) if periods else 0.,first_0p8_right_roll_deg=float(roll[:40].max()),first_2p5_drop_mm=float((initial[j,2]-q[:125,2].min())*1000)))
 summary=[dict(model=model,scenario=name,started=sum(r['started'] for r in records if r['model']==model and r['scenario']==name),rough_tracking=sum(r['rough_tracking'] for r in records if r['model']==model and r['scenario']==name),sustained=sum(r['sustained'] for r in records if r['model']==model and r['scenario']==name),total=B) for model in ['before_r2','adapt50'] for name,c in scenarios]
 (R/'adapt50_validation_results.json').write_text(json.dumps(dict(fresh_seed=1850091,flex_coordinates=48,stiffness_damping_dr_fraction=.1,ankle_sole_pitch_dr_deg=[-3,3],same_parameters_before_after=True,paired_uniform_noise_and_imu_delay_draws=True,native_training_sensor_noise_delay=True,standing599_calls=0,seconds=args.seconds,actual_observation_onnx_parity_error=max_parity_error,summary=summary,records=records),indent=2));print('FLEX_COMPARISON',summary,flush=True)
finally:e.close()
