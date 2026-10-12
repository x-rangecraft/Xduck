from pathlib import Path
import sys,argparse,hashlib
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'xduck_direction_speed_dr_20261010';sys.path.insert(0,str(BASE))
from engine import *
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['consumer','training_sensors'],default='training_sensors');ap.add_argument('--pose',choices=['symmetric','original'],default='symmetric');ap.add_argument('--commands',nargs='+',default=['forward','backward','left','right','forward_left','forward_right']);args=ap.parse_args()
commands={'forward':[.3,0,0],'backward':[-.3,0,0],'left':[0,0,.4],'right':[0,0,-.4],'forward_left':[.3,0,.4],'forward_right':[.3,0,-.4]}
bank=np.load(OUT/'dr_bank.npz');B=len(bank['qpos']);parity_session=actor('direction_r2_179.onnx');parent_path=BASE/'ppo_direction_r2_run/model_179.pt';before=hashlib.sha256(parent_path.read_bytes()).hexdigest();ck=torch.load(parent_path,map_location='cpu',weights_only=False);sd=ck['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean']);self.register_buffer('std',sd['obs_normalizer._std']);self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14));self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
actor_model=Actor().eval();goal_path=(OUT.parent/'xduck_symmetric_stand_20261010/SYMMETRIC_STANDING_TARGET.json') if args.pose=='symmetric' else BASE/'NEW_STANDING_TARGET.json';goal=np.array(json.loads(goal_path.read_text())['target_rad']);max_error=0.;summaries=[]
for command_name in args.commands:
 np.random.seed(1730091);torch.manual_seed(1730091);e=env(B);ps=[D.Policy() for _ in range(B)];cmd=np.zeros((B,3));rows=[]
 try:
  e.init_state();e.set_autoreset(False)
  from unilab.dr.types import ResetRandomizationPayload
  e._backend.set_state(np.arange(B,dtype=np.int32),bank['qpos'],bank['qvel'],ResetRandomizationPayload(**{k:bank[k] for k in FIELDS}))
  for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:getattr(e,'_'+k)[:]=bank[k]
  np.testing.assert_allclose(e.default_angles,D.REFERENCE,atol=1e-7);np.testing.assert_allclose(e._physical_scales,D.WALK_ACTION_SCALE,atol=1e-7)
  for p,f in zip(ps,frame_list(e,cmd)):p.reset({'joint_names':D.JOINTS},f)
  def hold(st):st.info['commands'][:,:3]=cmd;st.info['commands'][:,3:]=0
  e._maybe_resample_commands=hold;first=e._current_action_target.copy();measure=feet(e)
  for tick in range(200):
   hold(e.state);frames=frame_list(e,cmd);u=np.clip(tick/75,0,1);u=u*u*u*(10+u*(-15+6*u));target=(1-u)*first+u*goal;action=(target-D.REFERENCE)/e._physical_scales
   for j,(p,f) in enumerate(zip(ps,frames)):
    f['policy_context']['action']='stand';p.preprocess(f,None);p.postprocess({'actions':(target[j]-D.REFERENCE)/D.STAND_ACTION_SCALE},f)
   e.step(action);np.testing.assert_allclose(e._current_action_target,target,atol=4e-7)
  initial=e._backend.get_physics_state()[:,1:22].copy();standing_tilt=np.degrees(np.arccos(np.clip(1-2*(initial[:,4]**2+initial[:,5]**2),-1,1)))
  for tick in range(600):
   cmd[:]=commands[command_name];hold(e.state)
   if args.mode=='consumer':
    frames=frame_list(e,cmd);obs=np.array([p.preprocess(f,None)['obs'][0] for p,f in zip(ps,frames)])
   else:
    obs=e.state.obs['obs'].copy();obs[:,48:51]=cmd
   assert obs.shape==(B,61);np.testing.assert_allclose(obs[:,48:51],cmd,atol=1e-7)
   with torch.inference_mode():raw=actor_model(torch.tensor(obs,dtype=torch.float32)).numpy()
   assert np.isfinite(raw).all()
   if tick in [0,150,599]:
    check=np.linspace(0,B-1,8,dtype=int);onnx=np.concatenate([parity_session.run(None,{'obs':obs[j:j+1].astype(np.float32)})[0] for j in check]);error=float(abs(raw[check]-onnx).max());max_error=max(max_error,error);assert error<2e-5
   target=np.clip(D.REFERENCE+raw*D.WALK_ACTION_SCALE,D.TARGET_LOW,D.TARGET_HIGH)
   # Same consumer physical offset/history mapping, vectorized mean inference.
   for j,p in enumerate(ps):p.last_physical_offset=(raw[j]*D.WALK_ACTION_SCALE).copy()
   st=e.step(raw);np.testing.assert_allclose(e._current_action_target,target,atol=4e-7);np.testing.assert_allclose(st.obs['obs'][:,34:48],raw,atol=3e-6)
   q=e._backend.get_physics_state()[:,1:22].copy();rows.append(dict(qpos=q,velocity=e.get_local_linvel().copy(),clearance=e._foot_height.copy(),feet=measure(q),imu_delay_steps=e._imu_delay_lag.copy()))
   if (tick+1)%200==0:print(args.mode,args.pose,command_name,tick+1,flush=True)
  z={k:np.array([r[k] for r in rows]) for k in rows[0]};tag=args.mode+'_'+args.pose+'_'+command_name;np.savez_compressed(OUT/(tag+'_trace.npz'),**z,initial=initial)
  records=[]
  for j in range(B):
   q=z['qpos'][:,j];v=z['velocity'][:,j];h=z['clearance'][:,j];w,x,y,zz=q[:,3:7].T;tilt=np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)));yaw=np.unwrap(np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz)));rate=np.r_[0,np.diff(yaw)/.02];command=np.array(commands[command_name]);trans=bool(command[0]);signed=(np.sign(command[0])*v[:,0]) if trans else np.sign(command[2])*rate
   air=np.zeros(2,bool);peak=np.zeros(2);events=[]
   for t in range(600):
    leave=(~air)&(h[t]>.0015);peak=np.maximum(peak,h[t]);land=air&(h[t]<.0007)
    for side in np.flatnonzero(land):
     if peak[side]>=(.012 if trans else .008):events.append(dict(time_s=t*.02,side=int(side),peak_mm=float(peak[side]*1000)))
    peak[land]=0;air=(air|leave)&(~land)
   early=[r for r in events if r['time_s']<=2.5];late=[r for r in events if r['time_s']>=3];early_sides={r['side'] for r in early};late_sides=[sum(r['side']==s for r in late) for s in range(2)];window=np.convolve(signed,np.ones(50)/50,'valid');speed=float(signed[150:].mean())
   stable_start=bool(standing_tilt[j]<6 and tilt[:125].max()<20 and q[:125,2].min()>.25);stable_all=bool(tilt.max()<20 and q[:,2].min()>.25)
   started=bool(stable_start and len(early_sides)==2 and window[:76].max()>(.05 if trans else .10))
   sustained=bool(started and stable_all and min(late_sides)>=6 and speed>(.05 if trans else .10) and window[150:].min()>(.03 if trans else .05))
   heading_delta=float(np.degrees(yaw[-1]-yaw[0]));w0,x0,y0,z0=initial[j,3:7];heading0=np.arctan2(2*(w0*z0+x0*y0),1-2*(y0*y0+z0*z0));delta=q[-1,:2]-initial[j,:2];along=float(delta@np.array([np.cos(heading0),np.sin(heading0)]));sideways=float(delta@np.array([-np.sin(heading0),np.cos(heading0)]));angle=float(np.degrees(np.arctan2(abs(sideways),np.sign(command[0])*along))) if trans else None
   peaks=[[r['peak_mm'] for r in late if r['side']==s] for s in range(2)];width=(z['feet'][150:,j,0,1]-z['feet'][150:,j,1,1])*1000
   r=dict(case=j,label=str(bank['labels'][j]),group=str(bank['groups'][j]),started_by_2p5s=started,sustained_12s=sustained,standing_tilt_deg=float(standing_tilt[j]),stable_start=stable_start,stable_all=stable_all,tilt_max_deg=float(tilt.max()),min_height_m=float(q[:,2].min()),signed_speed_or_turn_rate=speed,body_vx_mean_mps=float(v[150:,0].mean()),body_vy_mean_mps=float(v[150:,1].mean()),min1s_signed_progress=float(window[150:].min()),yaw_change_deg=heading_delta,initial_direction_angle_deg=angle,rough_translation_direction=bool(sustained and np.sign(command[0])*along>.2 and angle<60 and abs(heading_delta)<90) if trans else None,early_full_steps=len(early),late_full_steps_by_side=late_sides,foot_peak_median_mm=[float(np.median(p)) if p else 0. for p in peaks],width_mean_mm=float(width.mean()),imu_delay_steps_seen=np.unique(z['imu_delay_steps'][:,j]).tolist())
   records.append(r)
  active=[r['signed_speed_or_turn_rate'] for r in records if r['sustained_12s']];median=float(np.median(active)) if active else None
  for r in records:r['speed_within_median_20pct']=bool(r['sustained_12s'] and .8*median<=r['signed_speed_or_turn_rate']<=1.2*median) if median is not None else False
  result=dict(mode=args.mode,pose=args.pose,model_sha256=before,command=command.tolist(),conditions=B,random_conditions=128,endpoint_conditions=30,nominal_conditions=1,stand599_calls=0,seconds=12,physical_hold_before_command_s=4,root_free=True,no_runtime_helper=True,actor_dim=61,action_dim=14,batched_torch_onnx_parity_max_error=max_error,uses_actual_training_sensor_pipeline=args.mode=='training_sensors',noise_level=float(e._cfg.noise_config.level) if args.mode=='training_sensors' else 0,active_signed_speed_median=median,records=records)
  (OUT/(tag+'_results.json')).write_text(json.dumps(result,indent=2));summaries.append(dict(tag=tag,total=B,started=sum(r['started_by_2p5s'] for r in records),sustained=sum(r['sustained_12s'] for r in records),speed_in_20pct=sum(r['speed_within_median_20pct'] for r in records),speed_range=[min(active),max(active)] if active else None));print('SUMMARY',summaries[-1],flush=True)
 finally:e.close()
assert before==hashlib.sha256(parent_path.read_bytes()).hexdigest()
(OUT/(args.mode+'_'+args.pose+'_summary.json')).write_text(json.dumps(summaries,indent=2))
