from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'xduck_direction_speed_dr_20261010';sys.path.insert(0,str(BASE))
from engine import *
model=mujoco.MjModel.from_binary_path(str(BASE.parent/'xduck_reward_audit_20261009/verified_rigid_model.mjb'))
a=np.load(BASE/'params_131.npz');b=np.load(BASE/'params_137.npz')
B=17;variants=json.loads((OUT/'pose_candidates.json').read_text());N=B*len(variants);ids=np.tile(np.arange(B),len(variants))
source={}
for k in FIELDS:
 nominal=np.asarray(getattr(model,k)).copy()
 source[k]=np.concatenate([nominal[None],a[k],b[k]])
for k in ['qpos','qvel','encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:
 nominal=a[k][0].copy()
 if k=='qvel' or k=='encoder_bias' or k=='sole_angles':nominal[:]=0
 if k=='imu_rotation':nominal[:]=np.eye(3)
 if k=='sole_delta_quat':nominal[:]=[1,0,0,0]
 source[k]=np.concatenate([nominal[None],a[k],b[k]])
e=env(N);ps=[D.Policy() for _ in range(N)];cmd=np.zeros((N,3));rows=[]
try:
 e.init_state();e.set_autoreset(False)
 from unilab.dr.types import ResetRandomizationPayload
 e._backend.set_state(np.arange(N,dtype=np.int32),source['qpos'][ids],source['qvel'][ids],ResetRandomizationPayload(**{k:source[k][ids] for k in FIELDS}))
 for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:getattr(e,'_'+k)[:]=source[k][ids]
 for p,f in zip(ps,frame_list(e,cmd)):p.reset({'joint_names':D.JOINTS},f)
 def hold(st):st.info['commands'][:]=0
 e._maybe_resample_commands=hold;first=e._current_action_target.copy();goals=np.repeat(np.array([v['target'] for v in variants]),B,axis=0);measure=feet(e)
 weight=source['body_mass'][ids].sum(1)*9.81
 for tick in range(600):
  hold(e.state);frames=frame_list(e,cmd);action=np.zeros((N,14));u=np.clip(tick/100,0,1);u=u*u*u*(10+u*(-15+6*u));target=(1-u)*first+u*goals
  for j,(p,f) in enumerate(zip(ps,frames)):
   f['policy_context']['action']='stand';p.preprocess(f,None);p.postprocess({'actions':(target[j]-D.REFERENCE)/D.STAND_ACTION_SCALE},f);action[j]=p.last_physical_offset/e._physical_scales
  st=e.step(action);np.testing.assert_allclose(e._current_action_target,target,atol=3e-7)
  q=e._backend.get_physics_state()[:,1:22].copy();forces=np.stack([np.linalg.norm(e._backend.get_sensor_data(n),axis=1) for n in e._cfg.sensor.feet_force],axis=1)
  rows.append(dict(qpos=q,velocity=e.get_local_linvel().copy(),feet=measure(q),clearance=e._foot_height.copy(),loaded=forces>.05*weight[:,None],force=forces,target=target.copy(),terminated=st.terminated.copy()))
  if (tick+1)%200==0:print('STAND',tick+1,flush=True)
 z={k:np.array([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(OUT/'stand_validation_trace.npz',**z,**{k:source[k][ids] for k in FIELDS},names=np.array([v['name'] for v in variants]),source_indices=ids)
 records=[]
 for j in range(N):
  q=z['qpos'][150:,j];w,x,y,zz=q[:,3:7].T;tilt=np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)));roll=np.degrees(np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y)));pitch=np.degrees(np.arcsin(np.clip(2*(w*y-zz*x),-1,1)))
  speed=np.linalg.norm(z['velocity'][-100:,j,:2],axis=1);foot=z['feet'][-100:,j];width=(foot[:,0,1]-foot[:,1,1])*1000
  r=dict(pose=variants[j//B]['name'],case=j%B,trace_index=j,nominal=j%B==0,body_tilt_max_deg=float(tilt.max()),body_roll_last2s_deg=float(roll[-100:].mean()),body_pitch_last2s_deg=float(pitch[-100:].mean()),min_height_m=float(q[:,2].min()),last2s_speed_max_mps=float(speed.max()),last2s_speed_mean_mps=float(speed.mean()),last2s_height_span_mm=float(np.ptp(q[-100:,2])*1000),width_mm=float(width.mean()),foot_fore_aft_difference_mm=float(abs((foot[:,0,0]-foot[:,1,0]).mean())*1000),double_support_last2s_fraction=float(z['loaded'][-100:,j].all(1).mean()),foot_geometry_contact_fraction=float((z['clearance'][-100:,j]<.003).all(1).mean()),target_tracking_abs_max_rad=float(abs(q[-100:,7:21]-goals[j]).max()))
  r['standing_pass']=bool(r['body_tilt_max_deg']<6 and r['min_height_m']>.28 and r['last2s_speed_max_mps']<.015 and r['last2s_height_span_mm']<2 and r['double_support_last2s_fraction']>.95 and r['foot_geometry_contact_fraction']>.95 and 200<r['width_mm']<222)
  records.append(r)
 (OUT/'stand_validation_results.json').write_text(json.dumps(dict(seconds=12,transition_seconds=2,hold_seconds=10,standing_policy_calls=0,root_free=True,state_edits_after_initialization=False,nominal_plus_16_DR_cases=True,physics_DR_course_multiplier=2.5,sensor_noise_not_used_for_constant_target_hold=True,records=records),indent=2))
 print('SUMMARY',[(v['name'],sum(r['standing_pass'] for r in records if r['pose']==v['name'])) for v in variants],flush=True)
finally:e.close()
