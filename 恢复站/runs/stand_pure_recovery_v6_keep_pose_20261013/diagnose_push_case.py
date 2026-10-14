from pathlib import Path
import sys,argparse,json
import numpy as np,torch
from validation_window import CRITERIA,pose_mask,assess_tail
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
prep=json.loads((R/'PREPARATION.json').read_text())
from unilab.utils.rotation import np_quat_apply
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--phase',default='discovery');ap.add_argument('--challenge-level',type=int,required=True);ap.add_argument('--force-N',type=float);ap.add_argument('--force-duration',type=float);ap.add_argument('--single-push',action='store_true');a=ap.parse_args()
E.ensure_registries();torch.set_num_threads(2);np.random.seed(41901);torch.manual_seed(41901);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.reset_mode='paired';c.env.discovery_phase=a.phase;c.env.dr_curriculum_enabled=a.challenge_level is not None
if a.challenge_level is not None:c.env.discovery_level=a.challenge_level
K=32;e=E.create_env(c,num_envs=K,env_cfg_override=E.build_ppo_env_cfg_override(c));actor=FrozenActor(R/a.checkpoint)
try:
 case_names=[prep['paired_cases'][i//2]['name'] for i in range(K)]
 if a.challenge_level is not None:
  level=a.challenge_level;assert 0<=level<=3;e._level=level;bank=e._banks[level];rng=np.random.default_rng(61317+level)
  indices=np.concatenate([rng.choice(np.flatnonzero(bank['kind']==kind),size=n,replace=True) for kind,n in [(0,4),(1,20),(2,8)]])
  e._paired={key:value[indices].copy() for key,value in bank.items()}
  # Fixed all-direction physical pushes at this course's upper force/duration.
  e._paired['fixture_force'][24:]=np.c_[np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)]*(a.force_N if a.force_N is not None else e.cfg.force_ranges_N[level][1])
  e._paired['fixture_duration'][24:]=a.force_duration if a.force_duration is not None else e.cfg.force_durations_s[level][1]
  case_names=['quiet']*4+['course_reset']*20+['course_push']*8
 e.init_state();e.set_autoreset(False);initial=e._backend.get_physics_state().copy();assert initial.shape[0]==K
 import hashlib
 initial_signature=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in [initial,e._mass_inertia_scale,e._foot_friction,e._imu_rotation,e._encoder_bias,e._sole_angles,e._motor.kp,e._motor.kd])).hexdigest()
 falls=np.zeros(K,bool);legacy=np.zeros(K,bool);first=np.full(K,-1,int);held=np.zeros(K);settle=np.full(K,np.nan);trace=[]
 for t in range(400):
  with torch.inference_mode():raw=actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  raw[K:2*K]=0;delta=(e.default_angles-e._current_action_target[2*K:])/e._increments;raw[2*K:]=np.arctanh(np.clip(delta,-.999999,.999999)) if c.env.decoder=='incremental' else 0
  st=e.step(raw)
  if a.single_push and t>=24:e._force_next[e._kind==2]=1000000
  q=e._backend.get_physics_state()[:,1:22].copy();tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));bad=e._bad.copy();hit=bad&~falls;first[hit]=t;falls|=bad;legacy|=(tilt>55)|(q[:,2]<.6*e._init_qpos[2])
  v=e.get_local_linvel();w=e.get_gyro();dv=e.get_dof_vel();contact=e._contact.copy();foot,_=e._backend.get_body_pose_w(e._sole_body_ids)
  stable=st.info['stand_pose_valid']&contact.all(1)&(np.linalg.norm(v,axis=1)<.03)&(np.linalg.norm(w,axis=1)<.2)&(np.sqrt(np.mean(dv**2,axis=1))<.15)&~falls
  held=np.where(stable,held+.02,0);hit=(held>=1)&np.isnan(settle);settle[hit]=t*.02
  _,head_quat=e._backend.get_body_pose_w(e._head_pose_id)
  head_up=np_quat_apply(head_quat[:,0],np.broadcast_to(e._head_axis,(e.num_envs,3)))[:,2]
  trace.append(dict(pose_valid_relaxed=pose_mask(q,head_up,e.default_angles,e.cfg.goal_height_m),head_up=head_up,qpos=q,velocity=v.copy(),gyro=w.copy(),dof_velocity=dv.copy(),contact=contact,foot_pos=foot.copy(),force=e._last_force.copy(),tilt=tilt,reward=st.reward.copy(),action_target=e._current_action_target.copy(),raw_action=raw.copy()))
 z={k:np.asarray([v[k] for v in trace]) for k in trace[0]};tail=slice(-round(CRITERIA['window_seconds']/.02),None)
 success,window_metrics=assess_tail(z['pose_valid_relaxed'][tail],z['contact'][tail],z['velocity'][tail],z['gyro'][tail],z['dof_velocity'][tail],falls)
 rows=[];modes=['model']
 for j in range(K):rows.append(dict(case=j,fixture=j%K,name=case_names[j%K],mode=modes[j//K],failed=bool(falls[j]),legacy_gate_failed=bool(legacy[j]),failure_s=None if first[j]<0 else float(first[j]*.02),nonfall_settled=bool(success[j]),strict_nonfall_settled=bool(not falls[j] and held[j]>=1.),completion_window_metrics={k:float(v[j]) for k,v in window_metrics.items()},settle_s=None if np.isnan(settle[j]) else float(settle[j])))
 summary={mode:dict(cases=K,falls=sum(x['failed'] for x in rows if x['mode']==mode),settled=sum(x['nonfall_settled'] for x in rows if x['mode']==mode)) for mode in modes}
 groups={name:dict(cases=sum(x['name']==name for x in rows),falls=sum(x['failed'] for x in rows if x['name']==name),settled=sum(x['nonfall_settled'] for x in rows if x['name']==name)) for name in ['quiet','course_reset','course_push']}
 quiet_rows=[x for x in rows if x['name']=='quiet']
 result=dict(checkpoint=a.checkpoint,phase=a.phase,push_diagnostic=dict(force_N=a.force_N,duration_s=a.force_duration,single_push=a.single_push,initial_signature=initial_signature),challenge_level=a.challenge_level,validation_criteria=CRITERIA,completion_requires_final_one_second=False,completion_uses_final_one_second_window=True,summary=summary,groups=groups,quiet_settled=sum(x['nonfall_settled'] for x in quiet_rows),quiet_cases=len(quiet_rows),rows=rows,nominal_diagnostic=a.challenge_level is None,domain_randomization_enabled=a.challenge_level is not None,effective_dr=e._effective_dr,limitations='Analytical diagnostic scenarios; current course includes configured DR when challenge_level is set. Not real walk handoff qualification. True failure is nonfoot-body ground contact, tilt>70deg or height<45%; legacy 55deg/60% gate separately reported.')
 (R/(a.tag+'_results.json')).write_text(json.dumps(result,indent=2)+'\n');z={k:np.asarray([v[k] for v in trace]) for k in trace[0]};np.savez_compressed(R/(a.tag+'_trace.npz'),**z);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limitations']}),flush=True)
finally:e.close()
