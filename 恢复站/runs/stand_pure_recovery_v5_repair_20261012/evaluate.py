from pathlib import Path
import sys,argparse,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
prep=json.loads((R/'PREPARATION.json').read_text())
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--phase',default='discovery');a=ap.parse_args()
E.ensure_registries();torch.set_num_threads(2);np.random.seed(41901);torch.manual_seed(41901);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.reset_mode='paired';c.env.discovery_phase=a.phase;c.env.sole_tilt_dr_enabled=False;c.env.noise_config.level=0.;c.env.latency.imu_delay_ms=[0.,0.]
for k,v in c.env.domain_rand.items():
 if isinstance(v,bool):c.env.domain_rand[k]=False
c.env.domain_rand.motor.enabled=False;c.env.domain_rand.imu_max_angle_deg=0
for k in ['foot_friction_range','dof_armature_multiplier_range','kp_multiplier_range','kd_multiplier_range','mass_inertia_scale_range']:c.env.domain_rand[k]=[1.,1.]
for k in ['com_range_stages','head_com_range_stages']:
 for stage in c.env.curriculum[k]:stage.value=0.
K=32;e=E.create_env(c,num_envs=K*3,env_cfg_override=E.build_ppo_env_cfg_override(c));actor=FrozenActor(R/a.checkpoint)
try:
 e.init_state();e.set_autoreset(False);initial=e._backend.get_physics_state().copy();assert np.max(np.abs(initial[:K,1:]-initial[K:2*K,1:]))<1e-6
 falls=np.zeros(K*3,bool);legacy=np.zeros(K*3,bool);first=np.full(K*3,-1,int);held=np.zeros(K*3);settle=np.full(K*3,np.nan);trace=[]
 for t in range(400):
  with torch.inference_mode():raw=actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  raw[K:2*K]=0;delta=(e.default_angles-e._current_action_target[2*K:])/e._increments;raw[2*K:]=np.arctanh(np.clip(delta,-.999999,.999999)) if c.env.decoder=='incremental' else 0
  st=e.step(raw);q=e._backend.get_physics_state()[:,1:22].copy();tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));bad=e._bad.copy();hit=bad&~falls;first[hit]=t;falls|=bad;legacy|=(tilt>55)|(q[:,2]<.6*e._init_qpos[2])
  v=e.get_local_linvel();w=e.get_gyro();dv=e.get_dof_vel();contact=e._contact.copy();foot,_=e._backend.get_body_pose_w(e._sole_body_ids)
  stable=st.info['stand_pose_valid']&contact.all(1)&(np.linalg.norm(v,axis=1)<.03)&(np.linalg.norm(w,axis=1)<.2)&(np.sqrt(np.mean(dv**2,axis=1))<.15)&~falls
  held=np.where(stable,held+.02,0);hit=(held>=1)&np.isnan(settle);settle[hit]=t*.02
  trace.append(dict(qpos=q,velocity=v.copy(),gyro=w.copy(),dof_velocity=dv.copy(),contact=contact,foot_pos=foot.copy(),force=e._last_force.copy(),tilt=tilt,reward=st.reward.copy(),action_target=e._current_action_target.copy(),raw_action=raw.copy()))
 rows=[];modes=['model','hold_initial_target','fixed_default_target']
 for j in range(K*3):rows.append(dict(case=j,fixture=j%K,name=prep['paired_cases'][(j%K)//2]['name'],mode=modes[j//K],failed=bool(falls[j]),legacy_gate_failed=bool(legacy[j]),failure_s=None if first[j]<0 else float(first[j]*.02),nonfall_settled=bool(not falls[j] and np.isfinite(settle[j])),settle_s=None if np.isnan(settle[j]) else float(settle[j])))
 summary={mode:dict(cases=K,falls=sum(x['failed'] for x in rows if x['mode']==mode),settled=sum(x['nonfall_settled'] for x in rows if x['mode']==mode)) for mode in modes}
 reference=rows[2*K:];eligible=[j for j in range(K) if reference[j]['failed']];rescued=[j for j in eligible if not rows[j]['failed']];quiet_rows=[x for x in rows[:K] if x['name']=='quiet']
 result=dict(checkpoint=a.checkpoint,phase=a.phase,summary=summary,reference_failed_fixtures=eligible,model_rescued_reference_failures=rescued,rescue_gain_fraction=len(rescued)/len(eligible) if eligible else 0.,quiet_settled=sum(x['nonfall_settled'] for x in quiet_rows),quiet_cases=len(quiet_rows),rows=rows,nominal_diagnostic=True,limitations='16 analytical scenarios, each repeated twice; nominal mechanics, not full DR or real walk handoff qualification. True failure is nonfoot-body ground contact, tilt>70deg or height<45%; legacy 55deg/60% gate separately reported.')
 (R/(a.tag+'_results.json')).write_text(json.dumps(result,indent=2)+'\n');z={k:np.asarray([v[k] for v in trace]) for k in trace[0]};np.savez_compressed(R/(a.tag+'_trace.npz'),**z);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limitations']}),flush=True)
finally:e.close()
