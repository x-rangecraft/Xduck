"""Finite diagnostic rollouts; no optimizer, parameter update, or training launch."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project'
sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.scratch_balance import ScratchReset
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor,OLD,WALK_SCALE

def main():
 assert json.loads((R/'status.json').read_text())['user_requires_explicit_resume']
 ck=R/'block_11/final.pt';before=hashlib.sha256(ck.read_bytes()).hexdigest()
 E.ensure_registries();torch.set_num_threads(2);torch.manual_seed(41012);np.random.seed(41012)
 c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.static_fractions=[1.]*6;c.env.stop_seconds=8.;c.env.max_episode_seconds=12.;c.env.curriculum_max_level=0
 # Isolate exploration and reward effects from mechanical/sensor randomization.
 for k,v in c.env.domain_rand.items():
  if isinstance(v,bool):c.env.domain_rand[k]=False
 c.env.domain_rand.motor.enabled=False;c.env.domain_rand.imu_max_angle_deg=0
 c.env.sole_tilt_dr_enabled=False;c.env.noise_config.level=0.;c.env.latency.imu_delay_ms=[0.,0.]
 for k in ['command_jitter_ms','feedback_jitter_ms']:
  if k in c.env.communication:c.env.communication[k]=[0,0]
 for k in ['com_range_stages','head_com_range_stages']:
  for v in c.env.curriculum[k]:v.value=0.
 modes=['mean600','fixed_reference','iid_small','iid_large','held_large','walker_zero','walker_2s_then_stand']
 scenarios=[dict(name='quiet',tilt=0,axis='pitch',speed=0,omega=0,force=0,duration=0),
  dict(name='pitch_push8',tilt=0,axis='pitch',speed=0,omega=0,force=8,duration=.4),
  dict(name='pitch_push16',tilt=0,axis='pitch',speed=0,omega=0,force=16,duration=.4),
  dict(name='roll_push8',tilt=0,axis='roll',speed=0,omega=0,force=8,duration=.4),
  dict(name='roll_push16',tilt=0,axis='roll',speed=0,omega=0,force=16,duration=.4),
  dict(name='pitch_moving10',tilt=10,axis='pitch',speed=.2,omega=.5,force=0,duration=0),
  dict(name='pitch_moving20',tilt=20,axis='pitch',speed=.3,omega=.8,force=0,duration=0),
  dict(name='roll_moving15',tilt=15,axis='roll',speed=.2,omega=.6,force=0,duration=0)]
 reps=3;K=len(scenarios)*reps;N=K*len(modes)
 m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);d=mujoco.MjData(m);q0=m.key_qpos[0].copy();ref=m.key_ctrl[0].copy();qi=m.jnt_qposadr[m.actuator_trnid[:,0]]
 feet=[]
 for side in ['left','right']:
  g=m.geom(side+'_foot_collision').id;mesh=m.geom_dataid[g];feet.append((g,m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()))
 qs=[];vs=[]
 for s in scenarios:
  for rep in range(reps):
   sign=1 if rep%2==0 else -1;q=q0.copy();v=np.zeros(m.nv);a=np.deg2rad(s['tilt'])*sign
   q[3:7]=[np.cos(a/2),np.sin(a/2) if s['axis']=='roll' else 0,np.sin(a/2) if s['axis']=='pitch' else 0,0]
   v[0 if s['axis']=='pitch' else 1]=s['speed']*sign;v[4 if s['axis']=='pitch' else 3]=s['omega']*sign
   d.qpos[:]=q;mujoco.mj_forward(m,d);bottom=min((verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,verts in feet);q[2]+=.001-bottom;qs.append(q);vs.append(v)
 qs=np.asarray(qs);vs=np.asarray(vs)
 original_reset=ScratchReset.build_reset_plan
 def fixed_reset(self,env,ids):
  plan=original_reset(self,env,ids);ids=np.asarray(ids);i=ids%K;plan.qpos[:]=qs[i];plan.qvel[:]=vs[i]
  actions=(qs[i][:,qi]-ref)/.12
  plan.info_updates['last_actions']=actions.copy();plan.info_updates['current_actions']=actions.copy()
  env._current_action_target[ids]=ref;env._previous_action_target[ids]=ref
  env._last_up[ids]=1-2*(qs[i,4]**2+qs[i,5]**2)
  return plan
 ScratchReset.build_reset_plan=fixed_reset
 e=E.create_env(c,num_envs=N,env_cfg_override=E.build_ppo_env_cfg_override(c));actor=FrozenActor(ck);walker=FrozenActor(R/'walk_parent.pt')
 rng=np.random.default_rng(41112);noise=np.zeros((N,14));tr=[];held=np.zeros(N);falls=np.zeros(N,bool);settle=np.full(N,np.nan);firstfail=np.full(N,-1,int)
 try:
  e._level=2;e.init_state();e.set_autoreset(False);e._zero_push[:]=True
  initial=e._backend.get_physics_state()[:,1:22].copy().reshape(len(modes),K,21)
  mismatch=float(np.max(np.abs(initial-initial[:1])));assert mismatch<1e-6,mismatch
  original_action=e.apply_action;clock=[0]
  def apply(actions,state):
   target=original_action(actions,state);force=np.zeros_like(e._last_force)
   for j in range(N):
    s=scenarios[(j%K)//reps];sign=1 if (j%reps)%2==0 else -1
    if .5<=clock[0]*.02<.5+s['duration']:force[j,0,0 if s['axis']=='pitch' else 1]=s['force']*sign
   e._backend.apply_body_force(e._push_ids,force);e._last_force[:]=force;return target
  e.apply_action=apply
  for t in range(400):
   clock[0]=t;obs=torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)
   with torch.inference_mode():raw=actor(obs).numpy()
   raw[K:2*K]=0
   # Explore only during the first 2.5 seconds, then permit a 5.5s quiet tail.
   if t<125:
    raw[2*K:3*K]+=rng.normal(0,.2,(K,14));raw[3*K:4*K]+=rng.normal(0,.8,(K,14))
    if t%8==0:noise[4*K:5*K]=rng.normal(0,.8,(K,14))
    raw[4*K:5*K]+=noise[4*K:5*K]
   wobs=obs[5*K:].numpy().copy();wobs[:,6:20]+=e.default_angles-OLD;wobs[:,34:48]=(e.default_angles+wobs[:,34:48]*.12-OLD)/WALK_SCALE;wobs[:,48:]=0
   with torch.inference_mode():wa=walker(torch.as_tensor(wobs,dtype=torch.float32)).numpy()
   converted=(OLD+wa*WALK_SCALE-e.default_angles)/.12
   raw[5*K:6*K]=converted[:K]
   if t<100:raw[6*K:]=converted[K:]
   st=e.step(raw);q=e._backend.get_physics_state()[:,1:22].copy();tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));bad=(tilt>55)|(q[:,2]<.6*q0[2]);new=bad&~falls;firstfail[new]=t;falls|=bad
   vel=e.get_local_linvel();gyro=e.get_gyro();dv=e.get_dof_vel();contact=e._contact.copy();foot,_=e._backend.get_body_pose_w(e._sole_body_ids)
   stable=(np.linalg.norm(vel[:,:2],axis=1)<.03)&(np.linalg.norm(gyro,axis=1)<.2)&(np.sqrt(np.mean(dv**2,axis=1))<.15)&contact.all(1)&(tilt<10)&~falls&(t>=25)
   held=np.where(stable,held+.02,0);hit=(held>=1)&np.isnan(settle);settle[hit]=t*.02
   tr.append(dict(qpos=q,tilt=tilt,velocity=vel.copy(),gyro=gyro.copy(),dof_velocity=dv.copy(),contact=contact,foot_pos=foot.copy(),force=e._last_force.copy(),reward=st.reward.copy(),action_target=e._current_action_target.copy(),action_raw=raw.copy()))
   if t%100==99:print(f'validation {t+1}/400 steps',flush=True)
  z={k:np.asarray([x[k] for x in tr]) for k in tr[0]};rows=[]
  for j in range(N):
   events=[]
   for side in range(2):
    ct=z['contact'][:,j,side];offs=np.flatnonzero(ct[:-1]&~ct[1:])+1
    for off in offs:
     scenario=scenarios[(j%K)//reps]
     if scenario['force'] and off<25:continue
     ons=np.flatnonzero(ct[off:]);
     if not len(ons):continue
     on=off+int(ons[0]);p=z['foot_pos'][:,j,side];travel=float(np.linalg.norm(p[on,:2]-p[off-1,:2]));lift=float(p[off:on+1,2].max()-min(p[off-1,2],p[on,2]))
     if firstfail[j]>=0 and on>=firstfail[j]:continue
     if (on-off)>=3 and travel>=.015 and lift>=.005:events.append(dict(side=side,off_s=off*.02,on_s=on*.02,travel_m=travel,after_actual_push=bool(scenario['force'] and off>=25)))
   end=int(firstfail[j])+1 if falls[j] else 400;rewards=z['reward'][:end,j].astype(float)
   rows.append(dict(case=j,mode=modes[j//K],scenario=scenarios[(j%K)//reps]['name'],rep=j%reps,failed=bool(falls[j]),nonfall_settled=bool(~falls[j] and np.isfinite(settle[j])),settle_s=None if np.isnan(settle[j]) else float(settle[j]),failure_s=None if firstfail[j]<0 else float(firstfail[j]*.02),steps=events,discounted_return=float(np.sum(rewards*.99**np.arange(end))),peak_tilt_deg=float(z['tilt'][:end,j].max())))
  summary=[dict(mode=mode,cases=K,falls=sum(x['failed'] for x in rows if x['mode']==mode),nonfall_settled=sum(x['nonfall_settled'] for x in rows if x['mode']==mode),cases_with_new_steps_before_failure=sum(bool(x['steps']) for x in rows if x['mode']==mode),nonfall_settled_cases_with_new_steps=sum(bool(x['steps']) and x['nonfall_settled'] for x in rows if x['mode']==mode)) for mode in modes]
  result=dict(ppo_updates=0,checkpoint_sha256_before=before,checkpoint_sha256_after=hashlib.sha256(ck.read_bytes()).hexdigest(),initial_qpos_max_difference_across_arms=mismatch,mechanics='XDuck original rigid GF43X40; 1ms mechanical timestep; nominal diagnostic DR and sensor noise disabled',horizon_s=8,exploration_window_s=2.5,quiet_tail_s=5.5,noise_held_seconds=.16,noise_std_action_small=.2,noise_std_action_large=.8,force_scenarios=scenarios,rows=rows,summary=summary,limitations='Finite open-loop exploration diagnostics are not trained policies or a PPO comparison. Cases after first failure are excluded from return; foot steps alone do not establish effective recovery. Same initial qpos/qvel and nominal mechanics, not a claim of bitwise identical GPU evolution.')
  assert result['checkpoint_sha256_before']==result['checkpoint_sha256_after']
  (R/'RECOVERY_DESIGN_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(R/'RECOVERY_DESIGN_VALIDATION_trace.npz',**z);print(json.dumps(summary),flush=True)
 finally:e.close()
if __name__=='__main__':main()
