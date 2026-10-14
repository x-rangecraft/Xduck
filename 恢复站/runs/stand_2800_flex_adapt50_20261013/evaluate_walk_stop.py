from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor,OLD,WALK_SCALE,COMMANDS
from unilab.envs.locomotion.xduck_gf43x40 import walk_stop_audit
from unilab.utils.rotation import np_quat_apply
from validation_window import CRITERIA,pose_mask,assess_tail
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--challenge-level',required=True,type=int);ap.add_argument('--phase',default='settle');a=ap.parse_args()
np.random.seed(77113);torch.manual_seed(77113);torch.set_num_threads(2);E.ensure_registries()
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.training.task_name='XDuckWalkStopAudit';c.env.discovery_level=a.challenge_level;c.env.discovery_phase=a.phase;c.env.dr_curriculum_enabled=True;c.env.reset_mode='paired';c.env.episode_seconds=20.
extra=np.array([[0,.3,0],[0,-.3,0],[.3/np.sqrt(2),.3/np.sqrt(2),0],[.3/np.sqrt(2),-.3/np.sqrt(2),0],[-.3/np.sqrt(2),.3/np.sqrt(2),0],[-.3/np.sqrt(2),-.3/np.sqrt(2),0],[0,0,.6917144638660746],[0,0,-.6917144638660746]])
command_cases=np.vstack([COMMANDS,extra]);K=4*len(command_cases);e=E.create_env(c,num_envs=K,env_cfg_override=E.build_ppo_env_cfg_override(c));stand=FrozenActor(R/a.checkpoint)
walker_path=R/'walker_flex_adapt_50.pt';assert hashlib.sha256(walker_path.read_bytes()).hexdigest()=='e959c81f9a362bc6c3c0a4f2cc626716c41d54fafad2ff74debe900e65458ad9';walk=FrozenActor(walker_path)
try:
 # Start from the user's new default stand, then walk physically; never reset at takeover.
 with np.load(e.cfg.handoff_bank) as bank:q0=bank['qpos'][0].copy()
 e._paired=dict(qpos=np.tile(q0,(K,1)),qvel=np.zeros((K,e._banks[e._level]['qvel'].shape[1])),kind=np.zeros(K,int),fixture_force=np.zeros((K,2)),fixture_duration=np.zeros(K))
 e.init_state();e.set_autoreset(False);commands=np.repeat(command_cases,4,axis=0)
 walk_frames=np.tile(np.array([40,65,105,160]),len(command_cases));starts=50+walk_frames;ends=starts+400
 walk_bad=np.zeros(K,bool);stop_bad=np.zeros(K,bool);moving=np.zeros(K,bool);incoming=np.zeros((K,8));raw_offset=np.zeros((K,14));first_jump=np.zeros(K);trace=[];prefix=[]
 for t in range(int(ends.max())):
  walking=(t>=50)&(t<starts);stopping=(t>=starts)&(t<ends);hit=t==starts;walk_begin=(t==50)
  if walk_begin:raw_offset[:]=e._current_action_target-OLD
  if hit.any():
   lv=e.get_local_linvel();gy=e.get_gyro();incoming[hit]=np.c_[lv[hit],gy[hit],e._contact[hit]]
   moving[hit]=(np.linalg.norm(lv[hit,:2],axis=1)>.03)|(np.abs(gy[hit,2])>.15)
  with torch.inference_mode():raw=stand(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  if walking.any():
   obs=e.state.obs['obs'][walking].copy();obs[:,6:20]+=e.default_angles-OLD
   obs[:,34:48]=raw_offset[walking]/WALK_SCALE;obs[:,48:]=0;obs[:,48:51]=commands[walking]
   with torch.inference_mode():wa=walk(torch.as_tensor(obs,dtype=torch.float32)).numpy()
   raw_offset[walking]=wa*WALK_SCALE;e.audit_walk_target[walking]=OLD+raw_offset[walking]
  e.audit_walking[:]=walking;previous=e._current_action_target.copy();s=e.step(raw)
  q=e._backend.get_physics_state()[:,1:1+len(e._init_qpos)].copy();v=e.get_local_linvel().copy();w=e.get_gyro().copy();dv=e.get_dof_vel().copy()
  walk_bad|=e._bad&~(t>=starts);stop_bad|=e._bad&stopping
  if hit.any():first_jump[hit]=np.max(np.abs(e._current_action_target[hit]-previous[hit]),axis=1)
  if walking.any():np.testing.assert_allclose(e._current_action_target[walking],np.clip(e.audit_walk_target[walking],e._target_low,e._target_high),atol=1e-7)
  if hit.any():
   assert not e.audit_walking[hit].any()
   expected=np.clip(previous[hit]+e._increments*np.tanh(raw[hit]),e._target_low,e._target_high)
   np.testing.assert_allclose(e._current_action_target[hit],expected,atol=1e-7)
  _,hq=e._backend.get_body_pose_w(e._head_pose_id);hu=np_quat_apply(hq[:,0],np.broadcast_to(e._head_axis,(K,3)))[:,2]
  trace.append(dict(nonfoot_ground=e._nonfoot_ground.copy(),qpos=q,velocity=v,gyro=w,dof_velocity=dv,contact=e._contact.copy(),head_up=hu,pose_valid_relaxed=pose_mask(q,hu,e.default_angles,e.cfg.goal_height_m,e._nonfoot_ground),action_target=e._current_action_target.copy(),raw_action=raw.copy(),walking=walking.copy(),stopping=stopping.copy(),tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)))))
 z={k:np.asarray([x[k] for x in trace]) for k in trace[0]};rows=[]
 for j in range(K):
  sl=slice(int(ends[j])-round(CRITERIA['window_seconds']/.02),int(ends[j]));success,metrics=assess_tail(z['pose_valid_relaxed'][sl,j:j+1],z['contact'][sl,j:j+1],z['velocity'][sl,j:j+1],z['gyro'][sl,j:j+1],z['dof_velocity'][sl,j:j+1],stop_bad[j:j+1])
  eligible=bool(not walk_bad[j] and moving[j]);rows.append(dict(case=j,command=commands[j].tolist(),walk_seconds=float(walk_frames[j]*.02),switch_s=float(starts[j]*.02),walk_failed=bool(walk_bad[j]),moving_at_switch=bool(moving[j]),eligible=eligible,incoming=incoming[j].tolist(),stop_failed=bool(stop_bad[j]),success=bool(success[0]),final_stand_success=bool(success[0]),first_target_jump_rad=float(first_jump[j]),completion_window_metrics={k:float(v[0]) for k,v in metrics.items()}))
 eligible=[x for x in rows if x['eligible']]
 summary=dict(attempted=K,command_linear_speed_cap_m_s=.3,eligible=len(eligible),walk_failed=sum(x['walk_failed'] for x in rows),not_moving_at_switch=sum(not x['moving_at_switch'] and not x['walk_failed'] for x in rows),stop_falls=sum(x['stop_failed'] for x in eligible),successful_stops=sum(x['success'] for x in rows),final_stand_successes=sum(x['success'] for x in rows),success_rate=sum(x['success'] for x in rows)/K)
 result=dict(checkpoint=a.checkpoint,level=a.challenge_level,effective_dr=e._effective_dr,summary=summary,rows=rows,success_definition='final_stand_with_ground_recovery_allowed',validation_criteria=CRITERIA,continuous_physics_and_motor_history=True,no_reset_at_switch=True,walker_checkpoint=str(walker_path),linear_speed_scope='Initial reset linear speed cap 0.3m/s; walking translation commands capped at total norm 0.3m/s',walker_sha256=hashlib.sha256(walker_path.read_bytes()).hexdigest(),used_for_training=False,teacher_or_trajectory_training=False,limitations='Frozen user walker tested on current rigid mechanics. Pre-switch walking failures and nonmoving starts are listed separately from valid handoff results.')
 (R/(a.tag+'_results.json')).write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(R/(a.tag+'_trace.npz'),**z,starts=starts,ends=ends,commands=commands);print(json.dumps(summary),flush=True)
finally:e.close()
