from pathlib import Path
import sys,json,numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
from validation_window import CRITERIA,pose_mask,assess_tail
from unilab.envs.locomotion.xduck_gf43x40.pure_reward import reward_rates,potential,GAMMA
import train_rsl_rl as E
q=np.zeros((50,1,21));q[:,:,2]=.32;q[:,:,3]=1.;q[:,:,7:]=1.;contact=np.zeros((50,1,2),bool);contact[:,:,0]=True;zero=np.zeros((50,1,3));dv=np.ones((50,1,14))*2
ok,_=assess_tail(pose_mask(q,np.zeros((50,1)),np.zeros(14),.319637239,np.zeros((50,1),bool)),contact,zero,zero,dv,np.ones(1,bool));assert ok[0] # Prior fall, joint deviations and motion do not deny final standing.
lying=q.copy();lying[:,:,2]=.08;assert not assess_tail(pose_mask(lying,np.ones((50,1)),np.zeros(14),.319637239),contact,zero,zero,dv,np.zeros(1,bool))[0][0]
assert not assess_tail(pose_mask(q,np.ones((50,1)),np.zeros(14),.319637239,np.ones((50,1),bool)),contact,zero,zero,dv,np.zeros(1,bool))[0][0]
rate=reward_rates(np.array([-1.]),np.array([.1]),np.zeros((1,3)),np.zeros((1,3)),np.zeros((1,14)),np.zeros((1,14)),'settle',np.zeros((1,14)),np.array([-1.]),allow_ground_recovery=True)
assert all(rate[k][0]==0 for k in ['head_return','hip_goal','leg_pose','head_pose']);assert sum(v[0] for v in rate.values())+.625<0
assert np.isclose(GAMMA*(-2.5)-(-2.5),.0125)
E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.allow_ground_recovery=True;c.env.reset_mode='paired';m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file)
B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013';z=np.load(B/'boundary2100_100deg_trace.npz');qbad=z['qpos'][int(np.argmin(z['qpos'][:,12,2])),12];e=E.create_env(c,num_envs=2,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e._paired=dict(qpos=np.stack([m.key_qpos[0],qbad]),qvel=np.zeros((2,20)),kind=np.zeros(2,int),fixture_force=np.zeros((2,2)),fixture_duration=np.zeros(2))
 e.init_state();e.set_autoreset(False);events=np.zeros(2,int);first_deadline=None
 for t in range(410):
  s=e.step(np.zeros((2,14),np.float32));assert np.isfinite(s.reward).all()
  if t<5:assert not s.terminated.any() and not s.truncated.any()
  if t==0:assert e._nonfoot_ground[1]
  if s.terminated.any() and first_deadline is None:first_deadline=t
  assert not s.truncated.any();events+=(e._last_final_bonus!=0)
 assert first_deadline is not None and first_deadline>=390 and np.all(events==1),(first_deadline,events)
 assert e.state.obs['obs'].shape==(2,61) and e.state.obs['critic'].shape==(2,76)
 out=dict(passed=True,temporary_ground_contact_not_terminal=True,deadline_terminal_and_no_bootstrap=True,one_endpoint_bonus_per_episode=events.tolist(),first_deadline_frame=first_deadline,lying_pose_cannot_farm_alignment_or_potential=True,head_joint_pose_not_success_gate=True,actor61_critic76=True,criteria=CRITERIA)
 (R/'FINAL_GOAL_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
finally:e.close()
