from pathlib import Path
import sys,json,numpy as np,mujoco,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.pure_reward import reward_rates
from unilab.envs.locomotion.xduck_gf43x40.pure_push_course import sample,specification
checks={};c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);d=mujoco.MjData(m);B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013';m0=mujoco.MjModel.from_binary_path(str(B.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'))
for key in ['gravity','iterations','tolerance','integrator','solver','disableflags','enableflags','timestep']:np.testing.assert_array_equal(getattr(m.opt,key),getattr(m0.opt,key))
checks['physics_options_unchanged']=True
sensor_ids=[m.sensor(name).id for name in c.env.contact_sensor_names]
assert all(m.sensor_refid[i]==int(m.sensor(i).name.rsplit('_',1)[1]) for i in sensor_ids);checks['all19_collision_sensor_refs_correct']=True
z=np.load(B/'boundary2100_100deg_trace.npz');qbad=z['qpos'][int(np.argmin(z['qpos'][:,12,2])),12].copy();d.qpos[:]=qbad;d.qvel[:]=0;mujoco.mj_forward(m,d)
found=np.array([d.sensordata[m.sensor_adr[i]] for i in sensor_ids]);assert found.max()>0,found;checks['CPU_true_body_ground_contact_detected']=True
base=dict(up=np.ones(1),height_ratio=np.ones(1),linvel=np.zeros((1,3)),gyro=np.zeros((1,3)),dof_vel=np.zeros((1,14)),action_delta=np.zeros((1,14)),phase='settle',joint_error=np.zeros((1,14)),head_up=np.ones(1),goal_ratio=1.,nominal_height=.319637239)
err=np.zeros((1,14));err[:,[1,10]]=.35
moving=reward_rates(**(base|dict(joint_error=err,return_weight=np.zeros(1))));settle=reward_rates(**(base|dict(joint_error=err,return_weight=np.ones(1))))
assert moving['hip_alignment'][0]==settle['hip_alignment'][0] and abs(moving['hip_alignment'][0])<.4;checks['hip_motion_cost_mild_and_not_discounted_by_danger']=True
head=np.zeros((1,14));head[:,5]=np.deg2rad(24);bad=reward_rates(**(base|dict(joint_error=head,return_weight=np.ones(1))));good=reward_rates(**(base|dict(return_weight=np.ones(1))))
assert good['head_return'][0]-bad['head_return'][0]>1.5 and bad['head_return_l1'][0]<-.5;checks['late_head_alignment_reward_matches_pose_acceptance']=True
noisy=reward_rates(**(base|dict(dof_vel=np.full((1,14),.9),return_weight=np.ones(1))))
assert noisy['quiet_bonus'][0]>.5;checks['quiet_reward_not_erased_by_stochastic_joint_motion']=True
rng=np.random.default_rng(71113);yaw=rng.uniform(-np.pi,np.pi,100000);spec=specification(3,1,[[3,18],[5,24],[8,30],[10,40]],[[.08,.4],[.08,.6],[.1,1],[.1,1.5]]);xy,dur,body,focus=sample(rng,yaw,spec,2);relative=np.mod(np.arctan2(xy[:,1],xy[:,0])-yaw,2*np.pi);sectors=np.mod(np.floor(relative/(np.pi/4)+.5).astype(int),8);counts=np.bincount(sectors[focus],minlength=8);assert np.all(abs(counts/counts.sum()-1/8)<.01);checks['all8_push_directions_balanced']=True
E.ensure_registries();c.env.reset_mode='paired';e=E.create_env(c,num_envs=4,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 q0=m.key_qpos[0].copy();e._paired=dict(qpos=np.stack([q0,qbad,q0,qbad]),qvel=np.zeros((4,20)),kind=np.zeros(4,int),fixture_force=np.zeros((4,2)),fixture_duration=np.zeros(4))
 e.init_state();e.set_autoreset(False);assert e.state.obs['obs'].shape==(4,61) and e.state.obs['critic'].shape==(4,76)
 observed=np.zeros(4,bool)
 for _ in range(4):s=e.step(np.zeros((4,14),np.float32));observed|=e._nonfoot_ground;assert np.isfinite(s.reward).all()
 assert observed[[1,3]].all() and not observed[[0,2]].any(),observed;checks['GPU_true_body_ground_contact_detected']=True
 assert e._increments[1]/.02==2. and e._increments[10]/.02==2.;checks['hip_roll_envelope_2rad_s']=True
 assert np.all((e.state.obs['critic'][:,-1]>=0)&(e.state.obs['critic'][:,-1]<=1));checks['critic_phase_cue_and_actor61_contract']=True
finally:e.close()
assert all(checks.values());out=dict(passed=True,checks=checks,GPU_ground_contact_flags=observed.tolist(),focused_direction_counts=counts.tolist(),no_PPO_updates=True)
(R/'REPAIR_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
