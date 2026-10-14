from pathlib import Path
import argparse,sys,json,time,hashlib
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.training.rsl_rl import RslRlVecEnvWrapper,attach_checkpoint_progress
ap=argparse.ArgumentParser();ap.add_argument('--updates',type=int,default=25);ap.add_argument('--run',required=True);ap.add_argument('--resume');ap.add_argument('--phase',default='discovery');ap.add_argument('--level',type=int,default=0);ap.add_argument('--smoke',action='store_true');ap.add_argument('--num-envs',type=int);ap.add_argument('--reset-goal-state',action='store_true');ap.add_argument('--push-stage',type=int,default=0);ap.add_argument('--seed',type=int,default=41021);a=ap.parse_args()
s=json.loads((R/'status.json').read_text());assert not s.get('user_requires_explicit_resume',False)
torch.set_num_threads(2);np.random.seed(a.seed);torch.manual_seed(a.seed);E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.discovery_level=a.level;c.env.discovery_phase=a.phase;c.env.push_course_stage=a.push_stage;c.env.discovery_seed=a.seed
assert c.algo.resume is False and c.algo.load_run is None and c.algo.resume_path is None
assert c.algo.num_steps_per_env==20 and c.env.rollout_steps==20
num_envs=32 if a.smoke else (a.num_envs or int(c.algo.num_envs))
e=E.create_env(c,num_envs=num_envs,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 wrapper=RslRlVecEnvWrapper(e,device='cuda:0');tc=E.normalize_ppo_train_cfg(OmegaConf.to_container(c.algo,resolve=True));E.apply_ppo_runtime_flags(tc,c,training_enabled=True)
 runner=E.OnPolicyRunner(wrapper,tc,log_dir=str(R/a.run),device='cuda:0')
 assert abs(runner.alg.gamma-.995)<1e-8 and abs(runner.alg.entropy_coef-.01)<1e-8
 if a.resume:
  assert Path(a.resume).parts[0].startswith('block_');runner.load(str(R/a.resume),map_location='cuda:0')
  # # Reward changed; retain policy, critic and normalizers, discard old optimizer moments.
 if a.reset_goal_state:
  runner.alg.optimizer.state.clear();normalizer=runner.alg.critic.obs_normalizer
  normalizer._mean.view(-1)[-2]=0.;normalizer._std.view(-1)[-2]=1.;normalizer._var.view(-1)[-2]=1.
  with torch.no_grad():runner.alg.critic.mlp[0].weight[:,-2]=0.
 initial={k:v.clone() for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.')};attach_checkpoint_progress(e,runner)
 if not a.resume:torch.save(dict(actor_state_dict=runner.alg.actor.state_dict(),critic_state_dict=runner.alg.critic.state_dict(),started_from_scratch=True),R/(a.run+'_initial_actor.pt'))
 updates=4 if a.smoke else a.updates;start=time.time();runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False);runner.save(str(R/a.run/'final.pt'))
 changed=max(float((initial[k]-v).abs().max()) for k,v in runner.alg.actor.state_dict().items() if k in initial);assert changed>0
 sd=runner.alg.actor.state_dict();std=torch.exp(sd['distribution.log_std_param']) if 'distribution.log_std_param' in sd else sd['distribution.std_param']
 result=dict(success_definition='final_stand_v3',allow_ground_recovery=e.cfg.allow_ground_recovery,goal_optimizer_reset=a.reset_goal_state,action_increment_rad=e._increments.tolist(),failure_tilt_deg=e.cfg.recovery_failure_tilt_deg,critic_privileged_phase_cue=True,push_training_profile=e._effective_push,focused_force_events=e._focused_force_events,training_reset_weights=list(e.cfg.training_reset_weights),reset_bank=e.cfg.reset_bank,effective_dr=e._effective_dr,seed=a.seed,updates=updates,num_envs=num_envs,rollout_steps=20,elapsed_seconds=time.time()-start,checkpoint=a.run+'/final.pt',started_from_scratch=not bool(a.resume),source_checkpoint=a.resume,optimizer_moments_reset=False,flexible_joints=e.cfg.flexible_dr_enabled,flexible_dr_fraction=e.cfg.flexible_dr_fraction,physics_microsteps=e.cfg.physics_microsteps,actor_weight_max_change=changed,std_min=float(std.min()),std_max=float(std.max()),level=a.level,phase=a.phase,force_events=e._force_events,physical_episode_samples=len(e._scores),training_recovery_window_rate=float(np.mean(e._scores)) if e._scores else None,teacher_or_trajectory_training=False,actor_dimension=61,entropy_type='tanh_increment_entropy',pose_goal_height_m=e.cfg.goal_height_m)
 (R/(a.run+'_completion.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
