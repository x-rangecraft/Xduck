from pathlib import Path
import argparse,sys,json,time
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.training.rsl_rl import RslRlVecEnvWrapper,attach_checkpoint_progress
ap=argparse.ArgumentParser();ap.add_argument('--updates',type=int,default=1000);ap.add_argument('--run',default='training1000');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',default=None);ap.add_argument('--level',type=int,default=0);a=ap.parse_args()
torch.set_num_threads(2);np.random.seed(32100);torch.manual_seed(32100);E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
if (R/'status.json').exists() and json.loads((R/'status.json').read_text()).get('user_requires_explicit_resume',False):
 raise SystemExit('Training is paused; only configuration was authorized')
class Wrapper(RslRlVecEnvWrapper):
 def _obs_to_tensordict(self,obs,info=None):
  td=super()._obs_to_tensordict(obs,info);td['stop_valid']=torch.as_tensor(self.env.stop_valid[:,None],device=self.device,dtype=torch.float32);td['stop_age']=torch.as_tensor(self.env.stop_age[:,None],device=self.device,dtype=torch.float32);return td
course=json.loads((R/'LEARNING_COURSE.json').read_text())['levels'][a.level]
c.algo.algorithm.exploration_std_min=course['std_min'];c.algo.algorithm.exploration_std_max=course['std_max'];c.algo.actor.distribution_cfg.init_std=course['entry_std']
c.env.curriculum_max_level=a.level
n=16 if a.smoke else int(c.algo.num_envs);e=E.create_env(c,num_envs=n,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e._level=a.level
 wrapped=Wrapper(e,device='cuda:0');tc=E.normalize_ppo_train_cfg(OmegaConf.to_container(c.algo,resolve=True));E.apply_ppo_runtime_flags(tc,c,training_enabled=True);runner=E.OnPolicyRunner(wrapped,tc,log_dir=str(R/a.run),device='cuda:0')
 # The first segment is fresh; subsequent segments restore only this scratch lineage.
 if a.resume:
  assert Path(a.resume).parts[0].startswith(('block_','smoke_'))
  runner.load(str(R/a.resume),map_location='cuda:0')
  previous=json.loads((R/(Path(a.resume).parts[0]+'_completion.json')).read_text())['curriculum_level']
  if a.level>previous:
   with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=course['entry_std'])
  runner.alg.project_std()
 initial={k:v.clone() for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.')};torch.save({'actor_state_dict':runner.alg.actor.state_dict(),'critic_state_dict':runner.alg.critic.state_dict(),'initialized_from_scratch':not bool(a.resume)},R/(a.run+'_start_actor.pt'));attach_checkpoint_progress(e,runner)
 assert e.stop_valid.sum()>0 and not c.algo.resume and c.algo.load_run is None
 updates=8 if a.smoke else a.updates;start=time.time();runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False)
 runner.save(str(R/a.run/'final.pt'))
 changed=max(float((initial[k]-v).abs().max()) for k,v in runner.alg.actor.state_dict().items() if k in initial);assert changed>0
 result=dict(reward_version='push_recover_settle_v3',std_min=float(runner.alg.actor.distribution.std_param.min()),std_max=float(runner.alg.actor.distribution.std_param.max()),updates=updates,elapsed_seconds=time.time()-start,initialized_from_scratch=not bool(a.resume),lineage_started_from_scratch=True,source_checkpoint=a.resume,checkpoint=str(Path(a.run)/'final.pt'),normalizers_learned_from_scratch=True,actor_weight_max_change=changed,actor_dim=61,critic_dim=76,flexible_dofs=0,mechanical_timestep_s=.001,force_events=e._force_events,curriculum_level=e._level,promotions=e._promotions,latest_episode_success_fraction=float(np.mean(e._end_scores)) if e._end_scores else None,qualified=False)
 (R/(a.run+'_completion.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
