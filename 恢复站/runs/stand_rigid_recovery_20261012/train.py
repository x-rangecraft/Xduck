from pathlib import Path
import argparse,sys,json,time,hashlib
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.training.rsl_rl import RslRlVecEnvWrapper,attach_checkpoint_progress
ap=argparse.ArgumentParser();ap.add_argument('--updates',type=int,default=300);ap.add_argument('--parent',default='stand_parent.pt');ap.add_argument('--run',default='training25');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
torch.set_num_threads(2);np.random.seed(26101);torch.manual_seed(26101);E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
class Wrapper(RslRlVecEnvWrapper):
 def _obs_to_tensordict(self,obs,info=None):
  td=super()._obs_to_tensordict(obs,info);td['stop_valid']=torch.as_tensor(self.env.stop_valid[:,None],device=self.device,dtype=torch.float32);td['stop_age']=torch.as_tensor(self.env.stop_age[:,None],device=self.device,dtype=torch.float32);return td
n=16 if a.smoke else int(c.algo.num_envs);e=E.create_env(c,num_envs=n,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 wrapped=Wrapper(e,device='cuda:0');tc=E.normalize_ppo_train_cfg(OmegaConf.to_container(c.algo,resolve=True));E.apply_ppo_runtime_flags(tc,c,training_enabled=True);runner=E.OnPolicyRunner(wrapped,tc,log_dir=str(R/a.run),device='cuda:0');parent=torch.load(R/a.parent,map_location='cuda:0',weights_only=True)
 runner.alg.actor.load_state_dict(parent['actor_state_dict'],strict=True);runner.alg.critic.load_state_dict(parent['critic_state_dict'],strict=True)
 for model in [runner.alg.actor,runner.alg.critic]:model.update_normalization=lambda obs:None
 norm={k:v.clone() for k,v in runner.alg.actor.state_dict().items() if k.startswith('obs_normalizer.')};attach_checkpoint_progress(e,runner)
 warm=0
 while e.stop_valid.sum()<max(1,n//4):
  with torch.inference_mode():action=e.prepare_actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  e.step(action);warm+=1
  if warm>350:raise RuntimeError('No valid Stand stage reached during live warmup')
 print('LIVE_WARMUP_FRAMES',warm,flush=True)
 updates=8 if a.smoke else a.updates;start=time.time();runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False)
 for k,v in norm.items():torch.testing.assert_close(v,runner.alg.actor.state_dict()[k],rtol=0,atol=0)
 if e._handoff_count==0:raise RuntimeError('No actual Walk -> Stand transitions occurred')
 result=dict(updates=updates,elapsed_seconds=time.time()-start,force_events=e._force_events,force_control_frames=e._force_control_frames,flexible_dr_enabled=False,actual_handoffs=e._handoff_count,incoming_state_samples=e._incoming,actor_dim=61,critic_dim=76,normalizer_frozen=True,walker_frozen=True,continuous_motor_and_physics=True,mask_fraction=runner.alg.valid_fraction,source_sha256=hashlib.sha256((R/a.parent).read_bytes()).hexdigest());(R/(a.run+'_completion.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['updates','actual_handoffs','mask_fraction','continuous_motor_and_physics']}),flush=True)
finally:e.close()
