"""Read-only mechanics/configuration probe: no optimizer or learner is constructed."""
from pathlib import Path
import sys,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
E.ensure_registries();np.random.seed(31201);torch.manual_seed(31201);torch.set_num_threads(2);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
results=[]
for level in [0,2]:
 e=E.create_env(c,num_envs=8,env_cfg_override=E.build_ppo_env_cfg_override(c))
 try:
  e._level=level;e.init_state();count=0;reward_terms=set()
  for t in range(32):
   with torch.inference_mode():action=e.prepare_actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
   st=e.step(action);assert np.isfinite(st.reward).all();assert e.state.obs['obs'].shape==(8,61)
   phase=st.info['balance_phase_weights'];np.testing.assert_allclose(phase.sum(1),1,atol=1e-6);assert np.all(phase>=0)
   reward_terms.update(k for k in st.info['log'] if k.startswith('Episode_Reward/'));count+=1
  results.append(dict(level=level,control_steps=count,actor_dimension=61,phase_weights_sum_to_one=True,finite_reward=True,active_reward_version=e.cfg.balance_reward.version,independent_zero_velocity_terms_present=all('Episode_Reward/'+k in reward_terms for k in ['zero_linear_velocity','zero_angular_velocity','linear_velocity_cost']),flexible_dofs=len(e._flex_joint_ids)))
 finally:e.close()
assert all(x['independent_zero_velocity_terms_present'] and x['flexible_dofs']==0 for x in results)
result=dict(passed=True,training_started=False,ppo_updates=0,mechanical_timestep_s=.001,results=results)
(R/'environment_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
