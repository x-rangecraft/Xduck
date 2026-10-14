from pathlib import Path
import sys,json,numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
E.ensure_registries();torch.set_num_threads(2);np.random.seed(41021);torch.manual_seed(41021);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
e=E.create_env(c,num_envs=32,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();assert not hasattr(e,'walk_actor') and not hasattr(e,'prepare_actor')
 assert e.state.obs['obs'].shape==(32,61);assert e.state.obs['critic_obs'].shape==(32,76) if 'critic_obs' in e.state.obs else True
 initial=e._current_action_target.copy();s=e.step(np.zeros((32,14)));np.testing.assert_allclose(e._current_action_target,initial,atol=1e-7)
 rewards=[s.reward.copy()];events=0
 for t in range(64):
  previous=e._current_action_target.copy();s=e.step(np.random.normal(0,1,(32,14)).astype(np.float32));assert np.isfinite(s.reward).all() and np.isfinite(e.state.obs['obs']).all();rewards.append(s.reward.copy());events+=int(s.terminated.sum())
  np.testing.assert_allclose(e.state.obs['obs'][:,34:48],(e._current_action_target-e.default_angles)/.12,atol=.01)
  assert np.all(e._current_action_target>=e._target_low-1e-6) and np.all(e._current_action_target<=e._target_high+1e-6)
 assert len(e._flex_joint_ids)==0
 result=dict(passed=True,actor_dim=61,critic_groups=list(e.state.obs),flexible_joints=0,no_teacher_actors=True,zero_action_holds_target=True,finite_rewards=True,target_history_matches_executed_target=True,contact_floor_sensor_count=len(c.env.contact_sensor_names),force_events=e._force_events,physical_failure_events=events,control_steps=65,ppo_updates=0)
 (R/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
