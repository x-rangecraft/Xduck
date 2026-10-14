from pathlib import Path
import sys,json,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
from unilab.envs.locomotion.xduck_gf43x40.pure_reward import reward_rates
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');t=np.array(c.env.posture_tolerance_rad)
def rates(error,head_up=1.,gate=1.,enabled=True):
 n=len(error)
 return reward_rates(np.ones(n),np.ones(n),np.zeros((n,3)),np.zeros((n,3)),np.zeros((n,14)),np.zeros((n,14)),'settle',error,np.full(n,head_up),return_weight=np.full(n,gate),allow_ground_recovery=True,posture_tolerance_rad=t if enabled else None,posture_head_world_tolerance_deg=10.)
base=rates(np.zeros((1,14)));inside=rates(np.tile(t*.5,(1,1)));bad=rates(np.tile(t*3,(1,1)),np.cos(np.deg2rad(35)))
keys=[k for k in base if k.startswith('posture_') or '_band_' in k]
assert all(np.allclose(base[k],inside[k]) for k in keys)
assert sum(base[k][0] for k in keys)>sum(bad[k][0] for k in keys)
released=rates(np.tile(t*3,(1,1)),gate=0.);assert all(released[k][0]==0 for k in keys)
rng=np.random.default_rng(29100);random=rates(rng.uniform(-3,3,(4096,14)),head_up=-1.)
assert np.min(sum(random[k] for k in keys))>=-1e-8
legacy=rates(np.tile(t*3,(1,1)),gate=0.,enabled=False)
assert all(np.allclose(released[k],legacy[k]) for k in legacy)
import train_rsl_rl as E
E.ensure_registries();e=E.create_env(c,num_envs=4,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();assert not e.cfg.flexible_dr_enabled and e.cfg.physics_microsteps==1
 for _ in range(20):
  s=e.step(np.zeros((4,14),np.float32));assert np.isfinite(s.reward).all()
 assert s.obs['obs'].shape==(4,61) and s.obs['critic'].shape==(4,76)
 result=dict(passed=True,within_tolerance_band_new_reward_flat=True,large_pose_error_reduces_reward=True,no_extra_constraint_during_recovery=True,new_pose_reward_sum_nonnegative_even_extreme_errors=True,cannot_gain_new_reward_by_deactivating_gate=True,rigid_physics_and_61D=True,new_terms=keys,aligned_new_reward_rate=float(sum(base[k][0] for k in keys)),bad_pose_new_reward_rate=float(sum(bad[k][0] for k in keys)))
 (R/'POSTURE_REWARD_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
