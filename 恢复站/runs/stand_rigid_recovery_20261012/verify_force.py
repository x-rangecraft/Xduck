"""Verify existing body-force contract at the GF/Warp boundary, outside learner lifecycle."""
from pathlib import Path
import sys,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
E.ensure_registries();torch.set_num_threads(2)
rows=[]
for sign in [0,1,-1]:
 np.random.seed(27101);torch.manual_seed(27101);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.static_episode_fraction=1.;c.env.sole_tilt_dr_enabled=False
 e=E.create_env(c,num_envs=1,env_cfg_override=E.build_ppo_env_cfg_override(c))
 try:
  e.init_state();b=e._backend;start=b.get_physics_state().copy();target=e.default_angles[None].copy();force=np.array([[[sign*.1*e._weight_n,0,0]]],np.float32)
  for _ in range(10):
   b.apply_body_force(np.array([b.get_body_id('trunk_base')],np.int32),force);b.step(target,e.cfg.sim_substeps)
  end=b.get_physics_state().copy();rows.append(dict(sign=sign,force_n=float(force[0,0,0]),initial=start.tolist(),final_base_velocity=b.get_base_lin_vel()[0].tolist(),final_root_position=b.get_base_pos()[0].tolist()))
 finally:e.close()
for row in rows[1:]:np.testing.assert_allclose(row['initial'],rows[0]['initial'],rtol=0,atol=1e-6)
positive=rows[1]['final_base_velocity'][0];negative=rows[2]['final_base_velocity'][0]
assert positive-negative>.001,(positive,negative)
result=dict(passed=True,duration_s=.2,positive_minus_negative_world_x_velocity_m_s=positive-negative,rows=rows)
(R/'external_force_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
