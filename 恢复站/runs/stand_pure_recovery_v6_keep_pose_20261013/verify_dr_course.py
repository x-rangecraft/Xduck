from pathlib import Path
import sys,json,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
from course_gate import passes,overall_success
E.ensure_registries();rows=[]
for level,m in enumerate([1.,1.5,2.,2.5]):
 c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.discovery_level=level;c.env.dr_curriculum_enabled=True
 e=E.create_env(c,num_envs=8,env_cfg_override=E.build_ppo_env_cfg_override(c))
 try:
  x=e._effective_dr;assert x['multiplier']==m
  assert np.allclose(x['kp_multiplier_range'],[.9,1.1])
  assert np.allclose(x['mass_inertia_scale_range'],[1-.05*m,1+.05*m])
  assert np.allclose(x['foot_friction_range'],[1-.3*m,1+.3*m])
  assert np.allclose(x['sole_tilt_range_deg'],[-3,3])
  assert x['noise_level']==2*m and x['imu_max_angle_deg']==12
  assert np.isclose(x['trunk_com_range_m'],.015675*m)
  assert np.isclose(x['head_com_range_m'],.01045*m)
  assert np.allclose(x['imu_delay_ms'],[0,20])
  assert all(np.allclose(bounds,[.9,1.1]) for bounds in x['motor'].values())
  e.init_state()
  for _ in range(4):assert np.isfinite(e.step(np.zeros((8,14),np.float32)).reward).all()
  assert np.max(np.abs(np.degrees(e._sole_angles)))<=3+1e-6
  assert e.state.obs['obs'].shape==(8,61) and not e.cfg.flexible_dr_enabled
  rows.append(x)
 finally:e.close()
def fixture(n):return dict(summary={'model':dict(cases=32,settled=n)})
assert not passes(fixture(28),fixture(29))
assert passes(fixture(31),fixture(27))
out=dict(passed=True,dr_by_level=rows,overall_success_definition='(nominal successes + current-course DR successes) / total evaluated model episodes',threshold=.9,minimum_successes_of_64=58,consecutive_passes_required=1)
(R/'DR_COURSE_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
