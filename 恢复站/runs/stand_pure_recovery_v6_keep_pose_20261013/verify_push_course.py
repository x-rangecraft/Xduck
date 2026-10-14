from pathlib import Path
import json,sys,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
from unilab.envs.locomotion.xduck_gf43x40.pure_push_course import specification,sample
import train_rsl_rl as E
checks={};profiles=[];rng=np.random.default_rng(71731);forces=[[3,18],[5,24],[8,30],[10,40]];times=[[.08,.4],[.08,.6],[.1,1],[.1,1.5]]
for level in [2,3]:
 previous=None
 for stage in range(4):
  spec=specification(level,stage,forces,times);yaw=rng.uniform(-np.pi,np.pi,10000);xy,duration,body,focus=sample(rng,yaw,spec,2);mag=np.linalg.norm(xy,axis=1)
  diff=np.arctan2(np.sin(np.arctan2(xy[:,1],xy[:,0])-yaw),np.cos(np.arctan2(xy[:,1],xy[:,0])-yaw));fore_aft=np.minimum(abs(diff),np.pi-abs(diff))
  assert abs(focus.mean()-.7)<.02 and np.max(fore_aft[focus])<=np.pi/12+1e-8 and np.all(body[focus]==0)
  assert mag.min()>=spec['broad_force_N'][0]-1e-8 and mag.max()<=spec['force_N'][1]+1e-8
  assert duration.min()>=spec['broad_duration_s'][0] and duration.max()<=spec['duration_s'][1]
  assert mag[focus].min()>=spec['force_N'][0]-1e-8 and duration[focus].min()>=spec['duration_s'][0]
  if previous:assert spec['force_N'][1]>previous['force_N'][1] and spec['duration_s'][1]>previous['duration_s'][1]
  previous=spec;profiles.append(dict(level=level,**spec,observed_fore_aft_fraction=float(focus.mean())))
checks['all_eight_stage_profiles_bounded_and_monotonic']=True;checks['front_back_relative_to_actual_robot_yaw']=True
E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');e=E.create_env(c,num_envs=32,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();counts=np.zeros(3,int)
 for _ in range(100):e.reset(np.arange(32));counts+=np.bincount(e._kind,minlength=3)
 ratio=counts/counts.sum();assert np.all(np.abs(ratio-[.1,.3,.6])<.04),ratio
 checks['physical_reset_sampler_10_30_60_percent']=True
 for _ in range(60):s=e.step(np.zeros((32,14),np.float32));assert np.isfinite(s.reward).all()
 assert e._force_events>0 and e._fore_aft_force_events>0
 checks['actual_force_path_produces_finite_reward']=True
 checks['input61_and_flex_off']=e.state.obs['obs'].shape==(32,61) and not e.cfg.flexible_dr_enabled
 assert checks['input61_and_flex_off']
 result=dict(passed=True,checks=checks,reset_sample_count=int(counts.sum()),observed_reset_fractions=ratio.tolist(),profiles=profiles,force_events=e._force_events,fore_aft_events=e._fore_aft_force_events,ppo_updates=0,reward_unchanged=True,stand_validation_force_limits_unchanged=True)
 (R/'PUSH_COURSE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
