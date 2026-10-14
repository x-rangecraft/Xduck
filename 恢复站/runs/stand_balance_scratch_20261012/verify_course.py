from pathlib import Path
import sys,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
E.ensure_registries();np.random.seed(29201);torch.manual_seed(29201);torch.set_num_threads(2);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.static_fractions=[1.]*6
# Probe curriculum forces using the retained source actor; this does not teach the fresh actor.
e=E.create_env(c,num_envs=16,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();e._level=1;e._zero_push[:]=False;e._force_next[:]=0;seen=set();peak=0.;forces=[]
 for t in range(64):
  with torch.inference_mode():raw=e.prepare_actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
  st=e.step(raw);force=e._last_force.copy();magnitude=np.linalg.norm(force,axis=2);seen.update(np.where(magnitude>0)[1].tolist());peak=max(peak,float(magnitude.max()));forces.append(force)
  assert np.isfinite(e._backend.get_physics_state()).all()
 assert seen=={0,1} and peak>0 and peak<=.04001*e._weight_n
 # Exercise the promotion gate on a separate probe, without changing actual training state.
 e._end_scores.clear();e._end_scores.extend([False]*128);e._next_check_update=0;e.cfg.curriculum_min_updates=0;old=e._level;e.step(raw);assert e._level==old
 e._end_scores.clear();e._end_scores.extend([True]*128);e._next_check_update=0;e.step(raw);assert e._level==old+1
 result=dict(passed=True,body_force_names=list(c.env.push_body_names),observed_body_slots=sorted(seen),peak_force_n=peak,mechanical_timestep_s=.001,failed_window_did_not_promote=True,passing_probe_window_promoted=True,probe_not_used_for_training=True)
 (R/'course_verification.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(R/'course_probe_forces.npz',force=np.array(forces));print(json.dumps(result))
finally:e.close()
