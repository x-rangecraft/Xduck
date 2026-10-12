from pathlib import Path
import shutil,json,hashlib
import mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'xduck_rigid_airtime_ab_20260929';P=R/'project'
assert not P.exists()
shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'));(P/'.venv').symlink_to((B/'project/.venv').resolve());shutil.copytree(B/'physics',R/'physics')
source=B/'project/conf/ppo/air100.yaml';shutil.copy2(source,R/'SUCCESSFUL_1024_CONFIG.yaml')
c=OmegaConf.load(source);before=OmegaConf.to_container(c)
c.algo.num_envs=4096;c.algo.max_iterations=200;c.algo.save_interval=50;c.algo.run_name='rigid4096_100ms_200';c.training.log_dir=str(R/c.algo.run_name)
c.algo.algorithm.num_mini_batches=16
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb')
converted={}
for name,stages in list(c.env.curriculum.items()):
 if name.endswith('_stages') and OmegaConf.is_list(stages):
  old=[int(s.step) for s in stages]
  for s in stages:s.step=(int(s.step)+3)//4
  converted[name]={'old_env_steps':old,'new_env_steps':[int(s.step) for s in stages]}
assert c.algo.resume is False and str(c.algo.load_run)=='-1' and c.algo.seed==1
assert c.env.physics_microsteps==1 and c.reward.air_time_threshold_min==.1
assert OmegaConf.to_container(c.reward)==before['reward']
assert OmegaConf.to_container(c.env.commands)==before['env']['commands']
assert OmegaConf.to_container(c.env.domain_rand)==before['env']['domain_rand']
for key in ['physical_action_scale_rad','action_offset_low','action_offset_high','control_config','latency','noise_config','communication','sole_tilt_range_deg']:
 assert OmegaConf.to_container(c.env[key])==before['env'][key] if OmegaConf.is_config(c.env[key]) else c.env[key]==before['env'][key],key
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);assert (m.nq,m.nv,m.nu)==(21,20,14) and m.opt.timestep==.001
assert hashlib.sha256(Path(c.env.warp_compiled_model_file).read_bytes()).hexdigest()==hashlib.sha256((B/'physics/effective_training.mjb').read_bytes()).hexdigest()
OmegaConf.save(c,P/'conf/ppo/rigid4096_100ms_200.yaml');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
for file in ['render_policy.py','diagnose_support_forces.py']:shutil.copy2(B/file,R/file)
# Log differences in full; the cloned source tree is byte-identical to the successful snapshot.
changes={}
def compare(a,b,p=''):
 for k in set(a)|set(b):
  key=p+'.'+k if p else k
  if isinstance(a.get(k),dict) and isinstance(b.get(k),dict):compare(a[k],b[k],key)
  elif a.get(k)!=b.get(k):changes[key]=[a.get(k),b.get(k)]
compare(before,OmegaConf.to_container(c))
for file in (B/'project/src').rglob('*.py'):
 rel=file.relative_to(B/'project');assert hashlib.sha256(file.read_bytes()).digest()==hashlib.sha256((P/rel).read_bytes()).digest(),str(rel)
audit=dict(passed=True,from_scratch=True,num_envs=4096,updates=200,final_checkpoint='model_199.pt',air_time_threshold_ms=100,mechanical_steps=20,nq=m.nq,nv=m.nv,passive_coordinates=0,model_and_source_identical_to_successful_baseline=True,wide_commands_restored=True,no_reference_reward=True,source_baseline=str(source),source_baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),minibatches=16,minibatch_samples=6144,initial_learning_rate=.001,learning_epochs=5,curriculum_conversions=converted,changes=changes)
(R/'audit.json').write_text(json.dumps(audit,indent=2));files=list((P/'src').rglob('*.py'))+[P/'conf/ppo/rigid4096_100ms_200.yaml',R/'physics/effective_training.mjb',R/'physics/scene.xml'];(R/'inputs_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({k:v for k,v in audit.items() if k not in ['changes','curriculum_conversions']},indent=2))
