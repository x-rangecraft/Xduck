from pathlib import Path
import shutil,json,hashlib,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'xduck_rigid4096_curriculum_20260930';P=R/'project'
assert not P.exists()
shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'));(P/'.venv').symlink_to((B/'project/.venv').resolve());shutil.copytree(B/'physics',R/'physics');shutil.copy2(B/'swing_reference.npz',R/'swing_reference.npz')
f=P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=f.read_text();old='self._cfg.reward_config.air_time_threshold_min=.1 if iteration<300 else (.14 if iteration<600 else .18)';new='self._cfg.reward_config.air_time_threshold_min=.1 if iteration<300 else (.14 if iteration<600 else (.18 if iteration<900 else .22))';assert s.count(old)==1;s=s.replace(old,new);compile(s,str(f),'exec');f.write_text(s)
ck=B/'rigid4096_curriculum/model_900.pt';saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['iter']==900 and saved['infos']['unilab_progress']['env_steps_per_env']==21624
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.algo.load_run=str(ck);c.algo.max_iterations=299;c.algo.run_name='rigid4096_220after900';c.training.log_dir=str(R/c.algo.run_name)
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.reference_path=str(R/'swing_reference.npz')
OmegaConf.save(c,P/'conf/ppo/rigid4096_curriculum.yaml');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
smoke=OmegaConf.create(OmegaConf.to_container(c));smoke.algo.max_iterations=2;smoke.algo.run_name='resume_smoke4096';smoke.training.log_dir=str(R/'resume_smoke4096');OmegaConf.save(smoke,P/'conf/ppo/resume_smoke4096.yaml')
for name in ['evaluate_directions.py','record_final.py','supervisor.py','test_curriculum.py']:shutil.copy2(B/name,R/name)
p=R/'supervisor.py';s=p.read_text();s=s.replace("'start_completed_updates':200,'completed_total_updates':200", "'start_completed_updates':901,'completed_total_updates':901").replace('total=200+n','total=901+n').replace('(180,2.5,.45)','(220,2.5,.45)').replace('(1000-n)','(299-n)').replace("R/'resume_smoke4096/model_201.pt',201","R/'resume_smoke4096/model_902.pt',902").replace('Restored environment progress to step 4800','Restored environment progress to step 21624').replace("s['run_completed_iterations']==1000","s['run_completed_iterations']==299").replace('additional_completed=1000','additional_completed=299').replace("R/'rigid4096_curriculum/model_1199.pt'","R/'rigid4096_220after900/model_1199.pt'")
s=s.replace("R/'setup.py',R/'add_cadence.py',R/'evaluate_directions.py'", "R/'setup.py',R/'evaluate_directions.py'")
# Preserve the original resumptive configuration name, but use the new run directory.
s=s.replace("str(r)","str(r)")
# The summary path must follow the changed algo.run_name.
s=s.replace("R/'rigid4096_curriculum/run_summary.json'","R/'rigid4096_220after900/run_summary.json'")
compile(s,str(p),'exec');p.write_text(s)
p=R/'test_curriculum.py';s=p.read_text().replace('(21600,180,2.5,.45)','(21600,220,2.5,.45)').replace('(120000,180,2.5,.45)','(120000,220,2.5,.45)');p.write_text(s)
a=json.loads((B/'audit.json').read_text());a.update(source_checkpoint=str(ck),source_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),start_completed_updates=901,additional_updates=299,actual_first_new_update_index=901,old_180ms_branch_preserved=True)
a['stages'][-1]['air_time_ms']=220
(R/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps({k:a[k] for k in ['total_updates','additional_updates','start_completed_updates','source_checkpoint','stages']},indent=2))
