"""Materialize archived legacy sources and relocate paths; never starts training."""
from pathlib import Path
import argparse,json,tarfile,shutil,hashlib
PACKAGE=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(archive,dest):
 dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive) as t:
  for member in t.getmembers():
   target=(dest/member.name).resolve()
   if not target.is_relative_to(dest.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe archive entry: '+member.name)
  t.extractall(dest,filter='data')
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--stage',default='all');p.add_argument('--verify-only',action='store_true');a=p.parse_args();manifest=json.loads((PACKAGE/'runtimes/snapshot_manifest.json').read_text());keys=list(manifest) if a.stage=='all' else [a.stage]
 if a.workspace.exists() and any(a.workspace.iterdir()):raise ValueError('Use a new empty workspace; existing directories are never overwritten')
 a.workspace.mkdir(parents=True,exist_ok=True);ws=a.workspace.resolve();verified={}
 for key in keys:
  desc=manifest[key];run=ws/desc['run_folder'];shutil.copytree(PACKAGE/'runs'/desc['run_folder'],run)
  extract(PACKAGE/'runtimes'/desc['base'],run/'project');extract(PACKAGE/'runtimes'/desc['overlay'],run/'project')
  for name in desc['deleted']:
   q=run/'project'/name
   if q.exists():q.unlink()
  actual={str(q.relative_to(run/'project')):digest(q) for q in (run/'project').rglob('*') if q.is_file()}
  if actual!=desc['hashes']:raise ValueError('Runtime snapshot mismatch for '+key)
  verified[key]=dict(files=len(actual),source_hashes_verified=True)
  physics='flexible' if key in ['04_flex1279','07_flex_final'] else 'rigid';extract(PACKAGE/'physics'/f'{physics}.tar.gz',run/'physics')
 # Historical scripts reference sibling experiments. Preserve these inputs too.
 for support in (PACKAGE/'support').iterdir():shutil.copytree(support,ws/support.name,dirs_exist_ok=True)
 extract(PACKAGE/'physics/reward_audit_rigid.tar.gz',ws/'xduck_reward_audit_20261009/rigid_physics')
 for q in (PACKAGE/'checkpoints').glob('*.pt'):shutil.copy2(q,ws/q.name)
 aliases={
 '00_model_199.pt':['xduck_rigid4096_100ms_200_20260930/rigid4096_100ms_200/model_199.pt'],
 '01_model_900.pt':['xduck_rigid4096_curriculum_20260930/rigid4096_curriculum/model_900.pt'],
 '02_model_1199.pt':['xduck_rigid4096_220after900_20260930/rigid4096_220after900/model_1199.pt'],
 '03_model_1229.pt':['xduck_speed_correction_20260930/mean_only/model_1229.pt'],
 '04_model_1279.pt':['xduck_meanonly_flex320_50_20260930/meanonly_flex50/model_1279.pt','xduck_clean_natural_gait_20261010/clean_parent.pt','xduck_direction_speed_dr_20261010/clean_parent.pt','xduck_r2_flex_adapt100_20261010/clean_parent.pt'],
 '05_clean_roll_seed.pt':['xduck_clean_natural_gait_20261010/clean_roll_seed.pt'],
 '05_clean_r3_80.pt':['xduck_clean_natural_gait_20261010/ppo_clean_r3_run/model_80.pt','xduck_direction_speed_dr_20261010/accepted_parent.pt'],
 '06_direction_150.pt':['xduck_direction_speed_dr_20261010/ppo_direction_run/model_150.pt'],
 '06_backward_bootstrap.pt':['xduck_direction_speed_dr_20261010/backward_bootstrap.pt'],
 '06_direction_r2_179.pt':['xduck_direction_speed_dr_20261010/ppo_direction_r2_run/model_179.pt','xduck_r2_flex_adapt100_20261010/parent_r2.pt'],
 '07_flex_model_50.pt':['xduck_r2_flex_adapt100_20261010/ppo_flex100/model_50.pt']}
 for source,names in aliases.items():
  for name in names:
   q=ws/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PACKAGE/'checkpoints'/source,q)
 for folder in ['xduck_clean_natural_gait_20261010','xduck_direction_speed_dr_20261010','xduck_r2_flex_adapt100_20261010']:
  run=ws/folder;run.mkdir(exist_ok=True)
  for name in ['xc_walking_v3_policy.py','clean_parent.onnx']:shutil.copy2(PACKAGE/'policy'/name,run/name)
 shutil.copy2(PACKAGE/'policy/clean_r3_80.onnx',ws/'xduck_clean_natural_gait_20261010/clean_r3_80.onnx')
 for name in ['parent_r2.onnx','flex_adapt_50.onnx']:shutil.copy2(PACKAGE/'policy'/name,ws/'xduck_r2_flex_adapt100_20261010'/name)
 # Restore the R2 reward/anchor owner overwritten by later rejected experiments.
 run=ws/'xduck_direction_speed_dr_20261010';final=ws/'xduck_r2_flex_adapt100_20261010'
 if (run/'project').exists():
  shutil.copy2(run/'direction_speed_r2_restored.py',run/'project/src/unilab/envs/locomotion/xduck_gf43x40/direction_speed.py')
  shutil.copy2(run/'direction_speed_r2_restored.py',run/'direction_speed.py')
  shutil.copy2(final/'masked_anchor.py',run/'project/src/unilab/algos/torch/masked_anchor.py')
 if not a.verify_only:
  for q in ws.rglob('*'):
   if q.is_file() and q.suffix in ['.py','.yaml','.yml','.json','.toml','.xml','.md','.txt']:
    try:text=q.read_text()
    except UnicodeError:continue
    updated=text.replace('/home/xingchen/Xduck/',str(ws)+'/')
    if updated!=text:q.write_text(updated)
 (ws/'RESTORE_REPORT.json').write_text(json.dumps(dict(verified=verified,source_snapshots_verified_before_relocation=True,r2_owner_restored_from_known_r2_reward=True,paths_relocated=not a.verify_only,training_started=False),indent=2));print('Restored',len(keys),'runtime stages; no training started:',ws)
if __name__=='__main__':main()
