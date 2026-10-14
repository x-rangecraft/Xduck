from pathlib import Path
import subprocess,sys,json,shutil,time
R=Path(__file__).resolve().parent
old=json.loads((R/'status.json').read_text());history=old.get('history',[])
done_files=sorted(R.glob('block_*_completion.json'));done=[json.loads(p.read_text()) for p in done_files]
completed=sum(x['updates'] for x in done);previous=done[-1]['checkpoint'] if done else None
phase=old.get('learner_phase',done[-1]['phase'] if done else 'discovery');level=old.get('level',0)
next_block=max([int(p.name.split('_')[1]) for p in done_files],default=-1)+1
assert json.loads((R/'REPAIR_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
assert not old.get('user_requires_explicit_resume',False)
def status(state,**kw):
 (R/'status.json').write_text(json.dumps(dict(phase=state,training_running=state=='training',user_requires_explicit_resume=False,user_authorized_new_training=True,completed_updates=completed,learner_phase=phase,level=level,history=history,teacher_or_trajectory_training=False,initialized_from_scratch=True,actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001,num_envs=4096,rollout_steps=20,entropy_type='tanh_increment_entropy',strict_pose_acceptance=True,initial_exploration_std=1.,entropy_coef=.01,checkpoint_every_updates=100,evaluation_every_updates=100,maximum_updates=600,**kw),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
def wait_for_gpu(name,updates):
 while True:
  line=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
  total,used=[int(x.strip()) for x in line.split(',')];free=total-used
  # Do not race another long-running supervisor between its training blocks.
  competitor=Path('/proc/1990736/cmdline')
  try:busy=b'head_quiet_fresh_20261012/supervise.py' in competitor.read_bytes()
  except FileNotFoundError:busy=False
  if free>=22528 and not busy:return
  status('waiting_for_gpu',block=name,block_target_updates=updates,candidate=previous,free_gpu_memory_MiB=free,required_free_gpu_memory_MiB=22528,competing_training_active=busy,reason='Keep 4096 environments; wait for the other sit/stand training to release GPU memory. No other job is interrupted.')
  time.sleep(30)
try:
 while completed<600:
  updates=min(100-completed%100,600-completed)
  name=f'block_{next_block:02d}';next_block+=1;wait_for_gpu(name,updates);status('training',block=name,block_target_updates=updates,candidate=previous)
  args=['--run',name,'--updates',str(updates),'--phase',phase,'--level',str(level)]
  if previous:args+=['--resume',previous]
  run('train.py',args,name+'.log');completed+=updates;previous=name+'/final.pt'
  shutil.copy2(R/previous,R/f'model_{completed}.pt');status('evaluating',candidate=previous)
  run('evaluate.py',['--checkpoint',previous,'--tag',name,'--phase',phase],name+'_eval.log')
  r=json.loads((R/(name+'_results.json')).read_text());d=json.loads((R/(name+'_completion.json')).read_text())
  history.append(dict(updates=completed,phase=phase,level=level,std_min=d['std_min'],std_max=d['std_max'],gain=r['rescue_gain_fraction'],rescued=r['model_rescued_reference_failures'],quiet_settled=r['quiet_settled'],summary=r['summary']))
  if phase=='discovery' and r['rescue_gain_fraction']>=.2 and len(r['model_rescued_reference_failures'])>=2:phase='settle'
  elif phase=='settle' and r['rescue_gain_fraction']>=.5 and r['quiet_settled']==r['quiet_cases']:phase='polish';level=min(level+1,3)
  elif phase=='polish' and r['rescue_gain_fraction']>=.7:level=min(level+1,3)
  if completed>=100 and max(x['gain'] for x in history)==0:
   status('stopped_for_diagnosis',candidate=previous,reason='No deterministic rescue of any fixed-reference failure at the 100-update review; do not extend blindly.');break
 else:status('budget_complete_review_required',candidate=previous)
except Exception as exc:status('failed',error=repr(exc));raise
