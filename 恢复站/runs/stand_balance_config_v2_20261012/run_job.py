from pathlib import Path
import sys,subprocess,json
R=Path(__file__).resolve().parent
for file in ['smoke_v2_completion.json','reward_verification.json','environment_verification.json']:assert (R/file).exists()
for file in ['reward_verification.json','environment_verification.json']:assert json.loads((R/file).read_text())['passed']
assert json.loads((R/'smoke_v2_completion.json').read_text())['initialized_from_scratch']
history=[]
def status(phase,**extra):(R/'status.json').write_text(json.dumps(dict(phase=phase,reward_version='tilt_brake_v2',user_requires_explicit_resume=False,training_running=phase=='training',lineage_started_from_scratch=True,planned_update_limit=1000,actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001,qualified=False,history=history,**extra),indent=2)+'\n')
def run(file,args,log):
 with (R/log).open('w') as f:subprocess.run([sys.executable,str(R/file),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 checkpoint=None;level=0;completed=0
 for block in range(10):
  name=f'block_{block:02d}';status('training',completed_updates=completed,block=name,level=level,candidate=checkpoint)
  args=['--updates','100','--run',name,'--level',str(level)]
  if checkpoint:args+=['--resume',checkpoint]
  run('train.py',args,name+'.log');done=json.loads((R/(name+'_completion.json')).read_text());checkpoint=done['checkpoint'];completed+=100;status('evaluating',completed_updates=completed,level=level,candidate=checkpoint)
  summaries={}
  for mode in ['none','short','sustained']:
   tag=f'{name}_static_{mode}';run('evaluate.py',['--checkpoint',checkpoint,'--level',str(level),'--mode',mode,'--tag',tag,'--seed',str(30301+block)],tag+'.log');summaries[tag]=json.loads((R/(tag+'_results.json')).read_text())['summary']
  if level>=2:
   for seconds in [1.2,3.6]:
    tag=f'{name}_walk_{seconds}';run('evaluate.py',['--checkpoint',checkpoint,'--level',str(level),'--mode','short','--walk-seconds',str(seconds),'--tag',tag,'--seed',str(30401+block)],tag+'.log');summaries[tag]=json.loads((R/(tag+'_results.json')).read_text())['summary']
  train_ready=done['latest_episode_success_fraction'] is not None and done['latest_episode_success_fraction']>=.9
  external_pass=all(x['eligible']>=22 and x['falls']==0 and x['settled']/x['eligible']>=.9 for x in summaries.values())
  history.append(dict(block=name,updates=completed,level=level,training_ready=train_ready,independent_pass=external_pass,summaries=summaries,candidate=checkpoint))
  if train_ready and external_pass and level<5:level+=1
 status('initial_training_budget_complete',completed_updates=completed,candidate=checkpoint,next_level=level,awaiting_analysis=True)
except Exception as exc:status('failed',error=repr(exc));raise
