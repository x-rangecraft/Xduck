from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent;history=[];completed=0;phase='discovery';level=0;previous=None
assert json.loads((R/'PREFLIGHT.json').read_text())['passed']
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
def status(state,**kw):
 (R/'status.json').write_text(json.dumps(dict(phase=state,training_running=state=='training',user_requires_explicit_resume=False,user_authorized_new_training=True,completed_updates=completed,learner_phase=phase,level=level,history=history,teacher_or_trajectory_training=False,initialized_from_scratch=True,actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001,initial_exploration_std=1.,entropy_coef=.01,planned_first_review_updates=100,maximum_updates=600,**kw),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 for b in range(24):
  name=f'block_{b:02d}';status('training',block=name,candidate=previous);args=['--run',name,'--updates','25','--phase',phase,'--level',str(level)]
  if previous:args+=['--resume',previous]
  run('train.py',args,name+'.log');completed+=25;previous=name+'/final.pt';status('evaluating',candidate=previous)
  run('evaluate.py',['--checkpoint',previous,'--tag',name,'--phase',phase],name+'_eval.log');r=json.loads((R/(name+'_results.json')).read_text());done=json.loads((R/(name+'_completion.json')).read_text())
  history.append(dict(updates=completed,phase=phase,level=level,std_min=done['std_min'],std_max=done['std_max'],gain=r['rescue_gain_fraction'],rescued=r['model_rescued_reference_failures'],quiet_settled=r['quiet_settled'],summary=r['summary']))
  if phase=='discovery' and r['rescue_gain_fraction']>=.2 and len(r['model_rescued_reference_failures'])>=2:phase='settle'
  elif phase=='settle' and r['rescue_gain_fraction']>=.5 and r['quiet_settled']==r['quiet_cases']:phase='polish';level=min(level+1,3)
  elif phase=='polish' and r['rescue_gain_fraction']>=.7:level=min(level+1,3)
  if completed>=100 and max(x['gain'] for x in history)==0:
   status('stopped_for_diagnosis',candidate=previous,reason='No deterministic rescue of any fixed-reference failure after 100 updates; do not extend blindly.');break
 else:status('budget_complete_review_required',candidate=previous)
except Exception as exc:status('failed',error=repr(exc));raise
