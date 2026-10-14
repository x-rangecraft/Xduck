"""Bounded first-stage training and offline evaluation; never enables flexibility."""
from pathlib import Path
import subprocess,sys,json,time
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');assert not c.env.flexible_dr_enabled;c.algo.max_iterations=50;c.training.log_dir=str(R/'training50');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
def status(phase,**extra):
 (R/'status.json').write_text(json.dumps(dict(phase=phase,planned_updates=50,flexible_dofs=0,flexible_dr_enabled=False,actor_dimension=61,qualified=False,**extra),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('training');run('train.py',['--updates','50','--run','training50'],'training50.log');candidate='training50/model_49.pt';assert (R/candidate).exists();status('evaluating',candidate=candidate)
 summaries={}
 for phase,seconds,seed in [('early',1.2,28101),('late',3.6,28103)]:
  for mode in ['none','short','sustained']:
   tag=f'baseline_{phase}_{mode}';args=['--walk-seconds',str(seconds),'--seed',str(seed),'--force-mode',mode,'--tag',tag];run('evaluate.py',args,tag+'.log')
   new=f'candidate_{phase}_{mode}';run('evaluate.py',args[:-1]+[new,'--candidate',candidate,'--base-tag',tag],new+'.log')
   summaries[new]=json.loads((R/(new+'_results.json')).read_text())['summary']
 status('first_stage_evaluation_complete',candidate=candidate,summaries=summaries,awaiting_result_analysis=True)
except Exception as exc:
 status('failed',error=repr(exc));raise
