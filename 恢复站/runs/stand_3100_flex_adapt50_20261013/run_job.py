from pathlib import Path
import json,subprocess,sys,shutil,time
R=Path(__file__).resolve().parent
assert json.loads((R/'PREFLIGHT.json').read_text())['passed']
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
def status(phase,completed=0,**extra):
 (R/'status.json').write_text(json.dumps(dict(phase=phase,training_running=phase=='training',user_requires_explicit_resume=False,source_lineage_updates=3100,completed_updates=completed,lineage_compute_updates=3100+completed,requested_updates=50,flexible_joints=True,flexible_dr_enabled=True,flexible_dr_fraction=.1,num_envs=4096,rollout_steps=20,actor_dimension=61,mechanical_dt=.001/16,control_dt=.02,**extra),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 while True:
  line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
  if int(line)>=16384:break
  status('waiting_for_gpu',free_gpu_memory_MiB=int(line));time.sleep(30)
 status('training',block='adapt_50',source_checkpoint='block_source/final.pt')
 run('train.py',['--run','adapt_50','--updates','50','--resume','block_source/final.pt','--phase','settle','--level','3','--push-stage','3','--seed','95113'],'training50.log')
 shutil.copy2(R/'adapt_50/final.pt',R/'model_3150.pt');shutil.copy2(R/'adapt_50/final.pt',R/'flex_adapt_50.pt')
 status('evaluating',50,latest_checkpoint='model_3150.pt')
 run('evaluate_course.py',['--checkpoint','block_source/final.pt','--tag','baseline3100_flex_course','--phase','settle','--challenge-level','3'],'baseline_flex_eval.log')
 run('evaluate_course.py',['--checkpoint','model_3150.pt','--tag','flex3150_course','--phase','settle','--challenge-level','3'],'course_eval.log')
 run('evaluate_walk_stop.py',['--checkpoint','model_3150.pt','--tag','flex3150_walk_stop','--phase','settle','--challenge-level','3'],'walk_stop_eval.log')
 run('posture_metrics.py',['baseline3100_flex_course','flex3150_course','flex3150_walk_stop'],'posture_comparison.log')
 p=json.loads((R/'POSTURE_COMPARISON.json').read_text())
 c=json.loads((R/'flex3150_course_results.json').read_text());w=json.loads((R/'flex3150_walk_stop_results.json').read_text())
 status('complete',50,latest_checkpoint='model_3150.pt',groups=c['groups'],walk_stop=w['summary'],overall_success_rate=(c['summary']['model']['settled']+w['summary']['successful_stops'])/112,baseline_flex_head_yaw_mean_abs_deg=p[0]['quiet_head_yaw_mean_abs_deg'],new_head_yaw_mean_abs_deg=p[1]['quiet_head_yaw_mean_abs_deg'],baseline_flex_head_roll_mean_abs_deg=p[0]['quiet_head_roll_mean_abs_deg'],new_head_roll_mean_abs_deg=p[1]['quiet_head_roll_mean_abs_deg'],new_head_tilt_mean_deg=p[1]['quiet_head_tilt_mean_deg'],quiet_head_axes_p95_within2deg_cases=sum(x['head_axes_p95_within2deg'] for x in p[1]['quiet_cases']),posture_report='POSTURE_COMPARISON.json')
except Exception as exc:status('failed',50 if (R/'model_3150.pt').exists() else 0,error=repr(exc));raise
