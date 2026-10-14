from pathlib import Path
import json,subprocess,sys,shutil,time
R=Path(__file__).resolve().parent
assert json.loads((R/'POSTURE_REWARD_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
def status(phase,completed=0,**extra):
 (R/'status.json').write_text(json.dumps(dict(phase=phase,training_running=phase=='training',user_requires_explicit_resume=False,source_lineage_updates=2900,completed_updates=completed,lineage_compute_updates=2900+completed,requested_updates=100,flexible_joints=False,flexible_dr_enabled=False,num_envs=4096,rollout_steps=20,actor_dimension=61,mechanical_dt=.001,control_dt=.02,**extra),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if free>=12288:break
  status('waiting_for_gpu',free_gpu_memory_MiB=free);time.sleep(30)
 status('training',block='adapt_100',source_checkpoint='block_source/final.pt')
 run('train.py',['--run','adapt_100','--updates','100','--resume','block_source/final.pt','--phase','settle','--level','3','--push-stage','3','--seed','90100'],'training100.log')
 shutil.copy2(R/'adapt_100/final.pt',R/'model_3000.pt')
 status('evaluating',100,latest_checkpoint='model_3000.pt')
 run('evaluate_course.py',['--checkpoint','model_3000.pt','--tag','head3000_course','--phase','settle','--challenge-level','3'],'course_eval.log')
 run('evaluate_walk_stop.py',['--checkpoint','model_3000.pt','--tag','head3000_walk_stop','--phase','settle','--challenge-level','3'],'walk_stop_eval.log')
 run('posture_metrics.py',['baseline2900_course','head3000_course','head3000_walk_stop'],'posture_comparison.log')
 c=json.loads((R/'head3000_course_results.json').read_text());w=json.loads((R/'head3000_walk_stop_results.json').read_text());p=json.loads((R/'POSTURE_COMPARISON.json').read_text())
 status('complete',100,latest_checkpoint='model_3000.pt',groups=c['groups'],walk_stop=w['summary'],overall_success_rate=(c['summary']['model']['settled']+w['summary']['successful_stops'])/112,baseline_quiet_head_tilt_mean_deg=p[0]['quiet_head_tilt_mean_deg'],new_quiet_head_tilt_mean_deg=p[1]['quiet_head_tilt_mean_deg'],baseline_head_yaw_mean_abs_deg=p[0]['quiet_head_yaw_mean_abs_deg'],new_head_yaw_mean_abs_deg=p[1]['quiet_head_yaw_mean_abs_deg'],baseline_head_roll_mean_abs_deg=p[0]['quiet_head_roll_mean_abs_deg'],new_head_roll_mean_abs_deg=p[1]['quiet_head_roll_mean_abs_deg'],posture_report='POSTURE_COMPARISON.json')
except Exception as exc:status('failed',100 if (R/'model_3000.pt').exists() else 0,error=repr(exc));raise
