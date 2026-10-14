from pathlib import Path
import subprocess,sys,json,shutil,time,hashlib
from dr_suite_gate import passes,overall_success
R=Path(__file__).resolve().parent;s=json.loads((R/'status.json').read_text());history=s.get('history',[]);completed=s.get('completed_updates',0);level=3;push_stage=s.get('push_stage',1)
manifest=json.loads((R/'REPAIR_MANIFEST.json').read_text());source_total=manifest['source_lineage_updates'];latest=s.get('latest_checkpoint','block_source/final.pt');reset_goal_pending=s.get('reset_goal_pending',False);best=s.get('best_checkpoint','block_source/final.pt')
assert json.loads((R/'FINAL_GOAL_VERIFICATION.json').read_text())['passed'];assert json.loads((R/'REWARD_GATE_VERIFICATION.json').read_text())['passed'];assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
assert not s.get('user_requires_explicit_resume',False)
def status(phase,**kw):
 (R/'status.json').write_text(json.dumps(dict(phase=phase,training_running=phase=='training',user_requires_explicit_resume=False,auto_stop_on_success=False,source_lineage_updates=source_total,completed_updates=completed,lineage_compute_updates=source_total+completed,level=level,push_stage=push_stage,history=history,latest_checkpoint=latest,best_checkpoint=best,num_envs=4096,rollout_steps=20,actor_dimension=61,critic_dimension=76,flexible_joints=False,mechanical_dt=.001,DR_multiplier=2.5,training_reset_weights=[.1,.3,.6],push_sampling='50% fore/aft; 20% lateral; 30% broad',save_every=100,evaluate_every=100,reward_version='ground_recovery_state_gated_pose',success_definition='final_stand_v3',allow_ground_recovery=True,reset_goal_pending=reset_goal_pending,contact_sensor_mapping_corrected=True,concurrent_gpu_training=True,**kw),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
def capacity():
 while True:
  line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0];total,used=map(int,line.split(','))
  if total-used>=12288:return
  status('waiting_for_gpu',free_gpu_memory_MiB=total-used);time.sleep(30)
try:
 if s.get('pending_source'):
  pending=s['pending_source'];done=Path(pending['completion_file']);status('waiting_for_previous_block',pending_source=pending)
  while not done.exists():
   if not Path(f'/proc/{pending["trainer_pid"]}').exists():raise RuntimeError('Previous trainer exited without completion checkpoint')
   time.sleep(5)
  d=json.loads(done.read_text());source=done.parent/d['checkpoint'];source_total=pending['previous_total_updates']+d['updates'];shutil.copy2(source,R/'block_source/final.pt')
  manifest.update(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lineage_updates=source_total);(R/'REPAIR_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');status('refreshed_source_smoke')
  run('train.py',['--run','refreshed_source_smoke','--smoke','--resume','block_source/final.pt','--phase','settle','--level','3','--push-stage',str(push_stage),'--seed','75114'],'refreshed_source_smoke.log')
 while True:
  capacity();name=f'block_{completed//100:02d}';status('training',block=name,candidate_source=latest)
  args=['--run',name,'--updates','100','--resume',latest,'--phase','settle','--level','3','--push-stage',str(push_stage),'--seed',str(76013+completed//100)]
  if reset_goal_pending:args+=['--reset-goal-state']
  run('train.py',args,name+'.log');reset_goal_pending=False
  completed+=100;latest=name+'/final.pt';shutil.copy2(R/latest,R/f'model_{source_total+completed}.pt');status('evaluating',candidate=latest)
  run('evaluate_course.py',['--checkpoint',latest,'--tag',name+'_course','--phase','settle','--challenge-level','3'],name+'_course_eval.log')
  run('evaluate_walk_stop.py',['--checkpoint',latest,'--tag',name+'_walk_stop','--phase','settle','--challenge-level','3'],name+'_walk_stop_eval.log')
  c=json.loads((R/(name+'_course_results.json')).read_text());w=json.loads((R/(name+'_walk_stop_results.json')).read_text());d=json.loads((R/(name+'_completion.json')).read_text());accepted=passes(c,w)
  history.append(dict(success_definition='final_stand_v3',lineage_updates=source_total+completed,new_updates=completed,push_stage=push_stage,course=c['summary']['model'],groups=c['groups'],walk_stop=w['summary'],overall_success_rate=overall_success(c,w),accepted=accepted,seconds_per_update=d['elapsed_seconds']/100,std_min=d['std_min'],std_max=d['std_max']))
  previous_best=max((x['overall_success_rate'] for x in history[:-1]),default=0.)
  if overall_success(c,w)>=previous_best:
   best=latest;shutil.copy2(R/latest,R/'best.pt')
  status('block_review_complete') # Continue even after passing 90%, as requested.

except Exception as exc:status('failed',error=repr(exc));raise
