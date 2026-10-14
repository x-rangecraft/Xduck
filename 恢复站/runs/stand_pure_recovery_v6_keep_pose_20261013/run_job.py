from pathlib import Path
import subprocess,sys,json,shutil,time
from dr_suite_gate import passes,overall_success,advance_course
R=Path(__file__).resolve().parent
old=json.loads((R/'status.json').read_text());history=old.get('history',[])
completed=old.get('completed_updates',0);level=old.get('level',0);push_stage=old.get('push_stage',0)
best=old.get('best_checkpoint','block_source/final.pt');latest=old.get('latest_checkpoint',best);phase='settle'
pending=old.get('pending_block');next_level_floor=old.get('next_course_level',0)
streak=old.get('pass_streak',0);rejected=old.get('consecutive_rejections',0)
assert json.loads((R/'REPAIR_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'DR_COURSE_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'DR_SUITE_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'PUSH_COURSE_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'push_course_smoke_completion.json').read_text())['actor_weight_max_change']>0
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
assert not old.get('user_requires_explicit_resume',False)
def status(state,**kw):
 (R/'status.json').write_text(json.dumps(dict(phase=state,training_running=state in ['training','finishing_active_block'],user_requires_explicit_resume=False,completed_updates=completed,source_lineage_updates=300,lineage_compute_updates=300+completed,learner_phase=phase,level=level,push_stage=push_stage,push_stage_every_updates=100,training_reset_weights=[.1,.3,.6],history=history,best_checkpoint=best,latest_checkpoint=latest,pass_streak=streak,consecutive_rejections=rejected,teacher_or_trajectory_training=False,initialized_from_scratch=False,source_model='v5/model300',actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001,num_envs=4096,rollout_steps=20,entropy_type='tanh_increment_entropy',entropy_coef=.01,checkpoint_every_updates=100,evaluation_every_updates=100,maximum_additional_updates=None,concurrent_gpu_training_authorized=True,required_free_gpu_memory_MiB=12288,completion_rule="level3_overall_success_at_least_90_percent",validation_criteria_version="window_v2",evaluation_scope="current_course_DR_and_real_walk_stop_only",dr_multipliers=[1.,1.5,2.,2.5],upgrade_overall_success_threshold=.9,**kw),indent=2)+'\n')
def run(script,args,log):
 with (R/log).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/script),*args],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,check=True)
def capacity():
 while True:
  line=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
  total,used=map(lambda x:int(x.strip()),line.split(','))
  if total-used>=12288:return
  status('waiting_for_gpu',free_gpu_memory_MiB=total-used);time.sleep(30)
try:
 while True:
  name=pending or f'block_{completed//100:02d}'
  if pending:
   status('finishing_active_block',block=name,next_course_level=next_level_floor,pending_block=pending,pending_trainer_pid=old['pending_trainer_pid'])
   while not (R/(name+'_completion.json')).exists():
    if not Path('/proc/'+str(old['pending_trainer_pid'])).exists():raise RuntimeError('Pending trainer exited without completion checkpoint')
    time.sleep(5)
  else:
   capacity();status('training',block=name,candidate_source=latest)
   run('train.py',['--run',name,'--updates','100','--phase',phase,'--level',str(level),'--push-stage',str(push_stage),'--resume',latest,'--seed',str(61013+completed//100)],name+'.log')
  completed+=100;candidate=name+'/final.pt';latest=candidate;shutil.copy2(R/candidate,R/f'adapt_{completed}.pt');status('evaluating',candidate=candidate)
  run('evaluate_course.py',['--checkpoint',candidate,'--tag',name+'_course_dr_only','--phase',phase,'--challenge-level',str(level)],name+'_course_dr_only_eval.log')
  run('evaluate_walk_stop.py',['--checkpoint',candidate,'--tag',name+'_walk_stop','--phase',phase,'--challenge-level',str(level)],name+'_walk_stop_eval.log')
  c=json.loads((R/(name+'_course_dr_only_results.json')).read_text());w=json.loads((R/(name+'_walk_stop_results.json')).read_text());d=json.loads((R/(name+'_completion.json')).read_text());accepted=passes(c,w)
  history.append(dict(additional_updates=completed,level=level,checkpoint=candidate,accepted=accepted,push_stage=push_stage,push_training_profile=d.get("push_training_profile"),overall_success_rate=overall_success(c,w),dr_multiplier=1.+.5*level,dr_only=True,course=c['summary']['model'],groups=c['groups'],walk_stop=w['summary'],std_min=d['std_min'],std_max=d['std_max'],seconds_per_update=d['elapsed_seconds']/100))
  if accepted:
   best=candidate;shutil.copy2(R/best,R/'best.pt');rejected=0;streak=0
  else:
   rejected+=1;streak=0 # Below 90%: continue learning at this same level.
  previous_level=level
  level,curriculum_complete=advance_course(level,accepted)
  push_stage=0 if level!=previous_level else min(push_stage+1,3)
  request_path=R/'USER_STAGE_TRANSITION.json'
  if request_path.exists():
   request=json.loads(request_path.read_text())
   if not request.get('applied',False) and completed+300>=request['after_total_updates']:
    assert request['target_level']==3
    level=3;push_stage=0;curriculum_complete=False
    history[-1]['user_authorized_next_level']=3
    history[-1]['pose_gate_overridden_for_transition']=True
    request.update(applied=True,applied_after_total_updates=completed+300,source_checkpoint=candidate)
    request_path.write_text(json.dumps(request,indent=2)+'\n')
  if pending:
   level=max(level,next_level_floor);pending=None
  if curriculum_complete:
   status("curriculum_complete",candidate=candidate,reason="Level3 DR and real walk-stop overall success reached 90 percent");break
  status('block_review_complete',candidate=candidate)
except Exception as exc:status('failed',error=repr(exc));raise
