from pathlib import Path
import os,subprocess,time,json,signal,re,statistics,traceback,hashlib
import torch
R=Path(__file__).resolve().parent;P=R/'project';UV=str(Path.home()/'.local/bin/uv');ENV=dict(os.environ,PYTHONPATH=str(P/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl');child=None
state={'target_total_updates':1200,'start_completed_updates':901,'completed_total_updates':901,'additional_completed':0,'num_envs':4096,'mechanical_steps':20,'passive_coordinates':0}
def save(**kw):
 state.update(kw,updated_unix=time.time());p=R/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(R/'status.json')
def run(args,name,training=False):
 global child
 with (R/(name+'.log')).open('x') as f:
  child=subprocess.Popen([UV,'run','--no-sync','python',*args],cwd=P,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(child_pid=child.pid)
  while child.poll() is None:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   if training:
    text=(R/(name+'.log')).read_text(errors='replace');times=[float(v) for v in re.findall(r'Iteration time:\s*([0-9.]+)s',text)];n=len(times);total=901+n
    stage=(100,1.,0.) if total<300 else ((140,1.5,.45) if total<600 else ((180,2.,.45) if total<900 else (220,2.5,.45)))
    save(additional_completed=n,completed_total_updates=total,air_time_threshold_ms=stage[0],dr_target_multiplier=stage[1],reference_weight=stage[2],cadence_weight=0. if total<300 else .15,last20_mean_seconds_per_update=statistics.mean(times[-20:]) if times else None,estimated_remaining_training_s=(299-n)*statistics.mean(times[-20:]) if times else None)
   time.sleep(10)
  if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}')
  save(child_pid=None)
def finite_ck(path,iteration):
 obj=torch.load(path,map_location='cpu',weights_only=False);assert obj['iter']==iteration and obj['infos']['unilab_progress']['env_steps_per_env']==(iteration+1)*24
 def check(v):
  if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
  elif isinstance(v,dict):
   for x in v.values():check(x)
  elif isinstance(v,(list,tuple)):
   for x in v:check(x)
 check(obj)
def main():
 save(phase='resume_validation',supervisor_pid=os.getpid(),started_unix=time.time());audit=json.loads((R/'audit.json').read_text());assert audit['total_updates']==1200
 assert json.loads((R/'curriculum_tests.json').read_text())['passed']
 assert hashlib.sha256(Path(audit['source_checkpoint']).read_bytes()).hexdigest()==audit['source_sha256']
 run(['scripts/train_rsl_rl.py','--config-name','resume_smoke4096'],'resume_smoke4096')
 summary=json.loads((R/'resume_smoke4096/run_summary.json').read_text());assert summary['status']=='completed' and summary['run_completed_iterations']==2
 finite_ck(R/'resume_smoke4096/model_902.pt',902);assert 'Restored environment progress to step 21624' in (R/'resume_smoke4096.log').read_text()
 files=list((P/'src').rglob('*.py'))+[R/'TRAINING_CONFIG.yaml',R/'physics/effective_training.mjb',R/'physics/scene.xml',R/'swing_reference.npz',R/'audit.json',R/'curriculum_tests.json',R/'setup.py',R/'evaluate_directions.py',Path(__file__)]
 (R/'inputs_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
 save(phase='training',resume_validation_passed=True)
 run(['scripts/train_rsl_rl.py','--config-name','rigid4096_curriculum'],'rigid4096_curriculum',True)
 s=json.loads((R/'rigid4096_220after900/run_summary.json').read_text());assert s['status']=='completed' and s['run_completed_iterations']==299
 ck=R/'rigid4096_220after900/model_1199.pt';finite_ck(ck,1199)
 save(phase='evaluating',completed_total_updates=1200,additional_completed=299)
 for tag,push in [('final1200_steady',False),('final1200_full_DR',True)]:
  save(evaluation=tag)
  run([str(R/'evaluate_directions.py'),'--checkpoint',str(ck),'--tag',tag,'--seconds','16',*(['--pushes'] if push else [])],tag)
 # Choose a fixed input using the median over four DR replicas; retain all scan results.
 report=json.loads((R/'final1200_steady.json').read_text());candidates=[]
 for row in report['directions']:
  if not row['name'].startswith('forward') or '_' in row['name']:continue
  valid=[c for c in row['cases'] if c['failure_s'] is None and c['mean_velocity'] is not None]
  if len(valid)!=4:continue
  score=float(sum(abs(c['mean_velocity'][0]-.175)+.02*abs((c['cadence_steps_per_s'] or 0)-3.55) for c in valid)/4)
  candidates.append((score,row['command'],row['name']))
 command=min(candidates)[1] if candidates else [.175,0,0]
 save(phase='recording',selected_video_command=command)
 run([str(R/'record_final.py'),'--checkpoint',str(ck),'--command',str(command[0])],'record_final')
 save(phase='completed',checkpoint=str(ck),direction_results=[str(R/'final1200_steady.json'),str(R/'final1200_full_DR.json')])
def stop(s,f):raise RuntimeError(f'Signal {s}')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:main()
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 save(phase='failed_or_stopped',child_pid=None,error=repr(exc),traceback=traceback.format_exc());raise
