from pathlib import Path
import os,subprocess,time,json,re,signal,traceback,hashlib
import torch
R=Path(__file__).resolve().parent;P=R/'project';child=None;state={}
env=dict(os.environ,PYTHONPATH=str(P/'src'),MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
def save(**kw):
 state.update(kw,updated_unix=time.time());(R/'status.tmp').write_text(json.dumps(state,indent=2));(R/'status.tmp').replace(R/'status.json')
def run(args,tag):
 global child
 with (R/(tag+'.log')).open('a') as f:
  child=subprocess.Popen([str(Path.home()/'.local/bin/uv'),'run','--no-sync','python',*args],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(phase=tag,child_pid=child.pid,completed_updates=0)
  while child.poll() is None:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   text=(R/(tag+'.log')).read_text(errors='replace');times=[float(x) for x in re.findall(r'Iteration time:\s*([0-9.]+)s',text)]
   save(completed_updates=len(times),last_iteration_s=times[-1] if times else None,remaining_training_s=(50-len(times))*sum(times[-5:])/len(times[-5:]) if times and tag=='training' else None)
   time.sleep(5)
  if child.returncode:raise RuntimeError(f'{tag} exit {child.returncode}')
  save(child_pid=None)
def verify(path,it):
 d=torch.load(path,map_location='cpu',weights_only=False);assert d['iter']==it and d['infos']['unilab_progress']['env_steps_per_env']==(it+1)*24
 def finite(v):
  if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
  elif isinstance(v,dict):
   for x in v.values():finite(x)
  elif isinstance(v,(list,tuple)):
   for x in v:finite(x)
 finite(d)
def stop(*args):raise RuntimeError('Stopped')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 save(supervisor_pid=os.getpid(),started_unix=time.time(),additional_updates=50,source_completed=1230,mechanical_steps=320,num_envs=4096)
 audit=json.loads((R/'audit.json').read_text());ck=Path(audit['source']);assert hashlib.sha256(ck.read_bytes()).hexdigest()==audit['sha256']
 run([str(R/'test_flex.py')],'flex_tests');assert json.loads((R/'flex_tests.json').read_text())['passed']
 run([str(R/'evaluate_directions.py'),'--checkpoint',str(ck),'--tag','before_steady','--seconds','16'],'before_eval')
 run(['scripts/train_rsl_rl.py','--config-name','flex_smoke'],'smoke4096');verify(R/'smoke4096/model_1231.pt',1231)
 run(['scripts/train_rsl_rl.py','--config-name','meanonly_flex50'],'training');ck=R/'meanonly_flex50/model_1279.pt';verify(ck,1279)
 for tag,push in [('after_steady',False),('after_full_DR',True)]:
  run([str(R/'evaluate_directions.py'),'--checkpoint',str(ck),'--tag',tag,'--seconds','16',*(['--pushes'] if push else [])],tag)
 save(phase='completed',completed_updates=50,total_completed=1280,checkpoint=str(ck))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 save(phase='failed',child_pid=None,error=repr(exc),traceback=traceback.format_exc());raise
