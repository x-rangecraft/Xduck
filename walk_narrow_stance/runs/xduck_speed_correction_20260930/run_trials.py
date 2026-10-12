from pathlib import Path
import os,subprocess,time,json,re,signal,traceback
import torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';child=None
state={};env=dict(os.environ,PYTHONPATH=str(P/'src'),MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
def save(**kw):
 state.update(kw,updated_unix=time.time());(R/'status.tmp').write_text(json.dumps(state,indent=2));(R/'status.tmp').replace(R/'status.json')
def run(args,tag):
 global child
 with (R/(tag+'.log')).open('x') as f:
  child=subprocess.Popen([str(Path.home()/'.local/bin/uv'),'run','--no-sync','python',*args],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(phase=tag,child_pid=child.pid)
  while child.poll() is None:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   text=(R/(tag+'.log')).read_text(errors='replace');times=re.findall(r'Iteration time:\s*([0-9.]+)s',text)
   save(completed_updates=len(times),last_iteration_s=float(times[-1]) if times else None)
   time.sleep(5)
  if child.returncode:raise RuntimeError(f'{tag} exit {child.returncode}')
  save(child_pid=None)
def verify(name,it,steps):
 ck=R/name/f'model_{it}.pt';d=torch.load(ck,map_location='cpu',weights_only=False)
 assert d['iter']==it and d['infos']['unilab_progress']['env_steps_per_env']==steps
 def finite(v):
  if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
  elif isinstance(v,dict):
   for x in v.values():finite(x)
  elif isinstance(v,(list,tuple)):
   for x in v:finite(x)
 finite(d)
 return ck
def stop(*args):raise RuntimeError('Stopped')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 save(supervisor_pid=os.getpid(),started_unix=time.time(),planned=['control','mean_only','mean_var'],updates_each=30)
 c=OmegaConf.load(P/'conf/ppo/mean_var.yaml');c.algo.max_iterations=2;c.algo.run_name='smoke';c.training.log_dir=str(R/'smoke');c.env.speed_correction_start_update=1190
 OmegaConf.save(c,P/'conf/ppo/speed_smoke.yaml')
 if not (R/'smoke/model_1201.pt').exists():run(['scripts/train_rsl_rl.py','--config-name','speed_smoke'],'smoke')
 verify('smoke',1201,28848)
 for name in ['control','mean_only','mean_var']:
  run(['scripts/train_rsl_rl.py','--config-name',name],name);ck=verify(name,1229,29520)
  run([str(R/'evaluate_directions.py'),'--checkpoint',str(ck),'--tag',name+'_steady','--seconds','16'],name+'_eval')
 save(phase='completed',child_pid=None)
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 save(phase='failed',child_pid=None,error=repr(exc),traceback=traceback.format_exc());raise
