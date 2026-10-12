from pathlib import Path
import subprocess,os,time,json,signal,re,statistics,traceback
import torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';UV=str(Path.home()/'.local/bin/uv')
ENV=dict(os.environ,PYTHONPATH=str(P/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl');child=None
state={'target_updates':200,'completed_updates':0,'num_envs':4096,'mechanical_steps':20,'passive_coordinates':0,'air_time_threshold_ms':100,'from_scratch':True}
def save(**kw):
 state.update(kw,updated_unix=time.time());p=R/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(R/'status.json')
def run(args,name,training=False):
 global child
 with (R/(name+'.log')).open('x') as f:
  child=subprocess.Popen([UV,'run','--no-sync','python',*args],cwd=P,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save(child_pid=child.pid)
  while child.poll() is None:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   if training:
    s=(R/(name+'.log')).read_text(errors='replace');times=[float(v) for v in re.findall(r'Iteration time:\s*([0-9.]+)s',s)];n=len(times)
    save(completed_updates=n,last20_mean_seconds_per_update=statistics.mean(times[-20:]) if times else None,estimated_remaining_training_s=(200-n)*statistics.mean(times[-20:]) if times else None)
   time.sleep(10)
  if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}')
  save(child_pid=None)
def main():
 save(phase='training',supervisor_pid=os.getpid(),started_unix=time.time());assert json.loads((R/'audit.json').read_text())['passed']
 run(['scripts/train_rsl_rl.py','--config-name','rigid4096_100ms_200'],'rigid4096_100ms_200',True)
 summary=json.loads((R/'rigid4096_100ms_200/run_summary.json').read_text());assert summary['status']=='completed' and summary['run_completed_iterations']==200
 ck=R/'rigid4096_100ms_200/model_199.pt';saved=torch.load(ck,map_location='cpu',weights_only=False);step=int(saved['infos']['unilab_progress']['env_steps_per_env']);assert step==4800 and saved['iter']==199
 assert 'Resuming from' not in (R/'rigid4096_100ms_200.log').read_text()
 c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
 for key,stages in list(c.env.curriculum.items()):
  if key.endswith('_stages') and OmegaConf.is_list(stages):
   selected=OmegaConf.to_container([s for s in stages if int(s.step)<=step][-1]);selected['step']=0;c.env.curriculum[key]=[selected]
 c.env.curriculum.step_cap=step;cfg=R/'eval_final200.yaml';OmegaConf.save(c,cfg);tag='rigid4096_100ms_final200';common=['--config',str(cfg),'--checkpoint',str(ck),'--tag',tag,'--seed','50021','--seconds','8']
 save(phase='evaluating',completed_updates=200);run([str(R/'diagnose_support_forces.py'),*common],'eval_final200')
 save(phase='recording');run([str(R/'render_policy.py'),*common,'--command','0.30','--completed-updates','200','--dr-label','baseline DR'], 'render_final200')
 save(phase='completed',checkpoint=str(ck),video=str(R/f'videos/{tag}_grid.mp4'))
def stop(s,f):raise RuntimeError(f'Signal {s}')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:main()
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 save(phase='failed_or_stopped',child_pid=None,error=repr(exc),traceback=traceback.format_exc());raise
