"""Frozen-policy collection timing only: no optimizer or training restart."""
from pathlib import Path
import sys,time,json,argparse
import numpy as np,torch
from omegaconf import OmegaConf
ap=argparse.ArgumentParser();ap.add_argument('--repaired',action='store_true');ap.add_argument('--steps',type=int);a=ap.parse_args()
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v4_20261012';P=(R if a.repaired else B)/'project'
sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
torch.set_num_threads(2);np.random.seed(42012);torch.manual_seed(42012);E.ensure_registries();c=OmegaConf.load((R if a.repaired else B)/'TRAINING_CONFIG.yaml');c.env.discovery_phase='settle'
e=E.create_env(c,num_envs=4096,env_cfg_override=E.build_ppo_env_cfg_override(c));ck=B/'block_06/final.pt';actor=FrozenActor(ck).to('cuda:0');sd=torch.load(ck,map_location='cuda:0',weights_only=True)['actor_state_dict'];std=sd['distribution.log_std_param'].exp()
try:
 e.init_state()
 def collect(n):
  a_time=0.;e_time=0.;start=time.perf_counter()
  for _ in range(n):
   t=time.perf_counter()
   with torch.inference_mode():raw=(actor(torch.as_tensor(e.state.obs['obs'],device='cuda:0',dtype=torch.float32))+torch.randn((4096,14),device='cuda:0')*std).cpu().numpy()
   a_time+=time.perf_counter()-t;t=time.perf_counter();e.step(raw);e_time+=time.perf_counter()-t
  torch.cuda.synchronize();return dict(steps=n,total_s=time.perf_counter()-start,policy_and_transfer_s=a_time,environment_s=e_time)
 collect(8);rows=[]
 for n in ([a.steps,a.steps] if a.steps else ([24,24] if a.repaired else [24,64,24])):rows.append(collect(n));print(json.dumps(rows[-1]),flush=True)
 result=dict(num_envs=4096,control_dt=.02,mechanical_dt=.001,optimizer_updates=0,training_remains_paused=True,rows=rows,estimated_ppo_learning_s_from_real_logs=.202,warning='Collection-only frozen current policy, large learned exploration std; compiler startup excluded. Timing is wall-clock on otherwise paused robot training.')
 result['repaired_posture_environment']=a.repaired
 filename=f'REPAIRED_SPEED_{a.steps}_BENCHMARK.json' if a.repaired and a.steps else ('REPAIRED_SPEED_BENCHMARK.json' if a.repaired else 'SPEED_BENCHMARK.json')
 (R/filename).write_text(json.dumps(result,indent=2)+'\n')
finally:e.close()
