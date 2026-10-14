from pathlib import Path
import shutil,json
from dataclasses import asdict
import numpy as np,mujoco
from omegaconf import OmegaConf
from balance_reward import BalanceRewardCfg
R=Path(__file__).resolve().parent;B=R.parent/'stand_balance_scratch_20261012';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
for name in ['stand_parent.pt','walk_parent.pt','bank_stand.npz']:shutil.copy2(B/name,R/name)
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40'
for name in ['scratch_balance.py','balance_reward.py']:shutil.copy2(R/name,owner/name)
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.env.balance_reward=asdict(BalanceRewardCfg());c.env.level_bank=str(R/'level_banks.npz');c.env.handoff_bank=str(R/'bank_stand.npz');c.env.walk_checkpoint=str(R/'walk_parent.pt');c.env.stand_checkpoint=str(R/'stand_parent.pt');c.training.log_dir=str(R/'training');c.algo.run_name='scratch_balance_tilt_brake_v2';c.algo.resume=False;c.algo.load_run=None;c.algo.resume_path=None
assert not c.env.flexible_dr_enabled and c.env.physics_microsteps==1
# Model mechanics, actuator and base DR remain unchanged. Generate generic low-height
# and moving states by cold-path kinematics, without executing any rise policy.
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);d=mujoco.MjData(m);ref=m.key_ctrl[0].copy();q0=m.key_qpos[0].copy();qi=m.jnt_qposadr[m.actuator_trnid[:,0]];limits=m.jnt_range[m.actuator_trnid[:,0]]
feet=[]
for side in ['left','right']:
 g=m.geom(side+'_foot_collision').id;mesh=m.geom_dataid[g];feet.append((g,m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()))
rng=np.random.default_rng(31101);ranges=[(2,.015,.02,.05),(4,.03,.05,.1),(8,.06,.12,.25),(12,.1,.2,.5),(18,.15,.35,.8),(25,.2,.5,1.2)];crouch_fraction=[0,0,.1,.15,.2,.25];banks={};coverage=[]
for level,(tilt,jitter,speed,angular) in enumerate(ranges):
 qs=[];vs=[];actions=[];low_count=0
 for i in range(1024):
  q=q0.copy();pose=ref+rng.uniform(-jitter,jitter,14);low=level>=2 and rng.random()<crouch_fraction[level]
  if low:
   bend=rng.uniform(.5,1.2);pose[[2,3,4,11,12,13]]+=bend*np.array([.5,-1,.5,-.5,1,-.5])
  q[qi]=np.clip(pose,limits[:,0]+.02,limits[:,1]-.02);roll,pitch=rng.uniform(-np.deg2rad(tilt),np.deg2rad(tilt),2)
  q[3:7]=[np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)]
  if i==0:q=q0.copy();low=False
  d.qpos[:]=q;mujoco.mj_forward(m,d);bottom=min((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,v in feet);q[2]+=.002-bottom
  if q[2]<.65*q0[2]:q=q0.copy();low=False
  v=np.zeros(m.nv);v[:2]=rng.uniform(-speed,speed,2);v[3:6]=rng.uniform(-angular,angular,3);v[6:]=rng.uniform(-2*jitter,2*jitter,14)
  if low:v[2]=rng.uniform(-.15,.2);v[6:]=rng.uniform(-.4,.4,14)
  if i==0:v[:]=0
  qs.append(q);vs.append(v);actions.append((q[qi]-ref)/.12);low_count+=int(q[2]<.9*q0[2])
 for key,value in [('qpos',qs),('qvel',vs),('actions',actions)]:banks[f'l{level}_{key}']=np.array(value)
 coverage.append(dict(level=level,samples=1024,height_min_m=float(np.array(qs)[:,2].min()),height_max_m=float(np.array(qs)[:,2].max()),below_90_percent_height=low_count,vertical_speed_min=float(np.array(vs)[:,2].min()),vertical_speed_max=float(np.array(vs)[:,2].max())))
np.savez_compressed(R/'level_banks.npz',**banks);OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
(R/'STATE_COVERAGE.json').write_text(json.dumps(dict(generic_initial_states=True,rise_policy_executed=False,nominal_height_m=float(q0[2]),levels=coverage),indent=2)+'\n')
(R/'status.json').write_text(json.dumps(dict(phase='configured_paused',training_running=False,user_requires_explicit_resume=True,reward_version='tilt_brake_v2',actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001,rise_model_role='reference_only',qualified=False),indent=2)+'\n')
print(json.dumps(coverage))
