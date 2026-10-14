from pathlib import Path
import shutil,json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_rigid_recovery_20261012';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','.git','__pycache__','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
shutil.copytree(B/'physics',R/'physics',dirs_exist_ok=True)
f=R/'physics/scene.xml';tree=ET.parse(f);tree.getroot().find('option').set('timestep','.001');tree.write(f,encoding='unicode')
m=mujoco.MjModel.from_binary_path(str(B/'physics/effective_training.mjb'));parent=mujoco.MjModel.from_binary_path(str(B/'physics/effective_training.mjb'));m.opt.timestep=.001;mujoco.mj_saveModel(m,str(R/'physics/effective_training.mjb'));assert (m.nq,m.nv,m.nu)==(21,20,14)
for name in ['body_mass','body_inertia','dof_armature','actuator_gainprm','actuator_forcerange']:np.testing.assert_array_equal(getattr(m,name),getattr(parent,name))
for name in ['bank_stand.npz','stand_parent.pt','walk_parent.pt']:shutil.copy2(B/name,R/name)
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40';shutil.copy2(R/'scratch_balance.py',owner/'scratch_balance.py');shutil.copy2(R/'masked_stop_ppo.py',P/'src/unilab/algos/torch/masked_stop_ppo.py');f=owner/'__init__.py';s=f.read_text();line='\nfrom .scratch_balance import ScratchBalanceCfg,ScratchBalanceEnv\n';f.write_text(s if line in s else s+line)
# Generate physically grounded noisy reset states on the cold path, never in rollout.
rng=np.random.default_rng(29101);d=mujoco.MjData(m);ref=m.key_ctrl[0].copy();q0=m.key_qpos[0].copy();jids=m.actuator_trnid[:,0];qi=m.jnt_qposadr[jids];limits=m.jnt_range[jids];banks={};ranges=[(2,.015,.02,.05),(4,.03,.05,.1),(8,.06,.12,.25),(12,.1,.2,.5),(18,.15,.35,.8),(25,.2,.5,1.2)]
foot=[]
for side in ['left','right']:
 g=m.geom(side+'_foot_collision').id;k=m.geom_dataid[g];verts=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].copy();foot.append((g,verts))
for level,(tilt,jitter,velocity,angular) in enumerate(ranges):
 qs=[];vs=[];acts=[]
 for i in range(1024):
  q=q0.copy();q[qi]=np.clip(ref+rng.uniform(-jitter,jitter,14),limits[:,0]+.02,limits[:,1]-.02);roll,pitch=rng.uniform(-np.deg2rad(tilt),np.deg2rad(tilt),2)
  q[3:7]=[np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)]
  if i==0:q=q0.copy()
  d.qpos[:]=q;mujoco.mj_forward(m,d);lowest=min((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,v in foot);q[2]+=.002-lowest
  v=np.zeros(m.nv);v[:2]=rng.uniform(-velocity,velocity,2);v[3:6]=rng.uniform(-angular,angular,3);v[6:]=rng.uniform(-2*jitter,2*jitter,14)
  if i==0:v[:]=0
  qs.append(q);vs.append(v);acts.append((q[qi]-ref)/.12)
 for name,value in [('qpos',qs),('qvel',vs),('actions',acts)]:banks[f'l{level}_{name}']=np.array(value)
np.savez_compressed(R/'level_banks.npz',**banks)
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.training.task_name='XDuckScratchBalance';c.training.log_dir=str(R/'training1000');c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=c.env.scene.model_file;c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.physics_microsteps=1;c.env.handoff_bank=str(R/'bank_stand.npz');c.env.level_bank=str(R/'level_banks.npz');c.env.walk_checkpoint=str(R/'walk_parent.pt');c.env.stand_checkpoint=str(R/'stand_parent.pt');c.env.handoff_seed=29102;c.env.static_episode_fraction=1.;c.env.force_weight_stages=[[0,0.,0.],[1,.01,.04],[2,.02,.08],[3,.04,.15],[4,.06,.25],[5,.08,.35]];c.env.pose_weight_stages=[[i,.1 if i<3 else .25] for i in range(6)];c.env.smooth_weight_stages=[[i,.01 if i<3 else .025] for i in range(6)];c.env.static_fractions=[1.,1.,.9,.8,.65,.5];c.env.curriculum_min_updates=25;c.env.curriculum_window_episodes=128;c.env.curriculum_pass_fraction=.9;c.env.curriculum_max_level=5;c.env.short_push_seconds=[.08,.25];c.env.sustained_push_seconds=[.4,1.5];c.env.push_body_names=['trunk_base','yaw_roll_motion'];c.env.stop_seconds=8.;c.algo.num_envs=512;c.algo.max_iterations=1000;c.algo.save_interval=100;c.algo.seed=29100;c.algo.resume=False;c.algo.load_run=None;c.algo.resume_path=None;c.algo.policy.init_noise_std=1.;c.algo.algorithm.entropy_coef=.01;c.algo.algorithm.learning_rate=3e-4;c.algo.algorithm.schedule='adaptive';c.algo.run_name='scratch_balance_rigid_1ms';OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
(R/'physics_contract.json').write_text(json.dumps(dict(actor_dim=61,critic_dim=76,nq=m.nq,nv=m.nv,nu=m.nu,flexible_dofs=0,flexible_dr_enabled=False,mechanical_timestep_s=.001,mechanical_steps_per_control=20,control_period_s=.02,total_mass_kg=float(m.body_mass.sum()),source_physics_preserved=True,initialized_from_scratch=True,reset_bank_samples_per_level=1024,force_maximum_n_by_level=[float(x[2]*m.body_mass.sum()*9.81) for x in c.env.force_weight_stages]),indent=2)+'\n');print('Prepared fresh actor curriculum, rigid 1ms mechanics, six grounded state banks')
