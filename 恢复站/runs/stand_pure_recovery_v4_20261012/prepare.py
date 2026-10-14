from pathlib import Path
import sys,shutil,json,hashlib,xml.etree.ElementTree as ET
from dataclasses import fields
import numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_balance_push_v3_20261012';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','.git','__pycache__','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40'
for name in ['pure_recovery.py','pure_reward.py']:shutil.copy2(R/name,owner/name)
shutil.copy2(R/'pure_distribution.py',P/'src/unilab/algos/torch/pure_distribution.py')
init=owner/'__init__.py';line='\nfrom .pure_recovery import PureRecoveryCfg,PureRecoveryEnv\n'
if line not in init.read_text():init.write_text(init.read_text()+line)
sys.path.insert(0,str(P/'src'))
from unilab.envs.locomotion.xduck_gf43x40.pure_recovery import PureRecoveryCfg
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');old=OmegaConf.to_container(c.env,resolve=True)
allowed={f.name for f in fields(PureRecoveryCfg)};env={k:v for k,v in old.items() if k in allowed}
physics=R/'physics';physics.mkdir(exist_ok=True);source=Path(old['scene']['model_file']);m0=mujoco.MjModel.from_binary_path(old['warp_compiled_model_file']);tree=ET.parse(source);root=tree.getroot();geoms=list(root.find('worldbody').iter('geom'));assert len(geoms)==m0.ngeom
sensor=root.find('sensor');names=[]
foot_bodies={int(m0.geom_bodyid[m0.geom(side+'_foot_collision').id]) for side in ['left','right']}
for i,g in enumerate(geoms):
 # Existing foot sensors use the ankle/foot subtree: ankle collision meshes are
 # part of the load-bearing foot assembly and touch in nominal standing.
 if not (m0.geom_contype[i] or m0.geom_conaffinity[i]) or i==m0.geom('floor').id or int(m0.geom_bodyid[i]) in foot_bodies:continue
 name=g.get('name') or f'pure_collision_geom_{i}';g.set('name',name);sn=f'pure_floor_found_{i}';names.append(sn)
 ET.SubElement(sensor,'contact',dict(name=sn,geom1='floor',geom2=name,data='found',num='1',reduce='none'))
tree.write(physics/'scene.xml');m=mujoco.MjModel.from_xml_path(str(physics/'scene.xml'))
# Copy calibrated compiled plant arrays; adding sensors must not change mechanics.
for attr in ['body_mass','body_inertia','dof_armature','dof_damping','dof_frictionloss','jnt_stiffness','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange','actuator_gaintype','actuator_biastype','actuator_dyntype','geom_friction']:
 getattr(m,attr)[:]=getattr(m0,attr)
m.opt.timestep=m0.opt.timestep
for attr in ['body_mass','body_inertia','body_pos','body_quat','jnt_axis','jnt_pos','jnt_range','dof_armature','actuator_ctrlrange','actuator_gainprm','key_qpos','key_ctrl']:
 np.testing.assert_allclose(getattr(m,attr),getattr(m0,attr),atol=1e-9)
assert (m.nq,m.nv,m.nu)==(21,20,14);assert not np.any(m.jnt_stiffness>0);assert len(names)>0
mujoco.mj_saveModel(m,str(physics/'effective_training.mjb'),None)
expected=np.array([0,.08,.394,-.061,-.455,.019,.06,0,0,0,-.08,-.394,.061,.455]);np.testing.assert_allclose(m.key_ctrl[0],expected,atol=1e-5)
d=mujoco.MjData(m);q0=m.key_qpos[0].copy();ref=m.key_ctrl[0].copy();qi=m.jnt_qposadr[m.actuator_trnid[:,0]];limits=m.jnt_range[m.actuator_trnid[:,0]];feet=[]
for side in ['left','right']:
 g=m.geom(side+'_foot_collision').id;mesh=m.geom_dataid[g];feet.append((g,m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()))
def make_state(axis='pitch',angle=0,speed=0,omega=0,jitter=0,bend=0,rng=None):
 q=q0.copy();v=np.zeros(m.nv)
 if jitter:q[qi]+=rng.uniform(-jitter,jitter,14)
 q[qi[[2,3,4,11,12,13]]]+=bend*np.array([.5,-1,.5,-.5,1,-.5]);q[qi]=np.clip(q[qi],limits[:,0]+.02,limits[:,1]-.02)
 a=np.deg2rad(angle);q[3:7]=[np.cos(a/2),np.sin(a/2) if axis=='roll' else 0,np.sin(a/2) if axis=='pitch' else 0,0]
 sign=1 if angle>=0 else -1;v[0 if axis=='pitch' else 1]=speed*sign*(1 if axis=='pitch' else -1);v[4 if axis=='pitch' else 3]=omega*sign
 mujoco.mj_resetData(m,d);d.qpos[:]=q;mujoco.mj_forward(m,d);bottom=min((verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,verts in feet);q[2]+=.006-bottom
 if q[2]<.6*q0[2]:return make_state(axis,angle,speed,omega,0,0,rng)
 return q,v
rng=np.random.default_rng(41021);data={};coverage=[]
for level,(maxangle,maxspeed,maxomega,jitter) in enumerate([(10,.25,.7,.06),(15,.35,1.,.09),(22,.5,1.3,.12),(30,.6,1.5,.16)]):
 qs=[];vs=[];kinds=[]
 for i in range(1024):
  u=rng.random();kind=0 if u<.2 else (1 if u<.7 else 2);axis='pitch' if rng.random()<.5 else 'roll';angle=rng.uniform(2,maxangle)*(1 if rng.random()<.5 else -1) if kind==1 else 0
  q,v=make_state(axis,angle,rng.uniform(.03,maxspeed) if kind==1 else 0,rng.uniform(.1,maxomega) if kind==1 else 0,jitter if kind==1 else .01,rng.uniform(.1,.4) if kind==1 and rng.random()<.15 else 0,rng)
  if kind==1:v[6:]=rng.uniform(-.4,.4,14)
  qs.append(q);vs.append(v);kinds.append(kind)
 for k,x in [('qpos',qs),('qvel',vs),('kind',kinds),('fixture_force',np.zeros((1024,2))),('fixture_duration',np.zeros(1024))]:data[f'l{level}_{k}']=np.asarray(x)
 coverage.append(dict(level=level,quiet=sum(np.array(kinds)==0).item(),moving=sum(np.array(kinds)==1).item(),push=sum(np.array(kinds)==2).item(),max_tilt_deg=maxangle))
cases=[]
for _ in range(2):cases.append(dict(name='quiet',axis='pitch',angle=0,speed=0,omega=0,kind=0,force=[0,0],duration=0))
for axis in ['pitch','roll']:
 for angle,speed,omega in [(6,.05,.15),(10,.2,.5)]:
  for sign in [1,-1]:cases.append(dict(name=f'{axis}_{angle}_{sign}',axis=axis,angle=angle*sign,speed=speed,omega=omega,kind=1,force=[0,0],duration=0))
for _ in range(2):cases.append(dict(name='crouch',axis='pitch',angle=0,speed=0,omega=0,kind=1,bend=.3,force=[0,0],duration=0))
for xy in [[16,0],[-16,0],[0,16],[0,-16]]:cases.append(dict(name=f'push_{xy}',axis='pitch',angle=0,speed=0,omega=0,kind=2,force=xy,duration=.4))
rows=[]
for case in cases:
 for repeat in range(2):
  q,v=make_state(case['axis'],case['angle'],case['speed'],case['omega'],0,case.get('bend',0),rng);rows.append((q,v,case['kind'],case['force'],case['duration']))
for k,index in [('qpos',0),('qvel',1),('kind',2),('fixture_force',3),('fixture_duration',4)]:data['paired_'+k]=np.asarray([row[index] for row in rows])
np.savez_compressed(R/'reset_bank.npz',**data);np.savez(R/'analytic_default_bank.npz',qpos=q0[None,:],qvel=np.zeros((1,m.nv)),actions=np.zeros((1,14)))
env.update(reset_bank=str(R/'reset_bank.npz'),handoff_bank=str(R/'analytic_default_bank.npz'),scene=dict(model_file=str(physics/'scene.xml'),visual_model_file=str(physics/'scene.xml'),fragment_files=[]),warp_compiled_model_file=str(physics/'effective_training.mjb'),contact_sensor_names=names,decoder='incremental',discovery_level=0,discovery_phase='discovery',discovery_seed=41021,episode_seconds=8.,discovery_gamma=.995)
env['max_episode_seconds']=8.;env['flexible_dr_enabled']=False;env['reference_path']=None;env['stage_schedule_enabled']=False
# Keep sole +/-3deg. Use half-width physical DR initially, then broaden later.
dr=env['domain_rand'];dr['motor']['torque_scale_range']=[.9,1.1];dr['motor']['friction_scale_range']=[.9,1.1];dr['motor']['response_time_scale_range']=[.9,1.1]
dr['foot_friction_range']=[.7,1.3];dr['dof_armature_multiplier_range']=[.9,1.1];dr['kp_multiplier_range']=[.9,1.1];dr['kd_multiplier_range']=[.9,1.1];dr['mass_inertia_scale_range']=[.95,1.05]
for key in ['com_range_stages','head_com_range_stages']:
 for stage in env['curriculum'][key]:stage['value']*=.5
c.env=env;c.training.task_name='XDuckPureRecovery';c.algo.num_envs=2048;c.algo.num_steps_per_env=64;c.algo.max_iterations=25;c.algo.save_interval=25;c.algo.resume=False;c.algo.load_run=None;c.algo.resume_path=None;c.algo.seed=41021;c.algo.run_name='pure_incremental_recovery_v4';c.algo.policy.init_noise_std=1.
c.algo.actor.distribution_cfg=dict(class_name='unilab.algos.torch.pure_distribution:PureGaussian',init_std=1.,std_type='log')
c.algo.critic=dict(class_name='rsl_rl.models.MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=True)
c.algo.algorithm.class_name='unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO'
for key in ['exploration_std_min','exploration_std_max','continuity_weight']:
 if key in c.algo.algorithm:del c.algo.algorithm[key]
c.algo.algorithm.entropy_coef=.01;c.algo.algorithm.gamma=.995;c.algo.algorithm.learning_rate=.0003
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
(R/'PREPARATION.json').write_text(json.dumps(dict(mass_kg=float(m.body_mass.sum()),mechanical_dt=m.opt.timestep,flex_joints=0,nonfoot_floor_sensors=len(names),mechanical_array_parity=True,new_stance=ref.tolist(),reset_coverage=coverage,paired_cases=cases,no_expert_checkpoints_or_actions=True),indent=2)+'\n')
(R/'status.json').write_text(json.dumps(dict(phase='preflight',training_running=False,user_authorized_new_training=True,user_requires_explicit_resume=False,completed_updates=0,initialized_from_scratch=True,teacher_or_trajectory_training=False,actor_dim=61,flexible_dofs=0,mechanical_timestep_s=.001),indent=2)+'\n');print('Prepared pure recovery owner',len(names),'floor sensors',coverage,flush=True)
