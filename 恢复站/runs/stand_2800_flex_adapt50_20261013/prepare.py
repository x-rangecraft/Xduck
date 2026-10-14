from pathlib import Path
import shutil,json,hashlib,copy,xml.etree.ElementTree as ET
import numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v8_ground_recover_20261013';F=R.parent/'stand624_flex50_20261010'
assert not (R/'status.json').exists()
shutil.copytree(B/'project',R/'project',symlinks=True)
for name in ['pure_recovery.py','pure_reward.py','pure_push_course.py','pure_dr_course.py','walk_stop_audit.py']:
 shutil.copy2(R/name,R/'project/src/unilab/envs/locomotion/xduck_gf43x40'/name)
shutil.copy2(R/'pure_distribution.py',R/'project/src/unilab/algos/torch/pure_distribution.py')
P=R/'physics';P.mkdir();tree=ET.parse(F/'physics/scene.xml');root=tree.getroot();root.find('option').set('timestep',str(.001/16))
oldxml=ET.parse(B/'physics/scene.xml');sensor=root.find('sensor')
if sensor is None:sensor=ET.SubElement(root,'sensor')
for el in list(sensor):
 if el.get('name','').startswith(('pure_floor_found_','track_')):sensor.remove(el)
for el in oldxml.getroot().find('sensor'):
 if el.get('name','').startswith(('pure_floor_found_','track_')):sensor.append(copy.deepcopy(el))
tree.write(P/'scene.xml');m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));nom=mujoco.MjModel.from_binary_path(str(F/'physics/effective_training.mjb'));rigid=mujoco.MjModel.from_binary_path(str(B/'physics/effective_training.mjb'))
assert (m.nq,m.nv,m.nu)==(69,68,14)
for name in ['body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat']:
 for i in range(m.nbody):getattr(m,name)[i]=getattr(rigid,name)[rigid.body(m.body(i).name).id]
for name in ['geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity']:
 for i in range(m.ngeom):getattr(m,name)[i]=getattr(rigid,name)[rigid.geom(m.geom(i).name).id]
for name in ['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange','actuator_gaintype','actuator_biastype','actuator_dyntype']:
 getattr(m,name)[:]=getattr(rigid,name)
active=m.actuator_trnid[:,0];old_active=rigid.actuator_trnid[:,0]
assert [m.joint(int(i)).name for i in active]==[rigid.joint(int(i)).name for i in old_active]
qi=m.jnt_qposadr[active];vi=m.jnt_dofadr[active];oq=rigid.jnt_qposadr[old_active];ov=rigid.jnt_dofadr[old_active]
for j,oldj in zip(active,old_active):
 m.qpos0[m.jnt_qposadr[j]]=rigid.qpos0[rigid.jnt_qposadr[oldj]]
 m.dof_armature[m.jnt_dofadr[j]]=rigid.dof_armature[rigid.jnt_dofadr[oldj]]
 m.dof_damping[m.jnt_dofadr[j]]=rigid.dof_damping[rigid.jnt_dofadr[oldj]]
 m.dof_frictionloss[m.jnt_dofadr[j]]=rigid.dof_frictionloss[rigid.jnt_dofadr[oldj]]
 m.jnt_range[j]=rigid.jnt_range[oldj]
flex=np.flatnonzero(m.jnt_stiffness>0);assert len(flex)==48
for j in flex:
 oldj=nom.joint(m.joint(j).name).id;m.jnt_stiffness[j]=nom.jnt_stiffness[oldj];m.dof_damping[m.jnt_dofadr[j]]=nom.dof_damping[nom.jnt_dofadr[oldj]]
q=np.zeros(m.nq);q[:7]=rigid.key_qpos[0,:7];q[qi]=rigid.key_qpos[0,oq];m.qpos0[:7]=rigid.qpos0[:7]
for k in range(m.nkey):m.key_qpos[k]=q;m.key_ctrl[k]=q[qi]
for el in root.findall('.//key'):el.set('qpos',' '.join(map(str,q)));el.set('ctrl',' '.join(map(str,q[qi])))
tree.write(P/'scene.xml');m.opt.timestep=.001/16;mujoco.mj_saveModel(m,str(P/'effective_training.mjb'),None)
def map_bank(path,destination):
 with np.load(path) as z:
  result={}
  for key in z.files:
   value=z[key]
   if key=='qpos' or key.endswith('_qpos'):
    assert value.shape[-1]==rigid.nq
    mapped=np.broadcast_to(q,(*value.shape[:-1],m.nq)).copy();mapped[...,:7]=value[...,:7];mapped[...,qi]=value[...,oq];result[key]=mapped
   elif key=='qvel' or key.endswith('_qvel'):
    assert value.shape[-1]==rigid.nv
    mapped=np.zeros((*value.shape[:-1],m.nv));mapped[...,:6]=value[...,:6];mapped[...,vi]=value[...,ov];result[key]=mapped
   else:result[key]=value
  np.savez_compressed(destination,**result)
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');map_bank(Path(c.env.reset_bank),R/'flex_reset_bank.npz');map_bank(Path(c.env.handoff_bank),R/'flex_default_bank.npz')
c.env.reset_bank=str(R/'flex_reset_bank.npz');c.env.handoff_bank=str(R/'flex_default_bank.npz');c.env.physics_microsteps=16;c.env.flexible_dr_enabled=True;c.env.flexible_dr_fraction=.1
c.env.warp_compiled_model_file=str(P/'effective_training.mjb');c.env.scene.model_file=str(P/'scene.xml');c.env.scene.visual_model_file=c.env.scene.model_file
c.training.log_dir=str(R/'training');c.algo.run_name='stand2800_flex_adapt50';c.algo.max_iterations=50
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
(R/'block_source').mkdir();source=B/'model_2800.pt';shutil.copy2(source,R/'block_source/final.pt')
for name in ['PREPARATION.json','EXTENDED_RESET_COVERAGE.json','walker_flex_adapt_50.pt']:shutil.copy2(B/name,R/name)
manifest=dict(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lineage_updates=2800,additional_updates=50,num_envs=4096,rollout_steps=20,actor_dimension=61,passive_flexible_coordinates=48,nq=m.nq,nv=m.nv,control_dt=.02,mechanical_dt=.001/16,mechanical_substeps_per_control=320,flexible_stiffness_dr=[.9,1.1],flexible_damping_dr=[.9,1.1],general_DR_multiplier=2.5,sole_installation_error_deg=[-3,3],reward_and_action_decoder_unchanged=True,optimizer_and_normalizer_state_preserved=True)
(R/'FLEX_ADAPT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'status.json').write_text(json.dumps(dict(phase='preflight',training_running=False,user_requires_explicit_resume=False,source_lineage_updates=2800,completed_updates=0,lineage_compute_updates=2800),indent=2)+'\n');print(json.dumps(manifest),flush=True)
