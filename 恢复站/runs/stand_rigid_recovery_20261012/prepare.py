from pathlib import Path
import shutil,json,xml.etree.ElementTree as ET
import mujoco,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand650_handoff_boost_20261012';F=R.parent/'stand624_flex50_20261010';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
(R/'physics').mkdir(exist_ok=True)
old=mujoco.MjModel.from_binary_path(str(F/'physics/effective_training.mjb'));tree=ET.parse(F/'physics/scene.xml');root=tree.getroot();root.find('compiler').set('meshdir',str(F/'physics/meshes'))
qi=old.jnt_qposadr[old.actuator_trnid[:,0]];vi=old.jnt_dofadr[old.actuator_trnid[:,0]];removed=[]
for parent in root.iter():
 for joint in list(parent):
  if joint.tag=='joint' and float(joint.get('stiffness','0'))>0:
   removed.append(joint.get('name'));parent.remove(joint)
assert len(removed)==48
for key in root.findall('.//key'):
 q=np.fromstring(key.get('qpos'),sep=' ');key.set('qpos',' '.join(map(str,np.r_[q[:7],q[qi]])))
 if key.get('qvel'):key.set('qvel',' '.join(map(str,np.zeros(20))))
sensor=root.find('sensor')
for body in root.findall('.//body'):
 label=body.get('name')
 if label:
  ET.SubElement(sensor,'framepos',name='track_pos_w_'+label,objtype='body',objname=label)
  ET.SubElement(sensor,'framequat',name='track_quat_w_'+label,objtype='body',objname=label)
xml=R/'physics/scene.xml';tree.write(xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(xml));assert (m.nq,m.nv,m.nu)==(21,20,14)
for name in ['body_mass','body_inertia','body_ipos','geom_pos','geom_quat','geom_friction','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']:
 dst=getattr(m,name);src=getattr(old,name)
 if name.startswith('geom_'):
  for i in range(old.ngeom):
   label=old.geom(i).name;new_id=m.geom(label).id;dst[new_id]=src[i]
 else:
  assert dst.shape==src.shape,(name,dst.shape,src.shape);dst[:]=src
m.dof_armature[:6]=old.dof_armature[:6];m.dof_armature[6:]=old.dof_armature[vi];m.dof_damping[:]=0;m.dof_frictionloss[:]=0;m.opt.timestep=old.opt.timestep
mujoco.mj_saveModel(m,str(R/'physics/effective_training.mjb'))
with np.load(F/'bank_stand.npz') as z:np.savez_compressed(R/'bank_stand.npz',qpos=np.c_[z['qpos'][:,:7],z['qpos'][:,qi]],qvel=np.c_[z['qvel'][:,:6],z['qvel'][:,vi]],actions=z['actions'])
for name in ['stand_parent.pt','walk_parent.pt']:shutil.copy2(B/name,R/name)
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40'
f=owner/'rigid_gpu.py';s=f.read_text().replace('assert len(self._flex_joint_ids)==48','assert len(self._flex_joint_ids)==(48 if cfg.flexible_dr_enabled else 0)');f.write_text(s)
f=owner/'stand_flex.py';s=f.read_text();s=(B/'project/src/unilab/envs/locomotion/xduck_gf43x40/stand_flex.py').read_text();start=s.index('        spread=env.cfg.flexible_dr_fraction');end=s.index('        return plan',start);s=s[:start]+'        if env.cfg.flexible_dr_enabled:\n'+''.join('    '+line+'\n' for line in s[start:end].splitlines())+s[end:];f.write_text(s)
shutil.copy2(R/'rigid_recovery.py',owner/'rigid_recovery.py');f=owner/'__init__.py';s=f.read_text();line='\nfrom .rigid_recovery import RigidRecoveryCfg,RigidRecoveryEnv\n';f.write_text(s if line in s else s+line)
shutil.copy2(R/'masked_stop_ppo.py',P/'src/unilab/algos/torch/masked_stop_ppo.py')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.training.task_name='XDuckRigidRecovery';c.env.scene.model_file=str(xml);c.env.scene.visual_model_file=str(xml);c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.flexible_dr_enabled=False;c.env.flexible_dr_fraction=0.;c.env.handoff_bank=str(R/'bank_stand.npz');c.env.walk_checkpoint=str(R/'walk_parent.pt');c.env.stand_checkpoint=str(R/'stand_parent.pt');c.env.stand_slew_rad=0.;c.env.rollout_steps=64;c.env.stop_seconds=10.;c.env.failure_tilt_deg=70.;c.env.minimum_root_height_ratio=.45;c.env.extra_target_rate_cost=0.;c.env.settled_bonus=0.;c.env.domain_rand.velocity_pushes=False;c.env.force_weight_stages=[[0,.02,.06],[50,.04,.12],[150,.06,.18]];c.env.force_duration_range=[.12,1.2];c.env.force_interval_range=[2.,4.];c.env.pose_weight_stages=[[0,.1],[150,.25],[250,.5]];c.env.smooth_weight_stages=[[0,.01],[150,.025],[250,.05]];c.env.static_episode_fraction=.35
c.algo.num_envs=256;c.algo.max_iterations=300;c.algo.save_interval=25;c.algo.algorithm.continuity_weight=0.;c.algo.algorithm.learning_rate=5e-5;c.algo.seed=26101;c.env.handoff_seed=26102;c.training.log_dir=str(R/'training300')
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
contract=dict(nq=m.nq,nv=m.nv,nu=m.nu,passive_flexible_joints=0,removed_joints=removed,total_mass_kg=float(m.body_mass.sum()),parent_total_mass_kg=float(old.body_mass.sum()),body_mass_inertia_equal=bool(np.array_equal(m.body_mass,old.body_mass) and np.array_equal(m.body_inertia,old.body_inertia)),active_armature_equal=bool(np.array_equal(m.dof_armature[6:],old.dof_armature[vi])),mechanical_timestep_s=m.opt.timestep,nominal_root_height_m=float(d.qpos[2]),actor_dim=61,flexible_dr_enabled=False,sole_installation_angle_dr_deg=[-3,3],force_unit='fraction of measured model weight (mass*g)',external_force_body='trunk_base',parent='original model650',updates_planned=300)
assert contract['body_mass_inertia_equal'] and contract['active_armature_equal'];(R/'physics_contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps(contract))
# Owner-layer implementation of the existing SimBackend.apply_body_force contract.
f=P/'src/unilab/base/backend/mjwarp/xduck_gf.py';s=f.read_text();needle='    def get_dr_capabilities(self):'
method='''    def apply_body_force(self,body_ids,force):
        ids=np.asarray(body_ids,dtype=np.int32).reshape(-1)
        value=np.asarray(force,dtype=np.float32)
        if value.shape!=(self._num_envs,len(ids),3) or not np.isfinite(value).all():
            raise ValueError('Invalid world-frame external force')
        pending=np.zeros((self._num_envs,self._cpu_model.nbody,6),dtype=np.float32)
        pending[:,ids,:3]=value
        self._device_data.xfrc_applied.assign(pending)

'''
if 'def apply_body_force' not in s:s=s.replace(needle,method+needle);f.write_text(s)
