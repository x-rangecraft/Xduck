from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'xduck_direction_speed_dr_20261010';sys.path.insert(0,str(BASE))
from engine import *
from unilab.utils.rotation import np_quat_mul
np.random.seed(1690091);torch.manual_seed(1690091);e=env(128)
try:
 e.init_state();e.set_autoreset(False);phi=e._backend.get_physics_state()
 random={k:getattr(e._backend._device_model,k).numpy().copy() for k in FIELDS}
 random.update(qpos=phi[:,1:22].copy(),qvel=phi[:,22:42].copy(),encoder_bias=e._encoder_bias.copy(),imu_rotation=e._imu_rotation.copy(),sole_angles=e._sole_angles.copy(),sole_delta_quat=e._sole_delta_quat.copy())
 m=mujoco.MjModel.from_binary_path(str(BASE.parent/'xduck_reward_audit_20261009/verified_rigid_model.mjb'))
 nominal={k:np.asarray(getattr(m,k)).copy() for k in FIELDS};nominal.update(qpos=random['qpos'][0].copy(),qvel=np.zeros(20),encoder_bias=np.zeros(14),imu_rotation=np.eye(3),sole_angles=np.zeros(2),sole_delta_quat=np.array([[1.,0,0,0],[1.,0,0,0]]))
 rows=[nominal];labels=['nominal'];groups=['nominal'];descriptions=[{}]
 def append(name,desc,apply):
  d={k:v.copy() for k,v in nominal.items()};apply(d);rows.append(d);labels.append(name);groups.append('endpoint');descriptions.append(desc)
 def sole(d,angles):
  d['sole_angles']=np.array(angles)
  for j,g in enumerate(e._sole_geom_ids):
   theta=angles[j];dq=np.r_[np.cos(theta/2),np.sin(theta/2)*e._sole_axes[j]];d['sole_delta_quat'][j]=dq;v=d['geom_pos'][g];u=dq[1:];d['geom_pos'][g]=v+2*np.cross(u,np.cross(u,v)+dq[0]*v);d['geom_quat'][g]=np_quat_mul(dq,d['geom_quat'][g])
 def imu(axis,deg):
  theta=np.deg2rad(deg);a=np.eye(3)[axis];K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]);return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
 for value in [.25,1.75]:append('friction_'+str(value),{'foot_friction':value},lambda d,v=value:d['geom_friction'].__setitem__((e._foot_geom_ids,0),v))
 for value in [.75,1.25]:append('armature_'+str(value),{'armature_ratio':value},lambda d,v=value:d.__setitem__('dof_armature',nominal['dof_armature']*v))
 for value in [.875,1.125]:
  def mass(d,v=value):d['body_mass'][e._trunk_body_id]*=v;d['body_inertia'][e._trunk_body_id]*=v
  append('trunk_mass_'+str(value),{'trunk_mass_inertia_ratio':value},mass)
 for body,ids in [('trunk',[e._trunk_body_id]),('head',e._head_body_ids)]:
  for axis in range(3):
   for sign in [-1,1]:
    def com(d,ids=ids,axis=axis,sign=sign):d['body_ipos'][ids,axis]+=sign*.015675
    append(body+'_com_'+str(axis)+'_'+str(sign),{body+'_com_axis':axis,'offset_m':sign*.015675},com)
 for angles in [(-3,-3),(-3,3),(3,-3),(3,3)]:append('sole_'+str(angles),{'sole_angles_deg':angles},lambda d,a=angles:sole(d,np.deg2rad(a)))
 for axis in range(3):
  for sign in [-1,1]:append('imu_'+str(axis)+'_'+str(sign),{'imu_axis':axis,'angle_deg':sign*6},lambda d,a=axis,s=sign:d.__setitem__('imu_rotation',imu(a,s*6)))
 for sign in [-1,1]:
  def combined(d,s=sign):
   d['geom_friction'][e._foot_geom_ids,0]=.25 if s<0 else 1.75;d['dof_armature']*=.75 if s<0 else 1.25;v=.875 if s<0 else 1.125;d['body_mass'][e._trunk_body_id]*=v;d['body_inertia'][e._trunk_body_id]*=v;d['body_ipos'][[e._trunk_body_id,*e._head_body_ids]]+=s*.015675;sole(d,np.deg2rad([s*3,-s*3]));d['imu_rotation']=imu(1,s*6)
  append('combined_'+str(sign),{'simultaneous_endpoints':True,'sign':sign},combined)
 for j in range(128):rows.append({k:v[j].copy() for k,v in random.items()});labels.append('random_'+str(j));groups.append('random');descriptions.append({'seed':1690091,'index':j})
 bank={k:np.array([r[k] for r in rows]) for k in nominal};np.savez_compressed(OUT/'dr_bank.npz',**bank,labels=np.array(labels),groups=np.array(groups))
 manifest=dict(total=len(rows),random_count=128,endpoint_count=groups.count('endpoint'),nominal_count=1,seed=1690091,course_multiplier=float(e._dr_multiplier),noise_config=dict(level=float(e._cfg.noise_config.level),gyro=e._cfg.noise_config.scale_gyro,gravity=e._cfg.noise_config.scale_gravity,joint_angle=e._cfg.noise_config.scale_joint_angle,joint_velocity=e._cfg.noise_config.scale_joint_vel),imu_delay_ms=list(e._cfg.latency.imu_delay_ms),joint_velocity_delay_ms=e._cfg.latency.joint_velocity_delay_ms,foot_friction_sample_range=[float(random['geom_friction'][:,e._foot_geom_ids,0].min()),float(random['geom_friction'][:,e._foot_geom_ids,0].max())],sole_sample_range_deg=np.degrees([random['sole_angles'].min(),random['sole_angles'].max()]).tolist(),labels=labels,groups=groups,descriptions=descriptions,not_all_continuous_combinations=True,flexible_and_motor_parameter_dr=False,external_pushes=False)
 (OUT/'dr_bank_manifest.json').write_text(json.dumps(manifest,indent=2));print('DR_BANK',len(rows),'random128 endpoints',groups.count('endpoint'),flush=True)
finally:e.close()
