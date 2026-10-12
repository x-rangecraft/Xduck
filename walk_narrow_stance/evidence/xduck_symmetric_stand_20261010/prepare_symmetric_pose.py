from pathlib import Path
import json,numpy as np,mujoco
R=Path(__file__).resolve().parent;B=R.parent/'xduck_direction_speed_dr_20261010'
old=np.array(json.loads((B/'NEW_STANDING_TARGET.json').read_text())['target_rad'])
projected=old.copy()
for left,right in [(0,9),(1,10),(2,11),(3,12),(4,13)]:
 projected[left]=(old[left]-old[right])/2;projected[right]=-projected[left]
projected[[7,8]]=0
# Round into a usable default and keep mirrored hip-knee-ankle pitch sum zero.
symmetric=np.array([0.,.08,.394,-.061,-.455,.019,.060,0.,0.,0.,-.08,-.394,.061,.455])
for left,right in [(0,9),(1,10),(2,11),(3,12),(4,13)]:assert symmetric[left]==-symmetric[right]
assert abs(symmetric[2]-symmetric[3]+symmetric[4])<1e-12
names=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
order=[0,1,2,3,4,9,10,11,12,13,5,6,7,8]
records=[]
m=mujoco.MjModel.from_binary_path(str(R.parent/'xduck_toeout_audit_20261009/verified_rigid_model.mjb'));d=mujoco.MjData(m)
for label,goal in [('original',old),('mirror_projection',projected),('symmetric_level',symmetric)]:
 mujoco.mj_resetDataKeyframe(m,d,m.key('STAND').id);d.qpos[3:7]=[1,0,0,0];d.qpos[7:21]=goal;mujoco.mj_forward(m,d)
 feet=[m.site(s+'_foot').id for s in ['left','right']];root=m.body('trunk_base').id;pos=d.site_xpos[feet]-d.xpos[root]
 axes=[]
 for name in names:axes.append(m.jnt_axis[m.joint(name).id].tolist())
 records.append(dict(name=label,target=goal.tolist(),joint_axes=axes,foot_site_relative_root_m=pos.tolist(),width_mm=float((pos[0,1]-pos[1,1])*1000),foot_fore_aft_difference_mm=float((pos[0,0]-pos[1,0])*1000),foot_height_difference_mm=float((pos[0,2]-pos[1,2])*1000),subtree_com_relative_root_m=(d.subtree_com[root]-d.xpos[root]).tolist()))
(R/'pose_candidates.json').write_text(json.dumps(records,indent=2))
(R/'SYMMETRIC_STANDING_TARGET.json').write_text(json.dumps(dict(target_rad=symmetric.tolist(),training_joint_order=names,motor_id_order_rad=symmetric[order].tolist(),motor_id_order_deg=np.degrees(symmetric[order]).tolist(),source='mirrored mean of previous standing pose, hip yaw/head yaw/head roll centered, rounded sagittal targets preserving zero nominal foot pitch',candidate_requires_physical_validation=True,standing599_called=False),ensure_ascii=False,indent=2))
print(json.dumps(records,indent=2))
