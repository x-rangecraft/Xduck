"""Success is final standing; temporary ground contact is allowed."""
import numpy as np
CRITERIA=dict(version='final_stand_v3',window_seconds=.2,minimum_standing_fraction=.8,
 height_min_offset_m=-.03,body_tilt_max_deg=20.,at_least_one_foot_support=True,
 require_body_clear_of_ground_at_end=True,temporary_ground_contact_allowed=True,
 joint_default_angles_and_velocity_not_hard_gates=True)

def pose_mask(q,head_up,default_angles,goal_height,nonfoot_ground=None):
 up=1-2*(q[...,4]**2+q[...,5]**2)
 valid=(q[...,2]>=goal_height+CRITERIA['height_min_offset_m'])&(up>np.cos(np.deg2rad(CRITERIA['body_tilt_max_deg'])))
 if nonfoot_ground is not None:valid&=~np.asarray(nonfoot_ground,dtype=bool)
 return valid

def assess_tail(pose,contact,velocity,gyro,dof_velocity,falls):
 support=np.any(contact,axis=-1);standing=np.asarray(pose,dtype=bool)&support
 metrics=dict(standing_fraction=np.mean(standing,axis=0),pose_fraction=np.mean(pose,axis=0),
  any_foot_fraction=np.mean(support,axis=0),both_feet_fraction=np.mean(np.all(contact,axis=-1),axis=0),
  mean_speed=np.mean(np.linalg.norm(velocity,axis=-1),axis=0),
  mean_gyro=np.mean(np.linalg.norm(gyro,axis=-1),axis=0),
  mean_joint_speed_rms=np.mean(np.sqrt(np.mean(dof_velocity**2,axis=-1)),axis=0))
 return metrics['standing_fraction']>=CRITERIA['minimum_standing_fraction'],metrics
