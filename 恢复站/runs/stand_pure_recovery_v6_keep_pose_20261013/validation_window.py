"""Episode completion tolerates short corrections; physical falls remain failures."""
import numpy as np

CRITERIA=dict(version='window_v2',window_seconds=1.,minimum_pose_fraction=.8,
 minimum_both_feet_fraction=.8,mean_speed_max_m_s=.08,mean_gyro_max_rad_s=.4,
 mean_joint_speed_rms_max_rad_s=.5,height_error_max_m=.03,body_tilt_max_deg=12.,
 head_tilt_max_deg=20.,head_joint_error_max_rad=.20,hip_yaw_error_max_rad=.12,
 hip_roll_error_max_rad=.10,leg_error_rms_max_rad=.30)

def pose_mask(q,head_up,default_angles,goal_height):
 err=q[...,7:]-default_angles;up=1-2*(q[...,4]**2+q[...,5]**2)
 return ((np.abs(q[...,2]-goal_height)<CRITERIA['height_error_max_m'])
  &(up>np.cos(np.deg2rad(CRITERIA['body_tilt_max_deg'])))
  &(head_up>np.cos(np.deg2rad(CRITERIA['head_tilt_max_deg'])))
  &(np.max(np.abs(err[...,5:9]),axis=-1)<CRITERIA['head_joint_error_max_rad'])
  &(np.max(np.abs(err[...,[0,9]]),axis=-1)<CRITERIA['hip_yaw_error_max_rad'])
  &(np.max(np.abs(err[...,[1,10]]),axis=-1)<CRITERIA['hip_roll_error_max_rad'])
  &(np.sqrt(np.mean(err[...,[0,1,2,3,4,9,10,11,12,13]]**2,axis=-1))<CRITERIA['leg_error_rms_max_rad']))

def assess_tail(pose,contact,velocity,gyro,dof_velocity,falls):
 metrics=dict(pose_fraction=np.mean(pose,axis=0),both_feet_fraction=np.mean(np.all(contact,axis=-1),axis=0),
  mean_speed=np.mean(np.linalg.norm(velocity,axis=-1),axis=0),
  mean_gyro=np.mean(np.linalg.norm(gyro,axis=-1),axis=0),
  mean_joint_speed_rms=np.mean(np.sqrt(np.mean(dof_velocity**2,axis=-1)),axis=0))
 success=(~np.asarray(falls,dtype=bool)
  &(metrics['pose_fraction']>=CRITERIA['minimum_pose_fraction'])
  &(metrics['both_feet_fraction']>=CRITERIA['minimum_both_feet_fraction'])
  &(metrics['mean_speed']<CRITERIA['mean_speed_max_m_s'])
  &(metrics['mean_gyro']<CRITERIA['mean_gyro_max_rad_s'])
  &(metrics['mean_joint_speed_rms']<CRITERIA['mean_joint_speed_rms_max_rad_s']))
 return success,metrics
