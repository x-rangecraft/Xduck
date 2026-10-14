"""Pure NumPy reward owner for quiet standing, braking and proactive recovery."""
from dataclasses import dataclass,field
import numpy as np

@dataclass
class BalanceRewardCfg:
 version: str='tilt_brake_v2'
 quiet_tilt_deg: float=6.
 recovery_tilt_deg: float=12.
 prediction_seconds: float=.2
 quiet_height_ratio: float=.95
 recovery_height_ratio: float=.8
 single_support_recovery: float=.35
 quiet_speed_scale: float=.15
 quiet_angular_scale: float=.35
 quiet_joint_speed_scale: float=.5
 linear_tracking_std: float=.2
 angular_tracking_std: float=.35
 velocity_huber_scale: float=.2
 upright_linear_weight: float=1.5
 upright_sharp_weight: float=2.5
 upright_sharp_std: float=.22360679774997896
 height_weight: float=1.
 height_std_ratio: float=.2
 height_l1_weight: float=.5
 linear_tracking_weight: float=2.
 angular_tracking_weight: float=2.
 velocity_cost_weight: float=.5
 vertical_velocity_cost_weight: float=.1
 angular_cost_weight: float=.1
 pose_weight_recovery: float=.2
 pose_weight_quiet: float=1.
 pose_std_quiet: list[float]=field(default_factory=lambda:[.1,.05,.15,.15,.1]*2)
 pose_std_recovery: list[float]=field(default_factory=lambda:[.3,.15,.4,.4,.3]*2)
 action_rate_quiet: float=.05
 action_rate_braking: float=.02
 action_rate_recovery: float=.005
 joint_velocity_quiet: float=.01
 joint_velocity_braking: float=.002
 joint_velocity_recovery: float=.0005
 head_tracking_weight: float=1.
 self_collision_cost_weight: float=1.
 limit_cost_weight: float=1.
 failure_cost_weight: float=5.
 upright_progress_weight: float=2.

 def validate(self):
  if not 0<=self.quiet_tilt_deg<self.recovery_tilt_deg<90:raise ValueError('Invalid tilt bands')
  if not 0<self.recovery_height_ratio<self.quiet_height_ratio:raise ValueError('Invalid height bands')
  if not 0<=self.prediction_seconds<=.5:raise ValueError('Invalid lookahead')
  for value in [self.quiet_speed_scale,self.quiet_angular_scale,self.quiet_joint_speed_scale,self.linear_tracking_std,self.angular_tracking_std,self.velocity_huber_scale,self.upright_sharp_std,self.height_std_ratio]:
   if value<=0:raise ValueError('Reward scales must be positive')
  for values in [self.pose_std_quiet,self.pose_std_recovery]:
   if len(values)!=10 or min(values)<=0:raise ValueError('Leg std must contain 10 positive entries')
  if not self.pose_weight_recovery>0 or not self.linear_tracking_weight>0 or not self.velocity_cost_weight>0:raise ValueError('Recovery must retain pose and zero-speed objectives')

def smoothstep(value,low,high):
 x=np.clip((value-low)/(high-low),0,1);return x*x*(3-2*x)

def score_balance(cfg,upvector,gyro,linvel,height_ratio,contact,joint_error,dof_vel,action_delta,limit_error):
 upvector=np.asarray(upvector);gyro=np.asarray(gyro);linvel=np.asarray(linvel)
 u=upvector/np.maximum(np.linalg.norm(upvector,axis=1,keepdims=True),1e-8)
 up=np.clip(u[:,2],-1,1);tilt=np.degrees(np.arccos(up))
 omega=np.linalg.norm(gyro,axis=1);axis=gyro/np.maximum(omega[:,None],1e-8);angle=omega*cfg.prediction_seconds
 # Projected up evolves as du/dt=-gyro cross u; do not mistake yaw for tipping.
 predicted=u*np.cos(angle)[:,None]-np.cross(axis,u)*np.sin(angle)[:,None]+axis*np.sum(axis*u,axis=1)[:,None]*(1-np.cos(angle))[:,None]
 predicted_tilt=np.degrees(np.arccos(np.clip(predicted[:,2],-1,1)))
 risk_tilt=np.maximum(tilt,predicted_tilt)
 angle_gate=1-smoothstep(risk_tilt,cfg.quiet_tilt_deg,cfg.recovery_tilt_deg)
 height_gate=smoothstep(height_ratio,cfg.recovery_height_ratio,cfg.quiet_height_ratio)
 contacts=np.asarray(contact,dtype=bool);support=np.where(contacts.sum(1)==0,0.,np.where(contacts.sum(1)==1,1-cfg.single_support_recovery,1.))
 posture=angle_gate*height_gate*support
 speed=np.linalg.norm(linvel[:,:2],axis=1)
 joint_speed=np.sqrt(np.mean(dof_vel**2,axis=1))
 motion_speed_sq=np.sum(linvel**2,axis=1)
 quiet=posture*np.exp(-motion_speed_sq/cfg.quiet_speed_scale**2-(omega/cfg.quiet_angular_scale)**2-(joint_speed/cfg.quiet_joint_speed_scale)**2)*contacts.all(1)
 braking=np.maximum(0.,posture-quiet);recovery=1-posture
 # Every phase has independent zero-speed rewards and continuous error costs.
 terms={
  'upright_linear':cfg.upright_linear_weight*up,
  'upright_sharp':cfg.upright_sharp_weight*np.exp(-(1-up*up)/cfg.upright_sharp_std**2)*np.clip(up,0,1),
  'height':cfg.height_weight*np.exp(-((height_ratio-1)/cfg.height_std_ratio)**2),
  'height_l1':-cfg.height_l1_weight*np.abs(height_ratio-1),
  'zero_linear_velocity':cfg.linear_tracking_weight*np.exp(-(np.sum(linvel[:,:2]**2,axis=1)+posture*linvel[:,2]**2)/cfg.linear_tracking_std**2),
  'zero_angular_velocity':cfg.angular_tracking_weight*np.exp(-np.sum(gyro**2,axis=1)/cfg.angular_tracking_std**2),
  'linear_velocity_cost':-cfg.velocity_cost_weight*(np.sqrt(1+(speed/cfg.velocity_huber_scale)**2)-1),
  'vertical_velocity_cost':-cfg.vertical_velocity_cost_weight*posture*linvel[:,2]**2,
  'angular_velocity_cost':-cfg.angular_cost_weight*(.1+.9*quiet)*np.sum(gyro**2,axis=1),
 }
 leg=[0,1,2,3,4,9,10,11,12,13]
 std=np.asarray(cfg.pose_std_quiet)[None]*quiet[:,None]+np.asarray(cfg.pose_std_recovery)[None]*(1-quiet[:,None])
 pose=np.exp(-np.mean((joint_error[:,leg]/std)**2,axis=1))
 terms['pose']=(cfg.pose_weight_recovery+(cfg.pose_weight_quiet-cfg.pose_weight_recovery)*quiet)*pose
 action_weight=cfg.action_rate_quiet*quiet+cfg.action_rate_braking*braking+cfg.action_rate_recovery*recovery
 joint_weight=cfg.joint_velocity_quiet*quiet+cfg.joint_velocity_braking*braking+cfg.joint_velocity_recovery*recovery
 terms['action_rate']=-action_weight*np.sum(action_delta**2,axis=1)
 terms['joint_velocity']=-joint_weight*np.sum(dof_vel**2,axis=1)
 terms['head_tracking']=cfg.head_tracking_weight*np.mean(np.exp(-(joint_error[:,5:9]/.5)**2),axis=1)
 terms['limits']=-cfg.limit_cost_weight*limit_error
 phase=dict(quiet=quiet,braking=braking,recovery=recovery,tilt_deg=tilt,predicted_tilt_deg=predicted_tilt)
 return terms,phase
