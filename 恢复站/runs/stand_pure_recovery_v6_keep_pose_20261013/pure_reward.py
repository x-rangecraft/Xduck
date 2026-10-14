"""Recovery plus measured stand endpoint; no fixed recovery path."""
import numpy as np
GAMMA=.995
LEGS=[0,1,2,3,4,9,10,11,12,13]
HIPS=[0,1,9,10]
LEG_STD=np.array([.35,.20,.55,.60,.40]*2)
HIP_STD=np.array([.10,.08,.10,.08])
PHASES={
 'discovery':dict(quiet=0.,action_rate=0.,joint_speed=0.,angular_speed=0.,sharp=.5,head=.15,head_world=.15,legs=.15,hip=.05,hip_goal=.15,head_l1=.05,leg_l1=.02,height_l1=.5,linear_cost=0.),
 'settle':dict(quiet=2.,action_rate=.005,joint_speed=.001,angular_speed=0.,sharp=1.5,head=.75,head_world=.5,legs=.75,hip=.4,hip_goal=2.,head_l1=.2,leg_l1=.1,height_l1=2.,linear_cost=.15),
 'polish':dict(quiet=3.,action_rate=.01,joint_speed=.002,angular_speed=.01,sharp=2.,head=1.,head_world=.75,legs=1.,hip=.6,hip_goal=3.,head_l1=.3,leg_l1=.15,height_l1=3.,linear_cost=.3),
}
def smoothstep(x,a,b):
 x=np.clip((x-a)/(b-a),0.,1.);return x*x*(3-2*x)
def upvector_from_q(q):
 w,x,y,z=np.asarray(q)[...,3:7].T
 return np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1)
def stand_pose_mask(height,goal_height,up,head_up,joint_error):
 return (np.abs(height-goal_height)<.02)&(up>np.cos(np.deg2rad(10)))&(head_up>np.cos(np.deg2rad(15)))&(np.max(np.abs(joint_error[:,5:9]),axis=1)<.15)&(np.max(np.abs(joint_error[:,[0,9]]),axis=1)<.10)&(np.max(np.abs(joint_error[:,[1,10]]),axis=1)<.08)&(np.sqrt(np.mean(joint_error[:,LEGS]**2,axis=1))<.25)
def potential(upvector,height_ratio,gyro,goal_ratio=1.):
 u=np.asarray(upvector);up=u[...,2];omega=np.linalg.norm(gyro,axis=-1);axis=gyro/np.maximum(omega[...,None],1e-8);a=.15*omega
 predicted=u*np.cos(a)[...,None]-np.cross(axis,u)*np.sin(a)[...,None]+axis*np.sum(axis*u,axis=-1)[...,None]*(1-np.cos(a))[...,None]
 risk=np.maximum(np.arccos(np.clip(up,-1,1)),np.arccos(np.clip(predicted[...,2],-1,1)))/(np.pi/2)
 height_progress=1-np.minimum(np.abs(height_ratio/goal_ratio-1)/.5,1)
 return 2*np.clip(up,-1,1)+height_progress-.5*np.clip(risk,0,1)**2
def reward_rates(up,height_ratio,linvel,gyro,dof_vel,action_delta,phase,joint_error,head_up,goal_ratio=1.,nominal_height=.32):
 cfg=PHASES[phase];up=np.asarray(up);herror=(height_ratio-goal_ratio)*nominal_height;omega2=np.sum(gyro*gyro,axis=-1);jrms2=np.mean(dof_vel*dof_vel,axis=-1)
 hb=np.exp(-(herror/.06)**2);hs=np.exp(-(herror/.025)**2)
 body=np.exp(-(1-up*up)/.05)*np.clip(up,0,1)
 head=np.exp(-np.mean((joint_error[:,5:9]/.35)**2,axis=1))
 legs=np.exp(-np.mean((joint_error[:,LEGS]/LEG_STD)**2,axis=1))
 head_world=np.exp(-(1-head_up*head_up)/.12)*np.clip(head_up,0,1)
 # Separate dense goal terms provide signal even when the completion product is
 # near zero. Wrong height/head/leg posture cannot collect full quiet credit.
 hip_error=np.max(np.abs(joint_error[:,HIPS])/HIP_STD,axis=1)
 hip_goal=np.exp(-.5*hip_error**2)
 goal=hs*body*head*legs*head_world*hip_goal
 quiet=goal*np.exp(-np.sum(linvel*linvel,axis=-1)/.08**2-omega2/.4**2-jrms2/.5**2)
 speed=np.linalg.norm(linvel[:,:2],axis=1)
 return dict(alive=np.ones_like(up),upright=2*np.clip(up,0,1),upright_sharp=cfg['sharp']*body,
  height_broad=hb,height_sharp=(.5 if phase=='discovery' else 1.)*hs,height_l1=-cfg['height_l1']*np.abs(herror)/(goal_ratio*nominal_height),
  head_pose=cfg['head']*head,head_upright=cfg['head_world']*head_world,leg_pose=cfg['legs']*legs,
  hip_goal=cfg['hip_goal']*hip_goal,
  hip_alignment=-cfg['hip']*hip_error,
  head_l1=-cfg['head_l1']*np.mean(np.abs(joint_error[:,5:9])/.35,axis=1),leg_l1=-cfg['leg_l1']*np.mean(np.abs(joint_error[:,LEGS])/LEG_STD,axis=1),
  quiet_bonus=cfg['quiet']*quiet,linear_velocity_cost=-cfg['linear_cost']*(np.sqrt(1+(speed/.2)**2)-1),
  action_rate=-cfg['action_rate']*np.sum(action_delta*action_delta,axis=-1),joint_speed=-cfg['joint_speed']*np.sum(dof_vel*dof_vel,axis=-1),angular_speed=-cfg['angular_speed']*omega2)
