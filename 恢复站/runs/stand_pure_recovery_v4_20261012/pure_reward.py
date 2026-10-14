"""Offline candidate reward. Not registered in a training environment."""
import numpy as np

GAMMA=.995
PHASES={
 'discovery':dict(quiet=0.,action_rate=0.,joint_speed=0.,angular_speed=0.),
 'settle':dict(quiet=1.,action_rate=.005,joint_speed=0.,angular_speed=0.),
 'polish':dict(quiet=2.,action_rate=.01,joint_speed=.001,angular_speed=.01),
}

def smoothstep(x,a,b):
 x=np.clip((x-a)/(b-a),0.,1.);return x*x*(3-2*x)

def upvector_from_q(q):
 w,x,y,z=np.asarray(q)[...,3:7].T
 return np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1)

def potential(upvector,height_ratio,gyro):
 u=np.asarray(upvector);up=u[...,2];omega=np.linalg.norm(gyro,axis=-1)
 # Conservative tipping-risk proxy, not a reward for a particular rescue path.
 axis=gyro/np.maximum(omega[...,None],1e-8);a=.15*omega
 predicted=u*np.cos(a)[...,None]-np.cross(axis,u)*np.sin(a)[...,None]+axis*np.sum(axis*u,axis=-1)[...,None]*(1-np.cos(a))[...,None]
 tilt=np.arccos(np.clip(up,-1,1));predicted_tilt=np.arccos(np.clip(predicted[...,2],-1,1))
 risk=np.clip(np.maximum(tilt,predicted_tilt)/(np.pi/2),0,1)
 return 2*np.clip(up,-1,1)+smoothstep(height_ratio,.45,.8)-.5*risk*risk

def reward_rates(up,height_ratio,linvel,gyro,dof_vel,action_delta,phase):
 cfg=PHASES[phase];up=np.asarray(up);omega2=np.sum(gyro*gyro,axis=-1)
 joint_rms2=np.mean(dof_vel*dof_vel,axis=-1)
 quiet=smoothstep(up,np.cos(np.deg2rad(15)),np.cos(np.deg2rad(5)))*smoothstep(height_ratio,.65,.8)
 quiet*=np.exp(-np.sum(linvel*linvel,axis=-1)/.05**2-omega2/.3**2-joint_rms2/.3**2)
 return dict(alive=np.ones_like(up),upright=2*np.clip(up,0,1),height=smoothstep(height_ratio,.45,.8),
  quiet_bonus=cfg['quiet']*quiet,
  action_rate=-cfg['action_rate']*np.sum(action_delta*action_delta,axis=-1),
  joint_speed=-cfg['joint_speed']*np.sum(dof_vel*dof_vel,axis=-1),
  angular_speed=-cfg['angular_speed']*omega2)

def rescore_trace(q,velocity,gyro,dof_velocity,target,nominal_height,first_failure_index,phase,initial_q,initial_gyro):
 up=1-2*(q[:,4]**2+q[:,5]**2);height=q[:,2]/nominal_height
 delta=np.diff(target,axis=0,prepend=target[:1])/.12
 terms=reward_rates(up,height,velocity,gyro,dof_velocity,delta,phase)
 phi=potential(upvector_from_q(q),height,gyro)
 initial_phi=potential(upvector_from_q(np.asarray(initial_q)[None,:]),np.asarray([initial_q[2]/nominal_height]),np.asarray(initial_gyro)[None,:])[0]
 prev=np.r_[initial_phi,phi[:-1]]
 next_phi=phi.copy();failure=np.zeros(len(phi),bool)
 if first_failure_index is not None:
  failure[first_failure_index]=True;next_phi[first_failure_index]=0
 rewards=.02*sum(terms.values())+GAMMA*next_phi-prev-10*failure
 end=len(phi) if first_failure_index is None else first_failure_index+1
 return float(np.dot(rewards[:end],GAMMA**np.arange(end)))
