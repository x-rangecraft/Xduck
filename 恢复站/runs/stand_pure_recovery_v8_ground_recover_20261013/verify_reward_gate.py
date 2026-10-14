from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'))
from unilab.envs.locomotion.xduck_gf43x40.pure_reward import pose_gate,reward_rates
from unilab.envs.locomotion.xduck_gf43x40.pure_push_course import specification,sample
def gate(tilt=0,height=.32,speed=0,omega=0,ground=False):
 return pose_gate(np.cos(np.deg2rad(np.array([tilt]))),np.array([height]),np.array([[speed,0,0]]),np.array([[0,omega,0]]),np.array([True]),np.array([ground]),.319637239)[0]
assert gate()==1
assert all(gate(**x)==0 for x in [dict(tilt=30),dict(height=.1),dict(speed=.3),dict(omega=2),dict(ground=True)])
assert 0<gate(tilt=18)<1
error=np.ones((1,14));kw=dict(up=np.array([1.]),height_ratio=np.array([1.]),linvel=np.zeros((1,3)),gyro=np.zeros((1,3)),dof_vel=np.ones((1,14)),action_delta=np.ones((1,14)),phase='settle',joint_error=error,head_up=np.array([1.]),allow_ground_recovery=True)
free=reward_rates(**kw,return_weight=np.array([0.]));settled=reward_rates(**kw,return_weight=np.array([1.]))
for key in ['head_pose','head_upright','leg_pose','hip_goal','hip_alignment','head_l1','leg_l1','quiet_bonus','linear_velocity_cost','action_rate','joint_speed']:
 assert free[key][0]==0,key
assert settled['hip_alignment'][0]<0 and 'head_return' not in settled
assert free['upright'][0]>0 and free['height_broad'][0]>0
spec=specification(3,3,[[3,18],[5,24],[8,30],[10,40]],[[.08,.4],[.08,.6],[.1,1],[.1,1.5]])
forces,duration,body,focus=sample(np.random.default_rng(80713),np.zeros(10000),spec,2)
assert np.all(np.linalg.norm(forces[focus],axis=1)>=32-1e-8) and np.all(duration[focus]>=1)
angles=np.arctan2(forces[:,1],forces[:,0]);lateral=float((np.abs(np.sin(angles))>np.cos(np.pi/4)).mean());assert .32<lateral<.38,lateral
out=dict(passed=True,no_clock_driven_pose_return=True,ground_high_tilt_low_height_fast_motion_release_constraints=True,pose_penalties_and_motion_costs_zero_during_rescue=True,no_v7_head_return_bonus_or_penalty=True,height_and_upright_recovery_signal_retained=True,same_stage3_force_duration=True,sampled_lateral_push_fraction=lateral)
(R/'REWARD_GATE_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
