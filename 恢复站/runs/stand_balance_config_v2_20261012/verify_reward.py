from pathlib import Path
import json
import numpy as np
from balance_reward import BalanceRewardCfg,score_balance
R=Path(__file__).resolve().parent;cfg=BalanceRewardCfg();cfg.validate()
def case(speed=0,tilt=0,pitch_rate=0,yaw_rate=0,height=1,contacts=(True,True),action=0,vertical_speed=0,joint_speed=0):
 a=np.deg2rad(tilt);up=np.array([[-np.sin(a),0,np.cos(a)]],float);gyro=np.array([[0,pitch_rate,yaw_rate]],float);lin=np.array([[speed,0,vertical_speed]],float)
 terms,phase=score_balance(cfg,up,gyro,lin,np.array([height]),np.array([contacts]),np.zeros((1,14)),np.full((1,14),joint_speed),np.full((1,14),action),np.zeros(1))
 return float(sum(terms.values())[0]),{k:float(v[0]) for k,v in terms.items()},{k:float(v[0]) for k,v in phase.items()}
rows=[]
for speed in [0,.05,.1,.2,.3,.5]:
 score,terms,phase=case(speed=speed);rows.append(dict(speed_m_s=speed,score=score,terms=terms,phase=phase))
assert all(rows[i]['score']>rows[i+1]['score'] for i in range(len(rows)-1))
assert rows[3]['terms']['zero_linear_velocity']>.7 and rows[3]['terms']['linear_velocity_cost']<-.2
stationary=case();moving=case(speed=.3);tipping=case(tilt=2,pitch_rate=1.5);recovering=case(tilt=20,pitch_rate=-1.);yaw=case(yaw_rate=2.);rising=case(height=.7);single=case(contacts=(True,False))
assert stationary[2]['quiet']>.99
assert moving[2]['braking']>.95 and moving[2]['quiet']<.05
assert tipping[2]['recovery']>.99 and tipping[2]['predicted_tilt_deg']>12
assert recovering[2]['recovery']>.99
assert yaw[2]['recovery']<1e-8 and yaw[2]['braking']>.99
assert rising[2]['recovery']>.99 and rising[1]['height_l1']<0
assert single[2]['quiet']==0
vertical_brake=case(vertical_speed=.2);leg_brake=case(joint_speed=1.)
assert vertical_brake[2]['braking']>.8 and vertical_brake[2]['quiet']<.2
assert leg_brake[2]['braking']>.95 and leg_brake[2]['quiet']<.05
quiet_action=case(action=1)[1]['action_rate'];recovery_action=case(tilt=20,action=1)[1]['action_rate'];assert abs(quiet_action/recovery_action-10)<1e-6
# Angle transition continuity: weights change smoothly across the two tilt bands.
for boundary in [6.,12.]:
 left=case(tilt=boundary-1e-4)[2];right=case(tilt=boundary+1e-4)[2];assert max(abs(left[k]-right[k]) for k in ['quiet','braking','recovery'])<1e-6
result=dict(passed=True,reward_version=cfg.version,stationary_vs_drift_scores=rows,quiet_action_sum_cost=quiet_action,recovery_action_sum_cost=recovery_action,small_tilt_fast_tipping=tipping[2],upright_yaw_braking=yaw[2],low_height_recovery=rising[2],single_support=single[2],vertical_motion_braking=vertical_brake[2],joint_motion_braking=leg_brake[2],tests=['drift reward strictly decreases with speed','zero-speed signal remains at 0.2m/s','tilt transition continuity','small tilt rapid tipping reacts early','yaw is braking, not tipping','low height cannot count as quiet stand','one foot support cannot count as quiet stand','sum-based action-rate cost relaxes during recovery'])
(R/'reward_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['stationary_vs_drift_scores','tests']}))
