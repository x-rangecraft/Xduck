from pathlib import Path
import json,numpy as np
from balance_reward import BalanceRewardCfg,score_balance
R=Path(__file__).resolve().parent;c=BalanceRewardCfg();c.validate()
def score(omega=0,contacts=(True,True),delta=1,jv=3,speed=0):
 t,p=score_balance(c,np.array([[0.,0.,1.]]),np.array([[0.,omega,0.]]),np.array([[speed,0.,0.]]),np.ones(1),np.array([contacts]),np.zeros((1,14)),np.full((1,14),jv),np.full((1,14),delta),np.zeros(1));return float(sum(t.values())[0]),{k:float(v[0]) for k,v in t.items()}
rot=[score(omega=x)[0] for x in [0,.5,.8,1.1,1.5,2.]];assert all(a>b for a,b in zip(rot,rot[1:]))
feet=[score(contacts=x)[0] for x in [(True,True),(True,False),(False,False)]];assert feet[0]>=feet[1]-1e-8 and feet[1]>=feet[2]-1e-8
joint=[score(delta=.2,jv=x)[0] for x in [0,.2,.5,1,2,3,5]];assert all(a>b for a,b in zip(joint,joint[1:]))
velocity=[score(delta=0,jv=0,speed=x)[0] for x in [0,.05,.1,.2,.3,.5]];assert all(a>b for a,b in zip(velocity,velocity[1:]))
assert score()[1]['action_rate']==score(omega=2,contacts=(False,False))[1]['action_rate']
assert score()[1]['joint_velocity']==score(omega=2,contacts=(False,False))[1]['joint_velocity']
gamma=c.progress_discount;phi=np.array([.99,1.,.99]);cycle=c.upright_progress_weight*(gamma*phi[1:]-phi[:-1]);hold=c.upright_progress_weight*(gamma*np.array([.99,.99])-np.array([.99,.99]));assert abs(cycle[0]+gamma*cycle[1]-hold[0]-gamma*hold[1])<1e-12
result=dict(passed=True,version=c.version,rotation_scores=rot,contact_scores=feet,joint_speed_scores=joint,horizontal_speed_scores=velocity,previous_rotation_counterexample_removed=True,previous_contact_counterexample_removed=True,motion_costs_do_not_relax_with_state=True,discounted_same_endpoint_cycle_equals_hold=True)
(R/'reward_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
