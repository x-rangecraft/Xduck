from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'))
from unilab.envs.locomotion.xduck_gf43x40.direction_speed import motion_metrics,COMMANDS
velocity=np.c_[.65*COMMANDS[:,:2],np.zeros(6)];yaw=COMMANDS[:,2]
moving,axis,speed,turn,target=motion_metrics(COMMANDS,velocity,yaw)
assert moving.tolist()==[True,True,False,False,True,True]
assert np.all(speed[moving]>.19) and np.all(turn[~moving]>.39)
np.testing.assert_allclose(target[1],[-.195,0],atol=1e-9)
_,_,bad_speed,bad_turn,_=motion_metrics(COMMANDS,-velocity,-yaw)
assert np.all(bad_speed[moving]<0) and np.all(bad_turn[~moving]<0)
assert np.all(np.where(moving,speed>.045,turn>.08))
(R/'direction_objective_checks.json').write_text(json.dumps(dict(forward_backward_symmetric=True,left_right_symmetric=True,legal_reverse_not_stall=True,pure_turn_not_requiring_forward_motion=True),indent=2));print('COMMAND_ALIGNED_OBJECTIVE_CHECKS_PASS')
