"""Bounded per-joint stiffness search for the supplied standing snapshot.

This is a diagnostic single-pose fit, not identification of hardware stiffness.
No IMU rotation, spring preload, joint zero, masses or motor readings are fitted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from unilab.training.xduck_standing_alignment import POLICY_JOINTS, run_case


def score(result):
    complete = result['simulation_time_s'] >= result['case']['seconds']-.02
    slow = max(result['last1s_gravity_std']) < .0005 and result.get('last1s_max_root_speed',1.) < .01
    # A grossly changed load-side posture cannot explain the supplied upright
    # standing photographs, even when the encoder channels remain unchanged.
    maximum_flex = max(abs(v) for v in result['flex_by_joint_deg'].values())
    posture_penalty = max(maximum_flex-10.,0.)
    tilt_penalty = max(abs(result['sim_roll_pitch_deg'][0])-5.,0.)
    return abs(result['roll_pitch_error_deg'][1]) + .05*abs(result['roll_pitch_error_deg'][0]) + posture_penalty + tilt_penalty + (0 if complete and slow else 20.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(exist_ok=True,parents=True)
    coefficients={n:[600.,1.] if n.startswith(('left','right')) else [75.,1.] for n in POLICY_JOINTS}
    coefficients['left_hip_pitch_joint']=[5.,1.]
    coefficients['right_hip_pitch_joint']=[1000.,1.]
    number=0
    def evaluate(co):
        nonlocal number
        number+=1
        return run_case({'name':f'fit_{number:03d}','coefficients':co,'seconds':8.,'dt':.00025},output)
    best=evaluate(coefficients)
    history=[best]
    for round_index in range(args.rounds):
        improved=False
        for name in POLICY_JOINTS:
            low=5. if ('pitch' in name and 'hip_yaw' not in name) else 20.
            high=1000. if name.startswith(('left','right')) else 300.
            current=coefficients[name][0]
            for k in sorted(set([float(np.clip(current/3,low,high)),float(np.clip(current*3,low,high))])):
                if abs(k-current)<1e-9:
                    continue
                co={n:list(v) for n,v in coefficients.items()}
                co[name][0]=k
                result=evaluate(co)
                history.append(result)
                if score(result)<score(best)-1e-4:
                    best=result
                    coefficients=co
                    improved=True
        checkpoint={'rounds_completed':round_index+1,'evaluations':number,
            'score':score(best),'best_case':best['case']['name'],
            'best_coefficients':coefficients,'best_pitch_deg':best['sim_roll_pitch_deg'][1],
            'pitch_error_deg':best['roll_pitch_error_deg'][1],
            'fitted_quantities':'Positive per-joint stiffness only; damping fixed at 1.',
            'search_bounds':'Pitch joints 5..1000 legs / 5..300 head; other joints 20..1000 legs / 20..300 head.',
            'acceptance':'Full duration; near-stationary gravity/root; maximum flex <=10deg; roll <=5deg.'}
        (output/'fit_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
        print('FIT_ROUND',json.dumps(checkpoint),flush=True)
        if not improved:
            break
    (output/'fit_history.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':
    main()
