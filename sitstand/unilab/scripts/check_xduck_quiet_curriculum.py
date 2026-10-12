"""Gate curriculum progression using completion and full-motion measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from unilab.envs.locomotion.xduck_gf43x40.acceptance import quiet_curriculum_gate

p=argparse.ArgumentParser()
p.add_argument('--baseline-geometry',type=Path,required=True)
p.add_argument('--baseline-trajectory',type=Path,required=True)
p.add_argument('--candidate-geometry',type=Path,required=True)
p.add_argument('--candidate-trajectory',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def summary(geometry,trajectory):
    d=json.loads(geometry.read_text());r=np.load(trajectory);out={}
    for c in ('descent','rise'):
        rows=[v for v in d['rows'] if v['case']==c]
        poses=np.concatenate([r[f'{c}_{s}_qpos'] for s in (6401,7401)],axis=1)
        tilt=np.concatenate([r[f'{c}_{s}_tilt'] for s in (6401,7401)],axis=1)
        excursion=np.rad2deg(np.ptp(poses[:,:,12:16],axis=0))
        out[c]=dict(successes=sum(v['successes'] for v in rows),falls=sum(v['falls'] for v in rows),
                    neck_excursion_deg=float(np.median(excursion[:,0])),
                    head_joint_excursion_median_deg=np.median(excursion,axis=0).tolist(),
                    body_peak_deg=float(np.rad2deg(np.median(tilt.max(axis=0)))),
                    body_peak_max_deg=float(np.rad2deg(tilt.max())))
    return out
b=summary(a.baseline_geometry,a.baseline_trajectory);c=summary(a.candidate_geometry,a.candidate_trajectory)
passed,checks=quiet_curriculum_gate(b,c)
result=dict(stage1_passed=passed,checks=checks,baseline=b,candidate=c,
            rule='At least90% of source successes, no additional falls or body peak median worsening, seated neck excursion down15%, rise body peak median down10%. Full trajectories include original reset noise; not endpoint-only acceptance.')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
