from pathlib import Path
import json, numpy as np, mujoco
R=Path(__file__).resolve().parent
rows=[]
for total in [2400,2500]:
 for profile in ['legacy','repaired']:
  tag=f'diag{total}_{profile}';d=json.loads((R/(tag+'_results.json')).read_text())
  rows.append(dict(model=total,action_profile=profile,groups=d['groups'],successes=d['summary']['model']['settled'],cases=32))
a=np.load(R/'diag2400_legacy_trace.npz');b=np.load(R/'diag2400_repaired_trace.npz')
initial_raw_difference=float(np.abs(a['raw_action'][0]-b['raw_action'][0]).max())
assert initial_raw_difference<1e-5,initial_raw_difference
m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));d=mujoco.MjData(m);floor=m.geom('floor').id
feet={int(m.geom_bodyid[m.geom(side+'_foot_collision').id]) for side in ['left','right']}
report=json.loads((R/'diag2400_legacy_results.json').read_text());recovered=[]
for row in report['rows']:
 if not row['final_stand_success'] or row['name']=='quiet':continue
 j=row['case'];hit=np.flatnonzero(a['nonfoot_ground'][:,j]);details=[]
 if len(hit):
  t=int(hit[0]);d.qpos[:]=a['qpos'][t,j];mujoco.mj_fwdPosition(m,d)
  for c in d.contact:
   if floor not in (c.geom1,c.geom2):continue
   g=int(c.geom2 if c.geom1==floor else c.geom1)
   if int(m.geom_bodyid[g]) not in feet and c.dist<=0:details.append(m.geom(g).name)
 recovered.append(dict(case=j,group=row['name'],body_contact_before_final_recovery=bool(len(hit)),first_contact_seconds=float(hit[0]*.02) if len(hit) else None,cpu_contact_geoms=details))
out=dict(controlled_course_tests=rows,identical_first_actor_outputs_max_difference=initial_raw_difference,successful_2400_disturbed_cases=recovered,successful_disturbed_cases_with_intermediate_contact=sum(x['body_contact_before_final_recovery'] for x in recovered),successful_disturbed_cases=len(recovered),training_at_2500_terminated_on_first_nonfoot_contact=True,training_not_started=True,scope='32 fixed DR3 course cases; excludes walk-stop; single deterministic seed; reward-only retraining ablation not yet done')
(R/'REPAIR_REGRESSION_DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
