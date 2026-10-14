from pathlib import Path
import json,numpy as np,mujoco
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013';V=R.parent/'stand_pure_recovery_v7_dynamic_rescue_20261013'
comparison=json.loads((V/'FINAL_STAND_TIMELINE_2100_2600.json').read_text());m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'))
out=[]
for total,block in [(2300,19),(2400,20)]:
 saved=next(x for x in comparison if x['total_updates']==total);success=saved['course_case_success'];d=json.loads((B/f'block_{block:02}_course_dr_only_results.json').read_text());z=np.load(B/f'block_{block:02}_course_dr_only_trace.npz');cases=[]
 for j in range(4,32):
  cutoff=(z['tilt'][:,j]>70)|(z['qpos'][:,j,2]<.45*m.key_qpos[0,2]);hit=np.flatnonzero(cutoff)
  description=d['tilt_case_parameters'][j-4] if j<24 else dict(axis='physical_push',world_direction_deg=(j-24)*45,force_N=40,duration_s=1.5)
  cases.append(dict(case=j,scenario=description,final_stand_success=success[j],first_old_training_cutoff_s=float(hit[0]*.02) if len(hit) else None,final_height_m=float(z['qpos'][-1,j,2]),final_tilt_deg=float(z['tilt'][-1,j]),final_joint_velocity_rms=float(np.sqrt(np.mean(z['dof_velocity'][-10:,j]**2))),mean_actor_output_abs=float(np.abs(z['raw_action'][:,j]).mean())))
 out.append(dict(total_updates=total,groups=saved['groups'],walk_stood_up=saved['walk_stood_up'],cases=cases))
z=np.load(B/'extended_reset_bank.npz');q=z['l3_qpos'];kind=z['l3_kind'];w,x,y,t=q[:,3:7].T
pitch=np.degrees(np.arcsin(np.clip(2*(w*y-t*x),-1,1)));roll=np.degrees(np.arctan2(2*(w*x+y*t),1-2*(x*x+y*y)));sel=kind==1
coverage={name:dict(bank_count=int((mask&sel).sum()),tilted_bank_count=int(sel.sum()),old_all_reset_fraction=float(.3*(mask&sel).sum()/sel.sum())) for name,mask in [('negative_pitch_beyond45',pitch<-45),('lateral_roll_beyond30',abs(roll)>30),('new_target_union',(pitch<-40)|(abs(roll)>25))]}
result=dict(source_models=out,training_coverage=coverage,confirmed_bottlenecks=['same directional tilt failures at2300 and2400','severe negative pitch and lateral reset coverage about1% each','old70deg or45%height termination removes floor-rise continuation','old sensor mapping cannot reliably reward actual final standing'],targeted_changes=json.loads((R/'REPAIR_MANIFEST.json').read_text()),limitations='Fixed saved DR audit trajectories; sampling and termination bottlenecks identified; per-change learning improvement remains to be measured.')
(R/'PLATEAU_AUDIT_2300_2400.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(coverage=coverage,failed_scenarios={x['total_updates']:[c for c in x['cases'] if not c['final_stand_success']] for x in out})),flush=True)
