from pathlib import Path
import json,numpy as np,mujoco
from validation_window import CRITERIA,pose_mask,assess_tail
R=Path(__file__).resolve().parent;c=json.loads((R/'REPAIR_MANIFEST.json').read_text());m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));d=mujoco.MjData(m);floor=m.geom('floor').id;foot_bodies={int(m.geom_bodyid[m.geom(side+'_foot_collision').id]) for side in ['left','right']};ref=m.key_qpos[0,7:];all_results=[]
def actual_ground(q):
 out=np.zeros(q.shape[:2],bool)
 for t in range(q.shape[0]):
  for j in range(q.shape[1]):
   d.qpos[:]=q[t,j];mujoco.mj_fwdPosition(m,d)
   out[t,j]=any(floor in (x.geom1,x.geom2) and int(m.geom_bodyid[x.geom2 if x.geom1==floor else x.geom1]) not in foot_bodies and x.dist<=0 for x in d.contact)
 return out
for root,block,total in [(R.parent/'stand_pure_recovery_v6_keep_pose_20261013','block_17',2100),(R.parent/'stand_pure_recovery_v6_keep_pose_20261013','block_18',2200),(R.parent/'stand_pure_recovery_v6_keep_pose_20261013','block_19',2300),(R.parent/'stand_pure_recovery_v6_keep_pose_20261013','block_20',2400),(R,'block_00',2500),(R,'block_01',2600)]:
 suffix='_course_dr_only' if total<=2400 else '_course';course=np.load(root/(block+suffix+'_trace.npz'));report=json.loads((root/(block+suffix+'_results.json')).read_text());q=course['qpos'][-10:];ground=actual_ground(q)
 cs,cm=assess_tail(pose_mask(q,course['head_up'][-10:],ref,.319637239,ground),course['contact'][-10:],course['velocity'][-10:],course['gyro'][-10:],course['dof_velocity'][-10:],np.zeros(q.shape[1],bool))
 walk=np.load(root/(block+'_walk_stop_trace.npz'));wr=json.loads((root/(block+'_walk_stop_results.json')).read_text());K=len(wr['rows']);tail={}
 for key in ['qpos','velocity','gyro','dof_velocity','contact','head_up']:tail[key]=np.stack([walk[key][int(end)-10:int(end),j] for j,end in enumerate(walk['ends'])],axis=1)
 wg=actual_ground(tail['qpos']);ws,wm=assess_tail(pose_mask(tail['qpos'],tail['head_up'],ref,.319637239,wg),tail['contact'],tail['velocity'],tail['gyro'],tail['dof_velocity'],np.zeros(K,bool))
 groups={name:dict(cases=sum(x['name']==name for x in report['rows']),stood_up=sum(bool(cs[x['case']]) for x in report['rows'] if x['name']==name)) for name in ['quiet','course_reset','course_push']}
 out=dict(total_updates=total,criterion=CRITERIA,groups=groups,course_stood_up=int(cs.sum()),course_cases=len(cs),walk_stood_up=int(ws.sum()),walk_attempts=K,total_stood_up=int(cs.sum()+ws.sum()),total_cases=len(cs)+K,success_rate=float((cs.sum()+ws.sum())/(len(cs)+K)),same_saved_trajectories=True,no_policy_rerun=True,course_case_success=cs.tolist(),walk_case_success=ws.tolist())
 all_results.append(out);print(json.dumps({k:v for k,v in out.items() if not k.endswith('_case_success') and k!='criterion'}),flush=True)
(R/'FINAL_STAND_TIMELINE_2100_2600.json').write_text(json.dumps(all_results,indent=2)+'\n')
