from pathlib import Path
import json,numpy as np,mujoco
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013'
m=mujoco.MjModel.from_binary_path(str(B.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m);floor=m.geom('floor').id
feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];footbodies={int(m.geom_bodyid[g]) for g in feet};result=[]
datasets=[(300,np.load(B/'baseline_trace.npz')),(400,np.load(B/'block_00_trace.npz')),(3100,np.load(B/'comparison3100_video_conditions_trace.npz'))]
assert all(np.array_equal(datasets[0][1]['force'],x[1]['force']) for x in datasets[1:])
for checkpoint,z in datasets:
 j=24;ground=np.zeros(400,bool);height=np.zeros((400,2));contacts=[]
 for t in range(400):
  d.qpos[:]=z['qpos'][t,j];mujoco.mj_fwdPosition(m,d)
  for side,g in enumerate(feet):
   mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
   height[t,side]=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()
  for c in d.contact:
   if floor not in (c.geom1,c.geom2):continue
   g=int(c.geom2 if c.geom1==floor else c.geom1)
   if m.geom_bodyid[g] not in footbodies and c.dist<=0:
    ground[t]=True;contacts.append(dict(time_s=(t+1)*.02,geom=m.geom(g).name))
 events=[]
 for side in range(2):
  air=(~z['contact'][:,j,side].astype(bool))&(height[:,side]>.001)
  for start in np.flatnonzero(air&~np.r_[False,air[:-1]]):
   end=int(start)
   while end<400 and air[end]:end+=1
   if end-start<2:continue
   landing=np.flatnonzero(z['contact'][end:,j,side]);land=end+int(landing[0]) if len(landing) else None
   if land is None:continue
   shift=float(np.linalg.norm(z['foot_pos'][land,j,side,:2]-z['foot_pos'][max(int(start)-1,0),j,side,:2]))
   ids=[0,1,2,3,4] if side==0 else [9,10,11,12,13]
   change=float(np.abs(z['action_target'][start:land+1,j][:,ids]-z['action_target'][23,j,ids]).max())
   pre_ground=bool(ground[24:land+1].any())
   events.append(dict(foot=['left','right'][side],takeoff_s=(int(start)+1)*.02,airtime_s=(end-int(start))*.02,landing_s=(land+1)*.02,landing_shift_m=shift,leg_target_change_rad=change,body_contact_before_landing=pre_ground,confirmed_precontact_step=bool(shift>=.015 and change>=.03 and not pre_ground)))
 result.append(dict(checkpoint=checkpoint,video_case=24,force_N=16,duration_s=.4,nominal_video_conditions=True,force_sequence_identical=True,body_ground_contact=bool(ground.any()),first_body_contacts=contacts[:3],confirmed_precontact_steps=sum(x['confirmed_precontact_step'] for x in events),events=events,max_body_tilt_deg=float(z['tilt'][:,j].max()),maximum_foot_clearance_m=height.max(0).tolist()))
O=R/'early300_review';O.mkdir(exist_ok=True);(O/'SAME_VIDEO_CONDITIONS_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
