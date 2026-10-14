from pathlib import Path
import numpy as np,mujoco,json
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);d=mujoco.MjData(m);q0=m.key_qpos[0].copy();qi=m.jnt_qposadr[m.actuator_trnid[:,0]];limits=m.jnt_range[m.actuator_trnid[:,0]];floor=m.geom('floor').id
feet=[];foot_bodies=set()
for side in ['left','right']:
 g=m.geom(side+'_foot_collision').id;mesh=m.geom_dataid[g];feet.append((g,m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()));foot_bodies.add(int(m.geom_bodyid[g]))
rng=np.random.default_rng(20261013)
def state(axis,angle,jitter=0):
 for attempt in range(100):
  q=q0.copy();q[qi]=np.clip(q[qi]+rng.uniform(-jitter,jitter,14),limits[:,0]+.02,limits[:,1]-.02);a=np.deg2rad(angle);q[3:7]=[np.cos(a/2),np.sin(a/2) if axis=='roll' else 0,np.sin(a/2) if axis=='pitch' else 0,0]
  d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
  bottom=min((verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,verts in feet);q[2]+=.006-bottom;d.qpos[:]=q;mujoco.mj_forward(m,d)
  body_ground=False
  for contact in d.contact:
   if floor not in (contact.geom1,contact.geom2):continue
   other=contact.geom2 if contact.geom1==floor else contact.geom1
   if int(m.geom_bodyid[other]) not in foot_bodies and contact.dist<0:body_ground=True
  if q[2]>.45*q0[2] and not body_ground:return q
 raise RuntimeError(f'Cannot build unpenetrated {axis} {angle} degree reset within bounded attempts')
with np.load(c.env.reset_bank) as old:data={k:old[k].copy() for k in old.files}
walk_vx=.3;walk_vy=.3;walk_wz=.6917144638660746;coverage=[]
for level,pmax,rmax,vxmax,vymax,wzmax,omega,jitter in [(2,50.,25.,walk_vx,walk_vy,walk_wz,1.3,.12),(3,60.,35.,.3,.3,.9,1.5,.16)]:
 kinds=np.concatenate([np.zeros(410,int),np.ones(1024,int),np.full(614,2,int)]);rng.shuffle(kinds);qs=[];vs=[];meta=[]
 for kind in kinds:
  axis='pitch' if rng.random()<.7 else 'roll';angle=0.
  if kind==1:
   maximum=pmax if axis=='pitch' else rmax;u=rng.random();lo,hi=(2.,min(20.,maximum)) if u<.4 else ((min(20.,maximum),min(35.,maximum)) if u<.8 else (min(35.,maximum),maximum))
   if hi<=lo:lo,hi=2.,maximum
   angle=rng.uniform(lo,hi)*rng.choice([-1,1])
  q=state(axis,angle,jitter if kind==1 else .01);v=np.zeros(20)
  if kind==1:
   speed=rng.uniform(0.,.3);direction=rng.uniform(-np.pi,np.pi);v[:2]=speed*np.array([np.cos(direction),np.sin(direction)]);v[3:6]=rng.uniform([-omega,-omega,-wzmax],[omega,omega,wzmax]);v[6:]=rng.uniform(-.4,.4,14)
   if rng.random()<.1:
    v[:2]=.3*np.array([np.cos(direction),np.sin(direction)])
  qs.append(q);vs.append(v);meta.append([angle,0 if axis=='pitch' else 1])
 for key,value in [('qpos',qs),('qvel',vs),('kind',kinds),('fixture_force',np.zeros((2048,2))),('fixture_duration',np.zeros(2048)),('metadata',meta)]:data[f'l{level}_{key}']=np.asarray(value)
 audit_q=[];audit_v=[];audit_meta=[]
 for axis,angles in [('pitch',[15,30,40,50] if level==2 else [20,35,50,60]),('roll',[rmax])]:
  for angle in angles:
   for sign in [-1,1]:
    for mode in ['low_speed','walk_edge_outward']:
     q=state(axis,sign*angle,0);v=np.zeros(20)
     if axis=='pitch':v[0]=sign*(.15 if mode=='low_speed' else vxmax);v[4]=sign*(.2 if mode=='low_speed' else .5)
     else:v[1]=-sign*(.1 if mode=='low_speed' else vymax);v[3]=sign*(.2 if mode=='low_speed' else .5)
     audit_q.append(q);audit_v.append(v);audit_meta.append(dict(axis=axis,angle_deg=sign*angle,mode=mode,linear_velocity=v[:3].tolist(),angular_velocity=v[3:6].tolist()))
 assert len(audit_q)==20
 data[f'audit_l{level}_qpos']=np.asarray(audit_q);data[f'audit_l{level}_qvel']=np.asarray(audit_v)
 tilted=np.asarray(meta)[kinds==1]
 coverage.append(dict(level=level,pitch_max_deg=pmax,roll_max_deg=rmax,vx_bounds=[-vxmax,vxmax],vy_bounds=[-vymax,vymax],yaw_rate_bounds=[-wzmax,wzmax],max_linear_speed_m_s=.3,verified_sampled_max_linear_speed_m_s=float(np.linalg.norm(np.asarray(vs)[:,:3],axis=1).max()),train_poses=2048,high_angle_fraction=.2,tilted_samples=len(tilted),sampled_pitch_range=[float(tilted[tilted[:,1]==0,0].min()),float(tilted[tilted[:,1]==0,0].max())],audit_cases=audit_meta))
path=R/'extended_reset_bank.npz';np.savez_compressed(path,**data)
manifest=dict(passed=True,reset_bank=str(path),levels=coverage,user_max_initial_linear_speed_m_s=.3,velocity_sampling='polar horizontal velocity, total 3D norm <=0.3m/s; vertical initial velocity zero',yaw_rate_bounds=[-walk_wz,walk_wz],generated_analytically=True,no_teacher_or_trajectory=True,nonfoot_ground_penetrations_rejected=True,minimum_height_ratio=.45,mechanics_unchanged=True,limitations='Nominal geometry screening of generated resets; recovery ability and DR feasibility require physical evaluation, not assumed.')
(R/'EXTENDED_RESET_COVERAGE.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='levels'}),flush=True)
