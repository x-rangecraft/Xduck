from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;O=R/'small_push_audit';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
from validation_window import pose_mask,assess_tail
E.ensure_registries();torch.set_num_threads(2);actor=FrozenActor(R/'model_3100.pt');summaries=[];reference_pre=None
for mode in ['policy','hold_target']:
 for force in [0,5,10,15,20]:
  np.random.seed(931031);torch.manual_seed(931031)
  c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.reset_mode='paired';c.env.discovery_seed=931031
  e=E.create_env(c,num_envs=8,env_cfg_override=E.build_ppo_env_cfg_override(c));m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file)
  try:
   directions=np.arange(8)*np.pi/4
   e._paired=dict(qpos=np.tile(m.key_qpos[0],(8,1)),qvel=np.zeros((8,m.nv)),kind=np.full(8,2),fixture_force=np.c_[np.cos(directions),np.sin(directions)]*force,fixture_duration=np.full(8,.3))
   e.seed_observation_noise(931032);e.init_state();e.set_autoreset(False);e._force_next[:]=100;trace=[];sent=False;hold=None
   # Inspect physical parameter equality once, outside the stepping loop.
   signatures={}
   for field in ['body_mass','body_inertia','body_ipos','geom_friction','dof_armature']:
    array=getattr(e._backend._device_model,field).numpy();signatures[field]=hashlib.sha256(array.tobytes()).hexdigest()
   for field,array in [('kp',e._motor.kp),('kd',e._motor.kd),('encoder_bias',e._encoder_bias),('sole_angles',e._sole_angles),('imu_rotation',e._imu_rotation)]:signatures[field]=hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
   for t in range(400):
    with torch.inference_mode():raw=actor(torch.as_tensor(e.state.obs['obs'],dtype=torch.float32)).numpy()
    if mode=='hold_target' and t>=99:raw[:]=0
    if t==99:hold=e._current_action_target.copy()
    s=e.step(raw);e._refresh_foot_height()
    if e._force_events and not sent:sent=True;e._force_next[:]=1000000
    q=e._backend.get_physics_state()[:,1:22].copy();v=e.get_local_linvel().copy();w=e.get_gyro().copy();dv=e.get_dof_vel().copy();_,hq=e._backend.get_body_pose_w(e._head_pose_id)
    from unilab.utils.rotation import np_quat_apply
    hu=np_quat_apply(hq[:,0],np.broadcast_to(e._head_axis,(8,3)))[:,2]
    feet,_=e._backend.get_body_pose_w(e._sole_body_ids)
    trace.append(dict(qpos=q,velocity=v,gyro=w,dof_velocity=dv,head_up=hu,foot_pos=feet.copy(),foot_height=e._foot_height.copy(),contact=e._contact.copy(),nonfoot_ground=e._nonfoot_ground.copy(),action_target=e._current_action_target.copy(),raw_action=raw.copy(),force=e._last_force.copy(),tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)))))
   z={k:np.asarray([r[k] for r in trace]) for k in trace[0]}
   z['contact']=z['contact'].astype(bool)
   if reference_pre is None:reference_pre=dict(qpos=z['qpos'][:99],parameters=signatures)
   assert signatures==reference_pre['parameters'],'DR differs between matched conditions'
   pre_diff=float(np.max(np.abs(z['qpos'][:99]-reference_pre['qpos'])));assert pre_diff<.02,pre_diff
   valid=pose_mask(z['qpos'][-10:],z['head_up'][-10:],e.default_angles,e.cfg.goal_height_m,z['nonfoot_ground'][-10:])
   success,_=assess_tail(valid,z['contact'][-10:],z['velocity'][-10:],z['gyro'][-10:],z['dof_velocity'][-10:],np.zeros(8,bool))
   rows=[]
   for j in range(8):
    events=[]
    for foot in range(2):
     airborne=(~z['contact'][:,j,foot])&(z['foot_height'][:,j,foot]>.005)
     # A real step needs >=60ms of airborne clearance, a new landing position,
     # leg target adjustment, and no prior body-ground contact in that segment.
     starts=np.flatnonzero(airborne&~np.r_[False,airborne[:-1]])
     for start in starts:
      if start<99:continue
      run_end=start
      while run_end<400 and airborne[run_end]:run_end+=1
      if run_end-start<3:continue
      lands=np.flatnonzero(z['contact'][run_end:,j,foot]);landing=run_end+int(lands[0]) if len(lands) else None
      if landing is None:continue
      take=max(start-1,99);shift=float(np.linalg.norm(z['foot_pos'][landing,j,foot,:2]-z['foot_pos'][take,j,foot,:2]))
      ids=[0,1,2,3,4] if foot==0 else [9,10,11,12,13]
      target_change=float(np.max(np.abs(z['action_target'][start:landing+1,j][:,ids]-hold[j,ids])))
      body_ground=bool(z['nonfoot_ground'][99:landing+1,j].any());upright=bool(z['tilt'][99:landing+1,j].max()<35)
      qualifies=shift>=.015 and target_change>=.03 and not body_ground and upright
      events.append(dict(foot='left' if foot==0 else 'right',takeoff_s=float((start+1)*.02),airborne_duration_s=float((run_end-start)*.02),landing_s=float((landing+1)*.02),landing_shift_m=shift,leg_target_change_rad=target_change,body_contact_before_landing=body_ground,upright_before_landing=upright,active_recovery_step=bool(qualifies)))
    active=[x for x in events if x['active_recovery_step']]
    rows.append(dict(case=j,direction_deg=j*45,final_stand_success=bool(success[j]),active_recovery_steps=len(active),airborne_events=events,body_touched_ground=bool(z['nonfoot_ground'][99:,j].any()),max_tilt_deg=float(z['tilt'][99:,j].max()),maximum_foot_horizontal_displacement_m=float(np.linalg.norm(z['foot_pos'][99:,j,:,:2]-z['foot_pos'][98,j,:,:2],axis=-1).max())))
   tag=f'{mode}_{force}N';np.savez_compressed(O/(tag+'_trace.npz'),**z)
   out=dict(model=3100,mode=mode,force_N=force,duration_s=.3,force_starts_s=2.,DR_multiplier=2.5,matched_DR=True,matching_note='Identical sampled parameters and noise seed; GPU contact solver can yield small pre-push numerical trajectory differences',pre_push_max_qpos_difference=pre_diff,cases=8,successful=int(success.sum()),cases_with_active_step=sum(x['active_recovery_steps']>0 for x in rows),active_steps=sum(x['active_recovery_steps'] for x in rows),rows=rows)
   (O/(tag+'_results.json')).write_text(json.dumps(out,indent=2)+'\n');summaries.append(out)
   (O/'SUMMARY.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
  finally:e.close()
