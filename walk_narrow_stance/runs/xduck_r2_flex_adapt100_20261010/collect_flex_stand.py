from flex_common import *
import argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='newstand_flex_bank');ap.add_argument('--seed',type=int,default=1810092);ap.add_argument('--n',type=int,default=64);args=ap.parse_args()
np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2);e=env(args.n,dr_seed=args.seed);m,qi,vi,names,local=metadata();goal=np.array(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad']);chunks=[];reports=[]
try:
 e.init_state();e.set_autoreset(False);bodyids=np.array([e._backend.get_body_id(n) for n in names],np.int32);first=e._current_action_target.copy()
 def hold(st):st.info['commands'][:]=0
 e._maybe_resample_commands=hold
 initial_q,initial_v=state(e,m);initial_fields={k:getattr(e._backend._device_model,k).numpy().copy() for k in FIELDS};initial_fields.update(qpos=initial_q,qvel=initial_v,**{k:getattr(e,'_'+k).copy() for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']});np.savez_compressed(R/(args.prefix+'_params.npz'),**initial_fields)
 start=time.time()
 for tick in range(200):
  hold(e.state);u=np.clip(tick/75,0,1);u=u*u*u*(10+u*(-15+6*u));targets=(1-u)*first+u*goal;raw=(targets-e.default_angles)/e._physical_scales;e.step(raw);np.testing.assert_allclose(e._current_action_target,targets,atol=4e-7)
  if tick not in [149,199]:continue
  q,v=state(e,m);tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));vel=e.get_local_linvel().copy();xyz=feet(e,q,bodyids,local);good=(tilt<6)&(np.linalg.norm(vel[:,:2],axis=1)<.02)&(q[:,2]>.28)&(q[:,2]<.4)
  saved={k:getattr(e._backend._device_model,k).numpy()[good].copy() for k in FIELDS};saved.update(qpos=q[good],qvel=v[good],target=targets[good].copy(),initial_width=xyz[good,0,1]-xyz[good,1,1],linvel=vel[good],**{k:getattr(e,'_'+k)[good].copy() for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']});chunks.append(saved);reports.append(dict(tick=tick+1,accepted=int(good.sum()),rejected=int((~good).sum())));print('FLEX_STAND',reports[-1],'elapsed',round(time.time()-start,1),flush=True)
 bank={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};assert len(bank['qpos'])>=args.n
 stiffness=bank['jnt_stiffness'][:,e._flex_joint_ids]/e._nominal_flex_stiffness[e._flex_joint_ids];damping=bank['dof_damping'][:,e._flex_dof_ids]/e._nominal_flex_damping[e._flex_dof_ids]
 assert .8999<stiffness.min()<stiffness.max()<1.1001 and .8999<damping.min()<damping.max()<1.1001
 np.savez_compressed(R/(args.prefix+'.npz'),**bank);(R/(args.prefix+'_manifest.json')).write_text(json.dumps(dict(frames=len(bank['qpos']),seed=args.seed,physical_free_root_hold=True,stand599_calls=0,reports=reports,nq=m.nq,nv=m.nv,stiffness_ratio=[float(stiffness.min()),float(stiffness.max())],damping_ratio=[float(damping.min()),float(damping.max())],sole_angle_deg_range=np.degrees([bank['sole_angles'].min(),bank['sole_angles'].max()]).tolist(),nonzero_passive_equilibrium=float(abs(bank['qpos'][:,e._passive_qpos_indices]).max()),elapsed_s=time.time()-start),indent=2));print('FLEX_STAND_BANK_READY',len(bank['qpos']),flush=True)
finally:e.close()
