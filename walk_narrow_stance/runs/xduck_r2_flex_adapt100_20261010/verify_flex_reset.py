from flex_common import *
torch.set_num_threads(2);np.random.seed(1870091);e=env(16,'PPO_FLEX100_CONFIG.yaml');m,qi,vi,names,local=metadata()
try:
 e.init_state();e.reset(np.arange(16,dtype=np.int32));q,v=state(e,m);bank=e._newstand_bank;distance=np.max(abs(q[:,None,:]-bank['qpos'][None,:,:]),axis=2);selected=distance.argmin(1);assert distance.min(1).max()<2e-6
 for field in FIELDS:np.testing.assert_allclose(getattr(e._backend._device_model,field).numpy(),bank[field][selected],atol=2e-6,rtol=1e-6)
 assert len(e._passive_qpos_indices)==48 and abs(q[:,e._passive_qpos_indices]).max()>1e-6
 goal=np.array(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad']);np.testing.assert_allclose(e._current_action_target,np.broadcast_to(goal,(16,14)),atol=4e-7);np.testing.assert_allclose(e.state.obs['obs'][:,34:48],np.broadcast_to((goal-e.default_angles)/e._physical_scales,(16,14)),atol=3e-6)
 k=e._backend._device_model.jnt_stiffness.numpy()[:,e._flex_joint_ids]/e._nominal_flex_stiffness[e._flex_joint_ids];d=e._backend._device_model.dof_damping.numpy()[:,e._flex_dof_ids]/e._nominal_flex_damping[e._flex_dof_ids]
 assert .8999<k.min()<k.max()<1.1001 and .8999<d.min()<d.max()<1.1001
 (R/'flex_reset_verification.json').write_text(json.dumps(dict(passive_coordinates=48,full_state_bank_restored=True,all_physics_randomization_arrays_match_bank=True,nonzero_passive_equilibrium_preserved=True,actual_k_ratio=[float(k.min()),float(k.max())],actual_damping_ratio=[float(d.min()),float(d.max())],sole_angle_range_deg=np.degrees([e._sole_angles.min(),e._sole_angles.max()]).tolist(),target_history_correct=True,stand599_calls=0),indent=2));print('FLEX_RESET_VERIFICATION_PASS',flush=True)
finally:e.close()
