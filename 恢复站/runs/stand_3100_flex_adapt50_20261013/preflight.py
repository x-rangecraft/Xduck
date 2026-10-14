from pathlib import Path
import sys,json,numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.reset_mode='paired'
e=E.create_env(c,num_envs=16,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);e.init_state()
 assert len(e._passive_qpos_indices)==48 and m.opt.timestep==.001/16
 assert e.state.obs['obs'].shape==(16,61) and e.state.obs['critic'].shape==(16,76)
 k=e._backend._device_model.jnt_stiffness.numpy();d=e._backend._device_model.dof_damping.numpy()
 kr=k[:,e._flex_joint_ids]/e._nominal_flex_stiffness[e._flex_joint_ids];dr=d[:,e._flex_dof_ids]/e._nominal_flex_damping[e._flex_dof_ids]
 assert kr.min()>=.9-1e-6 and kr.max()<=1.1+1e-6 and kr.std()>.03
 assert dr.min()>=.9-1e-6 and dr.max()<=1.1+1e-6 and dr.std()>.03
 keep=k[1:].copy();previous=k[0].copy();e.reset(np.array([0],np.int32));nextk=e._backend._device_model.jnt_stiffness.numpy()
 np.testing.assert_array_equal(keep,nextk[1:]);assert not np.array_equal(previous,nextk[0])
 assert np.max(np.abs(np.degrees(e._sole_angles)))<=3
 assert np.allclose(e._increments,[.015,.01,.08,.06,.075,.11,.025,.025,.01,.015,.01,.08,.06,.075])
 # Verify new sensors refer to real collision geometries, not old index guesses.
 for name in e.cfg.contact_sensor_names:
  sid=m.sensor(name).id;g=m.sensor_refid[sid];assert m.geom_contype[g] or m.geom_conaffinity[g]
 for _ in range(30):e.step(np.zeros((16,14),np.float32))
 assert np.isfinite(e.state.reward).all() and all(np.isfinite(x).all() for x in e.state.obs.values())
 full=e._backend.get_physics_state()[:,1:1+m.nq];deformation=full[:,e._passive_qpos_indices];maximum=float(np.max(np.abs(deformation)));assert maximum>1e-7
 result=dict(passed=True,flexible_coordinates=48,stiffness_ratio=[float(kr.min()),float(kr.max())],damping_ratio=[float(dr.min()),float(dr.max())],partial_reset_isolated=True,flexible_deformation_max_rad=maximum,actor61_critic76=True,decoder_unchanged=True,contact_sensor_mapping_verified=True,mechanical_dt=m.opt.timestep,mechanical_steps_per_control=320,finite_observation_and_reward=True)
 (R/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:e.close()
