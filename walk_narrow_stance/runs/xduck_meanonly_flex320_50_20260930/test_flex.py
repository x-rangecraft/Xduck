from pathlib import Path
import sys,json,numpy as np,torch,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'scripts'));import train_rsl_rl as E
E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');np.random.seed(1);torch.manual_seed(1)
e=E.create_env(c,num_envs=32,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();e.step_counter=29520;e._update_curriculum();e.reset(np.arange(32,dtype=np.int32))
 b=e._backend;m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file)
 assert e._cfg.physics_microsteps==16 and e._cfg.ctrl_dt/e._cfg.sim_dt==20 and m.opt.timestep==.0000625
 k=b._device_model.jnt_stiffness.numpy();d=b._device_model.dof_damping.numpy()
 kr=k[:,e._flex_joint_ids]/e._nominal_flex_stiffness[e._flex_joint_ids];dr=d[:,e._flex_dof_ids]/e._nominal_flex_damping[e._flex_dof_ids]
 assert kr.min()>=.9-1e-6 and kr.max()<=1.1+1e-6 and kr.std()>.04
 assert dr.min()>=.9-1e-6 and dr.max()<=1.1+1e-6 and dr.std()>.04
 assert np.all(d[:,m.jnt_dofadr[m.actuator_trnid[:,0]]]==0)
 assert np.abs(np.degrees(e._sole_angles)).max()<=3 and np.degrees(e._sole_angles).std()>.5
 keep=k[1:].copy();old=k[0].copy();e.reset(np.array([0],dtype=np.int32));k2=b._device_model.jnt_stiffness.numpy();np.testing.assert_array_equal(keep,k2[1:]);assert not np.array_equal(old,k2[0])
 for _ in range(10):e.step(np.zeros((32,14)))
 assert all(np.isfinite(v).all() for v in e.state.obs.values())
 q=b.get_physics_state()[:,1:70];assert q.shape==(32,69);flex=q[:,e._passive_qpos_indices];assert np.max(np.abs(flex))>1e-7
 # Contract filtering must reject unsupported new fields rather than silently claiming support.
 from unilab.dr.types import ResetRandomizationPayload,DomainRandomizationCapabilities
 payload=ResetRandomizationPayload(jnt_stiffness=k2,dof_damping=d)
 filtered,unsupported=DomainRandomizationCapabilities().filter_reset_payload(payload)
 assert filtered is None and unsupported==frozenset(['jnt_stiffness','dof_damping'])
 result=dict(passed=True,nq=m.nq,nv=m.nv,passive=48,mechanical_steps=320,stiffness_ratio=[float(kr.min()),float(kr.max())],damping_ratio=[float(dr.min()),float(dr.max())],sole_deg=[float(np.degrees(e._sole_angles).min()),float(np.degrees(e._sole_angles).max())],partial_reset_isolated=True,active_motor_damping_unchanged=True,passive_deflection_max_rad=float(np.abs(flex).max()),obs_shapes={k:list(v.shape) for k,v in e.state.obs.items()})
 (R/'flex_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
finally:e.close()
