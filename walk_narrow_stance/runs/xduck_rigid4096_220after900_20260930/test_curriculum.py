from pathlib import Path
import ast,json,sys
from types import SimpleNamespace as NS
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project'
# Test the actual reference owner method on direction/contact edge cases.
source=ast.parse((P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py').read_text());cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='HistoricalRigidEnv');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_compute_reward')
class Base:
 def _compute_reward(self,*args):return np.zeros(8)
node=ast.ClassDef(name='Probe',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[]);mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));ns={'np':np,'Base':Base};exec(compile(mod,'actual_reference_method','exec'),ns)
a=ns['Probe']();a._reference=dict(np.load(R/'swing_reference.npz'));a._reference_weight=.45;a._cadence_weight=0.;a._cadence_component=lambda state:np.zeros(8);a._cfg=NS(reference_height_std=.015,reference_pitch_std=np.radians(15),ctrl_dt=.02)
a._air_time=np.full((8,2),.14);a._contact=np.array([[False,True]]*5+[[False,False],[True,True],[False,True]])
a._foot_height=np.stack([np.interp(a._air_time[:,j]/a._reference['duration'][j],a._reference['phase'],a._reference['height'][:,j]) for j in range(2)],1)
a._sole_world_pitch=np.stack([np.interp(a._air_time[:,j]/a._reference['duration'][j],a._reference['phase'],a._reference['pitch'][:,j]) for j in range(2)],1)
cmd=np.array([[.2,0,0],[-.2,0,0],[0,.2,0],[0,0,.4],[0,0,0],[.2,0,0],[.2,0,0],[.2,.2,.3]])
state=NS(info={'commands':cmd});out=a._compute_reward(state,None,None,None,None);np.testing.assert_allclose(out,[.009,.0045,.0045,.0045,0,0,0,.0045],atol=1e-12)
a._reference_weight=0;np.testing.assert_array_equal(a._compute_reward(state,None,None,None,None),np.zeros(8))
# Cadence reward cannot be earned by standing, double-flight, side-stepping or stale history.
cm=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_cadence_component')
cn=ast.ClassDef(name='CadenceProbe',bases=[],keywords=[],body=[cm],decorator_list=[]);mm=ast.fix_missing_locations(ast.Module(body=[cn],type_ignores=[]));scope={'np':np};exec(compile(mm,'actual_cadence_method','exec'),scope)
k=scope['CadenceProbe']();k._cfg=NS(ctrl_dt=.02,cadence_target_cycle_s=.5632857142856981,cadence_cycle_std_s=.06);k.step_counter=100
k._contact=np.array([[True,False]]*5+[[False,False],[True,True],[True,False]])
k._cadence_previous_contact=k._contact.copy();k._landed_air_time=np.zeros((8,2));k._cadence_period_s=np.full((8,2),k._cfg.cadence_target_cycle_s);k._cadence_last_touch_s=np.full((8,2),1.9)
cc=np.array([[.175,0,0],[-.175,0,0],[0,.2,0],[0,0,.4],[0,0,0],[.175,0,0],[.175,0,0],[.175,0,0]])
np.testing.assert_allclose(k._cadence_component(NS(info={'commands':cc})),[1,0,0,0,0,0,0,1],atol=1e-12)
k._cadence_last_touch_s[:]=0.;np.testing.assert_array_equal(k._cadence_component(NS(info={'commands':cc})),np.zeros(8))
k._cadence_previous_contact[:]=[True,False];k._contact[:]=[False,True];k._landed_air_time[:]=[0,.12];k._cadence_last_touch_s[:]=[1.9,1.44];k._cadence_period_s[:]=0
k._cadence_component(NS(info={'commands':cc}));np.testing.assert_allclose(k._cadence_period_s[:,1],.56,atol=1e-12)

# Verify real GPU DR and all boundaries, including a resumed counter.
sys.path.insert(0,str(P/'scripts'));import train_rsl_rl as E
E.ensure_registries();np.random.seed(73921);torch.manual_seed(73921);c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');e=E.create_env(c,num_envs=64,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();m=e._backend._cpu_model;dm=e._backend._device_model;assert (m.nq,m.nv,m.nu)==(21,20,14)
 boundaries=[]
 for step,ms,dr,w in [(0,100,1,0),(4800,100,1,0),(7199,100,1,0),(7200,140,1.5,.45),(14399,140,1.5,.45),(14400,180,2,.45),(21599,180,2,.45),(21600,220,2.5,.45),(120000,220,2.5,.45)]:
  e.step_counter=step;e._update_curriculum();assert abs(e._cfg.reward_config.air_time_threshold_min*1000-ms)<1e-10 and e._dr_multiplier==dr and e._reference_weight==w
  boundaries.append(dict(step=step,ms=ms,dr=dr,reference=w))
 checks=[]
 for iteration,mult in [(200,1),(300,1.5),(600,2),(900,2.5)]:
  e.step_counter=iteration*24;e._update_curriculum();e.reset(np.arange(64,dtype=np.int32));mass=dm.body_mass.numpy()[:,e._trunk_body_id]/e._base_body_mass[e._trunk_body_id]
  np.testing.assert_allclose(mass,1+(e._mass_inertia_scale-1)*mult,atol=1e-6)
  mu=dm.geom_friction.numpy()[:,e._foot_geom_ids,0];np.testing.assert_allclose(mu,np.repeat((1+(e._foot_friction-1)*mult)[:,None],4,axis=1),atol=1e-6)
  com=dm.body_ipos.numpy()[:,e._trunk_body_id]-e._base_body_ipos[e._trunk_body_id];assert abs(com).max()<=.00627*mult+1e-6
  mask=e._base_dof_armature>0;arm=dm.dof_armature.numpy()[:,mask]/e._base_dof_armature[mask];assert arm.min()>=1-.1*mult-1e-6 and arm.max()<=1+.1*mult+1e-6
  angles=np.degrees(e._sole_angles);assert abs(angles).max()<=3*mult+1e-6
  assert e._cfg.noise_config.level==mult
  checks.append(dict(iteration=iteration,dr=mult,sole_minmax_deg=[float(angles.min()),float(angles.max())],mass_minmax=[float(mass.min()),float(mass.max())],friction_minmax=[float(mu.min()),float(mu.max())],constraints=int(e._backend._device_data.nefc.numpy().max())))
 # Audit directions bypass random command buckets and head/body movement in evaluation.
 for command in [[-.2,0,0],[0,.2,0],[0,-.2,0],[0,0,.5],[.2,.2,.4]]:
  e._cfg.audit_twist=tuple(command);np.testing.assert_array_equal(e._sample_velocity_commands(4),np.tile(command,(4,1)));assert not e._sample_head_commands(4).any() and not e._sample_body_commands(4).any()
 e._cfg.audit_twist=None
 e.step_counter=4800;e._update_curriculum();before_cmd=e._sample_velocity_commands(10000)
 assert before_cmd[:,0].min()<-.55 and before_cmd[:,0].max()>.55
 e.step_counter=7200;e._update_curriculum();after_cmd=e._sample_velocity_commands(10000)
 target=(after_cmd[:,0]>=.15)&(after_cmd[:,0]<=.20)&(after_cmd[:,1]==0)&(after_cmd[:,2]==0)
 assert .13<target.mean()<.17 and after_cmd[:,0].min()<-.55 and after_cmd[:,0].max()>.55 and after_cmd[:,2].min()<-.6 and after_cmd[:,2].max()>.6
 e.step_counter=4800;e._update_curriculum();e.reset(np.arange(64,dtype=np.int32));old_angles=e._sole_angles.copy();old_gpu=dm.geom_pos.numpy().copy()
 e.step_counter=21600;e._update_curriculum();np.testing.assert_array_equal(e._sole_angles,old_angles);np.testing.assert_array_equal(dm.geom_pos.numpy(),old_gpu)
 e.reset(np.array([0,1],dtype=np.int32));np.testing.assert_array_equal(dm.geom_pos.numpy()[2:],old_gpu[2:])
 st=e.step(np.zeros((64,14),dtype=np.float32));assert np.isfinite(st.reward).all() and all(np.isfinite(x).all() for x in st.obs.values())
 report=dict(passed=True,boundaries=boundaries,gpu_stages=checks,reference_direction_and_contact_tests=True,reference_zero_before300=True,cadence_direction_contact_stale_history_tests=True,target_command_bucket_fraction=float(target.mean()),wide_input_preserved=True,direction_audit_commands_tested=True,no_mid_episode_geometry_jump=True,partial_reset_isolated=True,finite=True)
 (R/'curriculum_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:e.close()
