from pathlib import Path
import sys,shutil,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','.git','__pycache__','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
for name in ['pure_reward.py','pure_recovery.py','pure_dr_course.py','pure_push_course.py','walk_stop_audit.py']:shutil.copy2(R/name,P/'src/unilab/envs/locomotion/xduck_gf43x40'/name)
shutil.copy2(R/'pure_distribution.py',P/'src/unilab/algos/torch/pure_distribution.py')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');s=json.loads((B/'status.json').read_text());source=B/s['latest_checkpoint'];(R/'block_source').mkdir(exist_ok=True);shutil.copy2(source,R/'block_source/final.pt')
shutil.copy2(B/'walker_flex_adapt_50.pt',R/'walker_flex_adapt_50.pt');shutil.copy2(B/'EXTENDED_RESET_COVERAGE.json',R/'EXTENDED_RESET_COVERAGE.json');shutil.copy2(B/'PREPARATION.json',R/'PREPARATION.json')
m0=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);tree=ET.parse(c.env.scene.model_file);changes=[]
for el in tree.getroot().find('sensor'):
 name=el.get('name','')
 if not name.startswith('pure_floor_found_'):continue
 idx=int(name.rsplit('_',1)[1]);expected=m0.geom(idx).name;changes.append(dict(sensor=name,old_geom=el.get('geom2'),new_geom=expected,expected_id=idx));el.set('geom2',expected)
physics=R/'physics';physics.mkdir(exist_ok=True);tree.write(physics/'scene.xml');m=mujoco.MjModel.from_xml_path(str(physics/'scene.xml'))
mechanical=['body_mass','body_inertia','body_pos','body_quat','body_ipos','jnt_axis','jnt_pos','jnt_range','dof_armature','dof_damping','dof_frictionloss','jnt_stiffness','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange','actuator_gaintype','actuator_biastype','actuator_dyntype','geom_friction','geom_pos','geom_quat','geom_contype','geom_conaffinity','key_qpos','key_ctrl']
# Preserve calibrated plant arrays; contact sensor selection is the only asset change.
for key in mechanical:getattr(m,key)[:]=getattr(m0,key)
m.opt.timestep=m0.opt.timestep
for key in mechanical:np.testing.assert_array_equal(getattr(m,key),getattr(m0,key))
for change in changes:
 sid=m.sensor(change['sensor']).id;assert m.sensor_refid[sid]==change['expected_id'];g=change['expected_id'];assert m.geom_contype[g] or m.geom_conaffinity[g]
assert len(changes)==19 and (m.nq,m.nv,m.nu)==(21,20,14)
mujoco.mj_saveModel(m,str(physics/'effective_training.mjb'),None)
c.env.scene.model_file=str(physics/'scene.xml');c.env.scene.visual_model_file=str(physics/'scene.xml');c.env.warp_compiled_model_file=str(physics/'effective_training.mjb')
c.env.recovery_failure_tilt_deg=100.;c.env.target_increment_rad=[.025,.04,.1,.08,.09,.11,.025,.025,.025,.025,.04,.1,.08,.09];c.env.discovery_level=3;c.env.push_course_stage=1;c.env.push_course_enabled=True;c.algo.algorithm.lam=.98;c.algo.num_steps_per_env=20;c.algo.num_envs=4096;c.training.log_dir=str(R/'training');c.algo.run_name='dynamic_rescue_contact_pose_repair'
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
manifest=dict(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lineage_updates=s['lineage_compute_updates'],sensor_changes=changes,mechanical_arrays_equal=mechanical,actor_dimension=61,critic_dimension=76,critic_phase_feature='last formerly-zero body-command slot',actor_observation_unchanged=True,actor_std_preserved=True,optimizer_moments_reset_at_first_resume=True,critic_new_phase_column_initialized_zero=True,pure_RL=True,teacher_or_trajectory_training=False,level=3,DR_multiplier=2.5,flexible_joints=False,mechanical_dt=.001,num_envs=4096,rollout_steps=20,save_and_evaluate_every=100,gaelambda=.98,termination='actual nonfoot contact or root height <45% or body tilt >100deg',pose_return_schedule='Tilted reset: after2s ramp0.5s; physical push: after release+0.2s ramp0.5s; independent of robot motion',all_eight_push_directions_balanced=True)
(R/'REPAIR_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');(R/'status.json').write_text(json.dumps(dict(phase='repair_preflight',training_running=False,user_requires_explicit_resume=False,level=3,push_stage=1,completed_updates=0,source_lineage_updates=s['lineage_compute_updates'],history=[],latest_checkpoint='block_source/final.pt',best_checkpoint='block_source/final.pt'),indent=2)+'\n');print(json.dumps(dict(source_updates=s['lineage_compute_updates'],sensor_count=len(changes),mechanical_parity=True)),flush=True)
