from pathlib import Path
import shutil,json,hashlib
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013';V=R.parent/'stand_pure_recovery_v7_dynamic_rescue_20261013'
assert not (R/'status.json').exists(),'Do not overwrite an existing run'
shutil.copytree(V/'project',R/'project',symlinks=True)
for name in ['pure_reward.py','pure_recovery.py','pure_dr_course.py','pure_push_course.py','walk_stop_audit.py']:
 shutil.copy2(R/name,R/'project/src/unilab/envs/locomotion/xduck_gf43x40'/name)
shutil.copy2(R/'pure_distribution.py',R/'project/src/unilab/algos/torch/pure_distribution.py')
shutil.copytree(V/'physics',R/'physics')
for name in ['walker_flex_adapt_50.pt','PREPARATION.json','EXTENDED_RESET_COVERAGE.json']:
 shutil.copy2(V/name,R/name)
source=B/'block_20/final.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha=='e98c34ae7c2568e48886ccb8f64378bd721d8561f9e780b3fa8d3d5672c462ed'
(R/'block_source').mkdir();shutil.copy2(source,R/'block_source/final.pt')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');OmegaConf.save(c,R/'BASELINE2400_CONFIG.yaml')
c.env.allow_ground_recovery=True;c.env.final_stand_bonus=20.;c.env.final_not_standing_cost=5.
c.env.target_increment_rad=[.015,.01,.08,.06,.075,.11,.025,.025,.01,.015,.01,.08,.06,.075]
c.env.recovery_failure_tilt_deg=100.;c.env.discovery_level=3;c.env.push_course_stage=3;c.env.discovery_phase='settle'
c.env.targeted_tilt_fraction=.5;c.env.targeted_pitch_min_deg=40.;c.env.targeted_roll_min_deg=25.
c.env.settle_tilt_soft_deg=10.;c.env.settle_tilt_off_deg=25.;c.env.settle_hold_s=.2
c.env.settle_speed_soft=.12;c.env.settle_speed_off=.25;c.env.settle_gyro_soft=.5;c.env.settle_gyro_off=1.2
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=c.env.scene.model_file
c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb')
c.algo.algorithm.lam=.95;c.algo.num_envs=4096;c.algo.num_steps_per_env=20;c.algo.max_iterations=100
c.algo.run_name='rollback2400_ground_recovery';c.training.log_dir=str(R/'training')
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
manifest=dict(source_checkpoint=str(source),source_sha256=sha,source_lineage_updates=2400,excluded_checkpoints=[2500,2600],actor_dimension=61,critic_dimension=76,old_action_increments_restored=True,gaelambda=.95,flexible_joints=False,mechanical_dt=.001,control_dt=.02,num_envs=4096,rollout_steps=20,DR_multiplier=2.5,push_stage=3,save_and_evaluate_every=100,termination='8-second task deadline only; temporary ground contact and tip allowed',posture_reward_gate='actual body tilt, height, foot support, linear and angular speed; 0.2s confirmation; immediately release when destabilized',targeted_tilt_fraction=.5,push_sampling='50% fore/aft;20% lateral;30% broad',extra_v7_head_return_terms_removed=True,trajectory_or_teacher=False,optimizer_moments_reset_for_new_objective=True)
(R/'REPAIR_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'status.json').write_text(json.dumps(dict(phase='audit_preflight',training_running=False,user_requires_explicit_resume=False,source_lineage_updates=2400,completed_updates=0,lineage_compute_updates=2400,level=3,push_stage=3,latest_checkpoint='block_source/final.pt',best_checkpoint='block_source/final.pt',reset_goal_pending=True,history=[]),indent=2)+'\n')
print(json.dumps(manifest),flush=True)
