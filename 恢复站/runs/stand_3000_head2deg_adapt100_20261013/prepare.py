from pathlib import Path
import shutil,json,hashlib
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_2900_head_axes_adapt100_20261013'
assert not (R/'status.json').exists()
shutil.copytree(B/'project',R/'project',symlinks=True)
for name in ['pure_recovery.py','pure_reward.py','pure_push_course.py','pure_dr_course.py','walk_stop_audit.py']:
 shutil.copy2(R/name,R/'project/src/unilab/envs/locomotion/xduck_gf43x40'/name)
shutil.copy2(R/'pure_distribution.py',R/'project/src/unilab/algos/torch/pure_distribution.py')
shutil.copytree(B/'physics',R/'physics')
for name in ['PREPARATION.json','EXTENDED_RESET_COVERAGE.json','walker_flex_adapt_50.pt','QUIET_POSTURE_AUDIT_2800.json']:shutil.copy2(B/name,R/name)
(R/'block_source').mkdir();source=B/'model_3000.pt';shutil.copy2(source,R/'block_source/final.pt')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.env.posture_band_enabled=True
c.env.posture_tolerance_rad=[.08726646,.08726646,.13962634,.17453293,.17453293,.17453293,.17453293,.17453293,.17453293,.08726646,.08726646,.13962634,.17453293,.17453293]
c.env.head_axes_enabled=True;c.env.head_axes_tolerance_rad=[.017453292519943295]*2;c.env.head_axes_excess_scale_rad=.026179938779914945;c.env.head_axes_bonus=3.;c.env.head_axes_penalty=.5;c.env.posture_tolerance_rad[7]=.017453292519943295;c.env.posture_tolerance_rad[8]=.017453292519943295
c.env.posture_head_world_tolerance_deg=1.;c.env.posture_bonus_weights=[4.,2.,2.];c.env.posture_penalty_weights=[.5,.5,.5]
c.env.flexible_dr_enabled=False;c.env.flexible_dr_fraction=0.;c.env.physics_microsteps=1
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=c.env.scene.model_file;c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb')
c.training.log_dir=str(R/'training');c.algo.run_name='stand3000_head2deg_adapt100';c.algo.max_iterations=100
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
manifest=dict(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lineage_updates=3000,additional_updates=100,flexible_joints=False,mechanical_dt=.001,control_dt=.02,num_envs=4096,rollout_steps=20,actor_dimension=61,DR_multiplier=2.5,action_decoder_unchanged=True,exploration_entropy_unchanged=.01,recovery_rewards_and_endpoint_bonus_unchanged=True,posture_only_when_body_stable=True,head_world_tolerance_deg=1.,head_joint_tolerance_deg=10.,hip_yaw_roll_tolerance_deg=5.,hip_pitch_tolerance_deg=8.,knee_ankle_tolerance_deg=10.,head_axes_tolerance_deg=1.,head_acceptance_deg=2.,head_axes_excess_scale_deg=1.5,head_axes_separate_bonus=3.,new_posture_terms_net_nonnegative=True,normalizers_and_critic_features_preserved=True,optimizer_moments_reset_for_new_reward=True)
(R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'status.json').write_text(json.dumps(dict(phase='preflight',training_running=False,user_requires_explicit_resume=False,source_lineage_updates=3000,completed_updates=0,lineage_compute_updates=3000,requested_updates=100,flexible_joints=False),indent=2)+'\n');print(json.dumps(manifest),flush=True)
