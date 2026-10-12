from pathlib import Path
import shutil,json,hashlib
from omegaconf import OmegaConf
import mujoco,numpy as np
R=Path(__file__).resolve().parent;B=R.parent/'xduck_direction_speed_dr_20261010';F=R.parent/'xduck_meanonly_flex320_50_20260930';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
if not (R/'physics').exists():shutil.copytree(F/'physics',R/'physics')
# Make copied XML includes/meshes resolve inside the isolated physics directory.
f=R/'physics/scene.xml';s=f.read_text().replace(str(F/'physics'),str(R/'physics'));f.write_text(s)
for name in ['common.py','engine.py','landing_metrics.py','xc_standing.onnx','xc_walking_v3_policy.py','export_walk_models.py']:
 shutil.copy2(B/name,R/name)
shutil.copy2(B/'ppo_direction_r2_run/model_179.pt',R/'parent_r2.pt');shutil.copy2(B/'clean_parent.pt',R/'clean_parent.pt');shutil.copy2(B/'direction_r2_179.onnx',R/'parent_r2.onnx');shutil.copy2(B/'direction_anchor_r2.npz',R/'direction_anchor.npz')
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40';shutil.copy2(R/'direction_speed.py',owner/'direction_speed.py')
# Preserve the R2 four-pool preservation algorithm, not later experimental yaw masks.
s=(B/'masked_anchor.py').read_text();a=s.index('  # Let PPO correct yaw');b=s.index('  self.pools=',a);s=s[:a]+s[b:];s=s.replace('weights=[2.,1.,.25,.25,1.5]','weights=[2.,1.,.5,2.]');(P/'src/unilab/algos/torch/masked_anchor.py').write_text(s);(R/'masked_anchor.py').write_text(s)
c=OmegaConf.load(B/'PPO_DIRECTION_R2_CONFIG.yaml');c.algo.seed=1810091;c.algo.num_envs=64;c.algo.max_iterations=100;c.algo.save_interval=25;c.algo.resume=False;c.algo.load_run='-1';c.algo.resume_path=None
c.algo.algorithm.anchor_dataset=str(R/'direction_anchor.npz');c.algo.algorithm.learning_rate=3e-5;c.algo.algorithm.fixed_learning_rate=3e-5;c.algo.algorithm.anchor_weight=.03
c.env.direction_seed=1810091;c.env.newstand_seed=1810091;c.env.smooth_head_target=[.019,.060];c.env.newstand_bank_path=str(R/'newstand_flex_bank.npz')
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.physics_microsteps=16;c.env.flexible_dr_enabled=True;c.env.flexible_dr_fraction=.1;c.env.flexible_dr_seed=1810092;c.env.sole_tilt_dr_enabled=True;c.env.sole_tilt_range_deg=[-3.,3.];c.env.dr_base_sole_tilt_deg=3.;c.env.sole_tilt_seed=1810093
c.env.reference_path=None;c.training.log_dir=str(R/'ppo_flex100');OmegaConf.save(c,R/'PPO_FLEX100_CONFIG.yaml');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
collect=OmegaConf.create(OmegaConf.to_container(c));collect.training.task_name='XDuckStartupFlex';collect.env.startup_training=False;collect.env.startup_reward=False;collect.env.toe_heading_reward=False;OmegaConf.save(collect,R/'COLLECT_CONFIG.yaml')
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);mx=mujoco.MjModel.from_xml_path(c.env.scene.model_file);assert (m.nq,m.nv,m.nu)==(69,68,14) and m.opt.timestep==.001/16;assert np.sum(mx.jnt_stiffness>0)==48
(R/'adaptation_manifest.json').write_text(json.dumps(dict(parent_checkpoint='parent_r2.pt',parent_sha256=hashlib.sha256((R/'parent_r2.pt').read_bytes()).hexdigest(),additional_ppo_updates=100,passive_flex_coordinates=48,nq=m.nq,nv=m.nv,mechanical_steps_per_control=320,mechanical_dt=m.opt.timestep,flex_stiffness_dr_fraction=.1,flex_damping_dr_fraction=.1,ankle_dr_definition='sole installation pitch geometry',ankle_or_sole_angle_range_deg=[-3,3],other_dr_course_multiplier=2.5,standing_pose='new symmetric target',fresh_optimizer=True,actor_normalizer_frozen=True,stand599_calls=0,third_fourth_round_weights_used=False,reward_family='R2 restored, no later heading/cadence experiment',stiffness_nominal=np.unique(m.jnt_stiffness[m.jnt_stiffness>0]).tolist(),passive_damping_nominal=np.unique(m.dof_damping[m.jnt_dofadr[m.jnt_stiffness>0]]).tolist()),indent=2));print('FLEX_ADAPTATION_PREPARED',flush=True)
