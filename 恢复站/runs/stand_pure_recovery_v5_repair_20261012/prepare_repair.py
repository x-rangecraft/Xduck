from pathlib import Path
import sys,shutil,json
import numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v4_20261012';P=R/'project'
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','.git','__pycache__','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
owner=P/'src/unilab/envs/locomotion/xduck_gf43x40'
for name in ['pure_reward.py','pure_recovery.py']:shutil.copy2(R/name,owner/name)
shutil.copy2(R/'pure_distribution.py',P/'src/unilab/algos/torch/pure_distribution.py')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.algo.num_steps_per_env=20;c.env.rollout_steps=20;c.algo.num_envs=4096;c.algo.save_interval=1000000;c.env.discovery_phase='discovery';c.training.log_dir=str(R/'training');c.algo.run_name='pure_recovery_v5_pose_entropy_repair'
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);d=mujoco.MjData(m);d.qpos[:]=m.key_qpos[0];mujoco.mj_forward(m,d)
axis=d.xmat[m.body('yaw_roll_motion').id].reshape(3,3).T@np.array([0.,0.,1.])
c.env.head_reference_axis=axis.tolist();c.env.goal_height_m=.319637239
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
(R/'status.json').write_text(json.dumps(dict(phase='repair_preflight',training_running=False,user_requires_explicit_resume=True,teacher_or_trajectory_training=False,num_envs=4096,rollout_steps=20,mechanical_dt=.001,checkpoint_every_updates=100,evaluation_every_updates=100,goal_height_m=.319637239,head_reference_axis=axis.tolist()),indent=2)+'\n');print('Prepared paused reward/speed repair',flush=True)
