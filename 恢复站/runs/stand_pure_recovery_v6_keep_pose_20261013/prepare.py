from pathlib import Path
import shutil,json
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v5_repair_20261012';P=R/'project'
assert not (R/'driver_pid.json').exists()
if not P.exists():shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','.git','__pycache__','outputs','logs'))
if not (P/'.venv').exists():(P/'.venv').symlink_to((B/'project/.venv').resolve())
for name in ['pure_reward.py','pure_recovery.py']:shutil.copy2(R/name,P/'src/unilab/envs/locomotion/xduck_gf43x40'/name)
shutil.copy2(R/'pure_distribution.py',P/'src/unilab/algos/torch/pure_distribution.py')
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.algo.num_envs=4096;c.algo.num_steps_per_env=20;c.env.rollout_steps=20;c.algo.save_interval=1000000;c.env.discovery_phase='settle';c.training.log_dir=str(R/'training');c.algo.run_name='pure_recovery_v6_keep_pose'
OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
print('Prepared isolated v6 project; source model300 retained unchanged.',flush=True)
