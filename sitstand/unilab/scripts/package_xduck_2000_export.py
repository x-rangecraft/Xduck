"""Bundle the selected checkpoint and exact inference interface for export."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.xduck_gf43x40.incremental import IncrementalCfg
from unilab.training import BackendAdapter, create_env, ensure_registries

p = argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
project = Path('.').resolve()
bundle = a.output
bundle.mkdir(parents=True,exist_ok=True)
config = json.loads((a.run/'stage2/run_config.json').read_text())
cfg = OmegaConf.create(config['config'])
env_cfg = IncrementalCfg()
apply_cfg_overrides(env_cfg,BackendAdapter(cfg,root_dir=project).build_task_env_cfg_override())
shutil.copy2(a.run/'stage2/model_1999.pt',bundle/'model_1999.pt')
shutil.copy2(a.run/'stage2/run_config.json',bundle/'run_config.json')
OmegaConf.save(cfg,bundle/'training_config.yaml')
(bundle/'environment_config.json').write_text(json.dumps(asdict(env_cfg),indent=2,default=str))
physics = Path(env_cfg.warp_compiled_model_file)
(bundle/'physics').mkdir(exist_ok=True)
shutil.copy2(physics,bundle/'physics'/physics.name)
m = mujoco.MjModel.from_binary_path(str(physics))
home_id = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'HOME')
assert home_id>=0
compiled_home = m.key_qpos[home_id,-14:].copy()
# The Warp environment supplies its own policy reference pose. The frozen
# binary HOME is not the actor's joint-observation/integrator reference.
ensure_registries()
env = create_env(cfg,num_envs=16,
    env_cfg_override=BackendAdapter(cfg,root_dir=project).build_task_env_cfg_override())
try:
    home = env.default_angles.copy()
    target_low = env._target_low.copy()
    target_high = env._target_high.copy()
finally:
    env.close()
names = list(env_cfg.asset.policy_joint_names)
assert len(names)==14
for i,name in enumerate(names):
    j = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
    assert j>=0 and m.jnt_qposadr[j]==7+i,name
assert not env_cfg.head_absolute_position and not env_cfg.neck_target_bounds_deg
io = dict(actor_observation_size=61,actor_output_size=14,policy_joint_names=names,
    units='radians, radians/second, seconds',control_dt_s=env_cfg.ctrl_dt,simulation_dt_s=env_cfg.sim_dt,
    default_angles_rad=home.tolist(),target_low_rad=target_low.tolist(),
    target_high_rad=target_high.tolist(),target_increment_rad=env_cfg.target_increment_rad,
    policy_reference_source='Actual registered training environment default_angles and target bounds',
    compiled_model_HOME_is_not_policy_reference=True,
    supersedes='Original xduck_sitstand_2000_export policy_io.json had incorrect policy reference and target bounds',
    target_history_scale_rad=env_cfg.target_history_scale_rad,
    target_mapping='target_next = clip(target_previous + increment * tanh(actor_output), target_low, target_high)',
    integrator_initialization='Initialize target_previous to actual joint positions, matching training reset.',
    actor_observation_layout={'gyro':[0,3],'negative_body_up':[3,6],
        'joint_position_minus_default':[6,20],'joint_velocity':[20,34],
        '(target_previous-default_angles)/0.12':[34,48],'commands':[48,61]},
    commands_order=['posture_0_stand_1_sit','velocity_y','velocity_yaw','neck_pitch_offset',
        'head_pitch_offset','head_yaw_offset','head_roll_offset','body_command_0','body_command_1',
        'body_command_2','body_command_3','body_command_4','body_command_5'],
    head_command_units='radians, offset from default joint posture',
    normalization='Actor obs_normalization=false (Identity); critic obs_normalization=true. Do not add actor normalization.',
    recent_video_neck_only_commands_deg={'descent':[32.591129302978516,0,0,0],
        'rise_initial':[3.408290386199951,0,0,0],
        'rise_after_3s':[-13.706999778747559,0,0,0]},
    recent_video_commands_are_evaluation_only=True)
(bundle/'policy_io.json').write_text(json.dumps(io,indent=2))
saved = torch.load(bundle/'model_1999.pt',map_location='cpu',weights_only=False)
assert saved['infos']['unilab_progress']['next_iteration']==2000
assert saved['infos']['unilab_progress']['env_steps_per_env']==48000
assert hashlib.sha256((bundle/'model_1999.pt').read_bytes()).hexdigest()=='b21eb0b82e6703dc1f28b080e2674a85d8f227f8656371a1858807bcb27bef27'
sources = ['src/unilab/envs/locomotion/xduck_gf43x40',
           'src/unilab/actuators/gf43x40',
           'src/unilab/envs/locomotion/microduck_dm4310/velocity.py',
           'src/unilab/envs/locomotion/microduck_xl330_official/velocity.py',
           'src/unilab/envs/locomotion/common/base.py',
           'src/unilab/utils/rotation.py']
for name in sources:
    src=project/name;dest=bundle/'source_snapshot'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if src.is_dir():
        shutil.copytree(src,dest,dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__','tmp*.xml'))
    else:
        shutil.copy2(src,dest)
versions={}
for package in ['torch','numpy','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','unirl','omegaconf','tensordict']:
    try:versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:versions[package]=None
(bundle/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
(bundle/'README.txt').write_text(
    'Xduck 2000轮导出资料包\n'
    'model_1999.pt：零起点编号1999，即2000次已完成PPO更新。完整训练检查点，包含actor、critic、优化器及critic归一化状态。\n'
    '来源：unchanged_resume 1500轮续训500轮；1500-1750使用1.5倍DR，1750-2000使用2倍DR。\n'
    'training_config.yaml/run_config.json：2000模型原训练配置，没有改成最近的视频评估命令。\n'
    'environment_config.json：合并默认值后的环境配置；policy_io.json：关节顺序、观测布局、目标限幅和增量解码。\n'
    '本包已修正原包policy_io.json中的默认角度及限幅错误，数值由实际训练环境提取。不要用编译物理模型HOME作为策略参考姿态。模型权重未改。\n'
    'actor的obs_normalization=false，即Identity；critic启用归一化。导出策略不要额外添加actor归一化。神经网络输出为增量控制输入；请按policy_io中的tanh、积分和限幅转换成关节目标。\n'
    '最近“只保留脖子命令”的视频输入单列在policy_io，改变输入命令没有改变模型权重。\n'
    'GF标定文件位于source_snapshot/src/unilab/actuators/gf43x40/calibration.json，GF参数未改。\n'
    'physics包含训练使用的Linux MuJoCo编译物理模型；source_snapshot是接口和执行器源码快照，并非完整独立UniLab安装。\n'
    '该包提供原PT模型和配置，没有额外生成ONNX/JIT。\n',encoding='utf-8')
manifest=dict(completed_updates=2000,env_steps_per_env=48000,
    policy_io_reference_corrected=True,policy_reference_verified_against_training_env=True,
    source_checkpoint=str(a.run/'stage2/model_1999.pt'),checkpoint_keys=list(saved.keys()),
    actor_observation_normalization_enabled=bool(cfg.algo.actor.obs_normalization),
    critic_observation_normalization_enabled=bool(cfg.algo.critic.obs_normalization),
    gf_unchanged=True,files={})
for file in sorted(bundle.rglob('*')):
    if file.is_file() and file.name!='manifest.json':
        manifest['files'][str(file.relative_to(bundle))]=dict(bytes=file.stat().st_size,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest())
(bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
zip_path=bundle.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(bundle.rglob('*')):
        if file.is_file():archive.write(file,arcname=str(Path(bundle.name)/file.relative_to(bundle)))
print(json.dumps(dict(archive=str(zip_path),bytes=zip_path.stat().st_size,
    actor_observation_normalization_enabled=manifest['actor_observation_normalization_enabled'],files=len(manifest['files']))),flush=True)
