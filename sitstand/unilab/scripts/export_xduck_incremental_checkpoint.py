"""Export the deterministic actor and verify the real incremental interface."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict
import torch

from unilab.training import BackendAdapter, create_env, ensure_registries

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path('.').resolve();out=a.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
ensure_registries()
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
progress=saved['infos']['unilab_progress']
completed=progress['next_iteration']
cfg=OmegaConf.create(json.loads((a.checkpoint.parent/'run_config.json').read_text())['config'])
assert cfg.algo.actor.class_name=='rsl_rl.models.MLPModel'
actor_cfg=OmegaConf.to_container(cfg.algo.actor,resolve=True)
actor_cfg.pop('class_name')
actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),
    {'actor':['actor']},'actor',14,**actor_cfg)
actor.load_state_dict(saved['actor_state_dict']);actor.eval()
module=actor.as_onnx(verbose=False).cpu().eval()
model_name=f'xduck_sitstand_{completed}_newstand.onnx'
onnx_path=out/model_name
with torch.inference_mode():
    torch.onnx.export(module,module.get_dummy_inputs(),str(onnx_path),
        input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
onnx.checker.check_model(onnx.load(str(onnx_path)))
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
np.random.seed(8701);torch.manual_seed(8701)
env=create_env(cfg,num_envs=16,env_cfg_override=BackendAdapter(cfg,root_dir=root).build_task_env_cfg_override())
obs_records=[];previous_records=[];max_decoder_error=0.
try:
    assert env.cfg.pure_policy_revision==6
    assert not env.cfg.head_absolute_position and not env.cfg.target_increment_start_rad
    assert not env.cfg.control_config.simulate_action_latency
    assert env.state is None
    env.step_counter=progress['env_steps_per_env'];env._update_curriculum();env.init_state();env.set_autoreset(False)
    default=env.default_angles.copy();low=env._target_low.copy();high=env._target_high.copy()
    increments=env._increments.copy();history_scale=env.cfg.target_history_scale_rad
    expected=np.array([0,.08,.394,-.061,-.455,.019,.06,0,0,0,-.08,-.394,.061,.455])
    np.testing.assert_allclose(default,expected,atol=1e-7)
    assert abs(low[5]+.29)<1e-7 and abs(high[5]-1.29)<1e-7
    names=list(env.cfg.asset.policy_joint_names)
    for _ in range(64):
        obs=env.state.obs['obs'].copy()
        previous=env._current_action_target.copy()
        with torch.inference_mode():
            action=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[16])).numpy()
        predicted=np.clip(previous+increments*np.tanh(action),low,high)
        env.step(action)
        max_decoder_error=max(max_decoder_error,float(np.max(np.abs(predicted-env._current_action_target))))
        obs_records.append(obs);previous_records.append(previous)
    real_obs=np.concatenate(obs_records).astype(np.float32)
    previous=np.concatenate(previous_records)
    rng=np.random.default_rng(8702)
    inputs=np.concatenate([real_obs,np.zeros((1,61),np.float32),rng.normal(0,1,(64,61)).astype(np.float32)])
    with torch.inference_mode():
        reference=actor(TensorDict({'actor':torch.from_numpy(inputs)},batch_size=[len(inputs)])).numpy()
        wrapped=module(torch.from_numpy(inputs)).numpy()
    wrapper_error=float(np.max(np.abs(reference-wrapped)))
    outputs=np.concatenate([session.run(['actions'],{'obs':row[None,:]})[0] for row in inputs])
    error=np.abs(reference-outputs)
    decoded_pt=np.clip(previous+increments*np.tanh(reference[:len(real_obs)]),low,high)
    decoded_onnx=np.clip(previous+increments*np.tanh(outputs[:len(real_obs)]),low,high)
    decoded_error=float(np.max(np.abs(decoded_pt-decoded_onnx)))
    assert wrapper_error<1e-6
    assert float(error.max())<1e-4
    assert decoded_error<1e-5 and max_decoder_error<1e-6
    validation=dict(completed_updates=completed,real_physics_observations=len(real_obs),
        sanity_observations=len(inputs)-len(real_obs),onnx_checked=True,
        input_name='obs',input_shape=[1,61],output_name='actions',output_shape=[1,14],
        deterministic_wrapper_vs_actor_max_abs_error=wrapper_error,
        onnx_vs_actor_max_abs_error=float(error.max()),onnx_vs_actor_mean_abs_error=float(error.mean()),
        onnx_vs_actor_decoded_target_max_abs_error_rad=decoded_error,
        reference_decoder_vs_actual_env_max_abs_error_rad=max_decoder_error,
        actor_obs_normalization=bool(cfg.algo.actor.obs_normalization),
        runtime_policy_reference_verified=True)
    (out/'export_validation.json').write_text(json.dumps(validation,indent=2))
    io=dict(completed_updates=completed,model_filename=model_name,
        input_name='obs',input_shape=[1,61],output_name='actions',output_shape=[1,14],
        actor_observation_normalization=bool(cfg.algo.actor.obs_normalization),
        control_dt_s=env.cfg.ctrl_dt,simulation_dt_s=env.cfg.sim_dt,policy_joint_names=names,
        default_angles_rad=default.tolist(),target_low_rad=low.tolist(),target_high_rad=high.tolist(),
        target_increment_rad=increments.tolist(),target_history_scale_rad=history_scale,
        reference_source='Actual registered training environment, not compiled HOME',
        actor_observation_layout={'body_gyro_rad_s':[0,3],'body_gravity_down_unit':[3,6],
            'joint_position_minus_default_rad':[6,20],'joint_velocity_rad_s':[20,34],
            '(previous_target-default)/history_scale':[34,48],'commands':[48,61]},
        command_layout={'posture_0_stand_1_sit':[48,49],'other_twist_slots':[49,51],
            'neck_head_pitch_yaw_roll_offsets_rad':[51,55],'body_command_six_slots':[55,61]},
        target_mapping='clip(previous_target + increment * tanh(raw_actions), low, high)',
        hardware_to_policy_indices=[0,1,2,3,4,10,11,12,13,5,6,7,8,9],
        policy_to_hardware_indices=[0,1,2,3,4,9,10,11,12,13,5,6,7,8],
        hardware_order='left leg five, right leg five, neck/head four; mouth excluded',
        stand_height_m=env.cfg.reward_config.stand_height,sit_height_m=env.cfg.reward_config.sit_height,
        policy_reference_applies_to_both_joint_observations_and_target_history=True)
    (out/'policy_io.json').write_text(json.dumps(io,indent=2))
finally:
    env.close()
shutil.copy2(a.checkpoint,out/a.checkpoint.name)
shutil.copy2(a.checkpoint.parent/'run_config.json',out/'run_config.json')
OmegaConf.save(cfg,out/'training_config.yaml')
(out/'physics').mkdir(exist_ok=True)
shutil.copy2(cfg.env.warp_compiled_model_file,out/'physics/pure_effective_newstand_linux.mjb')
for relative in ['src/unilab/envs/locomotion/xduck_gf43x40', 'src/unilab/actuators/gf43x40']:
    dst=out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(root/relative,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','tmp*.xml'))
(out/'reference_adapter.py').write_text('''"""Numerical reference only; integrate with the hardware API separately."""
import json
from pathlib import Path
import numpy as np
IO=json.loads(Path(__file__).with_name("policy_io.json").read_text())
DEFAULT=np.asarray(IO["default_angles_rad"],dtype=np.float32)
LOW=np.asarray(IO["target_low_rad"],dtype=np.float32)
HIGH=np.asarray(IO["target_high_rad"],dtype=np.float32)
INCREMENT=np.asarray(IO["target_increment_rad"],dtype=np.float32)
def hardware_to_policy(values):
    return np.asarray(values)[...,IO["hardware_to_policy_indices"]]
def policy_to_hardware(values):
    return np.asarray(values)[...,IO["policy_to_hardware_indices"]]
def initialize_target(measured_positions_policy_order):
    return np.clip(np.asarray(measured_positions_policy_order,dtype=np.float32),LOW,HIGH)
def build_observation(body_gyro,body_gravity_down,joint_positions,joint_velocities,previous_target,commands):
    obs=np.concatenate([body_gyro,body_gravity_down,np.asarray(joint_positions)-DEFAULT,
        joint_velocities,(np.asarray(previous_target)-DEFAULT)/IO["target_history_scale_rad"],commands],axis=-1).astype(np.float32)
    if obs.shape[-1]!=61:raise ValueError("Expected 61 observation values")
    return obs
def decode_target(previous_target,raw_actions):
    return np.clip(np.asarray(previous_target)+INCREMENT*np.tanh(np.asarray(raw_actions)),LOW,HIGH)
''')
versions={}
for name in ['torch','numpy','mujoco','rsl-rl-lib','onnx','onnxruntime','warp-lang']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
(out/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
(out/'README.txt').write_text(
    f'Xduck 新默认站姿 {completed} 轮导出包\n'
    f'{model_name}：确定性actor，输入obs float32[1,61]，输出actions float32[1,14]，opset17。\n'
    'actor观测归一化关闭。policy_io.json中的默认角与限幅来自实际训练环境，已核对新站姿。\n'
    'ONNX输出是原始动作，需做tanh、增量积分和限幅后才能成为关节目标，不要直接当目标位置。\n'
    '策略关节顺序：左腿5、头部4、右腿5；硬件日志顺序：左腿5、右腿5、头部4。转换索引见policy_io。嘴部不在14维策略中。\n'
    '策略接管时，用实际关节位置初始化积分目标；坐下/起立命令切换时保持积分状态，不要每周期重置。\n'
    'posture命令0=站立/起立，1=坐下；两个剩余twist槽为0；头部偏移和六个body命令槽按61维布局填入。\n'
    '参考数值函数见reference_adapter.py；它需要接入实际硬件API。控制周期20ms，GF参数未改。\n'
    '默认新站姿、积分历史参考、限幅都要一起使用本包policy_io，不要混用旧2000包配置。\n'
    '包含完整PT检查点、原训练配置、物理模型及GF源码快照。ONNX数值一致性证据见export_validation.json。\n',encoding='utf-8')
manifest=dict(completed_updates=completed,checkpoint_source=str(a.checkpoint),actor_file=model_name,
    policy_reference_verified=True,gf_unchanged=True,files={})
for file in sorted(out.rglob('*')):
    if file.is_file() and file.name!='manifest.json':
        manifest['files'][str(file.relative_to(out))]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive_path=out.with_suffix('.zip')
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(out.rglob('*')):
        if file.is_file():archive.write(file,arcname=str(Path(out.name)/file.relative_to(out)))
print(json.dumps(dict(archive=str(archive_path),bytes=archive_path.stat().st_size,validation=validation)),flush=True)
