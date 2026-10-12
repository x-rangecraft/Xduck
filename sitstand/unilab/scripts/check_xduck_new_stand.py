"""Materialize the user keyframe only, then check static GF hold (no PPO)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.utils.rotation import np_quat_apply

p=argparse.ArgumentParser()
p.add_argument('--source-run-config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path('.').resolve();a.output.mkdir(parents=True,exist_ok=True)
ensure_registries()
with initialize_config_dir(config_dir=str(root/'conf/ppo'),version_base='1.3'):
    new=compose(config_name='config',overrides=['task=xduck_gf43x40_sitstand_incremental_newstand_20261012/mjwarp'])
base=OmegaConf.create(json.loads(a.source_run_config.read_text())['config'])
source=Path(base.env.warp_compiled_model_file)
original=mujoco.MjModel.from_binary_path(str(source))
model=mujoco.MjModel.from_binary_path(str(source))
key=ET.parse(root/'src/unilab/envs/locomotion/xduck_gf43x40/stand_user_20261012.xml').getroot().find('keyframe/key')
idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'STAND')
assert idx>=0
model.key_qpos[idx]=np.fromstring(key.attrib['qpos'],sep=' ')
model.key_ctrl[idx]=np.fromstring(key.attrib['ctrl'],sep=' ')
binary=root/new.env.warp_compiled_model_file
binary.parent.mkdir(parents=True,exist_ok=True)
mujoco.mj_saveModel(model,str(binary),None)
loaded=mujoco.MjModel.from_binary_path(str(binary))
unchanged=['body_mass','body_inertia','body_ipos','body_pos','body_quat',
    'dof_armature','dof_damping','dof_frictionloss','jnt_range','jnt_axis',
    'geom_type','geom_size','geom_pos','geom_quat','actuator_gainprm',
    'actuator_biasprm','actuator_forcerange','sensor_type','key_qvel']
for name in unchanged:
    np.testing.assert_array_equal(getattr(original,name),getattr(loaded,name))
assert original.opt.timestep==loaded.opt.timestep
np.testing.assert_array_equal(original.opt.gravity,loaded.opt.gravity)
for j in range(original.nkey):
    if j!=idx:
        np.testing.assert_array_equal(original.key_qpos[j],loaded.key_qpos[j])
        np.testing.assert_array_equal(original.key_ctrl[j],loaded.key_ctrl[j])
shutil.copy2(binary,a.output/binary.name)
with open_dict(base):
    for name in ['reference_keyframe','scene','stand_reset_height','action_offset_low','action_offset_high']:
        base.env[name]=new.env[name]
    base.env.warp_compiled_model_file=str(binary)
    base.reward.stand_height=new.reward.stand_height
    base.algo.load_run=str(a.source_run_config.parent/'model_1999.pt')
    base.training.log_dir=str(a.output/'train')
OmegaConf.save(base,a.output/'newstand_config.yaml')
probe=OmegaConf.create(OmegaConf.to_container(base,resolve=True))
with open_dict(probe.env):
    probe.env.evaluation_case=0
    probe.env.reset_joint_noise_rad=0.
    probe.env.reset_height_jitter_m=0.
    probe.env.reset_velocity_max=0.
    probe.env.encoder_bias_rad=0.
    probe.env.nominal_env_fraction=1.
    probe.env.dr_level=0.
    probe.env.max_episode_seconds=6.
    probe.env.posture_dwell_seconds=[100.,100.]
    for stage in probe.env.curriculum.head_range_stages:stage.ranges=[[0.,0.]]*4
    for name in ['com_range_stages','head_com_range_stages','push_stages']:
        for stage in probe.env.curriculum[name]:stage.value=0.
env=create_env(probe,num_envs=16,env_cfg_override=BackendAdapter(probe,root_dir=root).build_task_env_cfg_override())
try:
    expected=model.key_qpos[idx,7:].copy()
    np.testing.assert_allclose(env.default_angles,expected,atol=1e-7)
    old_stand=original.key_qpos[idx,7:]
    # Re-express offsets; absolute policy targets must remain the old bounds.
    from unilab.envs.locomotion.xduck_gf43x40.incremental import IncrementalCfg
    old_cfg=IncrementalCfg()
    np.testing.assert_allclose(env._target_low,old_stand+old_cfg.action_offset_low,atol=1e-7)
    np.testing.assert_allclose(env._target_high,old_stand+old_cfg.action_offset_high,atol=1e-7)
    env.step_counter=48000;env._update_curriculum();env.init_state();env.set_autoreset(False)
    head=env._backend.get_body_ids(['head_roll'])
    axis=np.array([0.,.9999999999932535,3.673192317532603e-6])
    traces={k:[] for k in ['height','body_tilt','head_tilt','qvel','qpos']}
    for _ in range(250):
        env.step(np.zeros((16,14)))
        up=np_quat_apply(env._backend.get_body_quat_w(head)[:,0],np.broadcast_to(axis,(16,3)))
        values=dict(height=env._backend.get_base_pos()[:,2].copy(),
            body_tilt=np.degrees(env.last_tilt.copy()),
            head_tilt=np.degrees(np.arccos(np.clip(up[:,2],-1,1))),
            qvel=env.get_dof_vel().copy(),qpos=env.get_dof_pos().copy())
        for name,value in values.items():traces[name].append(value)
    t={k:np.asarray(v) for k,v in traces.items()}
    result=dict(stand_joint_angles_rad=env.default_angles.tolist(),target_low_rad=env._target_low.tolist(),
        target_high_rad=env._target_high.tolist(),reset_height_m=env.cfg.stand_reset_height,
        reward_stand_height_m=env.cfg.reward_config.stand_height,static_hold_s=5.,trials=16,
        late_height_mean_m=float(t['height'][-50:].mean()),
        late_body_tilt_mean_deg=float(t['body_tilt'][-50:].mean()),
        late_head_tilt_mean_deg=float(t['head_tilt'][-50:].mean()),
        late_joint_speed_rms_rad_s=float(np.sqrt(np.mean(t['qvel'][-50:]**2))),
        falls=int(((t['height']<.13)|(t['body_tilt']>75)).any(0).sum()),
        physical_arrays_verified_unchanged=unchanged,physical_system_unchanged=True,
        training_started=False,policy_reference_changed=True,source_weights_not_adapted=True,
        compiled_model_file=str(binary),compiled_model_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (a.output/'validation.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(a.output/'static_hold.npz',**t)
    print(json.dumps(result),flush=True)
finally:
    env.close()
