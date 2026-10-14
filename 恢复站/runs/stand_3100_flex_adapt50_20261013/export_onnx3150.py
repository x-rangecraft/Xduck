from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,onnx,onnxruntime as ort,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;D=R/'export3150';D.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'project/src'))
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
source=R/'model_3150.pt';actor=FrozenActor(source)
path=D/'model_3150.onnx'
torch.onnx.export(actor,torch.zeros(1,61),str(path),input_names=['obs'],output_names=['raw_action'],dynamic_axes={'obs':{0:'batch'},'raw_action':{0:'batch'}},opset_version=17,dynamo=False)
onnx.checker.check_model(onnx.load(str(path)))
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);assert m.nq==69 and c.env.flexible_dr_enabled
qi=m.jnt_qposadr[m.actuator_trnid[:,0]];default=m.key_qpos[0,qi]
config=OmegaConf.load(R.parent/'stand_pure_recovery_v8_ground_recover_20261013/delivery2800/INFERENCE_CONFIG.yaml')
np.testing.assert_allclose(config.default_angles_rad,default,atol=1e-6)
np.testing.assert_allclose(config.target_increment_rad,c.env.target_increment_rad,atol=1e-9)
np.testing.assert_allclose(config.target_low_rad,default+np.array(c.env.action_offset_low),atol=1e-6)
np.testing.assert_allclose(config.target_high_rad,default+np.array(c.env.action_offset_high),atol=1e-6)
assert list(config.joint_kp)==list(c.env.control_config.joint_kp) and list(config.joint_kd)==list(c.env.control_config.joint_kd)
config.model='model_3150.onnx';config.checkpoint='model_3150.pt';config.source_model_updates=3150
config.flexible_joints_enabled=True;config.flexible_dr_enabled=True;config.flexible_dr_fraction=float(c.env.flexible_dr_fraction);config.mechanical_dt_s=.001/16;config.physics_microsteps=16;config.control_dt_s=float(c.env.ctrl_dt)
OmegaConf.save(config,D/'INFERENCE_CONFIG.yaml')
rng=np.random.default_rng(93150);x=(actor.mean.numpy()+(actor.std.numpy()+.01)*rng.normal(size=(1024,61)).astype(np.float32)*3).astype(np.float32);x[:,48:]=0
with torch.inference_mode():expected=actor(torch.from_numpy(x)).numpy()
session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);actual=session.run(None,{'obs':x})[0]
error=float(np.abs(actual-expected).max());assert error<1e-4,error
inc=np.array(config.target_increment_rad);ref=np.array(config.default_angles_rad);low=np.array(config.target_low_rad);high=np.array(config.target_high_rad)
target_error=float(np.abs(np.clip(ref+inc*np.tanh(expected),low,high)-np.clip(ref+inc*np.tanh(actual),low,high)).max());assert target_error<1e-5,target_error
verification=dict(passed=True,source_checkpoint='model_3150.pt',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input='obs [batch,61]',output='raw_action [batch,14]',normalization_included=True,epsilon=.01,normalizer_do_not_apply_twice=True,samples=1024,max_raw_action_abs_difference=error,max_decoded_target_difference_rad=target_error,decoder='clip(previous_target + per_joint_increment * tanh(raw_action), low, high)',flexible_joints=True)
(D/'EXPORT_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification),flush=True)
