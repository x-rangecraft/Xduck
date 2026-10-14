from pathlib import Path
import sys,json,shutil,hashlib,xml.etree.ElementTree as ET
import numpy as np,torch,mujoco,onnx,onnxruntime as ort
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;D=R/'delivery2800';D.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'project/src'));sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.live_handoff import FrozenActor
from unilab.envs.locomotion.xduck_gf43x40.pure_dr_course import dr_summary
assert json.loads((R/'status.json').read_text())['phase']=='stopped_by_user'
source=R/'model_2800.pt';assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((R/'block_03/final.pt').read_bytes()).digest()
shutil.copy2(source,D/'model_2800.pt');actor=FrozenActor(source)
torch.onnx.export(actor,torch.zeros(1,61),str(D/'model_2800.onnx'),input_names=['obs'],output_names=['raw_action'],dynamic_axes={'obs':{0:'batch'},'raw_action':{0:'batch'}},opset_version=17,dynamo=False)
onnx.checker.check_model(onnx.load(str(D/'model_2800.onnx')))
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');E.ensure_registries();e=E.create_env(c,num_envs=1,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();rng=np.random.default_rng(82800)
 x=actor.mean.numpy()+(actor.std.numpy()+.01)*rng.normal(size=(1024,61)).astype(np.float32)*3;x[:,48:]=0
 x=np.vstack([x,e.state.obs['obs']]).astype(np.float32)
 with torch.inference_mode():expected=actor(torch.from_numpy(x)).numpy()
 session=ort.InferenceSession(str(D/'model_2800.onnx'),providers=['CPUExecutionProvider']);actual=session.run(None,{'obs':x})[0]
 difference=float(np.abs(actual-expected).max());assert difference<1e-4,difference
 inc=e._increments.copy();defaults=e.default_angles.copy();low=e._target_low.copy();high=e._target_high.copy()
 config=dict(model='model_2800.onnx',checkpoint='model_2800.pt',actor_input_dimension=61,actor_output_dimension=14,input_name='obs',output_name='raw_action',normalization_included_in_onnx=True,normalization_epsilon=.01,control_dt_s=.02,mechanical_dt_s=.001,flexible_joints_enabled=False,joint_order=list(e.cfg.asset.policy_joint_names),motor_ids_in_policy_order=[1,2,3,4,5,11,12,13,14,6,7,8,9,10],default_angles_rad=defaults.tolist(),target_increment_rad=inc.tolist(),target_low_rad=low.tolist(),target_high_rad=high.tolist(),joint_kp=list(e.cfg.control_config.joint_kp),joint_kd=list(e.cfg.control_config.joint_kd),decoder='incremental: clip(previous_target + increment_rad * tanh(raw_action), target_low, target_high)',handoff='Inherit last actual commanded target from previous policy; preserve motor state and sensor history',observation=[dict(slice=[0,3],name='gyro',units='rad/s, calibrated IMU/body sensor frame'),dict(slice=[3,6],name='projected_gravity',units='unit gravity vector; negative upvector; upright [0,0,-1]'),dict(slice=[6,20],name='joint_positions_minus_default',units='rad, policy order'),dict(slice=[20,34],name='joint_velocities',units='rad/s, policy order'),dict(slice=[34,48],name='previous_target_minus_default_divided_by_0_12',units='dimensionless; use commanded target, not raw network action'),dict(slice=[48,61],name='commands',units='13 zeros for standing')],training_latency=OmegaConf.to_container(c.env.latency,resolve=True),training_DR=e._effective_dr)
 OmegaConf.save(OmegaConf.create(config),D/'INFERENCE_CONFIG.yaml')
 np.savez(D/'observation_normalization.npz',mean=actor.mean.numpy(),std=actor.std.numpy(),epsilon=.01)
 previous=np.tile(defaults,(len(x),1));pt_target=np.clip(previous+inc*np.tanh(expected),low,high);onnx_target=np.clip(previous+inc*np.tanh(actual),low,high)
 decoder_diff=float(np.abs(pt_target-onnx_target).max());assert decoder_diff<1e-5
 verification=dict(passed=True,model_source='model_2800.pt == block_03/final.pt',onnx_checked=True,verification_observations=len(x),normalization_baked_into_onnx=True,max_raw_action_abs_difference=difference,max_decoded_target_difference_rad=decoder_diff,actor_input_dimension=61,actor_output_dimension=14)
 (D/'EXPORT_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
finally:e.close()
for name in ['TRAINING_CONFIG.yaml','REPAIR_MANIFEST.json','PLATEAU_AUDIT_2300_2400.json','FINAL_GOAL_VERIFICATION.json','REWARD_GATE_VERIFICATION.json']:
 shutil.copy2(R/name,D/name)
S=D/'source';S.mkdir(exist_ok=True)
for name in ['pure_recovery.py','pure_reward.py','pure_push_course.py','pure_dr_course.py','pure_distribution.py','walk_stop_audit.py','train.py','run_job.py','evaluate_course.py','evaluate_walk_stop.py','validation_window.py','dr_suite_gate.py']:
 shutil.copy2(R/name,S/name)
V=D/'validation';V.mkdir(exist_ok=True)
for name in ['block_03_completion.json','block_03_course_results.json','block_03_walk_stop_results.json']:
 shutil.copy2(R/name,V/name)
P=D/'physics';P.mkdir(exist_ok=True);shutil.copy2(R/'physics/effective_training.mjb',P/'effective_training.mjb')
tree=ET.parse(R/'physics/scene.xml');compiler=tree.getroot().find('compiler');meshdir=Path(compiler.get('meshdir'));compiler.set('meshdir','meshes')
(P/'meshes').mkdir(exist_ok=True)
for mesh in tree.getroot().findall('.//mesh'):
 if mesh.get('file'):
  src=meshdir/mesh.get('file');name=src.name;shutil.copy2(src,P/'meshes'/name);mesh.set('file',name)
tree.write(P/'scene.xml');mujoco.MjModel.from_xml_path(str(P/'scene.xml'))
# Binary preserves calibrated mechanics; XML is the portable visualization asset.
shutil.copy2(R.parent/'stand_pure_recovery_v6_keep_pose_20261013/extended_reset_bank.npz',D/'extended_reset_bank.npz')
handoff=Path(c.env.handoff_bank);shutil.copy2(handoff,D/'analytic_default_bank.npz')
shutil.copy2(R/'inference_example.py',D/'inference_example.py')
shutil.copy2(R/'DELIVERY_README.txt',D/'README.txt')
manifest={str(p.relative_to(D)):dict(size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(D.rglob('*')) if p.is_file()}
(D/'FILES_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=shutil.make_archive(str(R/'xduck_stand_2800_export'),'zip',root_dir=D)
print(json.dumps(dict(delivery=str(D),archive=archive,verification=verification)),flush=True)
