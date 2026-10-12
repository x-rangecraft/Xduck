from common import *
import torch,json
old=R.parent/'xduck_clean_natural_gait_20261010'
parts=[]
z=np.load(old/'startup_neural_handoff_trace.npz')
parts.append((z['obs'][:100,9],z['raw'][:100,9],np.ones((100,14)),0))
parts.append((z['obs'][150:,9],z['raw'][150:,9],np.ones((len(z['obs'])-150,14)),1))
z=np.load(old/'directions_switch_125_trace.npz');records=json.loads((old/'directions_switch_125_results.json').read_text())['records']
for rec in records:
 if rec['scenario'] not in ['forward_left','forward_right']:continue
 if not rec['rough_tracking_pass']:continue
 j=rec['trace_index'];o=z['obs'][450:,j];y=z['raw'][450:,j]
 parts.append((o,y,np.ones_like(y),1))
legacy=np.load(R.parent/'xduck_startup_demo_bc_20261009/dataset.npz');o=legacy['obs'];cmd=o[:,48:51]
valid=(cmd[:,0]<-.05)|(abs(cmd[:,2])>.15)
ix=np.flatnonzero(valid);rng=np.random.default_rng(1490091);ix=rng.choice(ix,min(4000,len(ix)),replace=False)
o=o[ix].astype(np.float32);clean=actor('clean_parent.onnx');y=np.array([clean.run(None,{'obs':x[None]})[0][0] for x in o])
mask=np.ones_like(y);mask[:,[1,5,6,7,8,10]]=0
parts.append((o,y,mask,2))
np.savez_compressed(R/'direction_anchor.npz',obs=np.concatenate([p[0] for p in parts]).astype(np.float32),mean=np.concatenate([p[1] for p in parts]).astype(np.float32),mask=np.concatenate([p[2] for p in parts]).astype(np.float32),phase=np.concatenate([np.full(len(p[0]),p[3]) for p in parts]))
cfg=OmegaConf.load(old/'PPO_CLEAN_R3_CONFIG.yaml')
cfg.training.task_name='XDuckDirectionSpeed';cfg.algo.num_envs=128;cfg.algo.seed=1490091
cfg.algo.algorithm.class_name='unilab.algos.torch.masked_anchor:DirectionAnchoredPPO'
cfg.algo.algorithm.anchor_dataset=str(R/'direction_anchor.npz');cfg.algo.algorithm.anchor_weight=.03;cfg.algo.algorithm.anchor_batch_size=384
cfg.algo.algorithm.learning_rate=5e-5;cfg.algo.algorithm.fixed_learning_rate=5e-5
cfg.env.newstand_bank_path=str(R/'newstand_bank.npz');cfg.env.newstand_seed=1490091
cfg.env.max_episode_seconds=10.;cfg.env.yaw_only_tracking=False
cfg.reward.scales.track_linear_velocity=0.;cfg.reward.scales.track_angular_velocity=0.;cfg.reward.scales.command_progress=0.
cfg.env.reference_weight_after300=0.;cfg.env.cadence_weight_after300=0.;cfg.env.speed_correction_mode='off'
OmegaConf.save(cfg,R/'PPO_DIRECTION_CONFIG.yaml')
(R/'training_manifest.json').write_text(json.dumps(dict(parent='accepted_parent.pt',clean_knowledge_teacher='clean_parent.onnx',legacy_states_only=True,legacy_dirty_actor_labels_used=False,anchor_phase_counts={str(i):sum(len(p[0]) for p in parts if p[3]==i) for i in range(3)},masked_legacy_rows=[1,5,6,7,8,10],preserve_pose='NEW_STANDING_TARGET.json',fresh_optimizer=True,actor_normalizer_frozen=True,standing599_training=False),indent=2))
print('DIRECTION_CONFIG_READY',flush=True)
