from common import *
import torch,json
cfg=OmegaConf.load(R.parent/'xduck_walk_smooth_newstand_20261010/PPO_LOWNOISE_CONFIG.yaml')
cfg.training.task_name='XDuckNaturalGait'
cfg.algo.algorithm.anchor_dataset=str(R/'clean_own_anchor.npz')
cfg.algo.algorithm.anchor_weight=.1
cfg.env.reference_weight_after300=0.
cfg.env.cadence_weight_after300=0.
cfg.env.speed_correction_mode='off'
cfg.env.startup_training=False;cfg.env.startup_reward=False
cfg.env.newstand_bank_path=str(R/'newstand_bank.npz')
cfg.env.newstand_seed=1200091
cfg.env.smooth_width_target_m=.212;cfg.env.smooth_width_weight=3.
cfg.env.smooth_height_weight=0.
cfg.env.smooth_head_target=[.018978888168931007,.059834010899066925]
cfg.env.curriculum.head_bias_stages=[dict(step=0,value=0.)]
cfg.env.max_episode_seconds=8.
z=np.load(R/'clean_seed_eval_trace.npz')
results=json.loads((R/'clean_seed_eval_results.json').read_text())['records']
ids=[x['trace_index'] for x in results if x['healthy'] and x['natural_width']]
assert ids,'No clean natural-width baseline; must validate a viable candidate before preserving its gait'
obs=np.concatenate([z['obs'][150:,j] for j in ids]).astype(np.float32)
mean=np.concatenate([z['raw'][150:,j] for j in ids]).astype(np.float32)
ck=torch.load(R/'clean_roll_seed.pt',map_location='cpu',weights_only=False)
np.savez_compressed(R/'clean_own_anchor.npz',obs=obs,mean=mean,std=ck['actor_state_dict']['distribution.std_param'].numpy())
OmegaConf.save(cfg,R/'PPO_CLEAN_CONFIG.yaml')
(R/'clean_training_manifest.json').write_text(json.dumps(dict(parent='clean_roll_seed.pt',original_parent='clean_parent.pt',new_stand599_inference=0,retention_frames=len(obs),own_qualified_cases=ids,old_actor_labels_used=False,old_reference_reward_disabled=True,fresh_optimizer=True),indent=2))
print('CLEAN_TRAINING_READY',ids,len(obs),flush=True)
