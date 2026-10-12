"""Use source actor leg coordination with diagnostic neutral-head control."""
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from unilab.training import BackendAdapter,create_env,ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.quiet_path_probe import summarize_quiet_probe

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];ensure_registries();torch.set_num_threads(2)
cfg=OmegaConf.create(json.loads((a.checkpoint.parent/'run_config.json').read_text())['config'])
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
actorcfg=OmegaConf.to_container(cfg.algo.actor,resolve=True);actorcfg.pop('class_name')
actor=MLPModel(TensorDict({'actor':torch.zeros(16,61)},batch_size=[16]),{'actor':['actor']},'actor',14,**actorcfg)
actor.load_state_dict(saved['actor_state_dict']);actor.eval();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for case,label in [(3,'descent'),(2,'rise')]:
    for seed in (6401,7401):
        np.random.seed(seed+case*100);torch.manual_seed(seed+case*100)
        c=copy.deepcopy(cfg);c.env.evaluation_case=case;c.env.max_episode_seconds=6.1;c.env.posture_dwell_seconds=[100.,100.];c.env.curriculum.step_cap=32424
        for item in c.env.curriculum.head_range_stages:item.ranges=[[0.,0.]]*4
        env=create_env(c,num_envs=16,env_cfg_override=BackendAdapter(c,root_dir=root).build_task_env_cfg_override())
        try:
            env.step_counter=32424;env._update_curriculum();env.init_state();env.set_autoreset(False)
            trace={k:[] for k in ('qpos','qvel','height','tilt','head_tilt','base_speed','target','fallen')}
            for step in range(300):
                with torch.inference_mode():action=actor(TensorDict({'actor':torch.from_numpy(env.state.obs['obs'].copy())},batch_size=[16])).numpy()
                # Only a diagnostic command; this branch is absent from PPO.
                action[:,5:9]=np.arctanh(np.clip(-env._current_action_target[:,5:9]/env._increments[5:9],-.999999,.999999))
                env.step(action)
                values=dict(qpos=env._backend.get_physics_state()[:,1:22],qvel=env.get_dof_vel(),height=env._backend.get_base_pos()[:,2],tilt=env.last_tilt,head_tilt=env._geometric_head_score()[0],base_speed=env.last_speed,target=env._current_action_target,fallen=env._failed)
                for k,v in values.items():trace[k].append(v.copy())
            d={k:np.asarray(v) for k,v in trace.items()};m=summarize_quiet_probe(d,env.cfg.reward_config.stand_height if case==2 else env.cfg.reward_config.sit_height,movement_start=0.)
            row=dict(case=label,seed=seed,trials=16,successes=int(m['success'].sum()),quiet_successes=int(m['quiet'].sum()),falls=int(m['fallen'].sum()),body_peak_median_deg=float(np.median(m['peak_body_deg'])),head_deviation_max_deg=float(m['head_deviation_deg'].max()))
            rows.append(row);np.savez_compressed(a.output/f'{label}_{seed}.npz',**d)
            (a.output/'results.json').write_text(json.dumps(dict(conditions='Source1351 actor legs, head targets held neutral by diagnostic controller; original GF/limits/mapping/DR. Not a modified learned policy. Full motion includes reset noise.',rows=rows),indent=2));print(json.dumps(row),flush=True)
        finally:env.close()
