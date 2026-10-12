"""Paired inference-only fixed-command sweep through the registered physics env."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.acceptance import geometric_success
from unilab.utils.rotation import np_quat_apply


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--project', type=Path, default=Path('.'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ensure_registries()
    torch.set_num_threads(2)
    original = OmegaConf.create(json.loads((a.checkpoint.parent/'run_config.json').read_text())['config'])
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    progress = saved['infos']['unilab_progress']['env_steps_per_env']
    actor_cfg = OmegaConf.to_container(original.algo.actor, resolve=True)
    actor_cfg.pop('class_name')
    actor = MLPModel(TensorDict({'actor': torch.zeros(8, 61)}, batch_size=[8]),
                     {'actor': ['actor']}, 'actor', 14, **actor_cfg)
    actor.load_state_dict(saved['actor_state_dict'])
    actor.eval()
    axis = np.asarray([0., .9999999999932535, 3.673192317532603e-6])
    rows = []
    pairing = []

    def run(candidates, count, seeds, cases, label):
        trajectories = {name: {} for name, command in candidates}
        for seed in seeds:
            for case, case_id, goal in cases:
                cfg = copy.deepcopy(original)
                cfg.env.evaluation_case = case_id
                cfg.env.posture_dwell_seconds = [100., 100.]
                cfg.env.max_episode_seconds = 6.1
                cfg.env.curriculum.step_cap = progress
                cfg.env.commands.head_resampling_time_range = [100., 100.]
                env = create_env(cfg, num_envs=count,
                    env_cfg_override=BackendAdapter(cfg, root_dir=a.project).build_task_env_cfg_override())
                head_id = env._backend.get_body_ids(['head_roll'])
                baseline_snapshot = None
                try:
                    for name, degrees in candidates:
                        radians = np.deg2rad(degrees).tolist()
                        if (np.any(np.asarray(radians) < np.asarray(env.cfg.action_offset_low[5:9]))
                                or np.any(np.asarray(radians) > np.asarray(env.cfg.action_offset_high[5:9]))):
                            print('SKIP command outside original valid command range '+name,flush=True)
                            continue
                        for stage in env.cfg.curriculum.head_range_stages:
                            # Xduck owner reverses curriculum axes before command sampling.
                            stage['ranges'] = [[-v, -v] for v in radians]
                        # Repeat exactly the same RNG draws, including startup mass DR.
                        if hasattr(env, '_startup_mass_scale'):
                            del env._startup_mass_scale
                        np.random.seed(seed + case_id*100)
                        torch.manual_seed(seed + case_id*100)
                        env.step_counter = progress
                        env._update_curriculum()
                        env.init_state()
                        env.set_autoreset(False)
                        snapshot = np.concatenate([
                            env._backend.get_physics_state().ravel(),
                            env._mass_inertia_scale.ravel(), env._foot_friction.ravel(),
                            env._imu_rotation.ravel(), env._encoder_bias.ravel(),
                            env._motor.actuator.friction_scale.ravel()])
                        if baseline_snapshot is None:
                            baseline_snapshot = snapshot.copy()
                        else:
                            assert np.allclose(snapshot, baseline_snapshot, atol=1e-6, rtol=0), (label, case, name)
                        failed = np.zeros(count, dtype=bool)
                        trace = {k: [] for k in ['qpos','height','qvel','target','commands','tilt','base_speed','head_tilt']}
                        for _ in range(300):
                            obs = TensorDict({'actor': torch.from_numpy(env.state.obs['obs'].copy())}, batch_size=[count])
                            with torch.inference_mode():
                                action = actor(obs).numpy()
                            state = env.step(action)
                            failed |= env._failed | state.terminated | state.truncated
                            assert np.allclose(state.info['commands'][:,3:7], radians, atol=1e-6)
                            quat = env._backend.get_body_quat_w(head_id)[:,0]
                            up = np_quat_apply(quat, np.broadcast_to(axis, (count,3)))
                            vals = dict(qpos=env._backend.get_physics_state()[:,1:22].copy(),
                                height=env._backend.get_base_pos()[:,2].copy(), qvel=env.get_dof_vel().copy(),
                                target=env._current_action_target.copy(), commands=state.info['commands'].copy(),
                                tilt=env.last_tilt.copy(), base_speed=env.last_speed.copy(),
                                head_tilt=np.arccos(np.clip(up[:,2],-1,1)))
                            for key, value in vals.items():
                                trace[key].append(value)
                        t = {k: np.asarray(v) for k,v in trace.items()}
                        success, conditions = geometric_success(t['height']-goal,t['tilt'],t['head_tilt'],t['base_speed'],t['qvel'])
                        success &= ~failed
                        upright_endpoint = (conditions['height'][-50:] & conditions['body_upright'][-50:] & conditions['head_upright'][-50:]).all(axis=0) & ~failed
                        height_endpoint = conditions['height'][-50:].all(axis=0) & ~failed
                        neck = np.degrees(t['qpos'][:,:,12])
                        speed = np.degrees(t['qvel'][:,:,5])
                        peak_head = np.degrees(t['head_tilt']).max(axis=0)
                        row = dict(phase=label, candidate=name, command_deg=degrees, case=case, seed=seed,
                            trials=count, falls=int(failed.sum()), geometric_successes=int(success.sum()),
                            upright_height_endpoint=int(upright_endpoint.sum()), height_endpoint=int(height_endpoint.sum()),
                            head_peak_tilt_median_deg=float(np.median(peak_head)),
                            head_over30_trials=int((peak_head>30).sum()),
                            late_head_tilt_mean_deg=float(np.degrees(t['head_tilt'][-50:]).mean()),
                            neck_excursion_median_deg=float(np.median(np.ptp(neck,axis=0))),
                            neck_speed_rms_deg_s=float(np.sqrt(np.mean(speed**2))),
                            body_peak_tilt_median_deg=float(np.median(np.degrees(t['tilt']).max(axis=0))))
                        rows.append(row)
                        for key, value in t.items():
                            trajectories[name][f'{case}_{seed}_{key}'] = value
                        (a.output/'results_partial.json').write_text(json.dumps(rows,indent=2))
                        print(json.dumps(row), flush=True)
                    pairing.append(dict(phase=label,seed=seed,case=case,initial_physics_and_sampled_DR_equal=True))
                finally:
                    env.close()
        for name, traces in trajectories.items():
            np.savez_compressed(a.output/f'{label}_{name}.npz',**traces)

    cases = [('descent',3,.2182),('rise',2,.3129)]
    candidates = [('zero',[0.,0.,0.,0.])]
    candidates += [(f'n{n:+d}_p{h:+d}',[float(n),float(h),0.,0.])
                   for n in [-40,-20,0,20,40] for h in [-40,-20,0,20,40] if (n,h)!=(0,0)]
    run(candidates,8,[6401],cases,'screen')
    def score(name):
        selected = [r for r in rows if r['phase']=='screen' and r['candidate']==name]
        if len(selected)!=2:
            return float('inf')
        return sum(100*r['falls'] + 10*(r['trials']-r['height_endpoint'])
                   + r['head_peak_tilt_median_deg'] + r['late_head_tilt_mean_deg']
                   + .2*r['neck_speed_rms_deg_s'] for r in selected)
    ranking = sorted(candidates,key=lambda item:score(item[0]))
    best = next(item for item in ranking if item[0]!='zero')
    (a.output/'screen_ranking.json').write_text(json.dumps([dict(name=n,command_deg=c,score=score(n)) for n,c in ranking],indent=2))
    print('VALIDATING '+json.dumps(best),flush=True)
    run([candidates[0],best],16,[6401,7401],
        cases+[('sit_hold',1,.2182),('stand_hold',0,.3129)],'validation')
    result = dict(checkpoint=str(a.checkpoint),progress=progress,training=False,
        constant_command_order=['neck_pitch','head_pitch','head_yaw','head_roll'],
        unchanged='GF, physics, action mapping, actor weights, DR2x, initial-state distribution',
        screening_trials_per_case=8,validation_trials_per_case=32,
        selected_candidate=best,pairing=pairing,rows=rows,
        limitation='Finite constant neck/head-pitch grid; no guarantee over arbitrary commands or timed command schedules.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETED',flush=True)


if __name__=='__main__':
    main()
