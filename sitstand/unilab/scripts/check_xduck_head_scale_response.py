"""Head response diagnostic, not PPO training or learned movement acceptance."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training import BackendAdapter, create_env, ensure_registries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physics-model', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ensure_registries()
    with initialize_config_dir(config_dir=str(root / 'conf/ppo'), version_base=None):
        base = compose(config_name='config', overrides=['task=xduck_gf43x40_sitstand_incremental_neck_limited/mjwarp'])
    base.env.warp_compiled_model_file = str(args.physics_model)
    recorded = np.load(args.trajectory)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for case, label in [(0, 'stand_hold'), (1, 'sit_hold')]:
        replay = recorded[f'{label}_6401_action'][-150:, :, 5:9]
        for scale in (['zero'] if args.baseline_only else [None, .25, .1, .05]):
            np.random.seed(8301 + case)
            cfg = copy.deepcopy(base)
            OmegaConf.set_struct(cfg.env, False)
            cfg.env.evaluation_case = case
            cfg.env.max_episode_seconds = 8.1
            cfg.env.posture_dwell_seconds = [100., 100.]
            cfg.env.head_absolute_position = scale is not None
            cfg.env.head_absolute_scale_rad = scale if isinstance(scale, float) else .25
            env = create_env(cfg, num_envs=16,
                             env_cfg_override=BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override())
            traces = {k: [] for k in ['head_position', 'head_velocity', 'head_target', 'body_tilt', 'height']}
            try:
                env.init_state()
                env.set_autoreset(False)
                # 1s settle, 2s common5deg slow target, 3s identical recorded
                # raw head outputs, 2s zero input. All leg outputs remain zero.
                for step in range(400):
                    action = np.zeros((16, 14))
                    if scale == 'zero':
                        pass
                    elif 50 <= step < 150:
                        desired = env.default_angles[5:9] + np.deg2rad(5.) * np.sin(2 * np.pi * .5 * ((step - 50) * .02))
                        if scale is None:
                            action[:, 5:9] = np.arctanh(np.clip((desired - env._current_action_target[:, 5:9]) / env._increments[5:9], -.999, .999))
                        else:
                            action[:, 5:9] = (desired - env.default_angles[5:9]) / scale
                    elif 150 <= step < 300:
                        action[:, 5:9] = replay[step - 150]
                    elif scale is None:
                        action[:, 5:9] = np.arctanh(np.clip((env.default_angles[5:9] - env._current_action_target[:, 5:9]) / env._increments[5:9], -.999, .999))
                    env.step(action)
                    up = -env._projected_gravity()
                    values = dict(head_position=env.get_dof_pos()[:, 5:9], head_velocity=env.get_dof_vel()[:, 5:9],
                                  head_target=env._current_action_target[:, 5:9],
                                  body_tilt=np.arccos(np.clip(up[:, 2], -1., 1.)), height=env._backend.get_base_pos()[:, 2])
                    for k, v in values.items():
                        traces[k].append(v.copy())
                data = {k: np.asarray(v) for k, v in traces.items()}
                velocity = data['head_velocity']
                rows.append(dict(case=label, mapping='zero_head_baseline' if scale == 'zero' else 'original_incremental' if scale is None else 'absolute', scale_rad=scale,
                                 slow_tracking_error_rms_deg=float(np.rad2deg(np.sqrt(np.mean((data['head_position'][50:150] - data['head_target'][50:150]) ** 2)))),
                                 replay_head_speed_rms_rad_s=float(np.sqrt(np.mean(velocity[150:300] ** 2))),
                                 replay_target_delta_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.diff(data['head_target'][150:300], axis=0) ** 2)))),
                                 quiet_head_speed_rms_rad_s=float(np.sqrt(np.mean(velocity[-50:] ** 2))),
                                 body_tilt_peak_median_deg=float(np.rad2deg(np.median(data['body_tilt'].max(axis=0)))),
                                 falls=int(((data['height'] < .13) | (data['body_tilt'] > np.deg2rad(75))).any(axis=0).sum())))
                print(json.dumps(rows[-1]), flush=True)
                name=f'{label}_{"incremental" if scale is None else str(scale)}'
                np.savez_compressed(args.output / f'{name}.npz', **data)
                (args.output / 'results.json').write_text(json.dumps(dict(
                    purpose='Open-loop head response only; not learned sit/stand success',
                    conditions='16env, fixed seeds, original physical/reset randomization and GF; legs hold sampled reset targets; same5deg slow targets and same recorded raw head outputs; no PPO',
                    limitations='Replay removes actor feedback. Smaller replay response alone does not establish a better trainable scale or eliminate closed-loop jitter.', rows=rows), indent=2))
            finally:
                env.close()


if __name__ == '__main__':
    main()
