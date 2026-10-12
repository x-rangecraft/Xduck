"""Read live solved Warp contacts without replaying/re-solving recorded poses.

Diagnostic-only instrumentation; does not change training or contact semantics.
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import torch
import mujoco
import mujoco_warp as mw
import warp as wp
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

R = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--tag', required=True)
p.add_argument('--seconds', type=float, default=8)
p.add_argument('--seed', type=int, default=50021)
a = p.parse_args()
sys.path.insert(0, str(R / 'project/scripts'))
import train_rsl_rl as E


def segments(mask, dt):
    """Only completed intervals; exclude censored intervals at either edge."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(t), float((t-s)*dt)) for s, t in zip(starts, ends)
            if s > 0 and t < len(mask)]


E.ensure_registries()
E.patch_rsl_rl_resume_state()
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.set_num_threads(2)
c = OmegaConf.load(a.config)
c.env.audit_velocity = .3
c.env.max_episode_seconds = max(20., a.seconds + 5)
e = E.create_env(c, num_envs=4, env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
    algo = E._algo_config_dict(c)
    wrapper = E._resolve_ppo_wrapper_cls(algo)(e, device='cpu')
    train = E.normalize_ppo_train_cfg(algo)
    train['logger'] = 'none'
    runner = OnPolicyRunner(wrapper, train, log_dir=None, device='cpu')
    runner.load(str(a.checkpoint), map_location='cpu')
    actor = runner.get_inference_policy(device='cpu')
    e.set_autoreset(False)
    b = e._backend
    m, dm, d = b._cpu_model, b._device_model, b._device_data
    passive_qpos_indices = np.setdiff1d(np.arange(7,m.nq), m.jnt_qposadr[m.actuator_trnid[:,0]])
    names = ['left_foot_collision', 'left_ankle_collision',
             'right_foot_collision', 'right_ankle_collision']
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    assert min(gids + [floor]) >= 0
    ids = wp.array(np.arange(d.naconmax, dtype=np.int32), dtype=wp.int32, device='cuda:0')
    forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device='cuda:0')
    trace = {k: [] for k in ('normal_N', 'vertical_N', 'found', 'sensor_found',
                            'passive_rad', 'foot_vel', 'bodyweight_N', 'valid')}
    alive = np.ones(4, dtype=bool)
    parity = []
    original_step = b.step

    def record():
        count = int(d.nacon.numpy()[0])
        if count > d.naconmax or np.any(d.nefc.numpy() > d.njmax):
            raise RuntimeError('Contact/constraint capacity exceeded')
        mw.contact_force(dm, d, ids, False, forces)
        local = forces.numpy()[:count].copy()
        geoms = d.contact.geom.numpy()[:count]
        worlds = d.contact.worldid.numpy()[:count]
        frames = d.contact.frame.numpy()[:count]
        normal = np.zeros((4, 4))
        vertical = np.zeros((4, 4))
        found = np.zeros((4, 4), dtype=bool)
        world_force = np.einsum('ni,nij->nj', local[:, :3], frames)
        for j, gid in enumerate(gids):
            mask = ((geoms[:, 0] == floor) & (geoms[:, 1] == gid)) | (
                (geoms[:, 1] == floor) & (geoms[:, 0] == gid))
            np.add.at(normal[:, j], worlds[mask], np.maximum(0, local[mask, 0]))
            sign = np.where(geoms[mask, 0] == floor, 1., -1.)
            np.add.at(vertical[:, j], worlds[mask], world_force[mask, 2] * sign)
            found[worlds[mask], j] = True
        # Validate the GPU force decoding against mj_contactForce on exported
        # solved data, without calling mj_forward or changing DR/velocities.
        if len(parity) < 3 and normal.sum() > 1:
            cpu_d = mujoco.MjData(m)
            mw.get_data_into(cpu_d, m, d, world_id=0)
            cpu_normal = np.zeros(4)
            wrench = np.zeros(6)
            for k in range(cpu_d.ncon):
                cg = cpu_d.contact[k].geom
                if floor not in cg:
                    continue
                other = int(cg[1] if cg[0] == floor else cg[0])
                if other in gids:
                    mujoco.mj_contactForce(m, cpu_d, k, wrench)
                    cpu_normal[gids.index(other)] += max(0, wrench[0])
            error = float(np.max(np.abs(cpu_normal - normal[0])))
            parity.append(error)
            if error > .01:
                raise RuntimeError(f'Force export mismatch: {error} N')
        weight = dm.body_mass.numpy().sum(axis=1) * abs(float(m.opt.gravity[2]))
        state = b.get_physics_state()
        values = dict(normal_N=normal, vertical_N=vertical, found=found,
                      sensor_found=np.concatenate([b.get_sensor_data(n) for n in e._cfg.sensor.feet_found], axis=1)>0,
                      passive_rad=state[:, 1 + passive_qpos_indices],
                      foot_vel=np.stack([b.get_sensor_data(n) for n in e._cfg.sensor.feet_vel], axis=1),
                      bodyweight_N=weight, valid=alive.copy())
        for key, value in values.items():
            trace[key].append(np.asarray(value).copy())

    def observed_step(ctrl, nsteps=1):
        # The owner already synchronizes and refreshes sensors after each
        # millisecond inside this loop. Split only to observe those states.
        for _ in range(nsteps):
            result = original_step(ctrl, nsteps=1)
            record()
        return result

    b.step = observed_step
    velocities, failures = [], []
    for i in range(round(a.seconds / c.env.ctrl_dt)):
        with torch.inference_mode():
            action = actor(wrapper.get_observations(), stochastic_output=False).numpy()
        state = e.step(action)
        alive &= ~np.asarray(state.terminated, dtype=bool)
        velocities.append(e.get_local_linvel()[:, 0].copy())
        failures.append(np.asarray(state.terminated).copy())
        if i % 100 == 0:
            print(f'force diagnostic {i*c.env.ctrl_dt:.1f}/{a.seconds}s', flush=True)
    z = {k: np.asarray(v) for k, v in trace.items()}
    z['vx'] = np.asarray(velocities)
    z['failure'] = np.asarray(failures)
    assert len(z['normal_N']) == round(a.seconds / c.env.sim_dt)
    if not all(np.isfinite(v).all() for v in z.values()):
        raise RuntimeError('Nonfinite force diagnostic')
    if len(parity) != 3:
        raise RuntimeError('Insufficient loaded samples for force parity validation')
    out = R / 'diagnostics'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'{a.tag}.npz', **z)
    rows = []
    dt = float(c.env.sim_dt)
    for j in range(4):
        valid = z['valid'][:, j] & (np.arange(len(z['valid'])) * dt >= 2.)
        f = z['normal_N'][valid, j]
        bw = z['bodyweight_N'][valid, j]
        if not len(f):
            rows.append({'env': j, 'error': 'failed before 2-second warmup'})
            continue
        load = np.stack([f[:, :2].sum(axis=1), f[:, 2:].sum(axis=1)], axis=1)
        geom = z['found'][valid, j]
        sensor = z['sensor_found'][valid, j]
        flex = z['passive_rad'][valid, j]
        row = {'env': j, 'valid_seconds_after_2s': len(f)*dt,
               'failure_seen': bool(z['failure'][:, j].any()),
               'mean_vx_after_2s_m_s': float(z['vx'][100:, j].mean()) if alive[j] else None,
               'mean_total_normal_over_bodyweight': float(np.mean(f.sum(axis=1)/bw)),
               'mean_force_N_per_geom': f.mean(axis=0).tolist(),
               'normal_impulse_share_per_geom': (f.sum(axis=0)/max(f.sum(), 1e-12)).tolist(),
               'normal_force_N_p95_per_geom': np.percentile(f, 95, axis=0).tolist(),
               'found_fraction_per_geom': geom.mean(axis=0).tolist(),
               'sensor_vs_geom_union_disagreement': (sensor != np.stack([geom[:, :2].any(axis=1), geom[:, 2:].any(axis=1)], axis=1)).mean(axis=0).tolist(),
               'thresholds': {}}
        for threshold in (.01, .05, .10):
            loaded = load > threshold*bw[:, None]
            single = loaded.sum(axis=1) == 1
            swing_events = []
            side_stats = []
            for side in (0, 1):
                sustained = (~loaded[:, side]) & loaded[:, 1-side] & (~geom[:,2*side:2*side+2].any(axis=1))
                events = segments(sustained, dt)
                durations = [event[2] for event in events]
                swing_events.extend((s, side) for s, t, duration in events if duration >= .181)
                sole, ankle = f[:, 2*side], f[:, 2*side+1]
                side_stats.append({
                    'ankle_load_without_sole_fraction': float(((ankle > threshold*bw) & (sole <= threshold*bw)).mean()),
                    'found_but_side_load_below_threshold_fraction': float((sensor[:, side] & ~loaded[:, side]).mean()),
                    'complete_supported_swing_count': len(durations),
                    'complete_supported_swing_ms_p50_p95': (1000*np.percentile(durations, [50, 95])).tolist() if durations else [],
                    'supported_swings_ge181ms': sum(x >= .181 for x in durations),
                    'supported_swings_ge100ms': sum(x >= .100 for x in durations),
                    'supported_swings_ge140ms': sum(x >= .140 for x in durations),
                    'supported_swings_ge220ms': sum(x >= .220 for x in durations),
                    'supported_swing_max_ms': 1000*max(durations, default=0.),
                })
            swing_events.sort()
            row['thresholds'][str(threshold)] = {
                'single_support_fraction': float(single.mean()),
                'double_support_fraction': float((loaded.sum(axis=1) == 2).mean()),
                'no_support_fraction': float((loaded.sum(axis=1) == 0).mean()),
                'left_right': side_stats,
                'qualifying_swing_sequence': [side for _, side in swing_events],
            }
        passive_ids = passive_qpos_indices
        order = np.argsort(np.percentile(np.abs(flex), 95, axis=0))[::-1][:8] if flex.shape[1] else []
        row['largest_passive_deformation'] = []
        for k in order:
            joint = int(np.flatnonzero(m.jnt_qposadr == passive_ids[k])[0])
            row['largest_passive_deformation'].append({
                'joint': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint),
                'abs_deg_p95': float(np.rad2deg(np.percentile(np.abs(flex[:, k]), 95))),
                'abs_deg_max': float(np.rad2deg(np.max(np.abs(flex[:, k]))))})
        rows.append(row)
    report = {'checkpoint': str(a.checkpoint), 'config': str(a.config), 'seed': a.seed,
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
              'sample_dt_s': dt, 'sole_tilt_deg': np.rad2deg(e._sole_angles).tolist(), 'swing_requires_no_sole_or_ankle_contact': True, 'seconds_per_env': a.seconds, 'geom_order': names,
              'method': 'live solved GPU contacts every motor millisecond; no mj_forward replay; no autoreset; exclude first 2s and samples after first failure',
              'caveats': [f'force sample dt={dt}s; mechanical dt={dt/int(c.env.physics_microsteps)}s',
                          'force thresholds are diagnostic sensitivity checks, not training changes',
                          'complete swing statistics omit intervals touching evaluation boundaries'],
              'gpu_export_mj_contactForce_max_abs_error_N': parity, 'rows': rows}
    (out / f'{a.tag}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
finally:
    e.close()
