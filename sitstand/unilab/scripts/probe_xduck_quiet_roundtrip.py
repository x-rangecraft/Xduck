"""Physical sit then rise with actuator/state continuity; no policy training."""
import argparse,json
from pathlib import Path
import numpy as np
from hydra import compose,initialize_config_dir
from unilab.training import BackendAdapter,create_env,ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.quiet_path_probe import (
    support_transfer_nodes,make_support_transfer_path,support_feedback_parameters,
    balance_targets,summarize_quiet_probe)
from unilab.envs.locomotion.xduck_gf43x40.motor_path import ContinuousMotorPath
from unilab.envs.locomotion.xduck_gf43x40.sitstand import LEG_IDS
from unilab.utils.rotation import np_quat_apply_inverse


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];ensure_registries();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.geometry.read_text())['nodes']
    candidate=np.array(json.loads(a.parameters.read_text())['cases']['descent']['parameters'])
    result=dict(conditions='Diagnostic bounded target paths, neutral heads, original GF/limits/mapping/DR. Original standing noisy reset. Sit then rise without reset or actuator-state replacement. Rise starts in actual reached low pose, not independently tilted seated reset. Not PPO.',rows=[])
    bundle={}
    for seed in (6401,7401):
        np.random.seed(seed+300)
        with initialize_config_dir(config_dir=str(root/'conf/ppo'),version_base=None):
            cfg=compose(config_name='config',overrides=['task=xduck_gf43x40_sitstand_head_upright/mjwarp'])
        cfg.env.warp_compiled_model_file=str(a.model);cfg.env.evaluation_case=3
        cfg.env.curriculum.step_cap=32424;cfg.env.max_episode_seconds=16.1
        cfg.env.posture_dwell_seconds=[100.,100.]
        for item in cfg.env.curriculum.head_range_stages:item.ranges=[[0.,0.]]*4
        env=create_env(cfg,num_envs=16,env_cfg_override=BackendAdapter(cfg,root_dir=root).build_task_env_cfg_override())
        try:
            env.step_counter=32424;env._update_curriculum();env.init_state();env.set_autoreset(False)
            stand=env.default_angles.copy();sit=stand.copy();sit[LEG_IDS]=env._sit_legs
            parameters=np.repeat(candidate[None],16,axis=0)
            nodes=support_transfer_nodes(stand,sit,rows,3,flat_endpoint=True)
            down=make_support_transfer_path(parameters,nodes,env._target_low,env._target_high)
            up=ContinuousMotorPath(down.nodes[:,::-1].copy(),down.durations[:,::-1].copy())
            reverse=parameters.copy();reverse[:,10:16]=parameters[:,28:34];reverse[:,28:34]=parameters[:,10:16]
            trace={k:[] for k in ('qpos','qvel','height','tilt','head_tilt','base_speed','target','fallen')}
            for step in range(800):
                if step==400:
                    env.state.info['commands'][:,0]=0.
                rising=step>=400;elapsed=max(0.,(step%400)*.02-1.)
                desired=(up if rising else down).sample(np.full(16,elapsed))
                gains=support_feedback_parameters(reverse if rising else parameters,elapsed)
                gyro=np_quat_apply_inverse(env._backend.get_base_quat(),env._backend.get_base_ang_vel())
                desired=balance_targets(desired,gains,-env._projected_gravity(),gyro,env._target_low,env._target_high)
                action=np.arctanh(np.clip((desired-env._current_action_target)/env._increments,-.999999,.999999))
                env.step(action)
                values=dict(qpos=env._backend.get_physics_state()[:,1:22],qvel=env.get_dof_vel(),height=env._backend.get_base_pos()[:,2],
                    tilt=env.last_tilt,head_tilt=env._geometric_head_score()[0],base_speed=env.last_speed,target=env._current_action_target,fallen=env._failed)
                for k,v in values.items():trace[k].append(v.copy())
            data={k:np.asarray(v) for k,v in trace.items()};metrics=[]
            for case,sl,goal in [('descent',slice(0,400),.2182),('rise',slice(400,800),.3129)]:
                part={k:v[sl] for k,v in data.items()};m=summarize_quiet_probe(part,goal);metrics.append(m)
                row=dict(case=case,seed=seed,trials=16,successes=int(m['success'].sum()),quiet_successes=int(m['quiet'].sum()),falls=int(m['fallen'].sum()),
                    body_peak_median_deg=float(np.median(m['peak_body_deg'])),body_peak_max_deg=float(m['peak_body_deg'].max()),head_deviation_max_deg=float(m['head_deviation_deg'].max()),
                    full_segment_body_peak_max_deg=float(np.rad2deg(part['tilt'].max())))
                result['rows'].append(row);np.savez_compressed(a.output/f'{case}_{seed}.npz',**part)
                for k,v in part.items():bundle[f'{case}_{seed}_{k}']=v
                print(json.dumps(row),flush=True)
            result['rows'].append(dict(case='roundtrip',seed=seed,trials=16,successes=int((metrics[0]['success']&metrics[1]['success']).sum()),quiet_successes=int((metrics[0]['quiet']&metrics[1]['quiet']).sum()),falls=int(data['fallen'].any(axis=0).sum())))
            (a.output/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'trajectory.npz',**bundle)
        finally:env.close()


if __name__=='__main__':main()
