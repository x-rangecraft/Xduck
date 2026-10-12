"""Search and independently validate quiet motor paths; no policy training."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.quiet_path_probe import make_quiet_path, make_support_transfer_path, support_transfer_nodes, balance_targets, support_feedback_parameters, summarize_quiet_probe
from unilab.envs.locomotion.xduck_gf43x40.sitstand import LEG_IDS
from unilab.utils.rotation import np_quat_apply_inverse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--physics-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--population', type=int, default=128)
    p.add_argument('--generations', type=int, default=6)
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--replicates', type=int, default=4)
    p.add_argument('--support-geometry',type=Path)
    p.add_argument('--phase-feedback',action='store_true')
    p.add_argument('--stepping-support',action='store_true')
    p.add_argument('--flat-foot-endpoint',action='store_true')
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    ensure_registries(); a.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(8301)
    # Cold-path FK measures actual CAD joint signs rather than assuming an
    # anatomical convention. Calibrate foot up in the validated stand pose.
    model=mujoco.MjModel.from_binary_path(str(a.physics_model));fk=mujoco.MjData(model)
    stand14=np.zeros(14);stand14[[2,4,11,13]]=[.384,-.384,-.384,.384]
    fk.qpos[:]=0.;fk.qpos[2]=.3129;fk.qpos[3]=1.;fk.qpos[7:]=stand14
    mujoco.mj_fwdPosition(model,fk)
    foot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_ankle_pitch_joint')
    local_up=fk.xmat[foot].reshape(3,3).T@np.array([0.,0.,1.])
    slopes=[]
    for joint in (2,3,4):
        fk.qpos[7:]=stand14;fk.qpos[7+joint]+=.01;mujoco.mj_fwdPosition(model,fk)
        up=fk.xmat[foot].reshape(3,3)@local_up;slopes.append(np.arctan2(up[0],up[2])/.01)
    print('foot_pitch_joint_slopes',slopes,flush=True)
    def run(case, seed, parameters, duration=8.):
        np.random.seed(seed)
        with initialize_config_dir(config_dir=str(root/'conf/ppo'),version_base=None):
            cfg=compose(config_name='config', overrides=['task=xduck_gf43x40_sitstand_head_upright/mjwarp'])
        cfg.env.warp_compiled_model_file=str(a.physics_model)
        cfg.env.evaluation_case=case
        if a.flat_foot_endpoint:
            row=json.loads(a.support_geometry.read_text())['nodes'][-1]
            if max(abs(v) for v in row['residual'])>=1e-5 or row['lowest_z'] < -1e-5 or row['margin_x'] <= 0:
                raise ValueError('Alternative endpoint fails geometric support checks')
            endpoint=np.zeros(14);endpoint[2:5]=row['angles'];endpoint[11:14]=-np.array(row['angles'])
            endpoint[[0,9]]=[row.get('yaw',0.),-row.get('yaw',0.)]
            OmegaConf.update(cfg,'env.sit_leg_angles',endpoint[LEG_IDS].tolist(),force_add=True)
        cfg.env.curriculum.step_cap=32424
        cfg.env.max_episode_seconds=duration+.1
        cfg.env.posture_dwell_seconds=[100.,100.]
        for item in cfg.env.curriculum.head_range_stages: item.ranges=[[0.,0.]]*4
        env=create_env(cfg,num_envs=len(parameters),env_cfg_override=BackendAdapter(cfg,root_dir=root).build_task_env_cfg_override())
        try:
            env.step_counter=32424;env._update_curriculum();env.init_state();env.set_autoreset(False)
            stand=env.default_angles.copy();sit=stand.copy();sit[LEG_IDS]=env._sit_legs
            start,goal=(sit,stand) if case==2 else (stand,sit)
            if a.support_geometry:
                rows=json.loads(a.support_geometry.read_text())['nodes']
                nodes=support_transfer_nodes(stand,sit,rows,case,a.stepping_support,a.flat_foot_endpoint)
                path=make_support_transfer_path(parameters,nodes,env._target_low,env._target_high)
            else:path=make_quiet_path(parameters,start,goal,env._target_low,env._target_high)
            trace={k:[] for k in ('qpos','qvel','height','tilt','head_tilt','base_speed','target','fallen')}
            for step in range(round(duration/.02)):
                desired=path.sample(np.full(len(parameters),max(0.,step*.02-1.)))
                gyro=np_quat_apply_inverse(env._backend.get_base_quat(),env._backend.get_base_ang_vel())
                gains=support_feedback_parameters(parameters,max(0.,step*.02-1.)) if a.phase_feedback else parameters
                desired=balance_targets(desired,gains,-env._projected_gravity(),gyro,env._target_low,env._target_high)
                # This controller only supplies ordinary bounded target increments.
                action=np.arctanh(np.clip((desired-env._current_action_target)/env._increments,-.999999,.999999))
                env.step(action)
                values=dict(qpos=env._backend.get_physics_state()[:,1:22],qvel=env.get_dof_vel(),
                    height=env._backend.get_base_pos()[:,2],tilt=env.last_tilt,head_tilt=env._geometric_head_score()[0],
                    base_speed=env.last_speed,target=env._current_action_target,fallen=env._failed)
                for k,v in values.items():trace[k].append(v.copy())
            data={k:np.asarray(v) for k,v in trace.items()}
            return data,summarize_quiet_probe(data,env.cfg.reward_config.stand_height if case==2 else env.cfg.reward_config.sit_height)
        finally:env.close()
    result=dict(conditions='Diagnostic paths and bounded balance feedback only. Original GF/hard limits/increment decoder, head targets zero, original physical/reset DR, clock1351. Not PPO or imitation. Search failure cannot prove physical impossibility. Movement quality measured after1s initial settling; full traces retain reset.',cases={})
    if a.support_geometry:
        result['path_family']='Upright flat-foot IK descent followed by hip support transfer; independent left/right sagittal corrections and free endpoint leg angles. Reverse for rise. Original noisy stand/sit reset poses retained.'
        result['phase_feedback']=a.phase_feedback
        result['foot_relocation_proposal']=a.stepping_support
        result['alternative_flat_foot_sit_reset']=a.flat_foot_endpoint
    for case,label in [(3,'descent'),(2,'rise')]:
        dimensions=34 if a.phase_feedback else (28 if a.support_geometry else 16)
        mean=np.zeros(dimensions);mean[9]=3.5 if a.support_geometry else 2.5
        if a.warm_start:
            values=np.array(json.loads(a.warm_start.read_text())['cases'][label]['parameters']);mean[:len(values)]=values
        if a.phase_feedback:mean[28:34]=mean[10:16]
        std=np.array([.35,.3,.4]*3+[1.,.8,.8,.08,.08,.4,.04])
        if a.support_geometry:std=np.r_[std,[.15]*9+[.3,.15,.3]]
        if a.phase_feedback:std=np.r_[std,[1.,1.,.08,.08,.6,.05]]
        if a.flat_foot_endpoint:
            # Stay near the independently solved support tube while jointly
            # exploring balance, speed, left/right coordination and endpoint.
            std[:9]=.06;std[16:28]=.03;mean[9]=4.5
        best=None;best_score=-np.inf;history=[]
        for generation in range(a.generations):
            params=rng.normal(mean,std,(a.population,dimensions));params[:,9]=np.clip(params[:,9],1.,5.)
            params[0]=mean
            if best is not None:params[1]=best
            if a.phase_feedback:
                params[2]=mean;params[2,28:34]=0.
                params[3]=mean;params[3,28:34]=-mean[10:16]
            # Seed coordinated ankle flattening while the hip support changes,
            # in addition to stochastic coupled waypoint/feedback proposals.
            for j in range(2,min(10,a.population)) if not a.support_geometry else []:
                params[j]=0.;params[j,9]=1.5+(j-2)*.4
                fractions=np.array([.25,.5,.75])
                stand=np.array([.384,0.,-.384]);sit=np.array([.625,-1.14,-.35])
                start,goal=(sit,stand) if case==2 else (stand,sit)
                nominal=start[None]+fractions[:,None]*(goal-start)[None]
                flat=stand[2]-(slopes[0]*(nominal[:,0]-stand[0])+slopes[1]*(nominal[:,1]-stand[1]))/slopes[2]
                params[j,:9]=np.stack([np.zeros(3),np.zeros(3),flat-nominal[:,2]],axis=1).reshape(-1)
            data,m=run(case,8301+generation,np.repeat(params,a.replicates,axis=0))
            scores=m['score'].reshape(a.population,a.replicates).mean(axis=1)
            idx=np.argsort(scores)[-max(8,a.population//8):]
            winner=int(idx[-1])
            if scores[winner]>best_score:best_score=float(scores[winner]);best=params[winner].copy()
            floors=np.array([.04]*9+[.15,.08,.08,.01,.01,.04,.005]+([.02]*12 if a.support_geometry else [])+([.08,.08,.01,.01,.04,.005] if a.phase_feedback else []))
            mean=params[idx].mean(axis=0);std=np.maximum(params[idx].std(axis=0),floors)
            row=dict(generation=generation,endpoint_successes=int(m['success'].sum()),quiet_successes=int(m['quiet'].sum()),best_score=best_score,best_body_peak_deg=float(np.median(m['peak_body_deg'].reshape(a.population,a.replicates)[winner])),best_head_deviation_deg=float(m['head_deviation_deg'].reshape(a.population,a.replicates)[winner].max()))
            history.append(row);print(label,json.dumps(row),flush=True)
            result['cases'][label]=dict(parameters=best.tolist(),search=history)
            (a.output/'results.json').write_text(json.dumps(result,indent=2))
        validations=[]
        for seed in (6401,7401):
            data,m=run(case,seed+case*100,np.repeat(best[None],16,axis=0))
            row=dict(seed=seed,trials=16,successes=int(m['success'].sum()),quiet_successes=int(m['quiet'].sum()),falls=int(m['fallen'].sum()),body_peak_median_deg=float(np.median(m['peak_body_deg'])),body_peak_max_deg=float(m['peak_body_deg'].max()),head_deviation_max_deg=float(m['head_deviation_deg'].max()))
            validations.append(row);np.savez_compressed(a.output/f'{label}_{seed}.npz',**data)
            print(label,'validation',json.dumps(row),flush=True)
        result['cases'][label]['validation']=validations
        (a.output/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
