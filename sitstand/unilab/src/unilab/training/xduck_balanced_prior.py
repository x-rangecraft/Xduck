"""Local trajectory-search owner with transient balance after real model63 failure.

Search does not select a deployable policy. It first seeks a smaller-tilt prior;
DR, independent validation and exported inference checks remain separate gates.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.training.xduck_calibrated_training import (
    CalibratedEnv, materialize, set_start, prior_reference,
)
from unilab.training.xduck_domain_randomization import PARENT, SOURCE
from unilab.training.xduck_curriculum import load_actor


def search(run, generations=8, population=24):
    torch.set_num_threads(4)
    run.mkdir(parents=True,exist_ok=True);materialize(run/'physics')
    actor=load_actor(SOURCE,OmegaConf.load(PARENT/'config.yaml'))
    rng=np.random.default_rng(301030)
    lo=np.array([.58,-.14,.65,.30,-.5,-.35,-.08,-.4])
    hi=np.array([1.05,.14,1.2,1.3,.5,.35,.08,.4])
    identity=np.array([1.,0.,1.,1.,0.,0.,0.,0.])
    means=np.tile(identity,(2,1));scales=np.tile((hi-lo)/3,(2,1))
    best=[None,None];results=[]
    env=CalibratedEnv(run/'physics',population*2,seed=13031,jitter=False)
    env.init_state();env.set_autoreset(False)
    try:
        for generation in range(generations):
            candidates=np.concatenate([np.clip(rng.normal(means[d],scales[d],(population,8)),lo,hi) for d in range(2)])
            for d in range(2): candidates[d*population]=best[d]['parameters'] if best[d] else identity
            set_start(env,[0]*population+[4]*population)
            alive=np.ones(env.num_envs,bool);survived=np.zeros(env.num_envs)
            peak=np.zeros(env.num_envs);tilt_area=np.zeros(env.num_envs);limit=np.zeros(env.num_envs,bool)
            angular_peak=np.zeros(env.num_envs);velocity_peak=np.zeros(env.num_envs)
            history=[]
            for tick in range(300):
                raw=env.state.obs['obs'].copy();action=prior_reference(actor,raw,candidates)
                env.step(action);alive&=~env.state.terminated;survived+=alive
                tilt=np.rad2deg(env.last_tilt);peak=np.maximum(peak,tilt)
                tilt_area+=np.maximum(tilt-8.,0)*.02
                angular_peak=np.maximum(angular_peak,np.linalg.norm(env._backend.get_base_ang_vel(),axis=1))
                velocity_peak=np.maximum(velocity_peak,np.max(abs(env._backend.get_dof_vel()[:,env.vi]),axis=1))
                limit|=env.last_limit_violation
                history.append(dict(obs=raw,actions=action.copy(),qpos=np.concatenate((env._backend.get_base_pos(),env._backend.get_base_quat(),env._backend.get_dof_pos()),axis=1).copy(),
                    metrics=np.stack((env.last_pose_error,env.last_height_error,env.last_tilt,env.last_speed),axis=1)))
            completed=alive&env.last_success
            # Completion is useful, but no reward for a large-tilt recovered fall.
            score=survived*.1+200*alive+1000*completed
            score-=60*np.maximum(peak-10.,0)+12*tilt_area+30*angular_peak
            score-=150*env.last_pose_error+200*abs(env.last_height_error)+200*limit
            metrics=[]
            for i in range(env.num_envs):
                metrics.append(dict(generation=generation,command=int(env.command[i]),parameters=candidates[i].tolist(),score=float(score[i]),
                    completed=bool(completed[i]),transient_pass=bool(completed[i] and peak[i]<=15.),peak_body_tilt_deg=float(peak[i]),
                    tilt_above8_deg_integral_deg_s=float(tilt_area[i]),peak_body_angular_speed_rad_s=float(angular_peak[i]),
                    peak_motor_speed_rad_s=float(velocity_peak[i]),pose_error_rad=float(env.last_pose_error[i]),
                    height_error_m=float(env.last_height_error[i]),endpoint_tilt_deg=float(np.rad2deg(env.last_tilt[i]))))
            results.extend(metrics)
            for d in range(2):
                ids=np.arange(d*population,(d+1)*population);elite=ids[np.argsort(score[ids])[-max(4,population//4):]];winner=elite[-1]
                means[d]=.25*means[d]+.75*candidates[elite].mean(axis=0)
                scales[d]=np.maximum(.25*scales[d]+.75*candidates[elite].std(axis=0),(hi-lo)*.025)
                if best[d] is None or score[winner]>best[d]['score']:
                    best[d]=metrics[winner]
                    np.savez_compressed(run/f'balanced_best_{d}.npz',**{k:np.asarray([r[k][winner] for r in history]) for k in history[0]})
            (run/'search.json').write_text(json.dumps(dict(source_checkpoint=str(SOURCE),seed=301030,
                reason='real autonomous model63 sit falls; reduce transient tilt before DR PPO',
                body_tilt_gate_deg=15,best=best,results=results),indent=2))
            print('BALANCED SEARCH',generation,best,flush=True)
    finally: env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--generations',type=int,default=8);p.add_argument('--population',type=int,default=24)
    a=p.parse_args();search(a.run,a.generations,a.population)


if __name__=='__main__':main()
