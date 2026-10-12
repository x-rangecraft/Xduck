"""Experimental cold-materialized domain ensemble using the standard PPO runner.

All domains are sampled together, fixed for each cohort, and used in every
training batch. No model mutation or XML access occurs during step/reset.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.training.xduck_calibrated_training import (
    ROOT, POLICY_JOINTS, CalibratedMixture, materialize, evaluate,
    evaluate_cycle, export_bundle,
)
from unilab.training.xduck_curriculum import load_actor
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.rsl_rl_reset import RslRlVecEnvWrapper

PARENT = ROOT/'training_runs/xduck_calibrated_policy_20260930/new_prior'
SOURCE = PARENT/'train/model_63.pt'
BOUNDS = dict(stiffness_nm_rad=[60.,90.], body_mass_and_inertia_scale=[.9,1.1],
    body_com_delta_m=[-.005,.005], sliding_friction=[.6,1.2],
    joint_zero_delta_deg=[-.5,.5], imu_mount_delta_deg=[-1.,1.],
    initial_motor_position_rad=[-.004,.004], initial_motor_velocity_rad_s=[-.015,.015])


def sample_domain(seed, scale=1.):
    rng = np.random.default_rng(seed)
    # CAD body names are accessed only once in the cold materialization path.
    base = ET.parse(PARENT/'physics/scene.xml').getroot()
    bodies = [b.get('name') for b in base.iter('body') if b.find('inertial') is not None]
    return dict(seed=seed, scale=scale,
        stiffness=75.+scale*rng.uniform(-15,15),
        mount_delta=(scale*rng.uniform(-1,1,2)).tolist(),
        friction=.9+scale*rng.uniform(-.3,.3),
        zero_delta_deg=(scale*rng.uniform(-.5,.5,14)).tolist(),
        body_mass_scale={n:float(1+scale*rng.uniform(-.1,.1)) for n in bodies},
        body_com_delta={n:(scale*rng.uniform(-.005,.005,3)).tolist() for n in bodies})


def materialize_domain(folder, domain, substeps=4, reference=True):
    info = materialize(folder, substeps=substeps, stiffness=domain['stiffness'],
                       mount_delta=domain['mount_delta'])
    scene = folder/'scene.xml'; root = ET.parse(scene).getroot()
    for body in root.iter('body'):
        inertial = body.find('inertial')
        if inertial is None: continue
        name = body.get('name'); factor = domain['body_mass_scale'][name]
        inertial.set('mass', str(float(inertial.get('mass'))*factor))
        for attr in ('diaginertia','fullinertia'):
            if inertial.get(attr):
                inertial.set(attr, ' '.join(map(str,np.fromstring(inertial.get(attr),sep=' ')*factor)))
        inertial.set('pos',' '.join(map(str,np.fromstring(inertial.get('pos'),sep=' ')+domain['body_com_delta'][name])))
    for name,delta in zip(POLICY_JOINTS,domain['zero_delta_deg']):
        joint = root.find(f'.//joint[@name="{name}"]')
        joint.set('ref',str(float(joint.get('ref','0'))-np.deg2rad(delta)))
    # Both interacting collision surfaces have the same sampled sliding friction.
    for geom in root.iter('geom'):
        if geom.get('name')=='floor' or geom.get('class')=='collision':
            geom.set('friction',f"{domain['friction']} .005 .0001")
    scene.write_text(ET.tostring(root,encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(scene))
    info['zero_offsets_rad'] = (np.asarray(info['zero_offsets_rad'])+np.deg2rad(domain['zero_delta_deg'])).tolist()
    info['domain_randomization'] = domain
    info['variant']['domain_seed'] = domain['seed']
    # Height targets stay physical CAD-frame goals; encoder starts stay raw.
    if reference:
        parent = json.loads((PARENT/'physics/physics.json').read_text())
        info['reference_pose_targets'] = parent['reference_pose_targets']
    info['compiled_total_mass_kg'] = float(model.body_mass.sum())
    (folder/'physics.json').write_text(json.dumps(info,indent=2))
    return info


def train(run, iterations=64, cohorts=12, per_cohort=2, resume=None):
    torch.set_num_threads(4); torch.manual_seed(300932); np.random.seed(300932)
    run.mkdir(parents=True,exist_ok=True)
    cfg = OmegaConf.load(run/'config.yaml' if resume else PARENT/'config.yaml')
    folders=[]; domains=[]
    for i in range(cohorts):
        # One nominal cohort preserves performance at the fitted center.
        domain=sample_domain(83000+i,scale=0. if i==0 else 1.)
        folder=run/f'domains/train_{i:02d}'
        materialize_domain(folder,domain); folders.append(folder); domains.append(domain)
    materialize(run/'physics')
    shutil.copy2(SOURCE,run/'initial.pt')
    env=CalibratedMixture(folders,per_cohort,seed=83001)
    cfg.algo.num_envs=env.num_envs; cfg.algo.num_steps_per_env=200; cfg.algo.save_interval=8
    cfg.algo.algorithm.learning_rate=3e-5; cfg.algo.algorithm.learning_rate_ceiling=3e-5
    cfg.algo.algorithm.critic_warmup_updates=4
    cfg.algo.algorithm.diagnostics_path=str(run/'feedback.jsonl')
    cfg.env=OmegaConf.structured(env.cfg)
    cfg.training.log_dir=str(run/'train')
    cfg.domain_randomization=dict(bounds=BOUNDS,domains=domains,seed=300932,
        sampling='jointly sampled cold domains; every PPO batch includes every cohort',
        physical_scene_count=cohorts,envs_per_cohort=per_cohort,
        source_checkpoint=str(SOURCE),physics_dt_s=.00025,policy_dt_s=.02,
        raw_encoder_and_imu=True,hardware_tested=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    runner=OnPolicyRunner(RslRlVecEnvWrapper(env,device='cpu'),
        normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo,resolve=True)),log_dir=str(run/'train'),device='cpu')
    if resume:
        runner.load(str(resume)); runner.current_learning_iteration+=1
    else:
        runner.alg.actor.load_state_dict(load_actor(SOURCE,OmegaConf.load(PARENT/'config.yaml')).state_dict())
        with torch.no_grad(): runner.alg.actor.distribution.std_param.fill_(.008)
    runner.alg.learning_rate=3e-5
    for group in runner.alg.optimizer.param_groups: group['lr']=3e-5
    cfg.algo.max_iterations=runner.current_learning_iteration+iterations
    OmegaConf.save(cfg,run/'config.yaml')
    (run/'domain_randomization.json').write_text(json.dumps(OmegaConf.to_container(cfg.domain_randomization),indent=2))
    shutil.copy2(Path(__file__),run/'source_snapshot.py')
    try: runner.learn(iterations)
    finally: env.close()


def suite(run,checkpoint,count=16,domains=16):
    torch.set_num_threads(4)
    actor=load_actor(checkpoint,OmegaConf.load(run/'config.yaml'))
    suffix='_baseline' if checkpoint.resolve()==SOURCE.resolve() else '_holdout'
    out=run/f'evaluation_{checkpoint.stem}{suffix}'; out.mkdir(exist_ok=True)
    result=dict(checkpoint=str(checkpoint),bounds=BOUNDS,training_domain_seed_range=[83000,83011],
        holdout_domain_seed_range=[94000,94000+domains-1],windows={},cycles={},passed=False,
        criterion='each domain and each direction >=90%; independent combined domains; continuous cycles >=90%',hardware_tested=False)
    for i in range(domains+2):
        label='center' if i==0 else 'center_fine' if i==1 else f'combined_{i-2:02d}'
        domain=sample_domain(94000+max(0,i-2),scale=0. if i<2 else 1.)
        folder=out/label; materialize_domain(folder,domain,substeps=8 if i==1 else 4)
        metrics=evaluate(actor,folder,count=count,seed=96000+i,jitter=True,balanced=True,record=folder/'rollout.npz')
        result['windows'][label]=metrics
        print('DR WINDOW',label,metrics['direction_rates'],flush=True)
        if i in [0,2,3,4]:
            result['cycles'][label]=evaluate_cycle(actor,folder,count=8,seed=97000+i)
        (out/'results.json').write_text(json.dumps(result,indent=2))
    result['passed']=all(min(w['direction_rates'].values())>=.9 for w in result['windows'].values()) and all(
        all(c['standing_survived']) and all(np.mean(s['success'])>=.9 for s in c['segments']) for c in result['cycles'].values())
    (out/'results.json').write_text(json.dumps(result,indent=2)); print('DR GATE',result['passed'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64); p.add_argument('--cohorts',type=int,default=12)
    p.add_argument('--per-cohort',type=int,default=2); p.add_argument('--resume',type=Path)
    p.add_argument('--suite',type=Path); p.add_argument('--count',type=int,default=16); p.add_argument('--domains',type=int,default=16)
    a=p.parse_args()
    if a.suite: suite(a.run,a.suite,a.count,a.domains)
    else: train(a.run,a.iterations,a.cohorts,a.per_cohort,a.resume)


if __name__=='__main__': main()
