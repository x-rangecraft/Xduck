"""Experimental, local-only adaptation to the 2026-09-29 dynamics hypothesis.

Uses the existing SimBackend, NpEnv reset/timeout contract and RSL PPO runner.
The backend retains all 16 hinge coordinates; only 14 encoder coordinates enter
the actor. No production task registration or hardware calibration is implied.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
import zipfile
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.actuators.gf43x40.actuator import GF43X40Parameters
from unilab.actuators.gf43x40.robot import GF43X40RobotMotor
from unilab.base.backend import create_backend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv
from unilab.base.scene import SceneCfg
from unilab.training import xduck_dynamics_identification as identification
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.rsl_rl_reset import RslRlVecEnvWrapper
from unilab.training.xduck_curriculum import load_actor
from unilab.training.xduck_imitation import actor_batch

ROOT = identification.ROOT
SOURCE_CFG = ROOT / 'training_runs/xduck_rise_com_continue100_20260929/com_1/config.yaml'
SOURCE_PT = SOURCE_CFG.parent / 'train/model_267.pt'
UPLOAD = Path('/Users/mac/Desktop/xc_upload_sitstand267_fix_20260929_1936')


def consumer_module():
    spec = importlib.util.spec_from_file_location('xduck_api_v2', UPLOAD / 'xc_sitstand_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize(folder, variant):
    """Cold path: persist the exact physical hypothesis and measured starts."""
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads((identification.OUT / 'metadata.json').read_text())
    pars = GF43X40Parameters.from_bundle()
    m, d, qi, vi, ai, fi = identification.model(variant, meta, pars)
    # mj_saveLastXML writes current model parameters, including the COM shift.
    mujoco.mj_saveLastXML(str(folder / 'scene.xml'), m)
    starts = []
    for label, rows in identification.trials():
        r = rows[0]
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0, 0, .32]
        d.qpos[3:7] = identification.quat(*r['attitude_rpy'][:2])
        d.qpos[qi] = np.asarray(r['positions'])[identification.INDEX]
        d.qvel[vi] = np.asarray(r['velocities'])[identification.INDEX]
        d.qvel[3:6] = r['imu_gyro']
        mujoco.mj_forward(m, d)
        lows = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                verts = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
                lows.append((verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
        d.qpos[2] -= min(lows)
        starts.append({'label': label, 'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist(), 'command': int('sit' in label)})
    info = {'variant': variant, 'joints': meta['policy_joints'], 'home': meta['home'],
            'actuator_indices': ai.tolist(), 'flex_indices': (fi-7).tolist(), 'starts': starts}
    (folder / 'physics.json').write_text(json.dumps(info, indent=2))
    return info


class FlexAdaptationEnv(NpEnv):
    """Partial actuation without changing the backend's full-state contract."""
    def __init__(self, folder, count=16, seed=30, jitter=True, reward_weights=None, start_probabilities=None):
        self.physics = json.loads((folder / 'physics.json').read_text())
        self.consumer = consumer_module()
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter
        self.start_probabilities = start_probabilities
        self.reward_weights = dict(reward_weights or {'upright':4,'height':3,'stillness':1,'speed':-2,'action_change':-.1,'success':2,'failure':-8})
        cfg = EnvCfg(scene=SceneCfg(str(folder / 'scene.xml')), sim_dt=.001 / self.physics['variant']['physics_substeps'], ctrl_dt=.02, max_episode_seconds=4.)
        backend = create_backend('mujoco', cfg.scene, count, cfg.sim_dt, base_name='base_link', add_body_sensors=True, chunk_size=1, adaptive_chunk_size=False)
        backend.materialize()
        super().__init__(cfg, backend, count)
        self.qi = backend.get_joint_dof_pos_indices(self.physics['joints'])
        self.vi = backend.get_joint_dof_vel_indices(self.physics['joints'])
        self.ai = np.asarray(self.physics['actuator_indices'])
        self.fi = np.asarray(self.physics['flex_indices'], dtype=int)
        self.base_ids = backend.get_body_ids(['base_link'])
        self.home = np.asarray(self.physics['home'])
        self.motor = GF43X40RobotMotor(count, 14, kp=self.consumer.KP, kd=self.consumer.KD, jitter_mode='measured_20260918', jitter_seed=seed, envelope_mode='continuous_conservative')
        self.command = np.zeros(count)
        self.age = np.zeros(count)
        self.last_action = np.zeros((count, 14))
        self.previous_velocity = np.zeros((count, 14))
        self.previous_imu = np.zeros((count, 6))
        self.selected_start = np.zeros(count, dtype=int)
        self.tick = 0
        self.native_ctrl = np.zeros((count, 14))
        self.hold = np.zeros(count, dtype=int)
        self.last_success = np.zeros(count, dtype=bool)
        self.last_limit_violation = np.zeros(count, dtype=bool)
        self.last_reward_terms = {}
        backend.set_pre_step_control(self.motor_tick)

    @property
    def obs_groups_spec(self):
        return {'obs': 62, 'critic': 74}

    @property
    def action_space(self):
        return gym.spaces.Box(-3., 3., shape=(14,), dtype=np.float32)

    def motor_tick(self, backend, target):
        if self.tick % self.physics['variant']['physics_substeps'] == 0:
            q, v = backend.get_dof_pos()[:, self.qi], backend.get_dof_vel()[:, self.vi]
            self.motor.finish_substep(q, v)
            tau = self.motor.begin_substep(target, q, v)
            self.native_ctrl[:, self.ai] = q + tau
        self.tick += 1
        return self.native_ctrl

    def measured(self):
        q = self.motor.quantize_position_feedback(self._backend.get_dof_pos()[:, self.qi])
        v = self.motor.quantize_velocity_feedback(self._backend.get_dof_vel()[:, self.vi])
        w, x, y, z = self._backend.get_base_quat().T
        gravity = np.stack((2*(w*y-x*z), -2*(w*x+y*z), 2*(x*x+y*y)-1), axis=1)
        # Explicitly use the local-frame sensor, not world angular velocity.
        gyro = self._backend.get_body_ang_vel_b(self.base_ids)[:, 0]
        return q, v, np.concatenate((gyro, gravity), axis=1)

    def observations(self, advance=False):
        q, v, imu = self.measured()
        command = np.zeros((self.num_envs, 13)); command[:, 0] = self.command
        obs = np.concatenate((self.previous_imu, q-self.home, self.previous_velocity, self.last_action, command, np.minimum(self.age[:, None]/4, 1)), axis=1)
        flex = np.zeros((self.num_envs, 2)); flexvel = flex.copy()
        if len(self.fi):
            flex[:] = self._backend.get_dof_pos()[:, self.fi]
            flexvel[:] = self._backend.get_dof_vel()[:, self.fi]
        # Privileged critic only; no flexible coordinate enters the actor.
        critic = np.concatenate((obs, self._backend.get_base_pos()[:, 2:3], self._backend.get_base_lin_vel(), flex, flexvel, imu[:, 3:6], self.hold[:, None]/25), axis=1)
        if advance:
            self.previous_velocity[:] = v
            self.previous_imu[:] = imu
        return {'obs': obs.astype(np.float32), 'critic': critic.astype(np.float32)}

    def reset(self, env_indices):
        ids = np.asarray(env_indices, dtype=int)
        # Equal sit/rise weighting despite two recorded descent trials.
        choice = np.where(self.rng.random(len(ids)) < .5, self.rng.integers(0, 2, len(ids)), 2)
        if self.start_probabilities is not None:
            choice = self.rng.choice(len(self.physics['starts']),len(ids),p=self.start_probabilities)
        self.selected_start[ids] = choice
        starts = self.physics['starts']
        q = np.asarray([starts[i]['qpos'] for i in choice]); v = np.asarray([starts[i]['qvel'] for i in choice])
        if self.jitter:
            q[:, self.qi+7] += self.rng.uniform(-.004, .004, (len(ids), 14))
            v[:, self.vi+6] += self.rng.uniform(-.015, .015, (len(ids), 14))
        self._backend.set_state(ids, q, v)
        self.motor.reset(ids)
        self.age[ids] = 0; self.last_action[ids] = 0; self.hold[ids] = 0
        self.command[ids] = [starts[i]['command'] for i in choice]
        _, _, imu = self.measured()
        # A reset must not reuse the preceding episode's transport feedback.
        qm = self.motor.quantize_position_feedback(q[:, self.qi+7], ids)
        vm = self.motor.quantize_velocity_feedback(v[:, self.vi+6], ids)
        self.previous_velocity[ids] = vm; self.previous_imu[ids] = imu[ids]
        obs = {k: a[ids].copy() for k, a in self.observations().items()}
        for a in obs.values():
            a[:, 6:20] = qm-self.home
        return obs, {}

    def apply_action(self, actions, state):
        if actions.shape != (self.num_envs, 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid policy action')
        self.action_change = np.square(actions-self.last_action).mean(axis=1)
        self.last_action[:] = actions
        return np.clip(self.home + actions, self.consumer.TARGET_LOW, self.consumer.TARGET_HIGH)

    def update_state(self, state):
        q = self._backend.get_dof_pos()[:, self.qi]
        v = self._backend.get_dof_vel()[:, self.vi]
        self.motor.finish_substep(q, v)
        self.age += .02
        _, _, imu = self.measured()
        tilt = np.arccos(np.clip(-imu[:, 5], -1, 1))
        height = self._backend.get_base_pos()[:, 2]
        speed = np.linalg.norm(self._backend.get_base_lin_vel(), axis=1)
        goal_height = np.where(self.command > .5, .2182, .3129)
        # Smooth transition-height target; reward upright balance throughout.
        progress = np.minimum(self.age/2.5, 1)
        start_height = np.where(self.command > .5, .3129, .2182)
        desired = start_height + (goal_height-start_height)*(3*progress**2-2*progress**3)
        upright = np.exp(-np.square(tilt/.18))
        height_reward = np.exp(-np.square((height-desired)/.035))
        still = np.exp(-np.square(speed/.08))
        limits = (q < self.consumer.TARGET_LOW-.01) | (q > self.consumer.TARGET_HIGH+.01)
        self.last_limit_violation = limits.any(axis=1)
        success = (np.abs(height-goal_height)<.01) & (tilt<np.deg2rad(5)) & (speed<.03)
        self.hold = np.where(success, self.hold+1, 0)
        self.last_success = self.hold >= 25
        failed = (tilt > np.deg2rad(40)) | (height < .13) | self.last_limit_violation
        # Reward scales are recorded by the run config; no imitation update
        # is applied after PPO, so recovery gradients cannot be erased.
        rw = self.reward_weights
        reward = .02*(rw['upright']*upright + rw['height']*height_reward + rw['stillness']*still + rw['speed']*speed + rw['action_change']*self.action_change) + rw['success']*(self.hold == 25) + rw['failure']*failed
        state.obs = self.observations(advance=True)
        state.reward = reward.astype(np.float32)
        state.terminated = failed
        state.info['log'] = {'balance/upright': float(upright.mean()), 'balance/height': float(height_reward.mean()), 'balance/success': float(self.last_success.mean())}
        return state

    def close(self):
        self._backend.set_pre_step_control(None)
        super().close()


def add_settled_standing_start(folder):
    """Cold-path reset bank entry after an actual five-second stand hold."""
    env = FlexAdaptationEnv(folder,1,seed=31,jitter=False)
    env.init_state(); env.set_autoreset(False)
    start = env.physics['starts'][0]
    env._backend.set_state(np.array([0]),np.asarray([start['qpos']]),np.asarray([start['qvel']]))
    env.motor.reset(np.array([0])); env.command[:] = 0
    alive = True
    for _ in range(250):
        env.step(np.zeros((1,14))); alive &= not env.state.terminated[0]
    if not alive: raise RuntimeError('Standing reset-bank preparation fell')
    backend = env._backend
    q = np.concatenate((backend.get_base_pos(),backend.get_base_quat(),backend.get_dof_pos()),axis=1)[0]
    v = np.concatenate((backend.get_base_lin_vel(),backend.get_body_ang_vel_b(env.base_ids)[:,0],backend.get_dof_vel()),axis=1)[0]
    physics = env.physics
    physics['starts'].append({'label':'stand5s_sit','qpos':q.tolist(),'qvel':v.tolist(),'command':1})
    (folder/'physics.json').write_text(json.dumps(physics,indent=2))
    env.close()


def evaluate(actor, folder, count=12, seed=701, jitter=False, record_path=None):
    env = FlexAdaptationEnv(folder, count, seed, jitter)
    env.init_state(); env.set_autoreset(False)
    # Include exactly equal counts of all three measured start cases.
    ids = np.arange(count); choices = ids % 3
    starts = env.physics['starts']
    q = np.asarray([starts[i]['qpos'] for i in choices]); vel = np.asarray([starts[i]['qvel'] for i in choices])
    if jitter:
        q[:, env.qi+7] += env.rng.uniform(-.004, .004, (count, 14))
        vel[:, env.vi+6] += env.rng.uniform(-.015, .015, (count, 14))
    env._backend.set_state(ids, q, vel)
    env.motor.reset(ids); env.command[:] = [starts[i]['command'] for i in choices]
    _, v, imu = env.measured(); env.previous_velocity[:] = v; env.previous_imu[:] = imu
    env.state.info['steps'][:] = 0; env.state.obs = env.observations()
    alive = np.ones(count, bool); first = np.full(count, -1); traces = []; limit = np.zeros(count, bool)
    goal = np.tile(env.home, (count, 1))
    goal[env.command > .5] += np.asarray([0,0,.241,-1.14,.034,0,0,0,0,0,0,-.241,1.14,-.034])
    held_pose = np.ones(count, bool); peak_flex = np.zeros(count)
    recorded_obs = []; recorded_actions = []; recorded_qpos = []
    actor.eval()
    begin = time.monotonic()
    for step in range(200):
        if record_path is not None:
            recorded_obs.append(env.state.obs['obs'].copy())
        with torch.no_grad(): a = actor(actor_batch(env.state.obs['obs'])).numpy()
        if record_path is not None: recorded_actions.append(a.copy())
        state = env.step(a)
        failed = alive & state.terminated; first[failed] = step+1; alive &= ~state.terminated
        limit |= env.last_limit_violation
        pose = np.sqrt(np.mean((env._backend.get_dof_pos()[:, env.qi]-goal)**2, axis=1))
        if step >= 175: held_pose &= pose < .08
        if len(env.fi): peak_flex = np.maximum(peak_flex,np.abs(env._backend.get_dof_pos()[:, env.fi]).max(axis=1))
        if record_path is not None:
            recorded_qpos.append(np.concatenate((env._backend.get_base_pos(),env._backend.get_base_quat(),env._backend.get_dof_pos()),axis=1).copy())
        quat = env._backend.get_base_quat(); w,x,y,z = quat.T
        traces.append(np.stack((np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1))),env._backend.get_base_pos()[:,2]),axis=1))
    success = alive & env.last_success & held_pose
    result = {'seed': seed, 'initial_jitter':jitter, 'seconds': time.monotonic()-begin, 'success': success.tolist(), 'alive': alive.tolist(), 'first_failure_s': [float(x*.02) if x>=0 else None for x in first], 'limit_violation':limit.tolist(), 'cases':[starts[i]['label'] for i in choices], 'rate':float(success.mean()), 'final_pose_rms_rad':pose.tolist(), 'peak_flex_deg':np.rad2deg(peak_flex).tolist()}
    if record_path is not None:
        np.savez_compressed(record_path,obs=np.asarray(recorded_obs),actions=np.asarray(recorded_actions),qpos=np.asarray(recorded_qpos),trace=np.asarray(traces))
    env.close()
    return result, np.asarray(traces)


def evaluate_cycle(actor, folder, count=6, seed=1709):
    """Five seconds holding stand, then sit and rise without physical resets."""
    env = FlexAdaptationEnv(folder,count,seed,jitter=False)
    env.init_state(); env.set_autoreset(False)
    ids = np.arange(count); starts = env.physics['starts']
    choice = ids % 2
    env._backend.set_state(ids,np.asarray([starts[i]['qpos'] for i in choice]),np.asarray([starts[i]['qvel'] for i in choice]))
    env.motor.reset(ids); env.command[:] = 0
    alive = np.ones(count,bool); result = {'standing_seconds':5,'seconds_per_action':6}
    for _ in range(250):
        env.step(np.zeros((count,14)))
        alive &= ~env.state.terminated
    result['standing_survived'] = alive.tolist()
    for label, command in [('sit',1),('rise',0)]:
        env.command[:] = command; env.age[:] = 0; env.last_action[:] = 0; env.hold[:] = 0
        _, vel, imu = env.measured(); env.previous_velocity[:] = vel; env.previous_imu[:] = imu
        env.state.obs = env.observations(); held_pose = np.ones(count,bool)
        goal = env.home + command*np.asarray([0,0,.241,-1.14,.034,0,0,0,0,0,0,-.241,1.14,-.034])
        for step in range(300):
            with torch.no_grad(): a = actor(actor_batch(env.state.obs['obs'])).numpy()
            env.step(a); alive &= ~env.state.terminated
            if step>=275:
                held_pose &= np.sqrt(np.mean((env._backend.get_dof_pos()[:,env.qi]-goal)**2,axis=1))<.08
        result[label] = {'success':(alive & env.last_success & held_pose).tolist(),'alive':alive.tolist()}
    env.close()
    return result


def evaluation_suite(run, checkpoint):
    """Predeclared center, nearby-physics, original nominal and cycle gates."""
    from unilab.training.xduck_imitation import rollout, stage_config
    cfg = OmegaConf.load(run/'config.yaml'); actor = load_actor(checkpoint,cfg)
    output = run/('evaluation_'+checkpoint.stem); output.mkdir(exist_ok=True)
    center = json.loads((run/'physics/physics.json').read_text())['variant']
    result = {'checkpoint':str(checkpoint),'criterion':'No termination/limits for 4s, last .5s height error <1cm, tilt <5deg, speed <3cm/s, pose RMS <.08rad; >=90% in every window','windows':{}}
    for label, changes in [('center',{}),('softer',{'stiffness':60}),('stiffer',{'stiffness':90}),('com_minus10mm',{'com_delta':[-.01,0,0]}),('com_minus30mm',{'com_delta':[-.03,0,0]}),('rigid_measured_start',{'flex':[],'com_delta':[0,0,0]})]:
        folder = output/label
        materialize(folder,{**center,**changes})
        metrics,_ = evaluate(actor,folder,count=12 if label=='center' else 6,seed=1709,jitter=True,record_path=folder/'rollout.npz')
        result['windows'][label] = metrics
        (output/'results.json').write_text(json.dumps(result,indent=2))
        print('WINDOW',label,metrics['rate'],metrics['alive'],flush=True)
    result['cycle'] = evaluate_cycle(actor,run/'physics')
    nominal,_ = rollout(stage_config(OmegaConf.load(SOURCE_CFG),0),ROOT,actor=actor,count=8,segments=4,seed=2709,independent=True)
    result['original_nominal'] = nominal
    result['passed'] = all(x['rate']>=.9 for x in result['windows'].values()) and all(nominal[k]['rate']>=.9 for k in ['rise','descent']) and all(np.mean(result['cycle'][k]['success'])>=.9 for k in ['sit','rise'])
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('GATE',result['passed'],'original',nominal,'cycle',result['cycle'],flush=True)
    return result


def export_bundle(run, checkpoint, evaluation):
    """Export an evaluated actor and retain the gate result in the bundle."""
    import onnxruntime as ort
    from unilab.training.xduck_pretrain import TensorPolicy
    cfg = OmegaConf.load(run/'config.yaml'); actor = load_actor(checkpoint,cfg)
    out = run/(('deployment_' if evaluation['passed'] else 'diagnostic_export_NOT_READY_')+checkpoint.stem); out.mkdir(exist_ok=True)
    policy = TensorPolicy(actor).eval()
    path = out/'xc_sitstand.onnx'
    torch.onnx.export(policy, (torch.zeros(1,62),), str(path), input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    session = ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    source = run/('evaluation_'+checkpoint.stem)/'center/rollout.npz'
    raw = np.load(source)['obs'].reshape(-1,62)
    raw = raw[np.linspace(0,len(raw)-1,256,dtype=int)]
    with torch.no_grad(): expected = policy(torch.from_numpy(raw)).numpy()
    actual = np.concatenate([session.run(None,{'obs':r[None]})[0] for r in raw])
    error = float(np.max(np.abs(actual-expected)))
    if not np.isfinite(actual).all() or error > 1e-5:
        raise RuntimeError(f'ONNX parity failed: {error}')
    old = load_actor(SOURCE_PT,OmegaConf.load(SOURCE_CFG))
    with torch.no_grad(): changed = float(torch.max(torch.abs(actor(actor_batch(raw))-old(actor_batch(raw)))))
    if changed <= 1e-6: raise RuntimeError('Actor did not change from model_267')
    if not torch.equal(actor.timing_head.weight,old.timing_head.weight):
        raise RuntimeError('Frozen timing prior changed unexpectedly')
    # Replay the real frame schema through the actual deployment consumer.
    consumer = consumer_module(); consumer_error = 0.; frame_count = 0
    for label, frames in identification.trials():
        instance = consumer.Policy()
        for index, row in enumerate(frames):
            frame = {'positions':row['positions'],'velocities':row['velocities'],
                     'imu':{'gyro':row['imu_gyro'],'gravity':row['imu_gravity']},
                     'policy_context':{'action':'sit' if 'sit' in label else 'rise'},'dt':.02}
            if index==0: instance.reset({'joint_names':consumer.JOINTS},frame)
            inputs = instance.preprocess(frame,None)
            action = session.run(None,inputs)[0]
            with torch.no_grad(): reference = policy(torch.from_numpy(inputs['obs'])).numpy()
            consumer_error = max(consumer_error,float(np.max(np.abs(action-reference))))
            targets = instance.postprocess({'actions':action},frame)
            if set(targets)!=set(consumer.CONTROLLED): raise RuntimeError('Consumer joint names changed')
            q = np.asarray([targets[n]['position'] for n in consumer.TRAINING_JOINTS])
            if not np.isfinite(q).all() or np.any(q<consumer.TARGET_LOW) or np.any(q>consumer.TARGET_HIGH):
                raise RuntimeError('Consumer emitted invalid target')
            frame_count += 1
    if consumer_error>1e-5: raise RuntimeError('Consumer replay parity failed')
    copied = (UPLOAD/'xc_sitstand_policy.py').read_text().replace('model_267 Sit/Rise','flex adaptation '+checkpoint.stem+' Sit/Rise')
    (out/'xc_sitstand_policy.py').write_text(copied)
    shutil.copy2(checkpoint,out/checkpoint.name)
    shutil.copy2(run/'config.yaml',out/'training_config.yaml')
    shutil.copy2(run/'physics/physics.json',out/'physics_hypothesis.json')
    shutil.copy2(__file__,out/'xduck_flex_adaptation_source.py')
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    verification = {'onnx_max_abs_error':error,'samples':256,'consumer_replay_frames':frame_count,'consumer_replay_max_abs_error':consumer_error,'consumer_targets_within_soft_limits':True,'frozen_timing_unchanged':True,'max_action_change_vs_model267_rad':changed,'input':[1,62],'output':[1,14],'period_us':20000,'api_version':2,'gate_passed':evaluation['passed']}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    (out/'README.txt').write_text(
        ('通过本轮仿真门槛，尚无真机验证。\n' if evaluation['passed'] else '未通过稳定性测试；这是研究检查点，不应当作已修复的真机部署版本。\n')+
        'XDuck 柔性关节与质心假设适应版\n'
        '部署文件：xc_sitstand.onnx 与 xc_sitstand_policy.py，需成对替换。\n'
        '沿用 API v2、20ms、原始 IMU 姿态、原关节顺序和原 KP/KD；无需提供仿真柔性关节角。\n'
        f'权重：{checkpoint.name}；从 model_267 的 actor 继续 PPO 训练。\n'
        f'完整仿真门槛通过：{evaluation["passed"]}。详细分项见 evaluation.json。\n'
        '候选物理参数不是实测标定：双踝 75Nm/rad、1Nm·s/rad，头颈质心后移20mm。\n'
        '没有新的真机验证；既往试验发生过后倒，首次复测须有防跌保护。\n'
        '训练配置含本机路径，复现实验需要现有 UniLab 工程及原始实验数据。部署两个文件可独立消费。\n'
    )
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
    suffix = '' if evaluation['passed'] else '_NOT_READY'
    archive = ROOT/'exports'/f'xduck_flex_adapt_20260930_{checkpoint.stem}{suffix}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p,arcname=out.name+'/'+p.name)
    print('BUNDLE',archive,verification,flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=36)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--suite-checkpoint', type=Path)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--minimum-initial-std', type=float)
    parser.add_argument('--initial-std', type=float)
    parser.add_argument('--training-physics-substeps', type=int, choices=[4,8], default=8)
    parser.add_argument('--emphasize-trial3', action='store_true')
    parser.add_argument('--train-subdir', default='train')
    args = parser.parse_args(); run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(300930); np.random.seed(300930)
    if args.suite_checkpoint:
        evaluation_suite(run,args.suite_checkpoint.resolve())
        return
    variant = json.loads((identification.OUT/'minimal_candidate_config.json').read_text())['parameters']
    materialize(run/'physics', variant)
    materialize(run/'training_physics',{**variant,'physics_substeps':args.training_physics_substeps})
    cfg = OmegaConf.load(SOURCE_CFG)
    actor = load_actor(SOURCE_PT, cfg)
    if not args.resume:
        baseline, trace = evaluate(actor, run/'physics', count=3)
        (run/'baseline.json').write_text(json.dumps(baseline, indent=2)); np.save(run/'baseline_trace.npy',trace)
        print('BASELINE',baseline,flush=True)
    if args.evaluate_only: return
    cfg.algo.num_envs = args.envs; cfg.algo.num_steps_per_env = 200
    cfg.algo.algorithm.diagnostics_path = str(run/'feedback.jsonl')
    cfg.algo.algorithm.learning_rate = args.learning_rate
    cfg.algo.algorithm.learning_rate_ceiling = args.learning_rate
    cfg.algo.save_interval = 8
    cfg.adaptation = {'status':'experimental hypothesis, not hardware calibration', 'physics': variant, 'start_checkpoint':str(SOURCE_PT), 'actor_obs':62, 'critic_obs':74, 'actions':14, 'critic_reset':True, 'optimizer_reset':True, 'seed':300930, 'envs':args.envs, 'iterations':args.iterations, 'reward':{'upright':4,'height':3,'stillness':1,'speed':-2,'action_change':-.1,'success':2,'failure':-8}}
    OmegaConf.save(cfg, run/'config.yaml')
    probabilities = [.15,.4,.25,.2] if args.emphasize_trial3 else None
    if args.emphasize_trial3:
        add_settled_standing_start(run/'training_physics')
    env = FlexAdaptationEnv(run/'training_physics',args.envs,seed=300930,reward_weights=cfg.adaptation.reward,start_probabilities=probabilities)
    wrapped = RslRlVecEnvWrapper(env,device='cpu')
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo,resolve=True))
    runner = OnPolicyRunner(wrapped,train_cfg,log_dir=str(run/args.train_subdir),device='cpu')
    # Transfer the trained actor exactly; the privileged critic has a new
    # observation meaning and is deliberately fitted afresh during warmup.
    runner.alg.actor.load_state_dict(actor.state_dict())
    if args.resume:
        runner.load(str(args.resume))
        runner.current_learning_iteration += 1
    runner.alg.learning_rate = args.learning_rate
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = args.learning_rate
    if args.minimum_initial_std is not None:
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.clamp_(min=args.minimum_initial_std)
        runner.alg.optimizer.state.pop(runner.alg.actor.distribution.std_param, None)
    if args.initial_std is not None:
        with torch.no_grad(): runner.alg.actor.distribution.std_param.fill_(args.initial_std)
        runner.alg.optimizer.state.pop(runner.alg.actor.distribution.std_param,None)
    cfg.adaptation.resume_checkpoint = str(args.resume) if args.resume else None
    cfg.adaptation.minimum_initial_std = args.minimum_initial_std
    cfg.adaptation.initial_std = args.initial_std
    cfg.adaptation.start_probabilities = probabilities or [.25,.25,.5]
    cfg.adaptation.first_iteration = runner.current_learning_iteration
    cfg.adaptation.last_iteration = runner.current_learning_iteration+args.iterations-1
    cfg.adaptation.physics_dt = env.cfg.sim_dt
    cfg.adaptation.policy_dt = env.cfg.ctrl_dt
    cfg.adaptation.max_episode_seconds = env.cfg.max_episode_seconds
    cfg.adaptation.sit_probability = sum((probabilities or [.25,.25,.5])[:2])
    cfg.env = OmegaConf.structured(env.cfg)
    cfg.reward = {'scales':dict(env.reward_weights)}
    cfg.training.task_name = 'experimental_xduck_flex_adaptation_module'
    cfg.training.log_dir = str(run/args.train_subdir)
    cfg.algo.max_iterations = runner.current_learning_iteration+args.iterations
    cfg.adaptation.training_physics = env.physics
    cfg.adaptation.initial_joint_position_jitter_rad = .004
    cfg.adaptation.initial_joint_velocity_jitter_rad_s = .015
    cfg.adaptation.imu_delay_seconds = .02
    cfg.adaptation.joint_velocity_delay_seconds = .02
    cfg.adaptation.motor_jitter_mode = 'measured_20260918'
    OmegaConf.save(cfg,run/'config.yaml')
    runner.learn(args.iterations)
    env.close()
    candidate, trace = evaluate(runner.alg.actor,run/'physics',count=12,seed=709)
    (run/'candidate.json').write_text(json.dumps(candidate,indent=2)); np.save(run/'candidate_trace.npy',trace)
    print('CANDIDATE',candidate,flush=True)


if __name__ == '__main__':
    main()
