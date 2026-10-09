"""Local training owner for the five-pose zero/mount hypothesis.

This is an experimental NpEnv/SimBackend/RSL-RL path. Encoder observations and
targets stay in firmware coordinates; joint references change geometry only.
Fourteen serial springs remain hidden from the 62-input deployment actor.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import importlib.util
import json
import shutil
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.actuators.gf43x40.actuator import GF43X40Parameters
from unilab.training.xduck_flex_adaptation import (
    FlexAdaptationEnv, ROOT, SOURCE_CFG, SOURCE_PT, UPLOAD, consumer_module, identification,
)
from unilab.training.xduck_standing_alignment import POLICY_JOINTS
from unilab.training.xduck_curriculum import load_actor
from unilab.training.xduck_imitation import actor_batch
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.rsl_rl_reset import RslRlVecEnvWrapper
from unilab.base.np_env import NpEnvState

CALIBRATION = ROOT/'training_runs/xduck_five_pose_calibration_20260930'


def materialize(folder, *, substeps=4, stiffness=75., mount_delta=(0., 0.)):
    """Persist a dynamics scene without diagnostic encoder equalities."""
    folder.mkdir(parents=True, exist_ok=True)
    profile = json.loads((CALIBRATION/'candidate.json').read_text())
    root = ET.parse(CALIBRATION/'trial1_stand_candidate.xml').getroot()
    for tag in ('equality', 'keyframe'):
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    # Keep a task-height reference at the historical CAD torso frame. The IMU
    # origin rebase must not lower success/failure heights by seventeen cm.
    base = root.find('worldbody/body[@name="base_link"]')
    ET.SubElement(base, 'site', name='task_torso', pos='.0912282641676487 0 .171100298369242', size='.001', group='3')
    ET.SubElement(root.find('sensor'), 'framepos', name='task_torso_position', objtype='site', objname='task_torso')
    ET.SubElement(root.find('sensor'), 'framelinvel', name='task_torso_velocity', objtype='site', objname='task_torso')
    from unilab.training.xduck_standing_alignment import quaternion
    mount = np.asarray(profile['imu_mount_roll_pitch_deg'])+mount_delta
    root.find('.//site[@name="imu_reported"]').set('quat', ' '.join(map(str, quaternion(*mount))))
    for joint in root.iter('joint'):
        if joint.get('name', '').endswith('_flex'):
            joint.set('stiffness', str(stiffness)); joint.set('damping', '1')
    scene = folder/'scene.xml'
    scene.write_text(ET.tostring(root, encoding='unicode'))
    m = mujoco.MjModel.from_xml_path(str(scene))
    pars = GF43X40Parameters.from_bundle()
    ids = np.array([m.joint(n).id for n in POLICY_JOINTS])
    qi, vi = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    ai = np.array([np.flatnonzero(m.actuator_trnid[:, 0]==j)[0] for j in ids])
    fi = np.array([m.jnt_qposadr[m.joint(n+'_flex').id] for n in POLICY_JOINTS])
    m.opt.timestep = .001/substeps
    m.dof_armature[vi] = pars.physics['armature']
    m.dof_damping[vi] = 0; m.dof_frictionloss[vi] = 0
    m.actuator_gainprm[:] = 0; m.actuator_gainprm[:, 0] = 1
    m.actuator_biasprm[:] = 0; m.actuator_biasprm[:, 1] = -1
    limit = pars.physics['mechanical_torque_limit']
    m.actuator_forcerange[:] = [-limit, limit]
    m.actuator_ctrlrange[:] = [-limit-pars.registers['pmax'], limit+pars.registers['pmax']]
    m.actuator_forcelimited[:] = 1; m.actuator_ctrllimited[:] = 1
    mujoco.mj_saveLastXML(str(scene), m)
    d = mujoco.MjData(m)
    starts = []
    for index, row in enumerate(profile['nonlinear_validation']):
        d.qpos[:] = row['final_qpos']; d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        height = float(d.sensor('task_torso_position').data[2])
        starts.append(dict(label=row['name'], qpos=d.qpos.tolist(), qvel=d.qvel.tolist(),
                           command=int(index!=4), height=height))
    c = consumer_module()
    info = dict(variant=dict(physics_substeps=substeps, stiffness=stiffness, damping=1.),
                joints=POLICY_JOINTS, home=c.REFERENCE.tolist(), actuator_indices=ai.tolist(),
                flex_indices=(fi-7).tolist(), zero_offsets_rad=np.deg2rad(profile['joint_zero_offsets_deg']).tolist(),
                imu_mount_roll_pitch_deg=mount.tolist(), starts=starts,
                stand_height=float(np.mean([s['height'] for s in starts[:4]])), sit_height=starts[4]['height'],
                stand_pose=c.REFERENCE.tolist(), sit_pose=d.qpos[qi].tolist(),
                calibration_sha256=hashlib.sha256((CALIBRATION/'candidate.json').read_bytes()).hexdigest(),
                calibration_status=profile['status'])
    (folder/'physics.json').write_text(json.dumps(info, indent=2))
    return info


def gravity(quat):
    w, x, y, z = quat.T
    return np.stack((2*(w*y-x*z), -2*(w*x+y*z), 2*(x*x+y*y)-1), axis=1)


class CalibratedEnv(FlexAdaptationEnv):
    def __init__(self, folder, count=16, seed=30, jitter=True, **kwargs):
        super().__init__(folder, count, seed, jitter, **kwargs)
        self._cfg.max_episode_seconds = 6.
        self.zero = np.asarray(self.physics['zero_offsets_rad'])
        self.stand_pose = np.asarray(self.physics['stand_pose'])
        self.sit_pose = np.asarray(self.physics['sit_pose'])
        self.goal_pose = np.zeros((count, 14))
        self.last_pose_error = np.zeros(count)
        self.last_height_error = np.zeros(count)
        self.last_tilt = np.zeros(count)
        self.last_speed = np.zeros(count)
        self.reference_targets = np.asarray(self.physics['reference_pose_targets']) if 'reference_pose_targets' in self.physics else None

    @property
    def obs_groups_spec(self):
        return {'obs': 62, 'critic': 98}

    def motor_tick(self, backend, target):
        if self.tick % self.physics['variant']['physics_substeps'] == 0:
            q, v = backend.get_dof_pos()[:, self.qi], backend.get_dof_vel()[:, self.vi]
            self.motor.finish_substep(q, v)
            tau = self.motor.begin_substep(target, q, v)
            # MuJoCo's joint actuator length is raw qpos, even with a nonzero
            # geometry reference. Its bias must cancel the encoder coordinate.
            self.native_ctrl[:, self.ai] = q + tau
        self.tick += 1
        return self.native_ctrl

    def measured(self):
        q = self.motor.quantize_position_feedback(self._backend.get_dof_pos()[:, self.qi])
        v = self.motor.quantize_velocity_feedback(self._backend.get_dof_vel()[:, self.vi])
        quat = self._backend.get_sensor_data('orientation_reported')
        gyro = self._backend.get_sensor_data('angular_velocity_reported')
        return q, v, np.concatenate((gyro, gravity(quat)), axis=1)

    def observations(self, advance=False):
        q, v, imu = self.measured()
        command = np.zeros((self.num_envs, 13)); command[:, 0] = self.command
        obs = np.concatenate((self.previous_imu, q-self.home, self.previous_velocity,
                              self.last_action, command, np.minimum(self.age[:, None]/4, 1)), axis=1)
        critic = np.concatenate((obs, self._backend.get_sensor_data('task_torso_position')[:, 2:3],
            self._backend.get_sensor_data('task_torso_velocity'), self._backend.get_dof_pos()[:, self.fi],
            self._backend.get_dof_vel()[:, self.fi], imu[:, 3:6], self.hold[:, None]/25), axis=1)
        if advance:
            self.previous_velocity[:] = v; self.previous_imu[:] = imu
        return {'obs': obs.astype(np.float32), 'critic': critic.astype(np.float32)}

    def reset(self, env_indices):
        # Equal directions even when policy-generated endpoints join the bank.
        commands = np.array([s['command'] for s in self.physics['starts']])
        self.start_probabilities = np.array([.5/np.sum(commands==c) for c in commands])
        return super().reset(env_indices)

    def update_state(self, state):
        q = self._backend.get_dof_pos()[:, self.qi]
        self.motor.finish_substep(q, self._backend.get_dof_vel()[:, self.vi])
        self.age += .02
        body_g = gravity(self._backend.get_base_quat())
        tilt = np.arccos(np.clip(-body_g[:, 2], -1, 1))
        height = self._backend.get_sensor_data('task_torso_position')[:, 2]
        speed = np.linalg.norm(self._backend.get_sensor_data('task_torso_velocity'), axis=1)
        progress = np.minimum(self.age/3., 1)
        smooth = progress**2*(3-2*progress)
        goal_height = np.where(self.command>.5, self.physics['sit_height'], self.physics['stand_height'])
        start_height = np.where(self.command>.5, self.physics['stand_height'], self.physics['sit_height'])
        desired_height = start_height+(goal_height-start_height)*smooth
        self.goal_pose[:] = self.stand_pose + self.command[:, None]*(self.sit_pose-self.stand_pose)
        start_pose = self.sit_pose+self.command[:, None]*(self.stand_pose-self.sit_pose)
        desired_pose = start_pose+(self.goal_pose-start_pose)*smooth[:, None]
        if self.reference_targets is not None:
            phase_index = np.minimum(self.age/4, 1)*(self.reference_targets.shape[1]-1)
            lower = np.floor(phase_index).astype(int)
            upper = np.minimum(lower+1, self.reference_targets.shape[1]-1)
            fraction = (phase_index-lower)[:, None]
            direction = (self.command<.5).astype(int)  # table order sit, rise
            desired_pose = self.reference_targets[direction, lower]*(1-fraction)+self.reference_targets[direction, upper]*fraction
        tracking = np.sqrt(np.square(q-desired_pose).mean(axis=1))
        pose = np.sqrt(np.square(q-self.goal_pose).mean(axis=1))
        self.last_limit_violation = ((q<self.consumer.TARGET_LOW-.01)|(q>self.consumer.TARGET_HIGH+.01)).any(axis=1)
        success = (np.abs(height-goal_height)<.015)&(tilt<np.deg2rad(10))&(speed<.04)&(pose<.10)
        self.hold = np.where(success, self.hold+1, 0); self.last_success = self.hold>=25
        failed = (tilt>np.deg2rad(40))|(height<.13)|self.last_limit_violation|(~np.isfinite(q).all(axis=1))
        reward = .02*(4*np.exp(-(tilt/.2)**2)+6*np.exp(-(tracking/.12)**2)
            +3*np.exp(-((height-desired_height)/.04)**2)+np.exp(-(speed/.10)**2)
            -.3*speed-.05*self.action_change)+2*(self.hold==25)-8*failed
        self.last_pose_error = pose; self.last_height_error = height-goal_height
        self.last_tilt = tilt; self.last_speed = speed
        state.obs = self.observations(advance=True); state.reward = reward.astype(np.float32)
        state.terminated = failed
        state.info['log'] = {'balance/success':float(self.last_success.mean()), 'balance/pose_error':float(pose.mean()),
                             'balance/tilt_deg':float(np.rad2deg(tilt).mean())}
        return state


class CalibratedMixture:
    """Local EnvState protocol composition; the standard runner owns collection.

    Each cohort is an ordinary NpEnv with its own fixed MuJoCo scene. No backend
    private methods, per-step XML changes, or second collector/learner exist.
    """
    def __init__(self, folders, count_per_cohort=8, seed=4501):
        self.envs = [CalibratedEnv(folder, count_per_cohort, seed+i*17) for i,folder in enumerate(folders)]
        self.num_envs = sum(e.num_envs for e in self.envs)
        self.cfg = self.envs[0].cfg
        self.obs_groups_spec = {'obs':62, 'critic':99}
        self.action_space = self.envs[0].action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(161,), dtype=np.float32)
        self.state = None
        self.offsets = np.cumsum([0]+[e.num_envs for e in self.envs])

    def _observation(self, e, obs):
        return {'obs':obs['obs'], 'critic':np.concatenate((obs['critic'],
            np.full((len(obs['critic']),1), e.physics['variant']['stiffness']/75.-1., np.float32)), axis=1)}

    def _combine(self):
        states = [e.state for e in self.envs]
        batches = [self._observation(e, s.obs) for e,s in zip(self.envs,states)]
        obs = {k:np.concatenate([b[k] for b in batches]) for k in self.obs_groups_spec}
        info = {'steps':np.concatenate([s.info['steps'] for s in states]), 'log':{}}
        for key in states[0].info.get('log',{}):
            info['log'][key] = sum(s.info.get('log',{}).get(key,0.)*e.num_envs for e,s in zip(self.envs,states))/self.num_envs
        final = None
        if any(s.final_observation is not None for s in states):
            terminal = [self._observation(e, s.final_observation if s.final_observation is not None else
                {k:np.zeros_like(v) for k,v in s.obs.items()}) for e,s in zip(self.envs,states)]
            final = {k:np.concatenate([b[k] for b in terminal]) for k in self.obs_groups_spec}
            info['final_observation'] = final
            info['_final_observation'] = np.concatenate([s.info.get('_final_observation',np.zeros(e.num_envs,bool)) for e,s in zip(self.envs,states)])
        self.state = NpEnvState(obs=obs, reward=np.concatenate([s.reward for s in states]),
            terminated=np.concatenate([s.terminated for s in states]), truncated=np.concatenate([s.truncated for s in states]),
            info=info, final_observation=final)
        return self.state

    def init_state(self):
        for e in self.envs: e.init_state()
        return self._combine()

    def reset(self, env_indices):
        ids = np.asarray(env_indices)
        for e,begin,end in zip(self.envs,self.offsets[:-1],self.offsets[1:]):
            local = ids[(ids>=begin)&(ids<end)]-begin
            if len(local):
                obs,info = e.reset(local)
                for key in obs: e.state.obs[key][local] = obs[key]
                e.state.info['steps'][local] = 0
                e.state.terminated[local] = False; e.state.truncated[local] = False
                e.state.final_observation = None
                e.state.info.pop('_final_observation',None)
        state = self._combine()
        return {k:v[ids].copy() for k,v in state.obs.items()}, {}

    def step(self, actions):
        for e,begin,end in zip(self.envs,self.offsets[:-1],self.offsets[1:]): e.step(actions[begin:end])
        return self._combine()

    def close(self):
        for e in self.envs: e.close()


def set_start(env, choices, command=None):
    ids = np.arange(env.num_envs); starts = env.physics['starts']
    q = np.asarray([starts[i]['qpos'] for i in choices]); v = np.asarray([starts[i]['qvel'] for i in choices])
    if env.jitter:
        q[:, env.qi+7] += env.rng.uniform(-.004, .004, (len(ids), 14))
        v[:, env.vi+6] += env.rng.uniform(-.015, .015, (len(ids), 14))
    env._backend.set_state(ids, q, v); env.motor.reset(ids)
    env.command[:] = [starts[i]['command'] for i in choices] if command is None else command
    env.age[:] = 0; env.last_action[:] = 0; env.hold[:] = 0
    _, v, imu = env.measured(); env.previous_velocity[:] = v; env.previous_imu[:] = imu
    env.state.info['steps'][:] = 0; env.state.obs = env.observations()


def evaluate(actor, folder, count=10, seed=1709, jitter=False, seconds=6., record=None, balanced=False):
    env = CalibratedEnv(folder, count, seed, jitter)
    env.init_state(); env.set_autoreset(False)
    choices = np.arange(count)%len(env.physics['starts'])
    if balanced:
        sit = [i for i,s in enumerate(env.physics['starts']) if s['command']==1]
        rise = [i for i,s in enumerate(env.physics['starts']) if s['command']==0]
        choices = np.array([sit[i%len(sit)] for i in range(count//2)]+[rise[i%len(rise)] for i in range(count-count//2)])
    set_start(env, choices)
    alive = np.ones(count, bool); limits = np.zeros(count, bool); rows = []; first = np.full(count, -1)
    begin = time.monotonic()
    for step in range(round(seconds/.02)):
        obs = env.state.obs['obs'].copy()
        with torch.no_grad(): actions = actor(actor_batch(obs)).numpy()
        env.step(actions)
        failed = alive&env.state.terminated; first[failed] = step; alive &= ~env.state.terminated
        limits |= env.last_limit_violation
        if record:
            rows.append(dict(obs=obs, actions=actions,
                qpos=np.concatenate((env._backend.get_base_pos(), env._backend.get_base_quat(), env._backend.get_dof_pos()), axis=1).copy(),
                metrics=np.stack((env.last_pose_error, env.last_height_error, env.last_tilt, env.last_speed), axis=1)))
    qpos = np.concatenate((env._backend.get_base_pos(), env._backend.get_base_quat(), env._backend.get_dof_pos()), axis=1).copy()
    contacts, contact_valid = final_contacts(folder, qpos, env.command)
    success = alive&env.last_success&contact_valid
    metrics = dict(seed=seed, jitter=jitter, seconds=seconds, wall_seconds=time.monotonic()-begin,
        cases=[env.physics['starts'][i]['label'] for i in choices], success=success.tolist(), alive=alive.tolist(),
        rate=float(success.mean()), limit_violation=limits.tolist(), first_failure_seconds=[float(i*.02) if i>=0 else None for i in first],
        pose_rms_rad=env.last_pose_error.tolist(), height_error_m=env.last_height_error.tolist(),
        body_tilt_deg=np.rad2deg(env.last_tilt).tolist(), speed_m_s=env.last_speed.tolist(),
        final_floor_contacts=contacts, valid_contact_mode=contact_valid.tolist(),
        direction_rates={name:float(success[env.command==command].mean()) for name,command in [('sit',1),('rise',0)]})
    if record:
        np.savez_compressed(record, **{key:np.asarray([row[key] for row in rows]) for key in rows[0]})
    env.close()
    return metrics


def final_contacts(folder, qpos, command):
    """Offline validation of support mode, outside the environment hot path."""
    m = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); d = mujoco.MjData(m)
    contacts = []; valid = []
    for q, cmd in zip(qpos, command):
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        floor = sorted({m.geom(c.geom[1] if m.geom(c.geom[0]).name=='floor' else c.geom[0]).name
            for c in d.contact if c.dist<.001 and 'floor' in (m.geom(c.geom[0]).name,m.geom(c.geom[1]).name)})
        feet = any('foot_collision' in n for n in floor)
        hips = any('hip_pitch_collision' in n for n in floor)
        forbidden = any('base_link_collision' in n or 'head_' in n or 'neck_' in n for n in floor)
        valid.append(feet and not forbidden and (hips if cmd>.5 else not hips))
        contacts.append(floor)
    return contacts, np.asarray(valid)


def evaluate_cycle(actor, folder, count=8, seed=3401):
    env = CalibratedEnv(folder, count, seed, jitter=True)
    env.init_state(); env.set_autoreset(False); set_start(env, np.arange(count)%4, command=0)
    alive = np.ones(count, bool); traces = []; output = dict(seed=seed, standing_seconds=3., segments=[])
    for _ in range(150):
        env.step(np.zeros((count, 14))); alive &= ~env.state.terminated
    output['standing_survived'] = alive.tolist()
    for command in [1, 0, 1, 0]:
        env.command[:] = command; env.age[:] = 0; env.last_action[:] = 0; env.hold[:] = 0
        _, v, imu = env.measured(); env.previous_velocity[:] = v; env.previous_imu[:] = imu
        env.state.obs = env.observations()
        for tick in range(300):
            with torch.no_grad(): actions = actor(actor_batch(env.state.obs['obs'])).numpy()
            env.step(actions); alive &= ~env.state.terminated
            traces.append(np.concatenate((env._backend.get_base_pos(), env._backend.get_base_quat(),env._backend.get_dof_pos()), axis=1).copy())
        floor, valid = final_contacts(folder, traces[-1], env.command)
        output['segments'].append(dict(command=command, success=(alive&env.last_success&valid).tolist(),
            alive=alive.tolist(), body_tilt_deg=np.rad2deg(env.last_tilt).tolist(), pose_rms_rad=env.last_pose_error.tolist(),
            height_error_m=env.last_height_error.tolist(), speed_m_s=env.last_speed.tolist(), final_floor_contacts=floor))
    np.savez_compressed(folder.parent/'cycle_qpos.npz', qpos=np.asarray(traces))
    env.close(); return output


def evaluate_suite(run, checkpoint, seed=4401, count=20, label=None):
    actor = load_actor(checkpoint, OmegaConf.load(run/'config.yaml'))
    out = run/('evaluation_'+checkpoint.stem+('_'+label if label else '')); out.mkdir(exist_ok=True)
    result = dict(checkpoint=str(checkpoint), criterion='>=90% EACH direction and continuous segment; survive6s, last0.5s torso height<1.5cm, bodytilt<10deg, speed<4cm/s, motor pose RMS<.1rad, correct feet/hips support, no torso ground contact', windows={})
    variants = [('center',75.,(0.,0.),4), ('center_fine',75.,(0.,0.),8), ('softer',60.,(0.,0.),4),
                ('stiffer',90.,(0.,0.),4), ('mount_plus1deg',75.,(1.,1.),4), ('mount_minus1deg',75.,(-1.,-1.),4)]
    for label, stiffness, mount, substeps in variants:
        folder = out/label; materialize(folder, stiffness=stiffness, mount_delta=mount, substeps=substeps)
        metrics = evaluate(actor, folder, count=count, seed=seed, jitter=True, record=folder/'rollout.npz', balanced=bool(label))
        result['windows'][label] = metrics
        (out/'results.json').write_text(json.dumps(result, indent=2))
        print('WINDOW', label, metrics['direction_rates'], metrics['valid_contact_mode'], flush=True)
    result['cycle'] = evaluate_cycle(actor, out/'center', count=32 if label else 8, seed=seed+1000 if label else 3401)
    result['passed'] = all(min(w['direction_rates'].values())>=.9 for w in result['windows'].values()) and all(np.mean(s['success'])>=.9 for s in result['cycle']['segments']) and all(result['cycle']['standing_survived'])
    (out/'results.json').write_text(json.dumps(result, indent=2)); print('GATE', result['passed'], result['cycle'], flush=True)
    return result


def augment_endpoint_starts(run, record):
    """Cold reset-bank entries produced by the policy itself, no fake IMU."""
    path = run/'physics/physics.json'; info = json.loads(path.read_text())
    if any(s['label'].startswith('policy_end_') for s in info['starts']): return
    m = mujoco.MjModel.from_xml_path(str(run/'physics/scene.xml')); d = mujoco.MjData(m)
    values = np.load(record)['qpos'][-1]
    original = list(info['starts'][:5])
    for index, start in enumerate(original):
        d.qpos[:] = values[index]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        info['starts'].append(dict(label='policy_end_'+start['label'], command=1-start['command'],
            qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), height=float(d.sensor('task_torso_position').data[2]),
            source_rollout=str(record), source_env_index=index))
    path.write_text(json.dumps(info, indent=2))


def train_mixture(run, checkpoint, iterations=64, count_per_cohort=8, resume=False):
    """Train one actor on soft/center/stiff models in every PPO batch."""
    torch.set_num_threads(4); torch.manual_seed(31011); np.random.seed(31011)
    out = run/'robust'; out.mkdir(exist_ok=True)
    if not (out/'physics/physics.json').exists():
        materialize(out/'physics', substeps=8)
        base = json.loads((out/'physics/physics.json').read_text())
        parent = json.loads((run/'physics/physics.json').read_text())
        base['reference_pose_targets'] = parent['reference_pose_targets']
        (out/'physics/physics.json').write_text(json.dumps(base, indent=2))
        augment_endpoint_starts(out, run/'initial_rollout.npz')
    center = json.loads((out/'physics/physics.json').read_text())
    folders = []
    for stiffness in [60.,75.,90.]:
        folder = out/f'physics_{stiffness:g}'
        if not (folder/'physics.json').exists():
            data = materialize(folder, substeps=8, stiffness=stiffness)
            data['starts'] = center['starts']; data['reference_pose_targets'] = center['reference_pose_targets']
            (folder/'physics.json').write_text(json.dumps(data, indent=2))
        folders.append(folder)
    cfg = OmegaConf.load(out/'config.yaml') if resume else OmegaConf.load(run/'config.yaml')
    cfg.algo.num_envs = 3*count_per_cohort; cfg.algo.num_steps_per_env = 200
    cfg.algo.save_interval = 8
    cfg.algo.algorithm.learning_rate = 1e-4; cfg.algo.algorithm.learning_rate_ceiling = 1e-4
    cfg.algo.algorithm.diagnostics_path = str(out/'feedback.jsonl')
    cfg.algo.algorithm.critic_warmup_updates = 4
    env = CalibratedMixture(folders, count_per_cohort)
    cfg.env = OmegaConf.structured(env.cfg); cfg.training.log_dir = str(out/'train')
    cfg.calibrated_mixture = dict(stiffness_nm_rad=[60.,75.,90.], damping_nm_s_rad=1., physics_dt=.000125,
        actuator_dt=.001, policy_dt=.02, critic_observations=99, actor_observations=62,
        source_checkpoint=str(checkpoint), optimizer_reset=not resume, source_calibration=center['calibration_sha256'],
        reset_bank_entries=len(center['starts']), physics_folders=[str(f) for f in folders],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env, device='cpu'), normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True)),
                           log_dir=str(out/'train'), device='cpu')
    if resume:
        runner.load(str(checkpoint)); runner.current_learning_iteration += 1
    else:
        runner.alg.actor.load_state_dict(load_actor(checkpoint, OmegaConf.load(run/'config.yaml')).state_dict())
        with torch.no_grad(): runner.alg.actor.distribution.std_param.clamp_(min=.008)
    runner.alg.learning_rate = 1e-4
    for group in runner.alg.optimizer.param_groups: group['lr'] = 1e-4
    cfg.algo.max_iterations = runner.current_learning_iteration+iterations
    cfg.calibrated_mixture.first_iteration = runner.current_learning_iteration
    cfg.calibrated_mixture.last_iteration = cfg.algo.max_iterations-1
    OmegaConf.save(cfg, out/'config.yaml')
    Path(out/'source_snapshot.py').write_text(Path(__file__).read_text())
    runner.learn(iterations); env.close()
    metrics = evaluate(runner.alg.actor, out/'physics', count=20, jitter=True, seed=4901, record=out/'last_rollout.npz')
    (out/'last_evaluation.json').write_text(json.dumps(metrics, indent=2)); print('ROBUST CANDIDATE', metrics, flush=True)
    return out


def train(run, iterations=64, count=16, resume=None, lr=3e-5):
    torch.set_num_threads(4); torch.manual_seed(300931); np.random.seed(300931)
    folder = run/'physics'
    if not (folder/'physics.json').exists(): materialize(folder)
    cfg = OmegaConf.load(run/'config.yaml') if resume and (run/'config.yaml').exists() else OmegaConf.load(SOURCE_CFG)
    cfg.algo.num_envs = count; cfg.algo.num_steps_per_env = 200
    cfg.algo.save_interval = 8; cfg.algo.max_iterations = iterations
    cfg.algo.algorithm.learning_rate = lr; cfg.algo.algorithm.learning_rate_ceiling = lr
    cfg.algo.algorithm.diagnostics_path = str(run/'feedback.jsonl')
    cfg.algo.actor.distribution_cfg.init_std = .015
    env = CalibratedEnv(folder, count, seed=300931)
    cfg.env = OmegaConf.structured(env.cfg)
    cfg.training.task_name = 'experimental_xduck_five_pose_calibrated_training'
    cfg.calibrated_training = dict(physics=env.physics, seed=300931, obs=62, critic=98, actions=14,
        source_checkpoint=str(SOURCE_PT), imu_transport_delay_s=.02, velocity_delay_s=.02,
        reward='pose/upright/height/speed; goals in encoder and historical torso frame', no_hardware_calibration=True)
    OmegaConf.save(cfg, run/'config.yaml')
    wrapped = RslRlVecEnvWrapper(env, device='cpu')
    runner = OnPolicyRunner(wrapped, normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True)),
                           log_dir=str(run/'train'), device='cpu')
    if resume:
        runner.load(str(resume)); runner.current_learning_iteration += 1
    else:
        runner.alg.actor.load_state_dict(load_actor(SOURCE_PT, OmegaConf.load(SOURCE_CFG)).state_dict())
        with torch.no_grad(): runner.alg.actor.distribution.std_param.fill_(.015)
    runner.alg.learning_rate = lr
    for group in runner.alg.optimizer.param_groups: group['lr'] = lr
    runner.learn(iterations)
    env.close()
    return runner.alg.actor


def search_ankle_bias(run, checkpoint=SOURCE_PT):
    """Diagnostic batch search of a neural prior's ankle bias, not deployment."""
    cfg = OmegaConf.load(SOURCE_CFG)
    actor = load_actor(checkpoint, cfg)
    values = np.linspace(-.20, .20, 17)
    count = 2*len(values)
    env = CalibratedEnv(run/'physics', count, seed=2001, jitter=False)
    env.init_state(); env.set_autoreset(False)
    choices = [0]*len(values)+[4]*len(values); set_start(env, choices)
    bias = np.tile(values, 2); alive = np.ones(count, bool); rows = []
    for tick in range(300):
        obs = env.state.obs['obs'].copy()
        with torch.no_grad(): action = actor(actor_batch(obs)).numpy()
        ramp = min((tick+1)*.02/.3, 1.)
        action[:, 4] += bias*ramp; action[:, 13] -= bias*ramp
        env.step(action); alive &= ~env.state.terminated
        rows.append(dict(obs=obs, actions=action.copy(), qpos=np.concatenate((env._backend.get_base_pos(),
            env._backend.get_base_quat(), env._backend.get_dof_pos()), axis=1).copy(),
            metrics=np.stack((env.last_pose_error, env.last_height_error, env.last_tilt, env.last_speed), axis=1)))
    result = [dict(command=int(env.command[i]), bias=float(bias[i]), alive=bool(alive[i]),
        success=bool(alive[i]&env.last_success[i]), pose=float(env.last_pose_error[i]), height=float(env.last_height_error[i]),
        tilt_deg=float(np.rad2deg(env.last_tilt[i])), speed=float(env.last_speed[i])) for i in range(count)]
    (run/'ankle_bias_search.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(run/'ankle_bias_search.npz', **{k:np.asarray([row[k] for row in rows]) for k in rows[0]})
    print('SEARCH', result, flush=True); env.close()


def smooth_reference(env, age, command, middle, gain, damping=.08, duration=4.):
    """Training-only teacher: endpoint interpolation and pitch stabilization."""
    # Use the same delayed report available to the actor, converted back with
    # this run's fixed observation rotation; no hidden spring state is used.
    from unilab.training.xduck_pose_calibration import mount_rotation
    rotation = mount_rotation(env.physics['imu_mount_roll_pitch_deg'])
    body_g = env.previous_imu[:, 3:6]@rotation.T
    gyro = env.previous_imu[:, :3]@rotation.T
    pitch = np.arctan2(body_g[:, 0], np.hypot(body_g[:, 1], body_g[:, 2]))
    phase = np.clip(age/duration, 0, 1)
    progress = phase**2*(3-2*phase)
    seated_fraction = np.where(command>.5, progress, 1-progress)
    target = env.stand_pose+(env.sit_pose-env.stand_pose)*seated_fraction[:, None]
    # Smooth midpoint ankle excursion vanishes at both endpoints.
    excursion = np.sin(np.pi*phase)**2*middle
    correction = gain*pitch+damping*gyro[:, 1]
    target[:, 4] += excursion+correction
    target[:, 13] -= excursion+correction
    return target-env.home


def search_smooth_reference(run):
    pairs = list(itertools.product([-.8, -.4, 0., .4, .8], [-1., -.5, 0., .5, 1.]))
    count = 2*len(pairs)
    env = CalibratedEnv(run/'physics', count, seed=2201, jitter=False)
    env.init_state(); env.set_autoreset(False)
    set_start(env, [0]*len(pairs)+[4]*len(pairs))
    middle = np.tile([x[0] for x in pairs], 2); gain = np.tile([x[1] for x in pairs], 2)
    alive = np.ones(count, bool); rows = []; peak_tilt = np.zeros(count)
    for tick in range(400):
        obs = env.state.obs['obs'].copy()
        actions = smooth_reference(env, env.age, env.command, middle, gain)
        env.step(actions); alive &= ~env.state.terminated
        peak_tilt = np.maximum(peak_tilt, np.rad2deg(env.last_tilt))
        rows.append(dict(obs=obs, actions=actions.copy(), qpos=np.concatenate((env._backend.get_base_pos(),
            env._backend.get_base_quat(), env._backend.get_dof_pos()), axis=1).copy(),
            metrics=np.stack((env.last_pose_error, env.last_height_error, env.last_tilt, env.last_speed), axis=1)))
    result = [dict(command=int(env.command[i]), middle=float(middle[i]), gain=float(gain[i]), alive=bool(alive[i]),
        success=bool(alive[i]&env.last_success[i]), pose=float(env.last_pose_error[i]), height=float(env.last_height_error[i]),
        tilt_deg=float(np.rad2deg(env.last_tilt[i])), peak_tilt_deg=float(peak_tilt[i]), speed=float(env.last_speed[i])) for i in range(count)]
    (run/'smooth_reference_search.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(run/'smooth_reference_search.npz', **{k:np.asarray([row[k] for row in rows]) for k in rows[0]})
    print('SMOOTH SEARCH', result, flush=True); env.close()


def prior_reference(actor, raw, parameters):
    """Offline trajectory search; the optimized result is fitted into a NN."""
    p = np.asarray(parameters)
    phase = np.clip(raw[:, 61]*p[:, 0], 0, 1)
    with torch.no_grad():
        modified = torch.from_numpy(raw.copy()); modified[:, 61] = torch.from_numpy(phase).float()
        out = actor.timing_head(actor.time_features(modified)).numpy()
        modified[:, 61] = torch.from_numpy(np.clip(phase-p[:, 1], 0, 1)).float()
        ankle = actor.timing_head(actor.time_features(modified)).numpy()
    out[:, [4, 13]] = ankle[:, [4, 13]]*p[:, 2:3]
    out[:, 5] *= p[:, 3]
    envelope = np.sin(np.pi*np.clip(phase/.625, 0, 1))**2
    out[:, 2] += p[:, 4]*envelope; out[:, 11] -= p[:, 4]*envelope
    out[:, 3] += p[:, 5]*envelope; out[:, 12] -= p[:, 5]*envelope
    ramp = np.clip(raw[:, 61]*4/.3, 0, 1)
    out[:, 4] += p[:, 6]*ramp; out[:, 13] -= p[:, 6]*ramp
    out[:, 6] += p[:, 7]*envelope
    return out.astype(np.float32)


def search_prior(run, generations=6, population=24):
    """CEM across actual GF/spring dynamics; preserves all failed proposals."""
    torch.set_num_threads(4)
    actor = load_actor(SOURCE_PT, OmegaConf.load(SOURCE_CFG))
    rng = np.random.default_rng(30931)
    lo = np.array([.65, -.14, .65, .6, -.3, -.3, -.08, -.5])
    hi = np.array([1.3, .14, 1.15, 1.5, .3, .3, .08, .5])
    initial = np.array([1., 0., 1., 1., 0., 0., 0., 0.])
    means = np.tile(initial, (2, 1)); scales = np.tile((hi-lo)/3, (2, 1))
    best = [None, None]; all_results = []
    env = CalibratedEnv(run/'physics', 2*population, seed=2401, jitter=False)
    env.init_state(); env.set_autoreset(False)
    for generation in range(generations):
        candidates = np.concatenate([np.clip(rng.normal(means[d], scales[d], (population, len(lo))), lo, hi) for d in range(2)])
        for d in range(2):
            candidates[d*population] = best[d]['parameters'] if best[d] else initial
        set_start(env, [0]*population+[4]*population)
        alive = np.ones(env.num_envs, bool); survived = np.zeros(env.num_envs)
        peak = np.zeros(env.num_envs); limits = np.zeros(env.num_envs, bool)
        last_good = np.zeros((env.num_envs, 4)); rows = []
        for tick in range(300):
            raw = env.state.obs['obs'].copy()
            actions = prior_reference(actor, raw, candidates)
            env.step(actions); alive &= ~env.state.terminated
            survived += alive; limits |= env.last_limit_violation
            peak = np.maximum(peak, np.rad2deg(env.last_tilt))
            metric = np.stack((env.last_pose_error, env.last_height_error, env.last_tilt, env.last_speed), axis=1)
            last_good[alive] = metric[alive]
            rows.append(dict(obs=raw, actions=actions.copy(), qpos=np.concatenate((env._backend.get_base_pos(),
                env._backend.get_base_quat(), env._backend.get_dof_pos()), axis=1).copy(), metrics=metric))
        # Reward completed posture and uninterrupted survival before minor
        # pose fidelity. Fall duration provides a useful gradient until viable.
        score = survived*.2+200*alive+1000*(alive&env.last_success)
        score -= np.rad2deg(env.last_tilt)+200*np.abs(env.last_height_error)+100*env.last_pose_error
        score -= 200*limits
        metrics = []
        for i in range(env.num_envs):
            metrics.append(dict(generation=generation, command=int(env.command[i]), parameters=candidates[i].tolist(),
                score=float(score[i]), alive=bool(alive[i]), success=bool(alive[i]&env.last_success[i]),
                surviving_seconds=float(survived[i]*.02), pose=float(env.last_pose_error[i]), height=float(env.last_height_error[i]),
                tilt_deg=float(np.rad2deg(env.last_tilt[i])), peak_tilt_deg=float(peak[i]), speed=float(env.last_speed[i])))
        all_results.extend(metrics)
        for d in range(2):
            cohort = np.arange(d*population, (d+1)*population)
            elite = cohort[np.argsort(score[cohort])[-max(4, population//4):]]
            win = elite[-1]
            means[d] = .3*means[d]+.7*candidates[elite].mean(axis=0)
            scales[d] = np.maximum(.3*scales[d]+.7*candidates[elite].std(axis=0), (hi-lo)*.025)
            if best[d] is None or score[win]>best[d]['score']:
                best[d] = metrics[win]
                np.savez_compressed(run/f'prior_search_best_{d}.npz', **{key:np.asarray([row[key][win] for row in rows]) for key in rows[0]})
        (run/'prior_search.json').write_text(json.dumps(dict(best=best, results=all_results), indent=2))
        print('CEM', generation, best, flush=True)
    env.close()


def initialize_searched_prior(run):
    """Fit the successful dynamics-search trajectories into neural weights."""
    torch.set_num_threads(4); torch.manual_seed(31001)
    searched = json.loads((run/'prior_search.json').read_text())
    if not all(x['success'] for x in searched['best']):
        raise RuntimeError('No successful two-direction prior to initialize')
    out = run/'new_prior'; out.mkdir(exist_ok=True)
    if not (out/'physics').exists(): shutil.copytree(run/'physics', out/'physics')
    source = load_actor(SOURCE_PT, OmegaConf.load(SOURCE_CFG))
    phases = np.linspace(0, 1, 2001)
    raw = np.zeros((2*len(phases), 62), dtype=np.float32)
    raw[:, 61] = np.tile(phases, 2); raw[:len(phases), 48] = 1
    parameters = np.concatenate([np.tile(x['parameters'], (len(phases), 1)) for x in searched['best']])
    target = prior_reference(source, raw, parameters)
    demonstrations = [np.load(run/f'prior_search_best_{i}.npz') for i in range(2)]
    np.savez_compressed(out/'demonstrations.npz', obs=np.concatenate([x['obs'] for x in demonstrations]),
                        actions=np.concatenate([x['actions'] for x in demonstrations]))
    cfg = OmegaConf.load(SOURCE_CFG)
    cfg.algo.actor.class_name = 'unilab.training.xduck_policy:TimingActor'
    cfg.algo.actor.timing_features = 192
    cfg.algo.actor.distribution_cfg.init_std = .005
    cfg.algo.algorithm.demonstration_path = str(out/'demonstrations.npz')
    cfg.algo.algorithm.diagnostics_path = str(out/'feedback.jsonl')
    cfg.calibrated_prior = dict(method='CEM in five-pose calibrated free-body GF/spring dynamics, then NN cosine regression',
                                searched_parameters=searched['best'], source_policy=str(SOURCE_PT))
    env = CalibratedEnv(out/'physics', 2, jitter=False)
    cfg.env = OmegaConf.structured(env.cfg)
    cfg.training.task_name = 'experimental_xduck_five_pose_calibrated_training'
    OmegaConf.save(cfg, out/'config.yaml')
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env, device='cpu'),
        normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True)), log_dir=str(out/'initialization'), device='cpu')
    actor = runner.alg.actor
    runner.logger.init_logging_writer()
    x = torch.from_numpy(raw)
    features = actor.time_features(x).double()
    weights = torch.linalg.lstsq(features, torch.from_numpy(target).double(), driver='gelsd', rcond=1e-9).solution
    with torch.no_grad():
        actor.timing_head.weight.copy_(weights.T.float())
        predicted = actor(actor_batch(x)).numpy()
    error = dict(rms_rad=float(np.sqrt(np.square(predicted-target).mean())), max_rad=float(np.abs(predicted-target).max()),
                 samples=len(raw), timing_features=192)
    (out/'initial_fit.json').write_text(json.dumps(error, indent=2))
    if error['max_rad']>.005: raise RuntimeError(f'NN timing regression too inaccurate: {error}')
    runner.current_learning_iteration = -1
    runner.save(str(out/'initial.pt'), infos={'calibrated_search_regression':error})
    runner.logger.stop_logging_writer()
    physics_path = out/'physics/physics.json'
    physics = json.loads(physics_path.read_text())
    physics['reference_pose_targets'] = (predicted.reshape(2, len(phases), 14)[:, ::10]+env.home).tolist()
    physics_path.write_text(json.dumps(physics, indent=2))
    env.close()
    metrics = evaluate(actor, out/'physics', count=20, seed=2801, jitter=True, record=out/'initial_rollout.npz')
    (out/'initial_evaluation.json').write_text(json.dumps(metrics, indent=2)); print('INITIAL NEURAL', error, metrics, flush=True)
    return out


def export_bundle(run, checkpoint, result_path):
    """Portable API-v2 inference pair, with real-consumer numerical replay."""
    import onnxruntime as ort
    from unilab.training.xduck_pretrain import TensorPolicy
    result = json.loads(result_path.read_text())
    if not result.get('passed'):
        raise RuntimeError('Export for deployment requires the simulation gate')
    cfg = OmegaConf.load(run/'config.yaml'); actor = load_actor(checkpoint, cfg)
    initial = load_actor(run/'initial.pt', cfg)
    if not torch.equal(actor.timing_head.weight,initial.timing_head.weight):
        raise RuntimeError('PPO changed its frozen searched timing prior')
    out = ROOT/'exports'/f'xduck_calibrated_sitstand_20260930_{checkpoint.stem}'
    out.mkdir(parents=True, exist_ok=True)
    policy = TensorPolicy(actor).eval()
    model = out/'xc_sitstand.onnx'
    torch.onnx.export(policy,(torch.zeros(1,62),),str(model),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    session = ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
    samples = np.load(result_path.parent/'center/rollout.npz')['obs'].reshape(-1,62)
    samples = samples[np.linspace(0,len(samples)-1,512,dtype=int)]
    with torch.no_grad(): expected = policy(torch.from_numpy(samples)).numpy()
    actual = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in samples])
    error = float(np.max(np.abs(expected-actual)))
    if error>1e-5 or not np.isfinite(actual).all(): raise RuntimeError(f'ONNX mismatch {error}')
    copied = (UPLOAD/'xc_sitstand_policy.py').read_text().replace('V1.1.2 model_267 Sit/Rise','V1.1.2 five-pose calibrated model_63 Sit/Rise')
    consumer_path = out/'xc_sitstand_policy.py'; consumer_path.write_text(copied)
    spec = importlib.util.spec_from_file_location('exported_xduck_consumer',consumer_path)
    consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
    replay_error = 0.; frames_count = 0
    for label,frames in identification.trials():
        instance = consumer.Policy()
        for index,row in enumerate(frames):
            frame = dict(positions=row['positions'],velocities=row['velocities'],
                imu=dict(gyro=row['imu_gyro'],gravity=row['imu_gravity']),
                policy_context=dict(action='sit' if 'sit' in label else 'rise'),dt=.02)
            if index==0: instance.reset({'joint_names':consumer.JOINTS},frame)
            inputs = instance.preprocess(frame,None); y = session.run(None,inputs)[0]
            with torch.no_grad(): ref = policy(torch.from_numpy(inputs['obs'])).numpy()
            replay_error = max(replay_error,float(np.max(np.abs(y-ref))))
            targets = instance.postprocess({'actions':y},frame)
            q = np.asarray([targets[n]['position'] for n in consumer.TRAINING_JOINTS])
            assert set(targets)==set(consumer.CONTROLLED)
            assert np.isfinite(q).all() and np.all(q>=consumer.TARGET_LOW) and np.all(q<=consumer.TARGET_HIGH)
            frames_count += 1
    if replay_error>1e-5: raise RuntimeError('Real consumer replay failed')
    with torch.no_grad(): feedback_change = float(torch.max(torch.abs(actor(actor_batch(samples))-initial(actor_batch(samples)))))
    if feedback_change<=1e-6: raise RuntimeError('No learned feedback change after PPO')
    for source,name in [(checkpoint,checkpoint.name),(run/'config.yaml','training_config.yaml'),
                        (run/'physics/physics.json','physics_hypothesis.json'),(result_path,'evaluation_holdout.json'),
                        (CALIBRATION/'candidate.json','five_pose_calibration_candidate.json'),
                        (Path(__file__),'xduck_calibrated_training_source.py')]: shutil.copy2(source,out/name)
    verification = dict(onnx_max_abs_error=error, onnx_samples=512, consumer_replay_frames=frames_count,
        consumer_replay_max_abs_error=replay_error, input_shape=[1,62], output_shape=[1,14], api_version=2,
        period_us=20000, transition_clock_seconds=4., frozen_timing_unchanged=True,
        ppo_feedback_max_action_change_rad=feedback_change, no_double_zero_or_imu_correction_in_consumer=True,
        gate_passed=True, hardware_tested=False)
    (out/'verification.json').write_text(json.dumps(verification, indent=2))
    (out/'README.txt').write_text('五姿态候选参数训练的坐下/起立策略；已通过记录的仿真门槛，尚未真机验证。\n'
        '部署使用 xc_sitstand.onnx + xc_sitstand_policy.py 两个文件，必须成对使用。\n'
        'API v2，20ms，输入62维，输出14个电机目标偏移；电机顺序、KP/KD及软限位沿用现有消费协议。\n'
        '训练模型含14关节K=75Nm/rad、D=1，以及五姿态拟合零位和有效IMU旋转。\n'
        '消费端继续读取原始编码器/IMU，不应再次叠加候选零位或IMU角度修正。\n'
        'model_63.pt用于UniLab续训；上传策略请选择ONNX与配套Python消费文件。\n'
        'training_config.yaml包含本机路径，复现实验还需要现有UniLab工程与CAD网格。\n')
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
        for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p,arcname=out.name+'/'+p.name)
    print('EXPORTED',archive,verification,flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluate', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--smooth-search', action='store_true')
    parser.add_argument('--prior-search', action='store_true')
    parser.add_argument('--initialize-prior', action='store_true')
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--mixture', type=Path)
    parser.add_argument('--mixture-resume', action='store_true')
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--export', type=Path)
    args = parser.parse_args(); run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    if not (run/'physics/physics.json').exists(): materialize(run/'physics')
    if args.export:
        export_bundle(run,args.export.resolve(),run/('evaluation_'+args.export.stem+'_holdout/results.json')); return
    if args.suite:
        evaluate_suite(run, args.suite.resolve(), seed=6401 if args.holdout else 4401,
                       count=64 if args.holdout else 20, label='holdout' if args.holdout else None); return
    if args.mixture:
        train_mixture(run, args.mixture.resolve(), args.iterations, max(2,args.envs//3), args.mixture_resume); return
    if args.search:
        search_ankle_bias(run); return
    if args.smooth_search:
        search_smooth_reference(run); return
    if args.prior_search:
        search_prior(run); return
    if args.initialize_prior:
        initialize_searched_prior(run); return
    if args.evaluate or args.baseline:
        cfg = OmegaConf.load(run/'config.yaml') if args.evaluate else OmegaConf.load(SOURCE_CFG)
        actor = load_actor(args.evaluate or SOURCE_PT, cfg)
    else:
        actor = train(run, args.iterations, args.envs, args.resume)
    metrics = evaluate(actor, run/'physics', count=5, record=run/'last_rollout.npz')
    (run/'last_evaluation.json').write_text(json.dumps(metrics, indent=2)); print('EVALUATION', metrics, flush=True)


if __name__ == '__main__': main()
