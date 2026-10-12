"""Exploratory shared encoder-zero / IMU-mount calibration from static poses.

Uses forward physics, never constrains the floating body to measured gravity.
Limited pose diversity means the solution is a regularized candidate, not a
hardware calibration. No source robot or hardware parameter is modified.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile
import mujoco
import numpy as np
from unilab.training.xduck_standing_alignment import (
    ROOT, MODEL, POLICY_JOINTS, MOTOR_TO_POLICY, assemble_scene, align_floor,
    gravity_from_quaternion, quaternion, roll_pitch,
)


def mount_rotation(rp_deg):
    """Columns are reported IMU-body axes expressed in mechanical body axes."""
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion(*rp_deg))
    return rotation.reshape(3, 3)


def read_poses():
    poses = []
    archive = Path('/Users/mac/Desktop/sit_rise_experiment_reports_20260929.zip')
    with zipfile.ZipFile(archive) as z:
        for i in (1, 2, 3):
            member = f'sit_rise_experiment_reports_20260929/experiment_{i:02d}/result.jsonl'
            rows = [json.loads(line) for line in z.read(member).splitlines()]
            # Last second BEFORE motion, not the last second of a sit/rise action.
            rows = [r for r in rows if r.get('type') == 'frame' and r['policy_label'] == 'stand'][-50:]
            q = np.array([r['positions'][:14] for r in rows])
            gyro = np.array([r['imu_gyro'] for r in rows])
            g = np.array([r['imu_gravity'] for r in rows])
            assert np.ptp(q, axis=0).max() < .003
            assert np.linalg.norm(gyro, axis=1).max() < .02
            poses.append(dict(name=f'trial{i}_stand', motor_readings_rad=q.mean(axis=0).tolist(),
                gravity=g.mean(axis=0).tolist(), frames=len(rows),
                source=str(archive)+'::'+member, synchronized=True,
                gyro_max_rad_s=float(np.linalg.norm(gyro,axis=1).max())))
    path = ROOT/'training_runs/xduck_static_imu_check_20260930/robot_state.jsonl'
    rows = [json.loads(line)['params'] for line in path.read_text().splitlines()]
    q = np.array([r['joints'][:14] for r in rows])
    assert np.ptp(q,axis=0).max() < .003
    poses.append(dict(name='current_stand', motor_readings_rad=q.mean(axis=0).tolist(),
        gravity=np.mean([r['safety']['gravity'] for r in rows],axis=0).tolist(),
        frames=len(rows), source=str(path), synchronized=True))
    for pose in poses:
        g = np.array(pose['gravity']); g /= np.linalg.norm(g)
        pose['gravity'] = g.tolist()
        pose['measured_roll_pitch_deg'] = roll_pitch(g).tolist()
    return poses


def bounded_regularized_fit(jacobian, target, center=None, joint_penalty=.15, mount_penalty=.02, mount_bound=10.):
    """Projected gradient solve with explicit degree bounds and weak priors."""
    n = jacobian.shape[1]
    assert n == 16
    bounds = np.array([3.]*14+[mount_bound,mount_bound])
    # A reported candidate needs a selection rule because it is underdetermined.
    regularization = np.array([joint_penalty]*14+[mount_penalty,mount_penalty])
    center = np.zeros(n) if center is None else np.asarray(center)
    h = jacobian.T@jacobian + np.diag(regularization**2)
    b = jacobian.T@(target+jacobian@center)
    x = center.copy()
    step = 1/np.linalg.eigvalsh(h)[-1]
    for _ in range(15000):
        new = np.clip(x-step*(h@x-b),-bounds,bounds)
        if np.max(np.abs(new-x)) < 1e-9:
            break
        x = new
    return x


def refine_mount(body_gravity, measured, initial, bound=10., penalty=.02):
    """Exact common rotation fit with fixed mechanical forward predictions."""
    rp=np.asarray(initial).copy()
    scale=180/np.pi
    for _ in range(20):
        prediction=body_gravity@mount_rotation(rp)
        residual=(measured-prediction).ravel()*scale
        j=np.empty((residual.size,2))
        for k in range(2):
            shifted=rp.copy();shifted[k]+=.01
            j[:,k]=((body_gravity@mount_rotation(shifted)-prediction)/.01).ravel()*scale
        delta=np.linalg.solve(j.T@j+np.eye(2)*penalty**2,j.T@residual-penalty**2*rp)
        rp=np.clip(rp+np.clip(delta,-.5,.5),-bound,bound)
        if np.max(np.abs(delta)) < 1e-8:
            break
    return rp


class PoseForward:
    def __init__(self, pose, output, dt=.0000625):
        self.pose = pose
        self.m = assemble_scene(dict(name=pose['name'], stiffness=75.,dt=dt,
            motor_readings_rad=pose['motor_readings_rad']), output)
        self.qi = np.array([self.m.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.enc = np.array(pose['motor_readings_rad'])[MOTOR_TO_POLICY]
        self.base_qpos0 = self.m.qpos0.copy()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[self.qi] = self.enc
        self.d.qpos[3:7] = quaternion(0.,0.)
        align_floor(self.m,self.d)

    def evaluate(self, offsets, warm=None, seconds=12.):
        if np.max(np.abs(offsets)) > 3.000001:
            raise ValueError('Encoder zeros outside +/-3 degrees')
        self.m.qpos0[self.qi] = self.base_qpos0[self.qi]-np.deg2rad(offsets)
        for name,value,offset in zip(POLICY_JOINTS,self.enc,offsets):
            self.m.eq_data[self.m.equality('encoder_lock_'+name).id,0]=value+np.deg2rad(offset)
        mujoco.mj_resetData(self.m,self.d)
        if warm is not None:
            self.d.qpos[:] = warm
        else:
            self.d.qpos[self.qi] = self.enc
            self.d.qpos[3:7] = quaternion(0.,0.)
            align_floor(self.m,self.d)
        mujoco.mj_step(self.m,self.d,nstep=round(seconds/self.m.opt.timestep))
        for _ in range(4):
            if np.max(np.abs(self.d.qvel)) < .01:
                break
            mujoco.mj_step(self.m,self.d,nstep=round(6./self.m.opt.timestep))
        qpos = self.d.qpos.copy()
        g = gravity_from_quaternion(qpos[3:7])
        velocity = float(np.max(np.abs(self.d.qvel)))
        err = float(np.max(np.abs(qpos[self.qi]-self.enc)))
        valid = np.isfinite(qpos).all() and velocity < .01 and err < 1e-5 and -g[2] > .9
        if not valid and self.m.opt.timestep > .000015625:
            self.m.opt.timestep /= 2
            return self.evaluate(offsets, warm=qpos, seconds=18.)
        if not valid:
            raise RuntimeError(f"{self.pose['name']}: unstable forward solve speed={velocity},encoder_error={err},rp={roll_pitch(g)},qvel={self.d.qvel.tolist()}")
        return g,qpos,dict(max_speed=velocity,encoder_error_rad=err,timestep=float(self.m.opt.timestep))


def fit(output, additional_poses=()):
    output.mkdir(parents=True,exist_ok=True)
    poses = read_poses()
    for path in additional_poses:
        pose=json.loads(Path(path).read_text())
        g=np.array(pose['gravity']);g/=np.linalg.norm(g)
        pose['gravity']=g.tolist()
        pose['measured_roll_pitch_deg']=roll_pitch(g).tolist()
        poses.append(pose)
    (output/'poses.json').write_text(json.dumps(poses,indent=2)+'\n')
    models = [PoseForward(p,output) for p in poses]
    cache_path=output/'forward_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    source_digest=hashlib.sha256((MODEL/'robot.xml').read_bytes()+(MODEL/'scene.xml').read_bytes()).hexdigest()
    executor=ThreadPoolExecutor(max_workers=4)
    def evaluate_batch(offsets,warms=None,seconds=12.):
        keys=[];futures=[]
        for i,model in enumerate(models):
            warm=None if warms is None else warms[i]
            key=json.dumps([model.pose['name'],np.asarray(offsets).tolist(),seconds,
                hashlib.sha256(np.asarray(warm).tobytes()).hexdigest() if warm is not None else None,
                'encoder-ref-v3-uniform75-adaptive62to15us', source_digest,
                model.pose['motor_readings_rad']])
            keys.append(key)
            futures.append(None if key in cache else executor.submit(model.evaluate,offsets,warm,seconds))
        results=[];errors=[]
        for key,future in zip(keys,futures):
            try:
                if future is not None:
                    g,q,info=future.result()
                    cache[key]=dict(g=g.tolist(),q=q.tolist(),info=info)
                    cache_path.write_text(json.dumps(cache)+'\n')
                r=cache[key];results.append((np.array(r['g']),np.array(r['q']),r['info']))
            except RuntimeError as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))
        return results
    measured = np.array([p['gravity'] for p in poses])
    zeros = np.zeros(14)
    baseline=[]; states=[]
    for model,(g,q,_) in zip(models,evaluate_batch(zeros)):
        baseline.append(g); states.append(q)
        print(json.dumps(dict(stage='baseline',pose=model.pose['name'],rp_deg=roll_pitch(g).tolist())),flush=True)
    baseline=np.array(baseline)
    # Difference in unit gravity times 180/pi is approximately angular degrees.
    scale=180/np.pi
    physical_j=np.empty((len(poses),3,14))
    for j,name in enumerate(POLICY_JOINTS):
        offset=zeros.copy();offset[j]=.25
        for i,(g,_,_) in enumerate(evaluate_batch(offset,states,seconds=6.)):
            physical_j[i,:,j]=(g-baseline[i])/.25
        print(json.dumps(dict(stage='sensitivity',joint=name)),flush=True)
    jac=np.zeros((len(poses),3,16));jac[:,:,:14]=physical_j
    for j in range(2):
        rp=[0.,0.];rp[j]=.25
        changed=baseline@mount_rotation(rp)
        jac[:,:,14+j]=(changed-baseline)/.25
    flat_j=jac.reshape(-1,16)*scale
    target=(measured-baseline).reshape(-1)*scale
    x=bounded_regularized_fit(flat_j,target)
    singular=np.linalg.svd(flat_j,compute_uv=False)
    rank=int(np.sum(singular>singular[0]*.01))
    # Hold one pose out at a time; these are linearized checks, separately labeled.
    loo=[]
    for i in range(len(poses)):
        mask=np.ones((len(poses),3),dtype=bool);mask[i]=False;mask=mask.ravel()
        fitted=bounded_regularized_fit(flat_j[mask],target[mask])
        predicted=baseline[i]+jac[i]@fitted;predicted/=np.linalg.norm(predicted)
        angle=np.rad2deg(np.arccos(np.clip(predicted@measured[i],-1,1)))
        loo.append(dict(held_out=poses[i]['name'],linearized_angle_error_deg=float(angle),
                        parameters_deg=fitted.tolist()))
    validation=[]; nonlinear=[]
    forward=evaluate_batch(x[:14],states,seconds=16.)
    x[14:]=refine_mount(np.array([item[0] for item in forward]),measured,x[14:])
    rotation=mount_rotation(x[14:])
    for i,(g,q,info) in enumerate(forward):
        prediction=rotation.T@g
        nonlinear.append(prediction)
        linear=baseline[i]+jac[i]@x;linear/=np.linalg.norm(linear)
        validation.append(dict(name=poses[i]['name'],measured_rp_deg=roll_pitch(measured[i]).tolist(),
            baseline_rp_deg=roll_pitch(baseline[i]).tolist(),
            predicted_rp_deg=roll_pitch(prediction).tolist(),
            body_rp_deg=roll_pitch(g).tolist(),
            error_rp_deg=(roll_pitch(prediction)-roll_pitch(measured[i])).tolist(),
            gravity_angle_error_deg=float(np.rad2deg(np.arccos(np.clip(prediction@measured[i],-1,1)))),
            nonlinear_vs_linear_deg=float(np.rad2deg(np.arccos(np.clip(prediction@linear,-1,1)))),
            final_qpos=q.tolist(),**info))
        print(json.dumps(dict(stage='validation',**{k:v for k,v in validation[-1].items() if k!='final_qpos'})),flush=True)
    result=dict(status='EXPLORATORY_UNDERDETERMINED_NOT_HARDWARE_CALIBRATION',
        joint_order=POLICY_JOINTS,joint_zero_offsets_deg=x[:14].tolist(),
        zero_definition='model geometric angle = measured encoder + offset; encoder coordinates stay measured',
        imu_mount_roll_pitch_deg=x[14:].tolist(),imu_mount_yaw_deg=0.,
        bounds=dict(joint_zero_deg=3.,imu_mount_rp_deg=10.),
        mount_bound_assumption='Exploratory +/-10 degrees, not a measured mounting tolerance.',
        stiffness=dict(all14_Nm_per_rad=75.,damping_Nm_s_per_rad=1.,fitted=False),
        poses=len(poses),parameters=16,singular_values=singular.tolist(),
        effective_rank_at_one_percent=rank,
        structural_observations_max=2*len(poses),
        regularization=dict(joint_zero_residual_deg_per_degree=.15,mount_residual_deg_per_degree=.02),
        fitting_method='Linearized joint-zero solve, full nonlinear mechanical validation, then exact common mount-rotation refinement with zeros fixed.',
        leave_one_pose_out_linearized=loo,nonlinear_validation=validation,
        limitations=[f'{len(poses)} static poses; still insufficient to identify sixteen parameters uniquely.',
            '14 encoder zeros and mount tilt are not uniquely identifiable.',
            'Accelerometer bias not fitted; unavailable raw acceleration and known orientations.',
            'IMU mount tilt can absorb accelerometer bias and unmodeled mechanics.',
            'Rigid model geometry, assumed stiffness75, contact and mass are not identified.',
            'Yaw mount fixed to zero; no three-axis excitation.',
            'Two screenshots excluded because encoder and IMU timestamps differ.',
            'Moving sit/rise data excluded from static fitting.'],
        source_robot_modified=False)
    (output/'candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'sensitivity.npz',jacobian=jac,baseline=baseline,measured=measured)
    executor.shutdown()
    return result


def validate_static_pose(pose, profile, output, label, start_body_pitch=0., dt=.00003125, seconds=20.):
    """Holdout prediction with fixed shared parameters and unconstrained body."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    name=label+'_pitch'+str(start_body_pitch).replace('.','p')
    model=assemble_scene(dict(name=name,stiffness=75.,dt=dt,
        motor_readings_rad=pose['motor_readings_rad'],
        joint_zero_offsets_deg=profile['joint_zero_offsets_deg'],
        imu_mount_roll_pitch_deg=profile['imu_mount_roll_pitch_deg']),output)
    data=mujoco.MjData(model)
    qi=np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    enc=np.array(pose['motor_readings_rad'])[MOTOR_TO_POLICY]
    data.qpos[qi]=enc;data.qpos[3:7]=quaternion(0.,start_body_pitch)
    align_floor(model,data)
    observed=np.array(pose['gravity']);observed/=np.linalg.norm(observed)
    mount=mount_rotation(profile['imu_mount_roll_pitch_deg'])
    trace=[];fell=False
    for _ in range(round(seconds/.02)):
        mujoco.mj_step(model,data,nstep=round(.02/model.opt.timestep))
        body_g=gravity_from_quaternion(data.qpos[3:7])
        reported=mount.T@body_g
        trace.append(dict(time=float(data.time),reported_g=reported.tolist(),
            rp_deg=roll_pitch(reported).tolist(),max_speed=float(np.max(np.abs(data.qvel))),
            encoder_error=float(np.max(np.abs(data.qpos[qi]-enc)))))
        if -body_g[2]<.5 or not np.isfinite(data.qpos).all():
            fell=True;break
    tail=trace[-min(50,len(trace)):]
    g=np.mean([row['reported_g'] for row in tail],axis=0);g/=np.linalg.norm(g)
    std=np.std([row['rp_deg'] for row in tail],axis=0)
    stable=not fell and len(trace)==round(seconds/.02) and max(row['max_speed'] for row in tail)<.01 and max(std)<.01
    mujoco.mj_forward(model,data)
    contacts=sorted(set(tuple(sorted([model.geom(c.geom[0]).name,model.geom(c.geom[1]).name]))
                       for c in data.contact if c.dist<.001))
    result=dict(candidate=label,start_body_pitch_deg=start_body_pitch,timestep_s=dt,
        stable=bool(stable),fell=bool(fell),duration_s=float(data.time),
        measured_rp_deg=roll_pitch(observed).tolist(),predicted_rp_deg=roll_pitch(g).tolist(),
        error_rp_deg=(roll_pitch(g)-roll_pitch(observed)).tolist(),
        gravity_angle_error_deg=float(np.rad2deg(np.arccos(np.clip(g@observed,-1,1)))),
        last1s_rp_std_deg=std.tolist(),last1s_max_speed=max(row['max_speed'] for row in tail),
        max_encoder_error_rad=max(row['encoder_error'] for row in trace),
        final_qpos=data.qpos.tolist(),final_qvel=data.qvel.tolist(),contacts=contacts,
        profile_fixed=True,measurement_used_as_body_constraint=False)
    (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/(name+'_trace.npz'),time=[row['time'] for row in trace],
        reported_g=[row['reported_g'] for row in trace],rp_deg=[row['rp_deg'] for row in trace])
    print(json.dumps({k:v for k,v in result.items() if k not in ('final_qpos','final_qvel','contacts')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pose',type=Path,action='append',default=[])
    args=parser.parse_args();fit(args.output.resolve(),args.pose)
