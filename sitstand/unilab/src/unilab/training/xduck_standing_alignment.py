"""Local static-standing diagnosis with measured motor angles and serial springs.

Encoder coordinates are constrained to the supplied measurements. Flexible
coordinates and the floating root remain dynamic; measured gravity is never a
constraint, target or fitting objective in the forward simulation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT.parent / 'Model/1.1.2/mjcf'
POLICY_JOINTS = [
    'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
    'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'neck_pitch_joint',
    'head_pitch_joint', 'head_yaw_joint', 'head_roll_joint',
    'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
    'right_knee_pitch_joint', 'right_ankle_pitch_joint',
]
MOTOR_TO_POLICY = [0, 1, 2, 3, 4, 10, 11, 12, 13, 5, 6, 7, 8, 9]
MOTOR_READINGS = np.array([-.061, -.008, .438, -.013, -.417,
                           -.033, .015, -.346, -.016, .314,
                           0., .072, -.004, -.004])
GRAVITY_RAW = np.array([-.064, -.034, -.997])


def gravity_from_quaternion(quat):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return rotation.reshape(3, 3).T @ np.array([0., 0., -1.])


def roll_pitch(gravity):
    gx, gy, gz = np.asarray(gravity)/np.linalg.norm(gravity)
    return np.rad2deg([np.arctan2(-gy, -gz), np.arctan2(gx, np.hypot(gy, gz))])


def quaternion(roll_deg, pitch_deg):
    roll, pitch = np.deg2rad([roll_deg, pitch_deg])/2
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    return np.array([cr*cp, sr*cp, cr*sp, -sr*sp])


def add_serial_flex(root, coefficients):
    """Cold-path XML overlay, one independent spring coordinate per motor."""
    added = []
    for body in root.iter('body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name')
            if name in coefficients:
                k, damping = coefficients[name]
                if body.find(f'joint[@name="{name}_flex"]') is None:
                    position = list(body).index(joint)+1
                    flex = ET.Element('joint', name=name+'_flex', type='hinge',
                        pos=joint.get('pos', '0 0 0'), axis=joint.get('axis', '0 0 1'),
                        limited='false', stiffness=str(k), damping=str(damping),
                        armature='0', frictionloss='0')
                    body.insert(position, flex)
                    added.append(name)
    assert set(added) <= set(POLICY_JOINTS)
    return added


def persist_full_flex():
    """Extend the active source MJCF; preserve its 14 motor coordinates and HOME."""
    path = MODEL/'robot.xml'
    root = ET.parse(path).getroot()
    old_joints = [node for node in root.iter('joint') if node.get('name')]
    old_order = [node.get('name') for node in old_joints]
    added = add_serial_flex(root, {name: (75., 1.) for name in POLICY_JOINTS})
    if added:
        for key in root.findall('keyframe/key'):
            old = np.fromstring(key.get('qpos'), sep=' ')
            # The freejoint contributes seven coordinates before hinge order.
            values = dict(zip(old_order, old[7:]))
            new_order = [j.get('name') for j in root.iter('joint') if j.get('name')]
            key.set('qpos', ' '.join(format(float(x), '.17g') for x in
                [*old[:7], *[values.get(n, 0.) for n in new_order]]))
        ET.indent(root, space='  ')
        path.write_text("<?xml version='1.0' encoding='utf-8'?>\n"+ET.tostring(root, encoding='unicode')+'\n')
    m = mujoco.MjModel.from_xml_path(str(MODEL/'scene.xml'))
    assert m.nq == 35 and m.nv == 34 and m.nu == 14
    for name in POLICY_JOINTS:
        assert m.joint(name+'_flex').id >= 0
    return added


def assemble_scene(case, output):
    robot = ET.parse(MODEL/'robot.xml').getroot()
    scene = ET.parse(MODEL/'scene.xml').getroot()
    scene.remove(scene.find('include'))
    # Flatten the single include, retaining source bodies, inertia and surfaces.
    for child in list(robot):
        if child.tag == 'worldbody':
            world = scene.find('worldbody')
            for body in list(child):
                world.append(copy.deepcopy(body))
        elif child.tag == 'asset':
            for asset in list(child):
                scene.find('asset').append(copy.deepcopy(asset))
        elif child.tag != 'keyframe':
            previous = scene.find(child.tag)
            if previous is not None:
                scene.remove(previous)
            scene.append(copy.deepcopy(child))
    scene.find('compiler').set('meshdir', str(MODEL/'assets'))
    opt = scene.find('option')
    opt.set('timestep', str(case.get('dt', .0005)))
    opt.set('iterations', '150')
    opt.set('tolerance', '1e-12')
    enabled = set(case.get('flex_joints', POLICY_JOINTS))
    stiffness = case.get('stiffness')
    coefficients = case.get('coefficients', {})
    for body in scene.iter('body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name', '')
            if name.endswith('_flex'):
                motor = name[:-5]
                if motor not in enabled:
                    body.remove(joint)
                else:
                    k, damping = coefficients.get(motor, [float(joint.get('stiffness')) if stiffness is None else stiffness,
                                                          float(joint.get('damping', '1'))])
                    joint.set('stiffness', str(k))
                    joint.set('damping', str(damping))
    # Measured encoder angle is an equality constraint, not a motor target whose
    # load-induced error could confound the motor/IMU comparison.
    eq = scene.find('equality')
    if eq is None:
        eq = ET.SubElement(scene, 'equality')
    motor_readings = np.asarray(case.get('motor_readings_rad', MOTOR_READINGS), dtype=float)
    if motor_readings.shape != (14,) or not np.isfinite(motor_readings).all():
        raise ValueError('Expected fourteen finite motor-ID-order encoder readings')
    zero_deg = np.asarray(case.get('joint_zero_offsets_deg', np.zeros(14)), dtype=float)
    if zero_deg.shape != (14,) or not np.isfinite(zero_deg).all() or np.max(np.abs(zero_deg)) > 3.0000001:
        raise ValueError('Joint zero offsets must be fourteen policy-order angles within +/-3 degrees')
    # Preserve measured motor coordinates: geometric angle = encoder + offset.
    # MuJoCo uses qpos-ref, so a positive geometric offset needs a negative ref.
    for name, offset in zip(POLICY_JOINTS, zero_deg):
        scene.find(f'.//joint[@name="{name}"]').set('ref', str(-np.deg2rad(offset)))
    if 'imu_mount_roll_pitch_deg' in case:
        mount = np.asarray(case['imu_mount_roll_pitch_deg'], dtype=float)
        if mount.shape != (2,) or not np.isfinite(mount).all() or np.max(np.abs(mount)) > 10.0000001:
            raise ValueError('Exploratory IMU mount roll/pitch must be within +/-10 degrees')
        # A reported-axis site independent of the CAD mesh's local axes.
        base = scene.find('worldbody/body[@name="base_link"]')
        ET.SubElement(base, 'site', name='imu_reported', pos='0 0 0',
                      quat=' '.join(map(str,quaternion(*mount))), size='.003', group='3')
        sensors = scene.find('sensor')
        if sensors is None:
            sensors = ET.SubElement(scene, 'sensor')
        ET.SubElement(sensors, 'framequat', name='orientation_reported', objtype='site', objname='imu_reported')
        ET.SubElement(sensors, 'gyro', name='angular_velocity_reported', site='imu_reported')
    for name, value, offset in zip(POLICY_JOINTS, motor_readings[MOTOR_TO_POLICY], zero_deg):
        # Joint equality is expressed relative to the joint reference angle.
        ET.SubElement(eq, 'joint', name='encoder_lock_'+name, joint1=name,
            polycoef=f'{value+np.deg2rad(offset):.17g} 0 0 0 0', solref='.002 1', solimp='.9999 .9999 .001')
    path = output/(case['name']+'.xml')
    path.write_text(ET.tostring(scene, encoding='unicode'))
    m = mujoco.MjModel.from_xml_path(str(path))
    if 'head_com_delta' in case:
        delta = np.array(case['head_com_delta'])
        data = mujoco.MjData(m)
        qi = [m.jnt_qposadr[m.joint(n).id] for n in POLICY_JOINTS]
        data.qpos[qi] = MOTOR_READINGS[MOTOR_TO_POLICY]
        mujoco.mj_forward(m, data)
        for name in ['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']:
            bid = m.body(name).id
            m.body_ipos[bid] += data.xmat[bid].reshape(3,3).T @ delta
        mujoco.mj_setConst(m, data)
    if 'disable_collision' in case:
        for name in case['disable_collision']:
            gid = m.geom(name).id
            m.geom_contype[gid] = m.geom_conaffinity[gid] = 0
    # Force feedback comes solely from equality locks in this diagnosis.
    m.actuator_gainprm[:] = 0
    m.actuator_biasprm[:] = 0
    mujoco.mj_saveLastXML(str(path), m)
    return m


def align_floor(m, d):
    mujoco.mj_forward(m, d)
    lows = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] and (m.geom_contype[gid] or m.geom_conaffinity[gid]) and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            lows.append((v @ d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid])[:,2].min())
    d.qpos[2] -= min(lows)
    d.qpos[2] += .001
    mujoco.mj_forward(m,d)


def run_case(case, output):
    begin = time.monotonic()
    m = assemble_scene(case, output)
    d = mujoco.MjData(m)
    qi = np.array([m.jnt_qposadr[m.joint(n).id] for n in POLICY_JOINTS])
    vi = np.array([m.jnt_dofadr[m.joint(n).id] for n in POLICY_JOINTS])
    flex_names = [m.joint(i).name[:-5] for i in range(m.njnt) if m.joint(i).name.endswith('_flex')]
    fi = np.array([m.jnt_qposadr[m.joint(n+'_flex').id] for n in flex_names],dtype=int)
    d.qpos[qi] = MOTOR_READINGS[MOTOR_TO_POLICY]
    d.qpos[3:7] = quaternion(*case.get('start_rp', [0.,0.]))
    align_floor(m,d)
    data = []
    final_time = case.get('seconds', 12.)
    stride = round(.02/m.opt.timestep)
    max_motor_error = 0.
    for tick in range(round(final_time/.02)+1):
        if tick:
            mujoco.mj_step(m,d,nstep=stride)
        gravity = gravity_from_quaternion(d.qpos[3:7])
        error = np.max(np.abs(d.qpos[qi]-MOTOR_READINGS[MOTOR_TO_POLICY]))
        max_motor_error = max(max_motor_error,float(error))
        row = {'t':float(d.time), 'gravity':gravity.tolist(), 'rp_deg':roll_pitch(gravity).tolist(),
               'root_pos':d.qpos[:3].tolist(),'root_speed':float(np.linalg.norm(d.qvel[:3])),
               'max_speed':float(np.max(np.abs(d.qvel))), 'encoder_error_rad':float(error),
               'flex_rad':d.qpos[fi].tolist(),'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'contacts':int(d.ncon)}
        data.append(row)
        if np.arccos(np.clip(-gravity[2],-1,1)) > np.deg2rad(50) or d.qpos[2] < .03:
            break
    tail = data[-min(50,len(data)):]
    mean_g = np.mean([r['gravity'] for r in tail],axis=0)
    real_g = GRAVITY_RAW/np.linalg.norm(GRAVITY_RAW)
    final_rp = roll_pitch(mean_g)
    stable = (len(data) >= round(final_time/.02) and max(r['root_speed'] for r in tail)<.002
              and max(r['max_speed'] for r in tail)<.01)
    tilt = float(np.rad2deg(np.arccos(np.clip(-mean_g[2]/np.linalg.norm(mean_g),-1,1))))
    result = {'case':case,'stable':stable,'simulation_time_s':float(d.time),
              'wall_seconds':time.monotonic()-begin,'real_gravity_raw':GRAVITY_RAW.tolist(),
              'real_gravity_normalized':real_g.tolist(), 'sim_gravity_mean_last1s':mean_g.tolist(),
              'gravity_difference_sim_minus_real':(mean_g-real_g).tolist(),
              'gravity_direction_error_deg':float(np.rad2deg(np.arccos(np.clip(np.dot(mean_g/np.linalg.norm(mean_g),real_g),-1,1)))),
              'sim_roll_pitch_deg':final_rp.tolist(),'real_roll_pitch_deg':roll_pitch(real_g).tolist(),
              'roll_pitch_error_deg':(final_rp-roll_pitch(real_g)).tolist(),
              'last1s_gravity_std':np.std([r['gravity'] for r in tail],axis=0).tolist(),
              'final_tilt_deg':tilt,'max_encoder_error_rad':max_motor_error,
              'final_encoder_error_rad':data[-1]['encoder_error_rad'],
              'final_encoder_rad_policy_order':d.qpos[qi].tolist(),
              'last1s_max_speed':max(r['max_speed'] for r in tail),
              'last1s_max_root_speed':max(r['root_speed'] for r in tail),
              'final_qvel':d.qvel.tolist(),
              'flex_by_joint_deg':dict(zip(flex_names,np.rad2deg(np.mean([r['flex_rad'] for r in tail],axis=0)).tolist())),
              'final_qpos':d.qpos.tolist()}
    # CAD site's axes differ from the controller body axes by a fixed rotation.
    site_id = m.site('imu').id
    mujoco.mj_forward(m,d)
    result['final_gravity_raw_cad_site_axes'] = (d.site_xmat[site_id].reshape(3,3).T@np.array([0.,0.,-1.])).tolist()
    result['scope'] = 'Gravity compared in unchanged body/controller axes; CAD site rotation is not treated as hardware mounting bias.'
    (output/(case['name']+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(output/(case['name']+'_trace.npz'),
                        time=[r['t'] for r in data],gravity=[r['gravity'] for r in data],
                        qpos=[r['qpos'] for r in data],qvel=[r['qvel'] for r in data],root_speed=[r['root_speed'] for r in data])
    print(json.dumps({'case':case['name'], **{k:result[k] for k in ['stable','simulation_time_s','sim_roll_pitch_deg',
          'gravity_direction_error_deg','final_encoder_error_rad','last1s_max_speed']}},ensure_ascii=False),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--persist-full-flex',action='store_true')
    parser.add_argument('--cases',type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.persist_full_flex:
        before = (MODEL/'robot.xml').read_text()
        (output/'robot_before_full_flex.xml').write_text(before)
        print('Added serial flexibility:',persist_full_flex(),flush=True)
    measurement = {'motor_source':'/Users/mac/Desktop/截屏2026-09-30 10.47.50.png',
       'gravity_source':'/Users/mac/Desktop/截屏2026-09-30 10.48.10.png',
       'motor_id_1_to_14_rad':MOTOR_READINGS.tolist(),'policy_joints':POLICY_JOINTS,
       'motor_to_policy':MOTOR_TO_POLICY,'gravity_raw':GRAVITY_RAW.tolist(),
       'real_roll_pitch_deg':roll_pitch(GRAVITY_RAW).tolist(),
       'assumption':'Screenshots 20s apart; stationary pose and body/controller gravity axes assumed.',
       'coefficients':'All mechanical springs initially 75 N m/rad, damping 1 N m s/rad; candidate assumptions, not measured per-joint stiffness.'}
    (output/'measurement.json').write_text(json.dumps(measurement,indent=2,ensure_ascii=False))
    cases = json.loads(args.cases.read_text()) if args.cases else [{'name':'all14_k75_upright','stiffness':75.}]
    results=[run_case(case,output) for case in cases]
    (output/'results.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
