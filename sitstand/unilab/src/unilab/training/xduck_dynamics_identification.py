"""Offline MuJoCo/GF target-replay sensitivity experiments; never deploys a policy.

Variants are hypotheses, not identified hardware parameters. Flexible hinges add
an unobserved serial deflection to a measured motor coordinate, with spring and
damper; motor armature remains on the encoder coordinate.
"""
import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from unilab.actuators.gf43x40.actuator import GF43X40Parameters
from unilab.actuators.gf43x40.robot import GF43X40RobotMotor

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'training_runs/xduck_dynamics_identification_20260929'
INDEX = [0,1,2,3,4,10,11,12,13,5,6,7,8,9]
KP = [90,90,75,100,90,75,90,75,90,90,90,75,100,90]
KD = [2.5,2.5,2.5,3,3,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3,3]


def quat(roll,pitch):
    cr,sr=np.cos(roll/2),np.sin(roll/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2)
    return np.array([cr*cp,sr*cp,cr*sp,-sr*sp])


def trials():
    result=[]
    with zipfile.ZipFile('/Users/mac/Desktop/sit_rise_experiment_reports_20260929.zip') as z:
        for e,label in [(1,'sit'),(3,'sit'),(2,'rise')]:
            rows=[x for line in z.read(f'sit_rise_experiment_reports_20260929/experiment_{e:02d}/result.jsonl').splitlines() if (x:=json.loads(line))['type']=='frame' and x['policy_label']==label]
            result.append((f'trial{e}_{label}',rows))
    return result


def model(variant,meta,pars):
    root=ET.parse(OUT/'source_scene.xml').getroot()
    root.find('compiler').set('meshdir',str(ROOT.parent/'Model/1.1.2/mjcf/assets'))
    for tag in ['keyframe','size']:
        node=root.find(tag)
        if node is not None:root.remove(node)
    flex=variant.get('flex',[])
    for body in root.iter('body'):
        for joint in list(body.findall('joint')):
            if joint.get('name') in flex:
                ET.SubElement(body,'joint',name=joint.get('name')+'_flex',type='hinge',axis=joint.get('axis','0 0 1'),pos=joint.get('pos','0 0 0'),limited='false',stiffness=str(variant['stiffness']),damping=str(variant.get('damping',1)),armature='0',frictionloss='0')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    m.opt.timestep=.001/variant.get("physics_substeps",1)
    joints=np.array([m.joint(n).id for n in meta['policy_joints']]);qi=m.jnt_qposadr[joints];vi=m.jnt_dofadr[joints]
    ai=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints])
    m.dof_armature[vi]=pars.physics['armature'];m.dof_damping[vi]=0;m.dof_frictionloss[vi]=0
    limit=pars.physics['mechanical_torque_limit'];m.actuator_gainprm[:]=0;m.actuator_gainprm[:,0]=1;m.actuator_biasprm[:]=0;m.actuator_biasprm[:,1]=-1
    m.actuator_forcerange[:]=[-limit,limit];m.actuator_ctrlrange[:]=[-limit-pars.registers['pmax'],limit+pars.registers['pmax']]
    m.actuator_forcelimited[:]=1;m.actuator_ctrllimited[:]=1
    # Changes are cold-path model configuration, never environment backend internals.
    foot=[m.geom(n).id for n in ['left_foot_collision','right_foot_collision']]
    m.geom_friction[foot,0]=variant.get('friction',1.)
    if 'solref' in variant:m.geom_solref[foot,0]=variant['solref']
    d=mujoco.MjData(m);d.qpos[qi]=meta['home'];d.qpos[3:7]=[1,0,0,0];mujoco.mj_forward(m,d)
    groups={'head':['neck_pitch','head_pitch','head_yaw','head_roll'],'upper':['base_link','neck_pitch','head_pitch','head_yaw','head_roll','control-box','Battery','IMU','power-strip'],'all':[m.body(i).name for i in range(1,m.nbody)]}
    names=groups.get(variant.get('group','upper'),[variant.get('group')])
    for name in names:
        if name is None:continue
        b=m.body(name).id
        delta=np.array(variant.get('com_delta',[0,0,0]))
        m.body_ipos[b]+=d.xmat[b].reshape(3,3).T@delta
        m.body_mass[b]*=variant.get('mass_scale',1.)
        m.body_inertia[b]*=variant.get('inertia_scale',variant.get('mass_scale',1.))
    if 'foot_shift' in variant:
        for g in foot:m.geom_pos[g]+=d.xmat[m.geom_bodyid[g]].reshape(3,3).T@np.array([variant['foot_shift'],0,0])
    for geom_name in variant.get('disable_collision',[]):
        g=m.geom(geom_name).id;m.geom_contype[g]=0;m.geom_conaffinity[g]=0
    for geom_name,delta in variant.get('collision_shift',{}).items():
        g=m.geom(geom_name).id;m.geom_pos[g]+=d.xmat[m.geom_bodyid[g]].reshape(3,3).T@np.array(delta)
    mujoco.mj_setConst(m,d)
    fi=np.array([m.jnt_qposadr[m.joint(n+'_flex').id] for n in flex],dtype=int)
    return m,d,qi,vi,ai,fi


def run(variants,name):
    meta=json.loads((OUT/'metadata.json').read_text());pars=GF43X40Parameters.from_bundle();ts=trials();cases=[]
    for variant in variants:
        for label,rows in ts:
            if variant.get('stand_prefix') and 'sit' in label:
                trial_number=int(label[5])
                with zipfile.ZipFile('/Users/mac/Desktop/sit_rise_experiment_reports_20260929.zip') as z:
                    rows=[r for line in z.read(f'sit_rise_experiment_reports_20260929/experiment_{trial_number:02d}/result.jsonl').splitlines() if (r:=json.loads(line))['type']=='frame']
            m,d,qi,vi,ai,fi=model(variant,meta,pars);r=rows[0]
            d.qpos[:3]=[0,0,.32];d.qpos[3:7]=quat(*r['attitude_rpy'][:2]) if variant.get('initial_mode','physical')=='physical' else np.array([1.,0,0,0]);d.qpos[qi]=np.array(r['positions'])[INDEX];d.qvel[vi]=np.array(r['velocities'])[INDEX];d.qvel[3:6]=r['imu_gyro']
            mujoco.mj_forward(m,d)
            # Actual convex-mesh vertices, including their compiled principal frame.
            low=[]
            for g in range(m.ngeom):
                if m.geom_bodyid[g]==0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];low.append(world[:,2].min())
            d.qpos[2]-=min(low);mujoco.mj_forward(m,d)
            cases.append(dict(variant=variant,label=label,rows=rows,m=m,d=d,qi=qi,vi=vi,ai=ai,fi=fi,trace=[]))
    motor=GF43X40RobotMotor(len(cases),14,kp=KP,kd=KD,parameters=pars,jitter_mode='measured_20260918',jitter_seed=0,envelope_mode='continuous_conservative')
    policies=None
    if any(c['variant'].get('closed_loop') for c in cases):
        import onnxruntime as ort
        upload=Path('/Users/mac/Desktop/xc_upload_sitstand267_fix_20260929_1936')
        spec=importlib.util.spec_from_file_location('xduck_consumer',upload/'xc_sitstand_policy.py');consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
        session=ort.InferenceSession(str(upload/'xc_sitstand.onnx'),providers=['CPUExecutionProvider'])
        policies=[consumer.Policy() for _ in cases]
    target=np.array([np.array(c['rows'][0]['applied_targets'])[INDEX] for c in cases]);length=max(len(c['rows']) for c in cases)
    def positions():return np.array([c['d'].qpos[c['qi']] for c in cases])
    def velocities():return np.array([c['d'].qvel[c['vi']] for c in cases])
    for step in range(length):
        for i,c in enumerate(cases):
            d,m=c['d'],c['m'];w,x,y,z=d.qpos[3:7];contacts=[]
            for k in range(d.ncon):
                con=d.contact[k]
                if m.geom_bodyid[con.geom1]!=0 and m.geom_bodyid[con.geom2]!=0:continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
                if force[0]>.1:contacts.append([float(con.pos[0]),float(con.pos[2]),float(force[0]),int(con.geom1),int(con.geom2)])
            c['trace'].append({'pitch':float(np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))),'q':d.qpos[c['qi']].copy(),'root':d.qpos[:3].copy(),'com':d.subtree_com[1].copy(),'flex':d.qpos[c['fi']].copy(),'qpos':d.qpos.copy(),'contacts':contacts})
            target[i]=np.array(c['rows'][min(step,len(c['rows'])-1)]['applied_targets'])[INDEX]
        if policies is not None:
            qp=motor.quantize_position_feedback(positions());qv=motor.quantize_velocity_feedback(velocities())
            for i,c in enumerate(cases):
                if not c['variant'].get('closed_loop') or c['rows'][min(step,len(c['rows'])-1)]['policy_label']=='stand':continue
                p=np.zeros(15);v=p.copy();p[INDEX]=qp[i];v[INDEX]=qv[i]
                w,x,y,z=c['d'].qpos[3:7];gravity=np.array([2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1])
                gyro=c['d'].qvel[3:6].copy()
                # One-frame IMU delay, within the training 0-20 ms interval.
                old=c.get('imu_previous',(gyro,gravity));c['imu_previous']=(gyro,gravity)
                frame={'positions':p,'velocities':v,'imu':{'gyro':old[0],'gravity':old[1]},'policy_context':{'action':'sit' if 'sit' in c['label'] else 'rise'},'dt':.02}
                if not c.get('policy_started'):
                    policies[i].reset({'joint_names':consumer.JOINTS},frame);c['policy_started']=True
                obs=policies[i].preprocess(frame,None);a=session.run(None,obs)[0];out=policies[i].postprocess({'actions':a},frame);target[i]=[out[n]['position'] for n in consumer.TRAINING_JOINTS]
        for sub in range(20):
            q,v=positions(),velocities();tau=motor.begin_substep(target,q,v)
            for i,c in enumerate(cases):c['d'].ctrl[c['ai']]=q[i]+tau[i];mujoco.mj_step(c['m'],c['d'],nstep=c['variant'].get('physics_substeps',1))
            motor.finish_substep(positions(),velocities())
        if step%50==0:print(name,'step',step,'/',length,'cases',len(cases),flush=True)
    result=[];arrays={}
    for i,c in enumerate(cases):
        rows=c['rows'];offset=next(j for j,r in enumerate(rows) if r['policy_label']!='stand');n=len(rows);trace=c['trace'][offset:n];rows=rows[offset:];n=len(rows);p=np.array([x['pitch'] for x in trace]);p+=np.rad2deg(rows[0]['attitude_rpy'][1]) if c['variant'].get('initial_mode')=='bias' else 0.;real=np.rad2deg([x['attitude_rpy'][1] for x in rows]);q=np.array([x['q'] for x in trace]);qr=np.array([x['positions'] for x in rows])[:,INDEX];mask=np.abs(real)<40;t=np.arange(n)*.02
        first=np.flatnonzero(p<-40);rfirst=np.flatnonzero(real<-40)
        # Both position fidelity and attitude matter; no scoring only "it fell".
        score=dict(variant=c['variant'],trial=c['label'],standing_prefix_frames=offset,action_entry_sim_pitch_deg=float(trace[0]['pitch']),pitch_rmse_pre40_deg=float(np.sqrt(np.mean((p[mask]-real[mask])**2))),pitch_rmse_deg=float(np.sqrt(np.mean((p-real)**2))),joint_rmse_pre40_deg=float(np.rad2deg(np.sqrt(np.mean((q[mask]-qr[mask])**2)))),backward40_s=float(t[first[0]]) if len(first) else None,real_backward40_s=float(t[rfirst[0]]) if len(rfirst) else None,peak_flex_deg=float(np.rad2deg(max([np.max(np.abs(x['flex'])) if len(x['flex']) else 0 for x in trace]))))
        result.append(score);arrays[f'{i}_pitch']=p;arrays[f'{i}_q']=q;arrays[f'{i}_real_pitch']=real;arrays[f'{i}_real_q']=qr;arrays[f'{i}_qpos']=np.array([x['qpos'] for x in trace]);arrays[f'{i}_root']=np.array([x['root'] for x in trace]);arrays[f'{i}_com']=np.array([x['com'] for x in trace]);arrays[f'{i}_flex']=np.array([x['flex'] for x in trace]);score['array_index']=i
        if c['variant']['name']=='baseline' or c['variant'].get('save_contacts'):(OUT/f'{name}_{c["variant"]["name"]}_{c["label"]}_contacts.json').write_text(json.dumps([x['contacts'] for x in trace]))
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/f'{name}.npz',**arrays)
    grouped={}
    for r in result:grouped.setdefault(r['variant']['name'],[]).append(r)
    for key,rs in sorted(grouped.items(),key=lambda kv:np.mean([x['pitch_rmse_pre40_deg'] for x in kv[1]])):
        print(key,[(r['trial'],round(r['pitch_rmse_pre40_deg'],2),round(r['joint_rmse_pre40_deg'],2),r['backward40_s'],round(r['peak_flex_deg'],2)) for r in rs],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variants',type=Path);p.add_argument('--name',required=True);args=p.parse_args();run(json.loads(args.variants.read_text()),args.name)
