from common import *
import json,time,argparse,torch,mujoco
from landing_metrics import metrics
a=argparse.Namespace(stage='search')
protocol={'training_seeds':[330091]};torch.set_num_threads(2)

def rotate(q,v):
    uv=np.cross(q[...,1:],v);return v+2*(q[...,:1]*uv+np.cross(q[...,1:],uv))

def feet(e):
    # Metadata access once per environment, never in the rollout hot path.
    m=mujoco.MjModel.from_binary_path(str(R.parent/'xduck_reward_audit_20261009/verified_rigid_model.mjb'))
    names=[m.body(int(m.site(s+'_foot').bodyid[0])).name for s in ['left','right']]
    bodyids=np.array([e._backend.get_body_id(n) for n in names],np.int32)
    local=np.array([m.site(s+'_foot').pos.copy() for s in ['left','right']])
    def measure(qpos):
        pos,quat=e._backend.get_body_pose_w(bodyids)
        world=pos+rotate(quat,np.broadcast_to(local,pos.shape))
        inverse=qpos[:,3:7].copy();inverse[:,1:]*=-1
        return rotate(inverse[:,None,:],world-qpos[:,None,:3])
    return measure

def summarize(z,j,onset=0):
    m=metrics(z['loaded'][:,j],z['velocity'][:,j],np.array([.3,0,0]),onset,fall=bool(z['terminated'][:,j].any()))
    start=max(onset+100,len(z['qpos'])-500);q=z['qpos'][start:,j]
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)))
    loc=z['feet'][start:,j];width=(loc[:,0,1]-loc[:,1,1])*1000;target=z['target'][start:,j]
    m.update(width_mm=float(width.mean()),width_ratio=float(width.mean()/242.15047973629743),
        tilt_mean=float(tilt.mean()),tilt_p95=float(np.percentile(tilt,95)),tilt_max=float(tilt.max()),
        min_height=float(q[:,2].min()),left_roll_saturation=float(np.isclose(target[:,1],-.11,atol=1e-6).mean()),
        right_roll_saturation=float(np.isclose(target[:,10],.19,atol=1e-6).mean()),
        foot_heading_abs_deg=float(np.degrees(abs(z['heading'][start:,j])).mean()),
        hip_roll_mean_deg=np.degrees(q[:,[8,17]]).mean(0).tolist(),
        lateral_foot_span_mm=(np.diff(np.percentile(loc[:,:,1],[5,95],axis=0),axis=0)[0]*1000).tolist(),
        vx=float(z['velocity'][start:,j,0].mean()),fall=bool(z['terminated'][:,j].any()))
    return m

def handoff(seed):
    tag=f'handoff_new_{seed}'
    if (R/(tag+'.npz')).exists():return tag
    np.random.seed(seed);torch.manual_seed(seed);e=env(16);ps=[D.Policy() for _ in range(16)];cmd=np.zeros((16,3));stand=actor('xc_standing.onnx')
    try:
        e.init_state();e.set_autoreset(False);np.random.seed(seed+4);e._sole_rng=np.random.default_rng(seed+2);e._flex_rng=np.random.default_rng(seed+3);e.reset(np.arange(16,dtype=np.int32))
        def hold(st):st.info['commands'][:,:3]=cmd;st.info['commands'][:,3:]=0
        e._maybe_resample_commands=hold
        for p,f in zip(ps,frame_list(e,cmd)):p.reset({'joint_names':D.JOINTS},f)
        for i in range(250):
            actions=np.zeros((16,14));targets=np.zeros_like(actions)
            for j,(p,f) in enumerate(zip(ps,frame_list(e,cmd))):
                f['policy_context']['action']='stand';o=p.preprocess(f,None);raw=stand.run(None,o)[0][0];out=p.postprocess({'actions':raw},f);actions[j]=p.last_physical_offset/e._physical_scales;targets[j]=[out[n]['position'] for n in D.TRAINING_JOINTS]
            st=e.step(actions);np.testing.assert_allclose(e._current_action_target,targets,atol=3e-7)
            assert not st.terminated.any()
        snapshot(e,ps,tag)
    finally:e.close()
    return tag

