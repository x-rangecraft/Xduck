from engine import *

ap=argparse.ArgumentParser()
ap.add_argument('--models',nargs='+',required=True)
ap.add_argument('--seed',type=int,default=1160091)
ap.add_argument('--seconds',type=float,default=12)
ap.add_argument('--prefix',default='walk_eval')
args=ap.parse_args()
B=8;N=B*len(args.models);ids=np.tile(np.arange(B),len(args.models))
np.random.seed(args.seed);torch.manual_seed(args.seed)
e=env(N);ps=[D.Policy() for _ in range(N)];cmd=np.zeros((N,3));rows=[];pre=[]
sessions=[actor(name) for name in args.models]
goal=np.asarray(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad'])
goal=np.tile(goal,(N,1));goal[B:]=D.REFERENCE;goal[B:,1]=.08;goal[B:,10]=-.08
try:
    e.init_state();e.set_autoreset(False)
    first_state=e._backend.get_physics_state()
    base={k:getattr(e._backend._device_model,k).numpy()[:B].copy() for k in FIELDS}
    base.update(encoder_bias=e._encoder_bias[:B].copy(),imu_rotation=e._imu_rotation[:B].copy(),
                sole_angles=e._sole_angles[:B].copy(),sole_delta_quat=e._sole_delta_quat[:B].copy())
    from unilab.dr.types import ResetRandomizationPayload
    e._backend.set_state(np.arange(N,dtype=np.int32),first_state[ids,1:22],first_state[ids,22:42],
                        ResetRandomizationPayload(**{k:base[k][ids] for k in FIELDS}))
    for k in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:
        getattr(e,'_'+k)[:]=base[k][ids]
    for p,f in zip(ps,frame_list(e,cmd)):
        p.reset({'joint_names':D.JOINTS},f)
    def hold(st):
        st.info['commands'][:,:3]=cmd;st.info['commands'][:,3:]=0
    e._maybe_resample_commands=hold
    start_target=e._current_action_target.copy();measure=feet(e)
    weight=base['body_mass'][ids].sum(1)*9.81
    for tick in range(200):
        hold(e.state);frames=frame_list(e,cmd);action=np.zeros((N,14))
        u=np.clip(tick/75,0,1);u=u*u*u*(10+u*(-15+6*u))
        targets=(1-u)*start_target+u*goal
        for j,(p,f) in enumerate(zip(ps,frames)):
            f['policy_context']['action']='stand';p.preprocess(f,None)
            p.postprocess({'actions':(targets[j]-D.REFERENCE)/D.STAND_ACTION_SCALE},f)
            action[j]=p.last_physical_offset/e._physical_scales
        e.step(action);np.testing.assert_allclose(e._current_action_target,targets,atol=3e-7)
        pre.append(dict(qpos=e._backend.get_physics_state()[:,1:22].copy(),velocity=e.get_local_linvel().copy(),feet=measure(e._backend.get_physics_state()[:,1:22])))
    initial=e._current_action_target.copy();initial_q=e._backend.get_physics_state()[:,1:22].copy()
    for tick in range(round(args.seconds/.02)):
        cmd[:,0]=.3;hold(e.state);frames=frame_list(e,cmd)
        raw=np.zeros((N,14),np.float32);obs=np.zeros((N,61),np.float32);target=np.zeros_like(raw);action=np.zeros_like(raw)
        for j,(p,f) in enumerate(zip(ps,frames)):
            ob=p.preprocess(f,None);obs[j]=ob['obs'][0]
            raw[j]=sessions[j//B].run(None,ob)[0][0]
            out=p.postprocess({'actions':raw[j]},f)
            target[j]=[out[name]['position'] for name in D.TRAINING_JOINTS]
            action[j]=p.last_physical_offset/e._physical_scales
        st=e.step(action);np.testing.assert_allclose(e._current_action_target,target,atol=3e-7)
        q=e._backend.get_physics_state()[:,1:22].copy()
        force=np.stack([np.linalg.norm(e._backend.get_sensor_data(name),axis=1) for name in e._cfg.sensor.feet_force],axis=1)
        rows.append(dict(qpos=q,obs=obs,raw=raw,target=target,velocity=np.c_[e.get_local_linvel()[:,:2],e.get_gyro()[:,2]],
                         loaded=force>.05*weight[:,None],clearance=e._foot_height.copy(),feet=measure(q),
                         heading=e._foot_heading.copy(),terminated=st.terminated.copy()))
        if (tick+1)%200==0:print(args.prefix,tick+1,flush=True)
    z={k:np.array([r[k] for r in rows]) for k in rows[0]}
    fields={k:base[k][ids] for k in FIELDS}
    np.savez_compressed(R/(args.prefix+'_trace.npz'),**z,**fields,names=np.array(args.models),source_indices=ids)
    np.savez_compressed(R/(args.prefix+'_stand_trace.npz'),**{k:np.array([r[k] for r in pre]) for k in pre[0]},**fields,names=np.array(args.models),source_indices=ids)
    records=[]
    for j in range(N):
        q=z['qpos'][:,j];v=z['velocity'][:,j,0];h=z['clearance'][:,j]
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)))
        width=(z['feet'][:,j,0,1]-z['feet'][:,j,1,1])*1000
        prior=np.zeros(2,bool);peak=np.zeros(2);events=[];micro=0;air_ticks=np.zeros(2,int)
        for t in range(len(q)):
            leaving=(~prior)&(h[t]>.0015)
            peak=np.maximum(peak,h[t]);air_ticks=np.where(prior|leaving,air_ticks+1,0)
            landing=prior&(h[t]<.0007)
            for side in np.flatnonzero(landing):
                if peak[side]>=.012 and air_ticks[side]>=4:
                    events.append(dict(time_s=t*.02,side=int(side),peak_mm=float(peak[side]*1000),width_mm=float(width[t])))
                elif peak[side]<.009:micro+=int(t<125)
            peak[landing]=0;prior=(prior|leaving)&(~landing)
        qw,qx,qy,qz=q[:,3:7].T;yaw=np.unwrap(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
        late=v[150:];horizon=min(126,len(q))
        steady_peaks=[ev['peak_mm'] for ev in events if ev['time_s']>3]
        steady_periods=[interval for side in range(2) for interval in np.diff([ev['time_s'] for ev in events if ev['side']==side and ev['time_s']>3])]
        r=dict(model=args.models[j//B],case=j%B,trace_index=j,seed=args.seed,stance_variant='pictured' if j<B else 'clean_default_with_natural_width',
            standing_tilt_max_deg=float(np.degrees(np.arccos(np.clip(1-2*(initial_q[j,4]**2+initial_q[j,5]**2),-1,1)))),
            tilt_max_deg=float(tilt.max()),min_height=float(q[:,2].min()),velocity_mean=float(late.mean()),
            min1s_velocity=float(np.convolve(late,np.ones(50)/50,'valid').min()),steady_width_mm=float(width[150:].mean()),
            steady_width_p95_mm=float(np.percentile(width[150:],95)),
            foot_heading_abs_deg=float(np.degrees(np.abs(z['heading'][150:,j])).mean()),
            startup_yaw_span_deg=float(np.degrees(np.ptp(yaw[:126]))),
            startup_width_min_mm=float(width[:126].min()),
            startup_width_max_mm=float(width[:126].max()),
            peak_swing_height_mm=float(np.median(steady_peaks)) if steady_peaks else 0.,
            cycle_seconds=float(np.median(steady_periods)) if steady_periods else 0.,
            startup_2p5_z_span_mm=float(np.ptp(q[:horizon,2])*1000),
            startup_2p5_drop_mm=float((initial_q[j,2]-q[:horizon,2].min())*1000),
            startup_2p5_vertical_speed_max=float(abs(np.diff(q[:horizon,2])/.02).max()),
            startup_2p5_acceleration_max=float(abs(np.diff(v[:horizon])/.02).max()),
            startup_min_velocity=float(v[:50].min()),initial_target_jump_rad=float(abs(target[0] if False else z['target'][0,j]-initial[j]).max()),
            shallow_startup_landings=micro,first_landings=events[:4],proper_steps=len(events),
            yaw_span_deg=float(np.degrees(np.ptp(yaw))),yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])))
        r['healthy']=bool(r['tilt_max_deg']<20 and r['min_height']>.25 and r['min1s_velocity']>.10 and len(events)>=12)
        r['narrow']=bool(r['healthy'] and 160<r['steady_width_mm']<205 and r['steady_width_p95_mm']<220)
        r['natural_width']=bool(r['healthy'] and 202<r['steady_width_mm']<222 and r['steady_width_p95_mm']<235)
        r['original_style']=bool(22<r['peak_swing_height_mm']<36 and .48<r['cycle_seconds']<.64 and .23<r['velocity_mean']<.32)
        r['straight']=bool(r['healthy'] and abs(r['yaw_change_deg'])<20 and r['yaw_span_deg']<30)
        r['smooth_start']=bool(r['natural_width'] and r['startup_2p5_drop_mm']<5 and r['startup_2p5_vertical_speed_max']<.16
            and r['startup_2p5_acceleration_max']<3. and r['startup_min_velocity']>-.03 and micro<=1
            and len(events)>1 and events[0]['time_s']<1.2 and events[1]['time_s']<1.8)
        r['user_goal_gate']=bool(r['smooth_start'] and r['original_style'] and r['straight'] and r['startup_width_min_mm']>195)
        records.append(r)
    (R/(args.prefix+'_results.json')).write_text(json.dumps(dict(
        standing_policy_inference_calls=0,walk_is_single_unassisted_neural_actor=True,
        prescribed_step_or_height_feedback=False,records=records),indent=2))
    print('SUMMARY',[(m,sum(r['healthy'] for r in records if r['model']==m),sum(r['smooth_start'] for r in records if r['model']==m),sum(r['straight'] for r in records if r['model']==m)) for m in args.models],flush=True)
finally:e.close()
