"""Render model_100 from the active 5090 Warp run at 1x playback speed."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import torch
import mujoco
import imageio_ffmpeg
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

R=Path(__file__).resolve().parent
OUT=R/'videos';OUT.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,default=R/'train_gpu4096_fresh5000/model_100.pt')
parser.add_argument('--config',type=Path,default=R/'project/conf/ppo/xduck_gpu_active.yaml')
parser.add_argument('--tag',default='gpu_model100_dr100_seed100401')
parser.add_argument('--seed',type=int,default=100401)
parser.add_argument('--command',type=float,default=.17)
parser.add_argument('--seconds',type=float,default=20.)
parser.add_argument('--completed-updates',type=int,default=101)
parser.add_argument('--dr-label',default='strong DR')
a=parser.parse_args();TAG=a.tag;CHECKPOINT=a.checkpoint
if not CHECKPOINT.is_file():raise FileNotFoundError(CHECKPOINT)
sys.path.insert(0,str(R/'project/scripts'))
import train_rsl_rl as E
np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
E.ensure_registries();E.patch_rsl_rl_resume_state()
c=OmegaConf.load(a.config)
c.env.audit_velocity=a.command
c.env.max_episode_seconds=max(20,a.seconds+5)
e=E.create_env(c,num_envs=4,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
    algo=E._algo_config_dict(c);wrapper=E._resolve_ppo_wrapper_cls(algo)(e,device='cpu')
    train=E.normalize_ppo_train_cfg(algo);train['logger']='none'
    runner=OnPolicyRunner(wrapper,train,log_dir=None,device='cpu')
    runner.load(str(CHECKPOINT),map_location='cpu')
    actor=runner.get_inference_policy(device='cpu')
    e.set_autoreset(False)
    geom_pos=e._backend._device_model.geom_pos.numpy().copy()
    geom_quat=e._backend._device_model.geom_quat.numpy().copy()
    angles=np.rad2deg(e._sole_angles.copy())
    trace={key:[] for key in ('qpos','vx','failure','target','contact')}
    for i in range(round(a.seconds/.02)):
        with torch.inference_mode():action=actor(wrapper.get_observations(),stochastic_output=False).numpy()
        state=e.step(action)
        values={'qpos':e._backend.get_physics_state()[:,1:1+e._backend._nq],
                'vx':e.get_local_linvel()[:,0],'failure':state.terminated,
                'target':e._current_action_target,'contact':e._contact}
        for key,value in values.items():trace[key].append(np.asarray(value).copy())
        if not all(np.isfinite(value).all() for value in values.values()):
            raise RuntimeError(f'Nonfinite playback at control step {i}')
    z={key:np.asarray(values) for key,values in trace.items()}
    np.savez_compressed(OUT/f'{TAG}.npz',**z)
    report={'checkpoint':str(CHECKPOINT),'completed_updates':a.completed_updates,
        'backend':'mjwarp','learner_checkpoint_device':'cuda','render_action':'deterministic',
        'dr_label':a.dr_label,'microsteps_per_motor_ms':int(c.env.physics_microsteps),'command_vx_m_s':a.command,
        'seconds_per_env':a.seconds,'sole_tilt_deg':angles.tolist(),'autoreset':False,'failed_events_per_env':z['failure'].sum(axis=0).astype(int).tolist(),
        'mean_vx_after_2s_m_s':z['vx'][100:].mean(axis=0).tolist(),
        'mean_target_delta_rad':float(np.abs(np.diff(z['target'],axis=0)).mean()),
        'single_support_after_2s':(z['contact'][100:].sum(axis=2)==1).mean(axis=0).tolist()}
    (OUT/f'{TAG}.json').write_text(json.dumps(report,indent=2))
finally:e.close()

m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'))
assert geom_pos.shape==(4,m.ngeom,3) and geom_quat.shape==(4,m.ngeom,4)
d=mujoco.MjData(m)
m.mat_rgba[:,3]=1.;m.geom_rgba[:,3]=1.
for name in ['left_foot_collision','right_foot_collision']:
    m.geom_group[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)]=2
renderer=mujoco.Renderer(m,height=480,width=640)
camera=mujoco.MjvCamera();camera.distance=1.25;camera.elevation=-18;camera.azimuth=120
option=mujoco.MjvOption();option.geomgroup[3]=0
path=OUT/f'{TAG}_grid.mp4'
video=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
    '-f','rawvideo','-vcodec','rawvideo','-s','1280x960','-pix_fmt','rgb24','-r','25','-i','-',
    '-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
try:
    for i in range(0,len(z['qpos']),2):
        canvas=Image.new('RGB',(1280,960))
        for j in range(4):
            m.geom_pos[:]=geom_pos[j];m.geom_quat[:]=geom_quat[j]
            d.qpos[:]=z['qpos'][i,j];mujoco.mj_forward(m,d)
            camera.lookat[:]=d.qpos[:3];camera.lookat[2]=.22
            renderer.update_scene(d,camera=camera,scene_option=option)
            frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,46),fill=(15,25,36))
            draw.text((8,5),f'5090 GPU {a.completed_updates} updates | {a.dr_label} | case {j+1}',fill='white')
            draw.text((8,24),f'cmd +{a.command:.2f} | vx {z["vx"][i,j]:+.3f} m/s | falls {int(z["failure"][:i+1,j].sum())} | 1x',fill='white')
            draw.text((8,465),f'sole DR L {angles[j,0]:+.2f} / R {angles[j,1]:+.2f} deg',fill='white')
            canvas.paste(frame,((j%2)*640,(j//2)*480))
        video.stdin.write(np.asarray(canvas).tobytes())
finally:
    video.stdin.close();code=video.wait();renderer.close()
if code:raise RuntimeError(f'ffmpeg exited {code}')
print(json.dumps({'video':str(path),'bytes':path.stat().st_size,'report':report},indent=2),flush=True)
