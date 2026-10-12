from pathlib import Path
import os
os.environ["MUJOCO_GL"]="egl"
import sys,json,subprocess
import numpy as np,torch,mujoco,imageio_ffmpeg
from omegaconf import OmegaConf
from PIL import Image,ImageDraw
from rsl_rl.runners import OnPolicyRunner
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'scripts'));import train_rsl_rl as E
TAG='checkpoint800_dr2_cmd03';ck=R/'rigid4096_curriculum/model_800.pt';saved=torch.load(ck,map_location='cpu',weights_only=False);step=int(saved['infos']['unilab_progress']['env_steps_per_env']);assert saved['iter']==800 and step==19224
E.ensure_registries();E.patch_rsl_rl_resume_state();np.random.seed(50021);torch.manual_seed(50021);torch.set_num_threads(2)
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml')
for key,stages in list(c.env.curriculum.items()):
 if key.endswith('_stages') and OmegaConf.is_list(stages):
  selected=OmegaConf.to_container([s for s in stages if int(s.step)<=step][-1]);selected['step']=0;c.env.curriculum[key]=[selected]
c.env.curriculum.step_cap=step;c.env.curriculum.standing_stages=[{'step':0,'value':0}];c.env.audit_twist=[.300,0,0];c.env.audit_velocity=None;c.env.max_episode_seconds=20
c.env.commands.resampling_time_range=[1000,1000];c.env.commands.head_resampling_time_range=[1000,1000];c.env.commands.body_resampling_time_range=[1000,1000]
OmegaConf.save(c,R/(TAG+'_config.yaml'));e=E.create_env(c,num_envs=4,env_cfg_override=E.build_ppo_env_cfg_override(c));e.init_state();e.step_counter=step;e._update_curriculum();e.reset(np.arange(4,dtype=np.int32))
try:
 algo=E._algo_config_dict(c);wrapped=E._resolve_ppo_wrapper_cls(algo)(e,device='cpu');train=E.normalize_ppo_train_cfg(algo);train['logger']='none';runner=OnPolicyRunner(wrapped,train,log_dir=None,device='cpu');runner.load(str(ck),map_location='cpu');actor=runner.get_inference_policy(device='cpu');e.set_autoreset(False)
 e._next_push_step[:]=step+np.ceil(np.random.default_rng(529).uniform(*e._cfg.domain_rand.velocity_push_interval_s,4)/.02).astype(int)
 geom_pos=e._backend._device_model.geom_pos.numpy().copy();geom_quat=e._backend._device_model.geom_quat.numpy().copy();angles=np.degrees(e._sole_angles.copy());assert e._dr_multiplier==2 and e._cfg.reward_config.air_time_threshold_min==.18
 rows=[];alive=np.ones(4,dtype=bool);failure=[None]*4
 for i in range(400):
  old=e._next_push_step.copy()
  with torch.inference_mode():action=actor(wrapped.get_observations(),stochastic_output=False).numpy()
  st=e.step(action);np.testing.assert_allclose(st.info['commands'][:,:3],np.tile([.300,0,0],(4,1)),atol=1e-6)
  for j in range(4):
   if st.terminated[j] and failure[j] is None:failure[j]=(i+1)*.02
  alive&=~np.asarray(st.terminated,dtype=bool)
  rows.append(dict(qpos=e._backend.get_physics_state()[:,1:22].copy(),vx=e.get_local_linvel()[:,0].copy(),contact=e._contact.copy(),alive=alive.copy(),push=(e._next_push_step!=old)))
  assert all(np.isfinite(v).all() for v in rows[-1].values())
  if i%100==0:print(f'checkpoint800 replay {(i+1)*.02:.2f}/8.00s',flush=True)
 z={k:np.array([r[k] for r in rows]) for k in rows[0]};OUT=R/'videos';OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/(TAG+'.npz'),**z,geom_pos=geom_pos,geom_quat=geom_quat,sole_tilt_deg=angles)
 report=dict(checkpoint=str(ck),completed_updates=801,command_m_s=.300,DR=2,threshold_ms=180,external_velocity_pushes=True,push_first_after_4_to_9s=True,seconds=8,sample_dt=.02,cases=[])
 for j in range(4):
  mask=z['alive'][:,j]&(np.arange(400)>=100);vx=z['vx'][:,j];avg=np.convolve(vx,np.ones(25)/25,'valid');avgmask=mask[24:];periods=[]
  for side in range(2):
   swing=mask&~z['contact'][:,j,side]&z['contact'][:,j,1-side];edge=np.diff(np.r_[False,swing,False].astype(int));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
   touches=[end*.02 for start,end in zip(starts,ends) if start>100 and end<400 and mask[end] and end-start>=5];periods.extend(np.diff(touches).tolist())
  periods=[p for p in periods if .25<=p<=1.2]
  report['cases'].append(dict(case=j,failure_s=failure[j],sole_tilt_deg=angles[j].tolist(),mean_vx=float(vx[mask].mean()) if mask.any() else None,raw_vx_std=float(vx[mask].std()) if mask.any() else None,moving05_vx_std=float(avg[avgmask].std()) if avgmask.any() else None,cadence_steps_per_s=2/float(np.median(periods)) if periods else None,single_contact_fraction=float((z['contact'][mask,j].sum(1)==1).mean()) if mask.any() else None,push_count=int(z['push'][:,j].sum())))
 (OUT/(TAG+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
finally:e.close()
m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));d=mujoco.MjData(m);m.mat_rgba[:,3]=1;m.geom_rgba[:,3]=1
for side in ['left','right']:m.geom_group[m.geom(side+'_foot_collision').id]=2
renderer=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.distance=1.25;cam.elevation=-18;cam.azimuth=120;opt=mujoco.MjvOption();opt.geomgroup[3]=0
path=OUT/(TAG+'_grid.mp4');ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s','1280x960','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
try:
 for i in range(0,400,2):
  canvas=Image.new('RGB',(1280,960))
  for j in range(4):
   fail=failure[j];frozen=fail is not None and (i+1)*.02>=fail;idx=min(i,max(0,round(fail/.02)-1)) if frozen else i
   m.geom_pos[:]=geom_pos[j];m.geom_quat[:]=geom_quat[j];d.qpos[:]=z['qpos'][idx,j];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.22;renderer.update_scene(d,camera=cam,scene_option=opt)
   im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,52),fill=(15,25,36));draw.text((8,6),f'checkpoint800 | rigid4096 | 180ms | DR2 | case{j+1} | 1x',fill='white')
   msg=f'FAILED {fail:.2f}s - FROZEN' if frozen else f'cmd +.300 | vx {z["vx"][idx,j]:+.3f} m/s | pushes {int(z["push"][:i+1,j].sum())}'
   draw.text((8,29),msg,fill='red' if frozen else 'white');canvas.paste(im,((j%2)*640,(j//2)*480))
  ff.stdin.write(np.asarray(canvas).tobytes())
finally:ff.stdin.close();rc=ff.wait();renderer.close()
assert rc==0;print(path,flush=True)
