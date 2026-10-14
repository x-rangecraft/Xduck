from pathlib import Path
import json,subprocess,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent;z=np.load(R/'block_12_course_dr_only_trace.npz');report=json.loads((R/'block_12_course_dr_only_results.json').read_text());ids=[18,19]
m=mujoco.MjModel.from_binary_path(str(R.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=640;m.vis.global_.offheight=480;m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=480,width=640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-12;cam.azimuth=90
name='model1600_dr2_pitch50_nonfall_recovery_1x';ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s','1280x480','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(R/(name+'.mp4'))],stdin=subprocess.PIPE)
try:
 for frame in range(200):
  i=min(round(frame/25/.02),399);canvas=Image.new('RGB',(1280,480))
  for col,j in enumerate(ids):
   q=z['qpos'][i,j];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];cam.lookat[2]=.38;renderer.update_scene(d,camera=cam,scene_option=opt);p=Image.fromarray(renderer.render());draw=ImageDraw.Draw(p);draw.rectangle((0,0,640,110),fill=(18,24,32));params=report['tilt_case_parameters'][j-4]
   draw.text((8,5),f'1600 TOTAL | DR 2x | INITIAL FORWARD PITCH +50deg | CASE {j} | 1x',fill=(150,255,180))
   draw.text((8,25),f'Initial Vx {params["linear_velocity"][0]:+.2f}m/s | initial pitch rate {params["angular_velocity"][1]:+.2f}rad/s | NO PUSH',fill='white')
   draw.text((8,45),f't={(i+1)*.02:.2f}s | body tilt {z["tilt"][i,j]:.1f}deg | root height {q[2]*100:.1f}cm',fill=(255,210,125))
   draw.text((8,65),f'Speed XY {np.linalg.norm(z["velocity"][i,j,:2]):.3f}m/s | body angular speed {np.linalg.norm(z["gyro"][i,j]):.3f}rad/s',fill='white')
   draw.text((8,85),'8s outcome: DID NOT FALL | full final pose acceptance: NO',fill=(150,255,180));draw.text((8,100),'SIDE VIEW | ORIGINAL SAVED DR EVALUATION TRAJECTORY',fill=(190,190,190));canvas.paste(p,(col*640,0))
  if frame==175:canvas.save(R/(name+'_preview.png'))
  ff.stdin.write(np.asarray(canvas).tobytes())
finally:ff.stdin.close();rc=ff.wait();renderer.close()
assert rc==0
(R/'video_model1600_pitch50_manifest.json').write_text(json.dumps(dict(checkpoint=report['checkpoint'],total_updates=1600,DR_multiplier=2.,cases=ids,initial_parameters=[report['tilt_case_parameters'][j-4] for j in ids],duration_s=8.,speed=1.,source='original saved evaluation trajectories; no resimulation',nonfall=True,full_final_pose_accepted=False),indent=2)+'\n');print(R/(name+'.mp4'),flush=True)
