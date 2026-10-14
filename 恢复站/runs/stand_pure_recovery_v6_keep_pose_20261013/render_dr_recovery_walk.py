from pathlib import Path
import json,subprocess,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent
m=mujoco.MjModel.from_binary_path(str(R.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m)
m.vis.global_.offwidth=640;m.vis.global_.offheight=400;m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=400,width=640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-13;cam.azimuth=15
course=np.load(R/'model800_course_dr_only_trace.npz');cr=json.loads((R/'model800_course_dr_only_results.json').read_text());walk=np.load(R/'model800_walk_stop_trace.npz');wr=json.loads((R/'model800_walk_stop_results.json').read_text())
q=course['qpos'][0,4:24];w,x,y,z=q[:,3:7].T;pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1));roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));tilt_ids=[4+int(np.argmax(abs(pitch))),4+int(np.argmax(abs(roll)))]
walk_ids=[]
for command in [[.3,0,0],[0,0,.4]]:
 candidates=[row for row in wr['rows'] if np.allclose(row['command'],command) and row['success']]
 assert candidates;walk_ids.append(max(candidates,key=lambda row:row['walk_seconds'])['case'])
manifest=[]
try:
 for kind,data,report,ids,duration,name in [('tilt',course,cr,tilt_ids,8.,'model800_dr15_tilt_recovery_1x'),('walk',walk,wr,walk_ids,9.,'model800_dr15_walk_stop_1x')]:
  ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s','1280x400','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(R/(name+'.mp4'))],stdin=subprocess.PIPE)
  try:
   for frame in range(round(duration*25)):
    canvas=Image.new('RGB',(1280,400))
    for col,j in enumerate(ids):
     offset=int(data['starts'][j])-50 if kind=='walk' else 0;i=offset+min(round(frame/25/.02),round(duration/.02)-1)
     q=data['qpos'][i,j];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];cam.lookat[2]=.4
     renderer.update_scene(d,camera=cam,scene_option=opt);p=Image.fromarray(renderer.render());draw=ImageDraw.Draw(p);draw.rectangle((0,0,640,112),fill=(18,24,32));row=report['rows'][j]
     phase=('WALKING' if i<data['starts'][j] else 'STANDING') if kind=='walk' else 'TILT + VELOCITY RECOVERY'
     speed=float(np.linalg.norm(data['velocity'][i,j,:2]));tilt=float(data['tilt'][i,j]);clock=(i-int(data['starts'][j]))*.02 if kind=='walk' else (i+1)*.02
     draw.text((8,5),f'800 TOTAL | DR 1.5x | CASE {j} | {phase} | 1x',fill='white')
     draw.text((8,25),f'{"Time relative to switch" if kind=="walk" else "Time"}: {clock:+.2f}s | height {q[2]*100:.1f}cm | tilt {tilt:.1f}deg',fill=(255,210,125))
     draw.text((8,45),f'Speed XY {speed:.3f}m/s | Body angular speed {np.linalg.norm(data["gyro"][i,j]):.3f}rad/s',fill='white')
     if kind=='walk':
      draw.text((8,65),f'Walk command {row["command"]} | switch after {row["walk_seconds"]:.2f}s walking',fill='white');draw.text((8,85),f'Continuous physics and motor history | stop fell={row["stop_failed"]} | final success={row["success"]}',fill=(150,255,180))
     else:
      draw.text((8,65),'Random tilted pose + linear/angular velocity + joint velocity',fill='white');draw.text((8,85),f'Fell={row["failed"]} | final stand success={row["nonfall_settled"]}',fill=(150,255,180))
     canvas.paste(p,(col*640,0))
    if frame==round(duration*25)-25:canvas.save(R/(name+'_preview.png'))
    ff.stdin.write(np.asarray(canvas).tobytes())
  finally:ff.stdin.close();rc=ff.wait()
  assert rc==0;manifest.append(dict(video=name+'.mp4',cases=ids,duration_s=duration,fps=25,speed=1.,DR_multiplier=1.5,source='original saved evaluation trajectories, no resimulation',checkpoint=report['checkpoint']))
finally:renderer.close()
(R/'video_model800_dr_recovery_walk_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
