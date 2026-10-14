import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,subprocess,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent;O=R/'small_push_audit/videos';O.mkdir(exist_ok=True)
policy=np.load(R/'small_push_audit/policy_15N_trace.npz');locked=np.load(R/'small_push_audit/hold_target_15N_trace.npz')
m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));d=mujoco.MjData(m);ref=m.key_qpos[0,7:]
W,H,FPS=640,480,25;m.vis.global_.offwidth=W;m.vis.global_.offheight=H
m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=H,width=W);option=mujoco.MjvOption();option.geomgroup[3]=0
cam=mujoco.MjvCamera();cam.distance=1.3;cam.elevation=-12
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
def panel(z,j,i,azimuth,title,subtitle,status):
 q=z['qpos'][i,j];d.qpos[:]=q;mujoco.mj_fwdPosition(m,d)
 cam.azimuth=azimuth;cam.lookat[:]=q[:3];cam.lookat[2]=.34
 renderer.update_scene(d,camera=cam,scene_option=option)
 if 'force' in z.files:
  force=z['force'][i,j];total=force.sum(0);mag=np.linalg.norm(total)
  if mag>1e-6:
   body_index=int(np.argmax(np.linalg.norm(force,axis=1)));body=['trunk_base','yaw_roll_motion'][body_index]
   end=d.xpos[m.body(body).id].copy();start=end-total/mag*.32
   g=renderer.scene.geoms[renderer.scene.ngeom]
   mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,.18,.06,1.]))
   mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.009,start,end);renderer.scene.ngeom+=1
 p=Image.fromarray(renderer.render());text=ImageDraw.Draw(p)
 text.rectangle((0,0,W,79),fill=(18,24,32));text.text((8,5),title,font=font,fill='white');text.text((8,28),subtitle,font=small,fill=(255,210,125))
 error=np.degrees(q[7:]-ref);head=np.degrees(np.arccos(np.clip(z['head_up'][i,j],-1,1)))
 text.text((8,49),f'Head tilt {head:.1f}deg | yaw {error[7]:+.1f}deg | roll {error[8]:+.1f}deg',font=small,fill='white')
 text.rectangle((0,H-27,W,H),fill=(18,24,32));text.text((8,H-24),status,font=small,fill=(180,220,190))
 return p
def render(name,seconds,build):
 output=O/(name+'.mp4');ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s',f'{W*2}x{H*2}','-pix_fmt','rgb24','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
 try:
  for frame in range(round(seconds*FPS)):
   canvas=Image.new('RGB',(W*2,H*2));panels=build(frame)
   for slot,p in enumerate(panels):canvas.paste(p,((slot%2)*W,(slot//2)*H))
   if frame==round(6*FPS):canvas.save(O/(name+'_preview.png'))
   ff.stdin.write(np.asarray(canvas).tobytes())
 finally:ff.stdin.close();rc=ff.wait()
 assert rc==0;print(str(output),flush=True)
def comparison(frame):
 i=min(399,50+round(frame/FPS/.02));panels=[]
 for j,direction in [(0,0),(4,180)]:
  for z,name in [(policy,'POLICY'),(locked,'FROZEN TARGET')]:
   panels.append(panel(z,j,i,15,f'3100 | {name} | push direction {direction}deg',f'DR3 x2.5 | 15N x0.3s | t={(i+1)*.02:.2f}s | 1x','Original poses; identical sampled DR parameters'))
 return panels
try:
 render('small_push_15N_policy_vs_hold',7.,comparison)
finally:renderer.close()
