import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,subprocess,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent;O=R/'videos3000';O.mkdir(exist_ok=True)
before=np.load(R/'baseline2900_course_trace.npz');course=np.load(R/'head3000_course_trace.npz');walk=np.load(R/'head3000_walk_stop_trace.npz')
cr=json.loads((R/'head3000_course_results.json').read_text());wr=json.loads((R/'head3000_walk_stop_results.json').read_text())
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
def quiet(frame):
 i=min(399,round(frame/FPS/.02));result=[]
 for label,z in [('2900',before),('3000',course)]:
  for azimuth,view in [(15,'front 3/4'),(90,'side')]:
   result.append(panel(z,2,i,azimuth,f'{label} | quiet standing | {view}',f'DR3 x2.5 | same case2 | t={(i+1)*.02:.2f}s | 1x','Original saved pose sequence; no resimulation'))
 return result
def recovery(frame):
 i=min(399,round(frame/FPS/.02));held=frame/FPS>=8.;tail='END: final frame held' if held else 'Original saved pose sequence | 1x'
 force=float(np.linalg.norm(course['force'][i,24].sum(0)))
 result=[panel(course,13,i,90,'3000 | tilt recovery | SUCCESS',f'Initial pitch -50deg, Vx -0.3m/s | t={(i+1)*.02:.2f}s',tail),
  panel(course,24,i,15,'3000 | push recovery | SUCCESS',f'40N x 1.5s | current force {force:.1f}N | t={(i+1)*.02:.2f}s',tail)]
 j=7;start=int(walk['starts'][j]);end=int(walk['ends'][j]);wi=min(end-1,start-50+round(frame/FPS/.02));age=(wi-start)*.02
 result.append(panel(walk,j,wi,15,'3000 | walking -> standing | SUCCESS',f'{"WALKING" if wi<start else "STANDING"} | switch age {age:+.2f}s | command 0.3m/s','Continuous physics; no reset at handoff'))
 result.append(panel(course,27,i,15,'3000 | remaining push failure',f'40N x 1.5s | direction +135deg | t={(i+1)*.02:.2f}s',tail))
 return result
assert cr['rows'][13]['final_stand_success'] and cr['rows'][24]['final_stand_success'] and not cr['rows'][27]['final_stand_success'] and wr['rows'][7]['final_stand_success']
try:
 render('standing_2900_vs3000',8.,quiet);render('recovery3000_selected',9.,recovery)
finally:renderer.close()
manifest=dict(model=3000,flexible_joints=False,DR_multiplier=2.5,fps=25,playback_speed=1.,source='original saved evaluation qpos sequences, not policy resimulation',standing_comparison=dict(case=2,top='2900',bottom='3000',left='front 3/4',right='side'),recovery_panels=['top-left: pitch-50deg recovery,case13','top-right:40N1.5s successful push,case24','bottom-left:0.3m/s walking to standing,case7','bottom-right:remaining failed push +135deg,case27'],course_panels_last_second_labeled_end_frame_hold=True,walking_handoff_preserves_full_history=True)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
