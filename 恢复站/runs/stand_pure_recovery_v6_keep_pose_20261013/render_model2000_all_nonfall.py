from pathlib import Path
import json,subprocess,shutil,math
import numpy as np,mujoco
from PIL import Image,ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent;O=R/'videos_model2000_nonfall';O.mkdir(exist_ok=True)
c=np.load(R/'block_16_course_dr_only_trace.npz');cr=json.loads((R/'block_16_course_dr_only_results.json').read_text());w=np.load(R/'block_16_walk_stop_trace.npz');wr=json.loads((R/'block_16_walk_stop_results.json').read_text())
quiet=[x['case'] for x in cr['rows'] if x['name']=='quiet' and not x['failed']];tilt=[x['case'] for x in cr['rows'] if x['name']=='course_reset' and not x['failed']];push=[x['case'] for x in cr['rows'] if x['name']=='course_push' and not x['failed']];valid_walk={x['case'] for x in wr['rows'] if x['eligible'] and not x['stop_failed']}
assert (len(quiet),len(tilt),len(push),len(valid_walk))==(4,12,4,77)
jobs=[('quiet_all',c,cr,[quiet],8.),('tilt_all',c,cr,[tilt[k:k+4] for k in range(0,len(tilt),4)],8.),('push_all',c,cr,[push],8.),('walk_stop_all',w,wr,[list(range(k,k+4)) for k in range(0,80,4)],9.)]
W,H=560,400;m=mujoco.MjModel.from_binary_path(str(R.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=W;m.vis.global_.offheight=H;m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=H,width=W);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-13
manifest=dict(model_total_updates=2000,DR_multiplier=2.5,all_nonfall_case_count=97,full_pose_success=False,source='original saved DR evaluation trajectories; no resimulation',playback_speed=1.,fps=25,videos=[],complete=False);seen=[]
def progress(name,chapter,total):
 (O/'progress.json').write_text(json.dumps(dict(video=name,completed_chapters=chapter,total_chapters=total),indent=2)+'\n')
try:
 for name,data,report,chapters,seconds in jobs:
  ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s',f'{W*2}x{H*2}','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(O/(name+'.mp4'))],stdin=subprocess.PIPE)
  chapter_meta=[]
  try:
   for chapter,ids in enumerate(chapters):
    good=[j for j in ids if name!='walk_stop_all' or j in valid_walk];seen.extend([(name,j) for j in good]);chapter_meta.append(dict(chapter=chapter+1,start_s=chapter*seconds,cases=good))
    for frame in range(round(seconds*25)):
     canvas=Image.new('RGB',(W*2,H*2),(18,24,32))
     for slot,j in enumerate(ids):
      row,col=divmod(slot,2)
      if name=='walk_stop_all' and j not in valid_walk:
       p=Image.new('RGB',(W,H),(18,24,32));draw=ImageDraw.Draw(p);draw.text((20,80),f'CASE {j}: WALKING PREFIX FAILED',fill=(180,180,180));draw.text((20,105),'Excluded from the 77 valid successful handoff clips',fill=(180,180,180));canvas.paste(p,(col*W,row*H));continue
      offset=int(data['starts'][j])-50 if name=='walk_stop_all' else 0;i=offset+round(frame/25/.02);q=data['qpos'][i,j];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.azimuth=90 if name=='tilt_all' else 15;cam.lookat[:]=q[:3];cam.lookat[2]=.39;renderer.update_scene(d,camera=cam,scene_option=opt);p=Image.fromarray(renderer.render());draw=ImageDraw.Draw(p);draw.rectangle((0,0,W,112),fill=(18,24,32));result=report['rows'][j]
      draw.text((7,4),f'2000 TOTAL | DR 2.5x | CASE {j} | GROUP {chapter+1}/{len(chapters)} | 1x',fill='white')
      if name=='tilt_all':
       params=report['tilt_case_parameters'][j-4];draw.text((7,23),f'INITIAL {params["axis"].upper()} {params["angle_deg"]:+.0f}deg | Vxy {params["linear_velocity"][:2]}',fill=(255,210,125))
      elif name=='push_all':
       f=data['force'][i,j].sum(0);draw.text((7,23),f'{"PUSH ON" if np.linalg.norm(f)>0 else "NO PUSH"} | {np.linalg.norm(f):.1f}N | XY {f[0]:+.1f}/{f[1]:+.1f}N | pulse 1.5s',fill=(255,210,125))
      elif name=='walk_stop_all':
       phase='STANDING' if i>=data['starts'][j] else ('WALKING' if data['walking'][i,j] else 'PREP');draw.text((7,23),f'{phase} | COMMAND {[round(x,3) for x in result["command"]]} | walked {result["walk_seconds"]:.1f}s',fill=(255,210,125))
      else:draw.text((7,23),'QUIET START | NO EXTERNAL PUSH',fill=(255,210,125))
      clock=(i-int(data['starts'][j]))*.02 if name=='walk_stop_all' else (i+1)*.02
      draw.text((7,43),f'{"Switch age" if name=="walk_stop_all" else "Time"} {clock:+.2f}s | tilt {data["tilt"][i,j]:.1f}deg | height {q[2]*100:.1f}cm',fill='white')
      draw.text((7,63),f'Speed XY {np.linalg.norm(data["velocity"][i,j,:2]):.3f}m/s | body angular speed {np.linalg.norm(data["gyro"][i,j]):.3f}rad/s',fill='white')
      draw.text((7,83),'OUTCOME: NO FALL | FINAL POSE ACCEPTANCE: NOT PASSED',fill=(150,255,180));draw.text((7,99),'ORIGINAL SAVED TRAJECTORY | CAMERA FOLLOWS BODY',fill=(180,180,180));canvas.paste(p,(col*W,row*H))
     if chapter==0 and frame==30:canvas.save(O/(name+'_preview.png'))
     ff.stdin.write(np.asarray(canvas).tobytes())
    progress(name,chapter+1,len(chapters));print(name,chapter+1,'/',len(chapters),flush=True)
  finally:ff.stdin.close();rc=ff.wait()
  assert rc==0;manifest['videos'].append(dict(file=name+'.mp4',duration_s=seconds*len(chapters),case_count=sum(len(x['cases']) for x in chapter_meta),chapters=chapter_meta));(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
finally:renderer.close()
assert len(seen)==97 and len(set(seen))==97
manifest['complete']=True;manifest['all_cases_shown_once']=True;(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('ALL 97 NONFALL CASES COMPLETE',flush=True)
