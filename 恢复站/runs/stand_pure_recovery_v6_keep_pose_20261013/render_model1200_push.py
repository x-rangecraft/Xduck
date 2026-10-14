from pathlib import Path
import json,subprocess,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
R=Path(__file__).resolve().parent;z=np.load(R/'block_08_course_dr_only_trace.npz');report=json.loads((R/'block_08_course_dr_only_results.json').read_text())
fixtures=[[26,30],[24,28]];labels=[['SURVIVED: WORLD +Y 30N','SURVIVED: WORLD -Y 30N'],['FELL: WORLD +X 30N','FELL: WORLD -X 30N']]
m=mujoco.MjModel.from_binary_path(str(R.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m);qi=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=m.key_ctrl[0]
m.vis.global_.offwidth=640;m.vis.global_.offheight=400;m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=400,width=640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera();cam.distance=1.50;cam.elevation=-13;cam.azimuth=15
name='model1200_dr2_push_survived_vs_fell_1x'
ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s','1280x800','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(R/(name+'.mp4'))],stdin=subprocess.PIPE)
try:
 for frame in range(200):
  i=min(round(frame/25/.02),399);canvas=Image.new('RGB',(1280,800))
  for row in range(2):
   for col,j in enumerate(fixtures[row]):
    q=z['qpos'][i,j];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];cam.lookat[2]=.40
    renderer.update_scene(d,camera=cam,scene_option=opt);p=Image.fromarray(renderer.render());draw=ImageDraw.Draw(p);draw.rectangle((0,0,640,113),fill=(18,24,32));result=report['rows'][j];f=z['force'][i,j].sum(0);mag=np.linalg.norm(f);err=q[qi]-ref
    draw.text((8,5),f'1200 TOTAL | DR 2x | {labels[row][col]} | CASE {j}',fill=(150,255,180) if row==0 else (255,150,120))
    draw.text((8,25),f't={(i+1)*.02:.2f}s | height {q[2]*100:.1f}cm | body tilt {z["tilt"][i,j]:.1f}deg | 1x',fill='white')
    draw.text((8,45),f'{"PUSH ON" if mag>0 else "NO PUSH"} | force {mag:.1f}N | world XY {f[0]:+.1f}/{f[1]:+.1f}N | each push 1.0s',fill=(255,210,125) if mag>0 else 'white')
    draw.text((8,65),f'Speed XY {np.linalg.norm(z["velocity"][i,j,:2]):.3f}m/s | head joint error {np.rad2deg(np.max(np.abs(err[5:9]))):.1f}deg',fill='white')
    draw.text((8,85),f'8s outcome: survived={not result["failed"]} | full final pose acceptance={result["nonfall_settled"]}',fill=(150,255,180) if row==0 else (255,150,120))
    draw.text((8,100),'ORIGINAL SAVED DR EVALUATION | SAME MODEL, DIFFERENT DIRECTIONS / DR SAMPLES',fill=(190,190,190));canvas.paste(p,(col*640,row*400))
  if frame==175:canvas.save(R/(name+'_preview.png'))
  ff.stdin.write(np.asarray(canvas).tobytes())
finally:ff.stdin.close();rc=ff.wait();renderer.close()
assert rc==0
(R/'video_model1200_push_manifest.json').write_text(json.dumps(dict(checkpoint=report['checkpoint'],total_lineage_updates=1200,DR_multiplier=2.,fixtures=fixtures,labels=labels,duration_s=8.,fps=25,playback_speed=1.,source='original saved DR evaluation trajectories; no resimulation',success_label_means_nonfall=True,full_final_pose_accepted=False,paired_DR_samples=False),indent=2)+'\n');print(R/(name+'.mp4'),flush=True)
