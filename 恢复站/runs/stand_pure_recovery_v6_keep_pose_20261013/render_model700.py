from pathlib import Path
import json,subprocess,shutil
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np,mujoco
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
fixtures=[[0,8,24],[0,28,31]];names=[['QUIET START','PITCH 10deg + INITIAL VELOCITY','PUSH +X 16 N'],['QUIET START + DR','DR PUSH FAILURE / CASE 28','DR PUSH FAILURE / CASE 31']]
datasets=[('700 TOTAL / NO DR',np.load(R/'block_03_trace.npz'),json.loads((R/'block_03_results.json').read_text())),('700 TOTAL / DR 1.5x',np.load(R/'block_03_course_trace.npz'),json.loads((R/'block_03_course_results.json').read_text()))]
m=mujoco.MjModel.from_binary_path(str(R.parent/'stand_pure_recovery_v4_20261012/physics/effective_training.mjb'));d=mujoco.MjData(m);qi=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=m.key_ctrl[0]
m.vis.global_.offwidth=640;m.vis.global_.offheight=400;m.geom_rgba[:,3]=1;m.mat_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=400,width=640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera();cam.distance=1.50;cam.elevation=-13;cam.azimuth=15
ff=subprocess.Popen([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-s','1920x800','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(R/'model700_nominal_dr15_review_1x.mp4')],stdin=subprocess.PIPE)
try:
 for frame in range(200):
  i=min(round(frame/25/.02),399);canvas=Image.new('RGB',(1920,800))
  for row,(updates,z,report) in enumerate(datasets):
   for col,j in enumerate(fixtures[row]):
    q=z['qpos'][i,j];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];cam.lookat[2]=.40
    renderer.update_scene(d,camera=cam,scene_option=opt);p=Image.fromarray(renderer.render());draw=ImageDraw.Draw(p);draw.rectangle((0,0,640,113),fill=(18,24,32))
    error=q[qi]-ref;head=float(np.rad2deg(np.max(np.abs(error[5:9]))));yaw=float(np.rad2deg(np.max(np.abs(error[[0,9]]))));roll=float(np.rad2deg(np.max(np.abs(error[[1,10]]))));speed=float(np.linalg.norm(z['velocity'][i,j,:2]));force=z['force'][i,j].sum(0);result=report['rows'][j]
    draw.text((8,5),f'{updates} UPDATES | {names[row][col]} | 1x',fill='white')
    draw.text((8,24),f't={(i+1)*.02:.2f}s | height {q[2]*100:.1f}cm | body tilt {z["tilt"][i,j]:.1f}deg',fill=(255,210,125))
    draw.text((8,43),f'Head max joint error {head:.1f}deg | hip yaw error {yaw:.1f}deg | roll error {roll:.1f}deg',fill='white')
    draw.text((8,62),f'Speed {speed:.3f}m/s | force {np.linalg.norm(force):.1f}N | world XY {force[0]:+.1f}/{force[1]:+.1f}N',fill='white')
    draw.text((8,81),f'8s: fell={result["failed"]} | final correct pose + quiet 1s={result["nonfall_settled"]}',fill=(150,255,180) if result['nonfall_settled'] else (255,150,120))
    draw.text((8,98),'MEAN POLICY | SAVED EVALUATION TRAJECTORY | CAMERA FOLLOWS BODY',fill=(190,190,190));canvas.paste(p,(col*640,row*400))
  if frame==175:canvas.save(R/'model700_nominal_dr15_preview.png')
  ff.stdin.write(np.asarray(canvas).tobytes())
finally:ff.stdin.close();rc=ff.wait();renderer.close()
assert rc==0
force_identical=bool(np.allclose(datasets[0][1]['force'][:,:32],datasets[1][1]['force'][:,:32]))
(R/'video_model700_manifest.json').write_text(json.dumps(dict(total_lineage_updates=700,additional_updates=400,rows=["no DR","DR1.5"],paired_scenarios=False,fixtures=fixtures,columns=names,source='original saved independent evaluation trajectories; no resimulation',playback_speed=1.,forces_identical_between_checkpoint_tests=force_identical,sampling_noise=False,domain_randomization_disabled=[True,False],full_handoff_qualification=False,checkpoints=[x[2]['checkpoint'] for x in datasets]),indent=2)+'\n')
print(R/'model700_nominal_dr15_review_1x.mp4')
