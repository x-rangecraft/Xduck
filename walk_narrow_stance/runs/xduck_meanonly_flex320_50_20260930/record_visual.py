from pathlib import Path
import argparse,json,subprocess
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--command',type=float,required=True);args=p.parse_args()
z=np.load(R/'after_steady_trace.npz');report=json.loads((R/'after_steady.json').read_text());names=z['command_names'].tolist()
index=min([i for i,n in enumerate(names) if n.startswith('forward') and '_' not in n],key=lambda k:abs(z['commands'][k*4,0]-args.command))
OUT=R/'videos';OUT.mkdir(exist_ok=True)
training_model=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));m=mujoco.MjModel.from_xml_path(str(R/'physics/scene.xml'));assert m.nq==training_model.nq;ids=np.array([m.geom(training_model.geom(i).name).id for i in range(training_model.ngeom)]);assert [m.joint(i).name for i in range(m.njnt)]==[training_model.joint(i).name for i in range(training_model.njnt)];d=mujoco.MjData(m);m.mat_rgba[:,3]=1.;m.geom_rgba[:,3]=1.
for side in ['left','right']:m.geom_group[m.geom(side+'_foot_collision').id]=2
renderer=mujoco.Renderer(m,height=480,width=640);camera=mujoco.MjvCamera();camera.distance=1.25;camera.elevation=-18;camera.azimuth=120;option=mujoco.MjvOption();option.geomgroup[3]=0

def frame(env,i,label):
 row=report['directions'][env//4]['cases'][env%4];fail=row['failure_s'];frozen=fail is not None and (i+1)*.02>=fail
 step=min(i,max(0,round(fail/.02)-1)) if frozen else i
 m.geom_pos[ids]=z['geom_pos'][env];m.geom_quat[ids]=z['geom_quat'][env];d.qpos[:]=z['qpos'][step,env];mujoco.mj_forward(m,d)
 camera.lookat[:]=d.qpos[:3];camera.lookat[2]=.22;renderer.update_scene(d,camera=camera,scene_option=option);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
 draw.rectangle((0,0,640,50),fill=(15,25,36));cmd=z['commands'][env];v=z['measured'][step,env]
 draw.text((8,6),f'1280 updates | flex320 | {label} | DR2.5, no external push | 1x',fill='white')
 txt=f'FAILED at {fail:.2f}s - FROZEN' if frozen else f'cmd {cmd[0]:+.3f},{cmd[1]:+.2f},{cmd[2]:+.2f} | measured {v[0]:+.3f},{v[1]:+.2f},{v[2]:+.2f}'
 draw.text((8,28),txt,fill='red' if frozen else 'white');return im

def write_video(path,envs,cols):
 rows=(len(envs)+cols-1)//cols;width=640*cols;height=480*rows
 ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
 try:
  for i in range(0,len(z['qpos']),2):
   canvas=Image.new('RGB',(width,height))
   for k,(env,label) in enumerate(envs):canvas.paste(frame(env,i,label),((k%cols)*640,(k//cols)*480))
   ff.stdin.write(np.asarray(canvas).tobytes())
 finally:ff.stdin.close();rc=ff.wait()
 assert rc==0
try:
 write_video(OUT/'flex50_cmd030_visual_grid.mp4',[(index*4+j,f'{names[index]} case{j+1}') for j in range(4)],2)
 chosen=[names[index],'backward','left','right','turn_left','walk_turn_left']

finally:renderer.close()
(R/'video_selection.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,selected_command=z['commands'][index*4].tolist(),source='recorded steady direction evaluation, no re-simulation',target_speed_grid='videos/flex50_cmd030_visual_grid.mp4',direction_grid='videos/flex50_directions_grid.mp4',all_cases_reported=True),indent=2))
