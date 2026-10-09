import os
os.environ.setdefault('MUJOCO_GL','cgl')
from pathlib import Path
import json
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
EVAL=HERE/'new_prior/evaluation_model_63_holdout'
OUT=HERE.parents[1]/'exports/xduck_calibrated_sitstand_20260930_model_63'
q=np.load(EVAL/'cycle_qpos.npz')['qpos'][:,0]
m=mujoco.MjModel.from_xml_path(str(EVAL/'center/scene.xml'));d=mujoco.MjData(m)
m.mat_rgba[:,3]=1;m.geom_rgba[:,3]=1
for foot in ['left_foot_collision','right_foot_collision']:
 gid=m.geom(foot).id;m.geom_group[gid]=2;m.geom_rgba[gid]=[.15,.18,.2,1]
opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[[0,2]]=1;opt.sitegroup[:]=0
m.vis.global_.offwidth=640;m.vis.global_.offheight=400
renderer=mujoco.Renderer(m,width=640,height=400)
camera=mujoco.MjvCamera();camera.azimuth=90;camera.elevation=-6;camera.distance=1.25
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',20)
small=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',17)
writer=imageio.get_writer(str(OUT/'simulation_two_cycles.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16)
stills={}
for tick in range(0,len(q),2):
 d.qpos[:]=q[tick];mujoco.mj_forward(m,d)
 torso=d.sensor('task_torso_position').data
 camera.lookat[:]=[torso[0],0,.29]
 renderer.update_scene(d,camera=camera,scene_option=opt)
 frame=Image.new('RGB',(640,480),'white');frame.paste(Image.fromarray(renderer.render()),(0,45))
 draw=ImageDraw.Draw(frame)
 action='坐下' if (tick//300)%2==0 else '起立'
 draw.text((12,8),f'model_63｜第{tick//600+1}轮 {action}｜{(tick+1)*.02:.2f}s',font=font,fill='#223747')
 draw.text((12,453),'实际策略与物理仿真轨迹回放；未经真机验证',font=small,fill='#435b6d')
 writer.append_data(np.asarray(frame))
 if tick in [0,298,598]: stills[tick]=frame.copy()
writer.close();renderer.close()
collage=Image.new('RGB',(1920,480),'white')
for i,tick in enumerate([0,298,598]):collage.paste(stills[tick],(i*640,0))
collage.save(OUT/'stand_sit_rise_preview.png')
print(OUT/'simulation_two_cycles.mp4',flush=True)
