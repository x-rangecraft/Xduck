from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;z=np.load(R/'stand_validation_trace.npz');m=mujoco.MjModel.from_binary_path(str(R.parent/'xduck_toeout_audit_20261009/verified_rigid_model.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=480;m.vis.global_.offheight=480;m.mat_rgba[:,3]=1;m.geom_rgba[:,3]=1
renderer=mujoco.Renderer(m,height=480,width=480);cam=mujoco.MjvCamera();opt=mujoco.MjvOption();opt.geomgroup[3]=0
canvas=Image.new('RGB',(1440,480))
try:
 for col,j,az,label in [(0,0,0,'BEFORE | asymmetric fixed target'),(1,34,0,'SYMMETRIC DEFAULT | front'),(2,34,90,'SYMMETRIC DEFAULT | side')]:
  d.qpos[:]=z['qpos'][-1,j];m.geom_pos[:]=z['geom_pos'][j];m.geom_quat[:]=z['geom_quat'][j];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.31;cam.azimuth=az;cam.elevation=-8;cam.distance=1.1;renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,480,28),fill=(15,25,36));dr.text((8,8),label,fill='white');canvas.paste(im,(col*480,0))
 canvas.save(R/'symmetric_default_pose.png')
finally:renderer.close()
print(R/'symmetric_default_pose.png')
