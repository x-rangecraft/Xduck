"""Render recorded checkpoint physics, selecting median height-error trials."""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser()
p.add_argument("--trajectory", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--label", required=True)
p.add_argument("--robot", choices=("microduck","xduck"), required=True)
p.add_argument("--motions-only",action="store_true")
p.add_argument("--paired-trial",action="store_true")
p.add_argument("--selection-trajectory",type=Path)
p.add_argument("--stand-height",type=float,default=.3129)
p.add_argument("--sit-height",type=float,default=.2182)
p.add_argument("--view",choices=("oblique","front"),default="oblique")
a=p.parse_args()
r=np.load(a.trajectory)
selection_record=np.load(a.selection_trajectory) if a.selection_trajectory else r
m=mujoco.MjModel.from_binary_path(str(a.model))
d=mujoco.MjData(m)
renderer=mujoco.Renderer(m,height=480,width=640)
option=mujoco.MjvOption()
if a.robot=="xduck":
    option.geomgroup[3]=1  # The frozen physics MJB keeps robot meshes in group 3.
if a.robot=="microduck":
    specs=[("sit_from_stand",11,.060),("rise_from_sit",11,.115)]
else:
    specs=[("descent",6401,a.sit_height),("rise",6401,a.stand_height),("sit_hold",6401,a.sit_height),("stand_hold",6401,a.stand_height)]
    if a.motions_only:specs=specs[:2]
views=[]
paired_idx=None
if a.paired_trial:
    error=sum(np.abs(r[f"{case}_{seed}_height"][-50:].mean(axis=0)-goal) for case,seed,goal in specs)
    paired_idx=int(np.argsort(error)[len(error)//2])
for case,seed,goal in specs:
    poses=r[f"{case}_{seed}_qpos"]
    heights=r[f"{case}_{seed}_height"]
    err=np.abs(selection_record[f"{case}_{seed}_height"][-50:].mean(axis=0)-goal)
    idx=paired_idx if paired_idx is not None else int(np.argsort(err)[len(err)//2])
    initial=poses[0,idx]
    w,x,y,z=initial[3:7]
    yaw=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    cam=mujoco.MjvCamera()
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[initial[0],initial[1],.14 if a.robot=="microduck" else .28]
    cam.distance=.65 if a.robot=="microduck" else 1.4
    cam.elevation=-16
    cam.azimuth=float(yaw+115)
    if a.view=="front":
        cam.azimuth=float(yaw)
        cam.elevation=-8
        cam.distance=.65 if a.robot=="microduck" else 1.1
    views.append((case,poses,heights,goal,idx,cam))
a.output.parent.mkdir(parents=True,exist_ok=True)
with imageio.get_writer(a.output,fps=25,codec="libx264",quality=8) as writer:
    for step in range(0,min(len(v[1]) for v in views),2):
        canvas=Image.new("RGB",(1280,480*((len(views)+1)//2)))
        for n,(case,poses,heights,goal,idx,cam) in enumerate(views):
            d.qpos[:]=poses[step,idx]
            d.qvel[:]=0
            mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=option)
            frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,44),fill=(20,20,20))
            selection='paired median height error' if paired_idx is not None else ('same reference trial' if a.selection_trajectory else 'median height error')
            draw.text((10,6),f"{a.label} / {case} / trial {idx} ({selection})",fill="white")
            draw.text((10,25),f"t={step*.02:.2f}s height={heights[step,idx]:.3f}m target={goal:.3f}m",fill="white")
            canvas.paste(frame,((n%2)*640,(n//2)*480))
        writer.append_data(np.asarray(canvas))
        if step==0: canvas.save(a.output.with_suffix(".png"))
renderer.close()
print(a.output,flush=True)
