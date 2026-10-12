"""Render paired recorded physics; same trial and camera for both commands."""
import argparse
import json
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = [np.load(a.baseline), np.load(a.candidate)]
model = mujoco.MjModel.from_binary_path(str(a.model))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)
option = mujoco.MjvOption()
option.geomgroup[3] = 1
views = {}
selection = []
for case, goal in [('descent', .2182), ('rise', .3129), ('sit_hold', .2182), ('stand_hold', .3129)]:
    key = f'{case}_6401_'
    error = sum(np.abs(r[key+'height'][-50:].mean(axis=0)-goal) for r in records)
    idx = int(np.argsort(error)[len(error)//2])
    initial = records[0][key+'qpos'][0,idx]
    w,x,y,z = initial[3:7]
    yaw = np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [initial[0],initial[1],.28]
    camera.distance = 1.4
    camera.elevation = -16
    camera.azimuth = float(yaw+115)
    views[case] = (key,idx,camera)
    selection.append(dict(case=case,seed=6401,trial=idx,selection='median combined endpoint height error, same index in both paired runs'))
a.output.parent.mkdir(parents=True, exist_ok=True)
with imageio.get_writer(a.output, fps=25, codec='libx264', quality=8) as writer:
    for cases in [('descent','rise'), ('sit_hold','stand_hold')]:
        for step in range(0,300,2):
            canvas = Image.new('RGB',(1280,960))
            for row,case in enumerate(cases):
                key,idx,camera = views[case]
                for column,record in enumerate(records):
                    data.qpos[:] = record[key+'qpos'][step,idx]
                    data.qvel[:] = 0
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,640,56),fill=(20,20,20))
                    command = 'ZERO head command' if column==0 else 'neck +20 / head pitch +20 deg'
                    draw.text((10,5),f'2000 / DR2x / {command}',fill='white')
                    draw.text((10,22),f'{case} / paired trial {idx} / t={step*.02:.2f}s',fill='white')
                    tilt = np.degrees(record[key+'head_tilt'][step,idx])
                    height = record[key+'height'][step,idx]
                    draw.text((10,39),f'head tilt={tilt:.1f}deg / height={height:.3f}m',fill='white')
                    canvas.paste(frame,(column*640,row*480))
            writer.append_data(np.asarray(canvas))
            if cases[0]=='descent' and step==0:
                canvas.save(a.output.with_suffix('.png'))
renderer.close()
a.output.with_suffix('.json').write_text(json.dumps(selection,indent=2))
print(a.output,flush=True)
