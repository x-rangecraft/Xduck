from pathlib import Path
import json,subprocess,numpy as np,mujoco
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;archive=np.load(R/'adapt50_validation_trace.npz');z={k:archive[k] for k in archive.files};archive.close();audit=json.loads((R/'FLEX50_CAPABILITY_AUDIT.json').read_text());rs=json.loads((R/'adapt50_validation_results.json').read_text())['records'];case=audit['representative_video_case'];physics=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'));m=mujoco.MjModel.from_xml_path(str(R/'physics/scene.xml'));assert m.nq==physics.nq and [m.joint(i).name for i in range(m.njnt)]==[physics.joint(i).name for i in range(physics.njnt)];geom_ids=np.array([m.geom(physics.geom(i).name).id for i in range(physics.ngeom)]);m.geom_group[[m.geom(side+'_foot_collision').id for side in ['left','right']]]=2;d=mujoco.MjData(m);m.vis.global_.offwidth=400;m.vis.global_.offheight=300;m.mat_rgba[:,3]=1;m.geom_rgba[:,3]=1;renderer=mujoco.Renderer(m,height=300,width=400);opt=mujoco.MjvOption();opt.geomgroup[3]=0;cam=mujoco.MjvCamera()
def panel(j,t,label):
 u=t/.02;i=min(int(u),len(z['qpos']));f=u-int(u)
 q0=z['initial'][j] if i==0 else z['qpos'][i-1,j];q1=z['qpos'][min(i,len(z['qpos'])-1),j];d.qpos[:]=(1-f)*q0+f*q1;d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);m.geom_pos[geom_ids]=z['geom_pos'][j];m.geom_quat[geom_ids]=z['geom_quat'][j];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
 w,x,y,zz=z['initial'][j,3:7];yaw=np.degrees(np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz)));cam.azimuth=yaw+65;cam.elevation=-9;cam.distance=1.45;cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.31;renderer.update_scene(d,camera=cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,400,42),fill=(15,25,36));dr.text((8,5),label,fill='white');dr.text((8,23),f't={t:.2f}s | 1x | true closed-loop flex physics',fill='white');return im
scenarios=['forward','backward','left','right','forward_left','forward_right']
def write(name,cols,rows,entries):
 print('RENDERING',name,flush=True)
 width=cols*400;height=rows*300;ff=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(R/name)],stdin=subprocess.PIPE)
 try:
  for frame in range(300):
   if frame%100==0:print('FRAME',name,frame,flush=True)
   im=Image.new('RGB',(width,height));t=frame/25
   for k,(j,label) in enumerate(entries):im.paste(panel(j,t,label),((k%cols)*400,(k//cols)*300))
   if frame==75:im.save(R/(Path(name).stem+'_preview.png'))
   ff.stdin.write(np.asarray(im).tobytes())
 finally:ff.stdin.close();assert ff.wait()==0
try:
 six=[]
 for s in scenarios:
  p=next(r for r in rs if r['model']=='adapt50' and r['case']==case and r['scenario']==s);six.append((p['trace_index'],f'FLEX51 | {s} | DR case {case}'))
 write('flex51_six_commands_1x.mp4',3,2,six)
 forward=[(r['trace_index'],f'FLEX51 forward | DR case {r["case"]}') for r in rs if r['model']=='adapt50' and r['scenario']=='forward'];write('flex51_forward_8dr_1x.mp4',4,2,forward)
finally:renderer.close()
print('VIDEOS_READY',case)
