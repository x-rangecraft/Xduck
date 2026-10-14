from pathlib import Path
import sys,json,numpy as np,mujoco
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);names=[m.joint(int(x)).name for x in m.actuator_trnid[:,0]];ref=m.key_qpos[0,7:];tol=np.array(c.env.posture_tolerance_rad);results=[]
for tag in sys.argv[1:]:
 z=np.load(R/(tag+'_trace.npz'));report=json.loads((R/(tag+'_results.json')).read_text());rows=[]
 for j in range(z['qpos'].shape[1]):
  end=int(z['ends'][j]) if 'ends' in z.files else len(z['qpos']);tail=slice(end-50,end)
  q=z['qpos'][tail,j];err=q[:,7:]-ref;head_angle=np.degrees(np.arccos(np.clip(z['head_up'][tail,j],-1,1)))
  band=(np.abs(err)<=tol).all(1)&(head_angle<=c.env.posture_head_world_tolerance_deg)&z['pose_valid_relaxed'][tail,j]&z['contact'][tail,j].any(1)
  rows.append(dict(case=j,name=report['rows'][j].get('name','walk_stop'),head_world_tilt_mean_deg=float(head_angle.mean()),head_world_tilt_p95_deg=float(np.percentile(head_angle,95)),joint_abs_error_mean_deg=dict(zip(names,np.degrees(np.abs(err).mean(0)).tolist())),pose_within_band_fraction=float(band.mean()),recovery_success=bool(report['rows'][j].get('final_stand_success',False))))
 quiet=[x for x in rows if x['name']=='quiet'];out=dict(tag=tag,window_seconds=1.,posture_separate_from_recovery_score=True,quiet_head_tilt_mean_deg=float(np.mean([x['head_world_tilt_mean_deg'] for x in quiet])) if quiet else None,quiet_cases=quiet,rows=rows);results.append(out)
(R/'POSTURE_COMPARISON.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in results]),flush=True)
