from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parent
z=np.load(R.parent/'xduck_clean_natural_gait_20261010/startup_connection_review_trace.npz');j=1
h=z['clearance'][:,j,0];air=False;peak=0;events=[]
for t in range(len(h)):
 if not air and h[t]>.0015:air=True
 if air:
  peak=max(peak,h[t])
  if h[t]<.0007:
   if t>150 and peak>.022:events.append(t)
   air=False;peak=0
cycles=[];poses=[];rolls=[]
q=z['qpos'][:,j];w,x,y,zz=q[:,3:7].T;roll=np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y))
for lo,hi in zip(events,events[1:]):
 if not 20<=hi-lo<=32:continue
 old=np.arange(lo,hi+1);sample=lo+np.arange(25)/25*(hi-lo)
 cycles.append(np.array([np.interp(sample,old,z['target'][old,j,k]) for k in range(14)]).T)
 poses.append(np.array([np.interp(sample,old,q[old,7+k]) for k in range(14)]).T)
 rolls.append(np.interp(sample,old,roll[old]))
assert len(cycles)>=4
mean=np.mean(cycles,axis=0);actual=np.mean(poses,axis=0);velocity=(np.roll(actual,-1,axis=0)-np.roll(actual,1,axis=0))/.04
np.savez_compressed(R/'reverse_cycle.npz',target=mean,actual=actual,forward_joint_velocity=velocity,roll=np.mean(rolls,axis=0))
(R/'reverse_cycle_manifest.json').write_text(json.dumps(dict(source='accepted forward gait trace; actual motor targets, not video reversal',cycles=len(cycles),nominal_cycle_seconds=.5,purpose='physical backward teacher candidate; free root and actual contact required'),indent=2));print('REVERSE_CYCLE_READY',len(cycles))
