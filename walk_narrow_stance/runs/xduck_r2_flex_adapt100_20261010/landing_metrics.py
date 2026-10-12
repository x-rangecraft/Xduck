"""Count walking support transfers, including direct sampled single-support swaps."""
import numpy as np

def metrics(loaded,velocity,goal,onset,dt=.02,fall=False):
 norm=float(np.linalg.norm(goal[:2]));direction=goal[:2]/max(norm,1e-9)
 along=velocity[:,:2]@direction;air=np.zeros(2);previous=np.ones(2,bool);last_side=-1;alternations=[];touches=[];progress=0.;success=None
 for i in range(onset,len(loaded)):
  now=loaded[i]
  # The other foot may have unloaded at the same sampled transition.
  # Accept support at either endpoint, but reject a landing after sampled double flight.
  supported=previous[::-1]|now[::-1]
  touchdown=now&(~previous)&(air>=.08-1e-9)&supported
  if touchdown.sum()==1:
   side=int(touchdown.argmax());touches.append((i,side,float(air[side])))
   if last_side>=0 and side!=last_side:alternations.append(i*dt)
   last_side=side
  progress+=along[i]*dt
  if success is None and len(alternations)>=2 and progress>=.04:success=(i-onset+1)*dt
  air=np.where(now,0.,air+dt);previous=now.copy()
 final_begin=len(loaded)*dt-2.;last_count=sum(t>=final_begin for t in alternations)
 return {'startup_success_s':success,'walking_at_end':bool(norm>=.05 and not fall and last_count>=2 and along[-100:].mean()>.02) if norm>=.05 else None,'final_alternations':last_count,'alternation_times_s':alternations,'touch_events':touches,'last2s_along_speed':float(along[-100:].mean()),'last2s_double_flight_fraction':float((loaded[-100:].sum(1)==0).mean())}

def self_test():
 # Direct left/right support swaps have no double-support frames, but valid 0.3s air times.
 z=np.array([[i//15%2==0,i//15%2==1] for i in range(200)]);v=np.tile([.15,0,0],(200,1));g=np.array([.3,0,0])
 a=metrics(z,v,g,0);assert a['walking_at_end'] and a['final_alternations']>=5
 assert not metrics(np.ones((200,2),bool),v*0,g,0)['walking_at_end']
 assert not metrics(np.tile([True,False],(200,1)),v,g,0)['walking_at_end']
 hop=np.zeros((200,2),bool);hop[::15]=True;assert not metrics(hop,v,g,0)['walking_at_end']
 # Contact flicker every 20ms cannot pass the 80ms swing duration.
 flicker=np.array([[i%2==0,i%2==1] for i in range(200)]);assert not metrics(flicker,v,g,0)['walking_at_end']
 assert not metrics(z,v,g,0,fall=True)['walking_at_end']
 return True
if __name__=='__main__':print(self_test())
