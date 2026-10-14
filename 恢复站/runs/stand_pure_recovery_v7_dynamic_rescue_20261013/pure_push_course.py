"""Balanced fore/aft sustained pushes; fixed-world force during each push."""
import numpy as np

STAGES={
 2:[(18.,24.,.30,.60),(20.,26.,.40,.75),(22.,28.,.50,.90),(24.,30.,.60,1.00)],
 3:[(24.,30.,.60,1.00),(28.,34.,.70,1.15),(30.,37.,.80,1.30),(32.,40.,1.00,1.50)],
}

def specification(level,stage,force_ranges,durations):
 assert 0<=level<=3 and 0<=stage<=3
 if level in STAGES:lo,hi,tlo,thi=STAGES[level][stage]
 else:lo,hi=force_ranges[level];tlo,thi=durations[level]
 return dict(stage=stage,force_N=[lo,hi],duration_s=[tlo,thi],
  broad_force_N=[force_ranges[level][0],hi],broad_duration_s=[durations[level][0],thi],
  focused_direction_fraction=.85,direction_angle_halfwidth_deg=10.,balanced_eight_directions=True)

def sample(rng,yaw,spec,body_count):
 n=len(yaw);focus=rng.random(n)<spec['focused_direction_fraction']
 angle=rng.uniform(-np.pi,np.pi,n)
 sectors=rng.integers(8,size=int(focus.sum()))
 angle[focus]=yaw[focus]+sectors*(np.pi/4)+rng.uniform(-np.pi/18,np.pi/18,int(focus.sum()))
 mag=rng.uniform(*spec['broad_force_N'],n);duration=rng.uniform(*spec['broad_duration_s'],n)
 mag[focus]=rng.uniform(*spec['force_N'],int(focus.sum()));duration[focus]=rng.uniform(*spec['duration_s'],int(focus.sum()))
 body=rng.integers(body_count,size=n);body[focus]=0
 return np.c_[np.cos(angle),np.sin(angle)]*mag[:,None],duration,body,focus
