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
  fore_aft_fraction=.5,lateral_fraction=.2,fore_aft_angle_halfwidth_deg=15.)

def sample(rng,yaw,spec,body_count):
 n=len(yaw);draw=rng.random(n);fore=draw<spec['fore_aft_fraction'];side=(draw>=spec['fore_aft_fraction'])&(draw<spec['fore_aft_fraction']+spec['lateral_fraction']);focus=fore|side
 angle=rng.uniform(-np.pi,np.pi,n)
 angle[fore]=yaw[fore]+rng.choice([0.,np.pi],size=int(fore.sum()))+rng.uniform(-np.pi/12,np.pi/12,int(fore.sum()))
 angle[side]=yaw[side]+rng.choice([-np.pi/2,np.pi/2],size=int(side.sum()))+rng.uniform(-np.pi/12,np.pi/12,int(side.sum()))
 mag=rng.uniform(*spec['broad_force_N'],n);duration=rng.uniform(*spec['broad_duration_s'],n)
 mag[focus]=rng.uniform(*spec['force_N'],int(focus.sum()));duration[focus]=rng.uniform(*spec['duration_s'],int(focus.sum()))
 body=rng.integers(body_count,size=n);body[focus]=0
 return np.c_[np.cos(angle),np.sin(angle)]*mag[:,None],duration,body,focus
