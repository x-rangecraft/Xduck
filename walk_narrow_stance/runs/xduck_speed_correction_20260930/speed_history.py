"""Reward-only velocity history. No policy observation or action changes."""
import numpy as np

class SpeedHistory:
    def __init__(self, n):
        self.raw = np.zeros((25,n,2)); self.filtered = np.zeros((50,n,2))
        self.count = np.zeros(n,dtype=int); self.fcount = np.zeros(n,dtype=int)
        self.command = np.full((n,3),np.nan); self.push = np.full(n,-1,dtype=np.int64)
        self.push_age = np.zeros(n,dtype=int); self.i = 0; self.j = 0
    def reset(self, ids):
        self.raw[:,ids]=0; self.filtered[:,ids]=0
        self.count[ids]=0; self.fcount[ids]=0
        self.command[ids]=np.nan; self.push[ids]=-1; self.push_age[ids]=0
    def update(self, velocity, command, next_push):
        changed=np.any(np.abs(command-self.command)>1e-7,axis=1)|np.isnan(self.command).any(1)
        self.reset(np.flatnonzero(changed)); self.command[:]=command
        pushed=(self.push!=next_push)&(self.push>=0)
        self.push_age+=1; self.push_age[pushed]=0; self.push[:]=next_push
        self.raw[self.i]=velocity[:,:2]; self.i=(self.i+1)%25
        self.count=np.minimum(self.count+1,25)
        mean=self.raw.sum(0)/self.count[:,None]
        ready=self.count==25
        self.filtered[self.j]=np.where(ready[:,None],mean,0); self.j=(self.j+1)%50
        self.fcount=np.minimum(self.fcount+ready,50)
        direction=command[:,:2]/np.maximum(np.linalg.norm(command[:,:2],axis=1)[:,None],1e-9)
        along=np.einsum('tni,ni->tn',self.filtered,direction)
        variance=along.var(0)
        return mean,ready,variance,(self.fcount==50)&(self.push_age>=75)&(np.linalg.norm(command[:,:2],axis=1)>.01)
