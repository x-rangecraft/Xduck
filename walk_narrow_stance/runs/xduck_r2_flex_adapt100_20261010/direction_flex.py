"""Command-aligned locomotion with accepted gait and speed across DR."""
from dataclasses import dataclass
import numpy as np
from unilab.base import registry
from .new_stand_smooth import NewStandSmoothCfg,NewStandSmoothEnv,NewStandBankReset
from .startup_flex import StartupFlexEnv
import mujoco
from .startup_flex import rotate

COMMANDS=np.array([[.3,0,0],[-.3,0,0],[0,0,.4],[0,0,-.4],[.3,0,.4],[.3,0,-.4]])

def motion_metrics(command,velocity,yaw_rate,gain=.65):
    norm=np.linalg.norm(command[:,:2],axis=1)
    translating=norm>.05
    axis=command[:,:2]/np.maximum(norm[:,None],1e-6)
    signed_speed=np.sum(velocity[:,:2]*axis,axis=1)
    signed_turn=np.sign(command[:,2])*yaw_rate
    return translating,axis,signed_speed,signed_turn,gain*command[:,:2]

@registry.envcfg('XDuckDirectionFlex')
@dataclass
class DirectionFlexCfg(NewStandSmoothCfg):
    direction_speed_gain: float=.65
    direction_seed: int=1490091
    direction_switch_fraction: float=.6
    smooth_width_target_m: float=.212
    smooth_width_weight: float=2.
    smooth_height_weight: float=0.
    smooth_head_target: tuple[float,float]=(.018978888,.059834011)
    direction_stall_seconds: float=3.

@registry.env('XDuckDirectionFlex',sim_backend='mjwarp')
class DirectionFlexEnv(StartupFlexEnv):
    def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
        self._dir_rng=np.random.default_rng(cfg.direction_seed)
        self._dir_commands=np.zeros((num_envs,13))
        self._dir_old_commands=np.zeros((num_envs,3))
        self._dir_kind=np.zeros(num_envs,int)
        self._dir_age=np.zeros(num_envs,int)
        self._dir_next_switch=np.full(num_envs,np.inf)
        self._dir_mean_v=np.zeros((num_envs,2));self._dir_mean_w=np.zeros(num_envs)
        self._dir_land_time=np.full((num_envs,2),-1.)
        self._dir_land_position=np.zeros((num_envs,2,3))
        self._dir_land_axis=np.zeros((num_envs,2,3))
        self._dir_last_full=np.full(num_envs,-1.)
        self._dir_stall_at=np.full(num_envs,3.)
        self._dir_heading0=np.zeros(num_envs)
        if cfg.newstand_bank_path is None:
            raise ValueError('A physically collected new-standing bank is required')
        with np.load(cfg.newstand_bank_path) as z:
            self._newstand_bank = {k:z[k].copy() for k in z.files}
        self._newstand_rng = np.random.default_rng(cfg.newstand_seed)
        self._smooth_age = np.zeros(num_envs,int)
        self._smooth_z0 = np.zeros(num_envs)
        self._smooth_heading0 = np.zeros(num_envs)
        self._smooth_width0 = np.zeros(num_envs)
        self._smooth_previous_v = np.zeros(num_envs)
        self._smooth_air = np.zeros((num_envs,2),bool)
        self._smooth_peak = np.zeros((num_envs,2))
        self._smooth_liftoff = np.zeros((num_envs,2,3))
        self._smooth_axis = np.zeros((num_envs,2,3))
        self._smooth_landings = np.zeros(num_envs,int)
        # Model and site metadata are read only on initialization.
        m = mujoco.MjModel.from_binary_path(cfg.warp_compiled_model_file)
        self._smooth_site_local = np.array([m.site(s+'_foot').pos.copy() for s in ['left','right']])
        names = [m.body(int(m.site(s+'_foot').bodyid[0])).name for s in ['left','right']]
        super().__init__(cfg,num_envs=num_envs,backend_type=backend_type)
        self._smooth_body_ids = np.array([self._backend.get_body_id(n) for n in names],np.int32)
        assert len(self._passive_qpos_indices)==48 and cfg.flexible_dr_enabled and cfg.physics_microsteps==16

    def _make_dr_provider(self):return DirectionFlexReset()

    def _set_direction(self,ids,initial=False):
        ids=np.asarray(ids,dtype=int);n=len(ids)
        kinds=self._dir_rng.choice(6,n,p=[.25,.35,.10,.10,.10,.10])
        if not initial:
            same=kinds==self._dir_kind[ids]
            kinds[same]=(kinds[same]+self._dir_rng.integers(1,6,same.sum()))%6
        commands=COMMANDS[kinds].copy()
        varied=self._dir_rng.random(n)<.2
        commands[varied,:2]*=self._dir_rng.choice([2/3,7/6],varied.sum())[:,None]
        commands[varied,2]*=self._dir_rng.choice([.75,1.25],varied.sum())
        self._dir_old_commands[ids]=0 if initial else self._dir_commands[ids,:3]
        self._dir_commands[ids]=0;self._dir_commands[ids,:3]=commands
        self._dir_kind[ids]=kinds;self._dir_age[ids]=0
        self._dir_mean_v[ids]=0;self._dir_mean_w[ids]=0
        self._dir_land_time[ids]=-1;self._dir_last_full[ids]=-1
        self._dir_stall_at[ids]=self._dir_rng.uniform(2.8,3.8,n)
        now=self._smooth_age[ids]*self.cfg.ctrl_dt
        self._dir_next_switch[ids]=np.where(self._dir_rng.random(n)<self.cfg.direction_switch_fraction,now+self._dir_rng.uniform(4.,6.,n),np.inf)
        if not initial:
            q=self._backend.get_base_quat()[ids]
            self._dir_heading0[ids]=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))

    def _maybe_resample_commands(self,state):
        super()._maybe_resample_commands(state)
        ids=np.flatnonzero(self._smooth_age*self.cfg.ctrl_dt>=self._dir_next_switch)
        if len(ids):self._set_direction(ids)
        state.info['commands'][:]=self._dir_commands

    def _reset_episode_history(self,ids):
        super()._reset_episode_history(ids)
        self._smooth_age[ids]=0;self._smooth_air[ids]=False;self._smooth_peak[ids]=0;self._smooth_landings[ids]=0
        self._dir_mean_v[ids]=0;self._dir_mean_w[ids]=0
        self._dir_land_time[ids]=-1;self._dir_last_full[ids]=-1
        self._dir_age[ids]=0

    def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
        # Reuse the infrastructure/base regularizers, skipping all previous
        # positive-forward-only startup/style/progress rewards.
        reward=StartupFlexEnv._compute_reward(self,state,linvel,gyro,upvector,dof_pos)
        dt=self.cfg.ctrl_dt;episode_age=self._smooth_age*dt;age=self._dir_age*dt
        cmd=state.info['commands'][:,:3]
        q=self._backend.get_base_quat();w,x,y,z=q.T
        roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        yaw_rate=(gyro[:,1]*np.sin(roll)+gyro[:,2]*np.cos(roll))/np.maximum(np.cos(pitch),.5)
        translating,axis,speed,turn,target=motion_metrics(cmd,linvel,yaw_rate,self.cfg.direction_speed_gain)
        self._dir_mean_v=.95*self._dir_mean_v+.05*linvel[:,:2]
        self._dir_mean_w=.95*self._dir_mean_w+.05*yaw_rate
        u=np.clip(age/1.2,0,1);shape=u*u*u*(10+u*(-15+6*u))
        desired=(1-shape[:,None])*self._dir_old_commands[:,:2]*self.cfg.direction_speed_gain+shape[:,None]*target
        velocity_score=np.exp(-np.sum((self._dir_mean_v-desired)**2,axis=1)/.065**2)
        instant_score=np.exp(-np.sum((linvel[:,:2]-desired)**2,axis=1)/.16**2)
        desired_w=(1-shape)*self._dir_old_commands[:,2]+shape*cmd[:,2]
        angular_score=np.exp(-((self._dir_mean_w-desired_w)/.22)**2)
        progress=np.where(translating,np.clip(speed,-.2,np.linalg.norm(target,axis=1)),.5*np.clip(turn,-.4,.6))
        # Compare normalized errors for the same command bins across DR.
        bins=self._dir_kind*10+np.rint(np.maximum(np.linalg.norm(cmd[:,:2],axis=1),np.abs(cmd[:,2]))*10).astype(int)
        normalized=np.where(translating,np.sum(self._dir_mean_v*axis,axis=1)/np.maximum(np.linalg.norm(target,axis=1),.05),np.sign(cmd[:,2])*self._dir_mean_w/np.maximum(abs(cmd[:,2]),.1))
        consistency=np.zeros(self.num_envs)
        for b in np.unique(bins[age>2.]):
            members=(bins==b)&(age>2.)&(normalized>.25)
            if members.sum()>=3:consistency[members]=np.minimum((normalized[members]-np.median(normalized[members]))**2,4.)
        pos,quat=self._backend.get_body_pose_w(self._smooth_body_ids)
        world=pos+rotate(quat,np.broadcast_to(self._smooth_site_local,pos.shape))
        inverse=q.copy();inverse[:,1:]*=-1
        width=np.abs(rotate(inverse,world[:,0]-world[:,1])[:,1])
        width_cost=np.minimum(((width-.212)/np.where(translating,.03,.045))**2,9.)
        height=self._backend.get_base_pos()[:,2]
        down_cost=np.minimum((np.maximum(self._smooth_z0-height-.004,0)/.012)**2,9.)*(episode_age<2.)
        world_velocity=rotate(q,linvel)
        vertical_cost=np.minimum((np.maximum(abs(world_velocity[:,2])-.14,0)/.10)**2,9.)*(episode_age<2.)
        initial_roll_cost=np.minimum((np.maximum(abs(roll)-np.deg2rad(6),0)/.08)**2,9.)*(episode_age<1.)
        h=self._foot_height;leaving=(~self._smooth_air)&(h>.0015)
        self._smooth_peak=np.maximum(self._smooth_peak,h)
        landing=self._smooth_air&(h<.0007)
        known=self._dir_land_time>=0
        stride=np.sum((world-self._dir_land_position)*self._dir_land_axis,axis=2)
        threshold=np.where(translating,.020,.008)
        proper=landing&(self._smooth_peak>=threshold[:,None])&((~translating[:,None])|(~known)|(stride>.04))
        peak_goal=np.where(translating,.032,.018)
        peak_score=np.exp(-((self._smooth_peak-peak_goal[:,None])/.009)**2)
        period=episode_age[:,None]-self._dir_land_time
        timing=np.exp(-((period-.50)/.14)**2)
        stride_goal=np.maximum(np.linalg.norm(target,axis=1)*.50,.04)
        stride_score=np.exp(-((stride-stride_goal[:,None])/.04)**2)*translating[:,None]
        events=(proper*(1.5*peak_score+.5*timing*known+.5*stride_score*known)).sum(1)
        short_cost=(landing&(self._smooth_peak<np.where(translating,.014,.005)[:,None])).sum(1)*.5
        overheight=(landing*np.minimum((np.maximum(self._smooth_peak-.045,0)/.015)**2,4.)).sum(1)*.8
        direction3=np.c_[axis,np.zeros(self.num_envs)]
        world_axis=rotate(q,direction3)
        self._dir_land_axis[landing]=np.broadcast_to(world_axis[:,None,:],world.shape)[landing]
        self._dir_land_position[landing]=world[landing]
        self._dir_land_time[landing]=np.broadcast_to(episode_age[:,None],landing.shape)[landing]
        self._dir_last_full[proper.any(1)]=episode_age[proper.any(1)]
        self._smooth_peak[landing]=0;self._smooth_air=(self._smooth_air|leaving)&(~landing)
        single=(h>.003).sum(1)==1
        swing=np.minimum(h.max(1)/peak_goal,1.)*single
        head=(dof_pos+self._encoder_bias)[:,[5,6]]
        head_score=np.exp(-np.mean(((head-np.array(self.cfg.smooth_head_target))/.15)**2,axis=1))
        # A small deviation is allowed; penalize large sustained turn on straight commands.
        straight=(np.abs(cmd[:,2])<.05)&translating
        unwanted_turn=np.minimum((np.maximum(abs(self._dir_mean_w)-.10,0)/.15)**2,4.)*straight
        overspeed=np.minimum((np.maximum(np.sum(self._dir_mean_v*axis,axis=1)-np.linalg.norm(target,axis=1)-.02,0)/.05)**2,4.)*translating*(age>1.2)
        density=(6*velocity_score+instant_score+2*angular_score+4*progress
            -2*consistency-3*overspeed-2*width_cost-2*down_cost-.5*vertical_cost-.5*initial_roll_cost
            +1.5*swing-2*(h>.003).all(1)+.6*head_score-.5*unwanted_turn)
        reward=reward+(density*dt+events-short_cost-overheight).astype(reward.dtype)
        state.info.setdefault('log',{}).update({'direction/vx':float(linvel[:,0].mean()),'direction/mean_speed_score':float(velocity_score.mean()),'direction/turn_score':float(angular_score.mean()),'direction/width_mm':float(width.mean()*1000),'direction/dr_consistency_cost':float(consistency.mean()),'direction/backward_fraction':float((cmd[:,0]<-.05).mean()),'direction/proper_step_fraction':float(proper.any(1).mean())})
        self._smooth_age+=1;self._dir_age+=1
        return reward

    def update_state(self,state):
        result=super().update_state(state)
        cmd=state.info['commands'][:,:3]
        translating,axis,speed,turn,target=motion_metrics(cmd,np.c_[self._dir_mean_v,np.zeros(self.num_envs)],self._dir_mean_w,self.cfg.direction_speed_gain)
        age=self._dir_age*self.cfg.ctrl_dt;episode=self._smooth_age*self.cfg.ctrl_dt
        movement=np.where(translating,speed>.045,turn>.08)
        stalled=(age>self._dir_stall_at)&((~movement)|(self._dir_last_full<0)|((episode-self._dir_last_full)>1.5))
        newly=stalled&(~result.terminated)
        return result.replace(terminated=result.terminated|stalled,reward=result.reward-(3*newly).astype(result.reward.dtype))

class DirectionFlexReset(NewStandBankReset):
    def build_reset_plan(self,env,ids):
        plan=super().build_reset_plan(env,ids)
        env._set_direction(ids,initial=True)
        plan.info_updates['commands'][:]=env._dir_commands[ids]
        q=plan.qpos[:,3:7]
        env._dir_heading0[ids]=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        return plan
