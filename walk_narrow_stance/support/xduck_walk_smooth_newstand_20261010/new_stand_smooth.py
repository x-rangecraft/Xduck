"""Experimental Walk owner: physically settled new-pose resets and smooth starts.

The actor contract remains 61 observations and 14 actions. World heights,
contacts and velocities are reward information only, never actor inputs.
"""
from dataclasses import dataclass
import mujoco
import numpy as np
from unilab.base import registry
from .rigid_startcourse import RigidStartCourseCfg, RigidStartCourseEnv
from .rigid_gpu import StagedSoleTiltReset
from .startup_flex import rotate

FIELDS = ['geom_pos','geom_quat','jnt_stiffness','dof_damping','body_mass',
          'body_inertia','body_ipos','geom_friction','dof_armature']

@registry.envcfg('XDuckWalkSmoothNewStand')
@dataclass
class NewStandSmoothCfg(RigidStartCourseCfg):
    newstand_bank_path: str|None = None
    newstand_seed: int = 1150091
    smooth_start_seconds: float = 2.5
    smooth_speed_ramp_seconds: float = 1.2
    smooth_width_target_m: float = .183
    smooth_width_weight: float = 2.
    smooth_down_weight: float = 2.
    smooth_vertical_weight: float = .5
    smooth_height_weight: float = 1.5
    smooth_walk_height_m: float = .33
    smooth_acceleration_weight: float = .3
    smooth_head_weight: float = .6
    smooth_slide_weight: float = 1.
    smooth_head_target: tuple[float,float] = (-.12,-.12)
    newstand_speed_low: float = .18
    newstand_speed_high: float = .35

@registry.env('XDuckWalkSmoothNewStand',sim_backend='mjwarp')
class NewStandSmoothEnv(RigidStartCourseEnv):
    def __init__(self,cfg,num_envs=1,backend_type='mjwarp'):
        if cfg.newstand_bank_path is None:
            raise ValueError('A physically collected new-standing bank is required')
        with np.load(cfg.newstand_bank_path) as z:
            self._newstand_bank = {k:z[k].copy() for k in z.files}
        self._newstand_rng = np.random.default_rng(cfg.newstand_seed)
        self._smooth_age = np.zeros(num_envs,int)
        self._smooth_z0 = np.zeros(num_envs)
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

    def _make_dr_provider(self):
        return NewStandBankReset()

    def _reset_episode_history(self,ids):
        super()._reset_episode_history(ids)
        self._smooth_age[ids] = 0
        self._smooth_air[ids] = False
        self._smooth_peak[ids] = 0
        self._smooth_landings[ids] = 0

    def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
        reward = super()._compute_reward(state,linvel,gyro,upvector,dof_pos)
        cfg = self._cfg
        age = self._smooth_age*cfg.ctrl_dt
        early = age<cfg.smooth_start_seconds
        u = np.clip(age/cfg.smooth_speed_ramp_seconds,0,1)
        shape = u*u*u*(10+u*(-15+6*u))
        cmd = state.info['commands']
        # During startup, reward gradual forward acceleration instead of a jump
        # immediately to commanded speed. The command itself remains positive.
        old_score = np.exp(-np.sum((linvel[:,:2]-cmd[:,:2])**2,axis=1)/cfg.reward_config.linear_velocity_std**2)
        desired = cmd[:,:2]*shape[:,None]
        new_score = np.exp(-np.sum((linvel[:,:2]-desired)**2,axis=1)/cfg.reward_config.linear_velocity_std**2)
        speed_delta = (new_score-old_score)*cfg.reward_config.scales['track_linear_velocity']*early
        pos,quat = self._backend.get_body_pose_w(self._smooth_body_ids)
        world = pos+rotate(quat,np.broadcast_to(self._smooth_site_local,pos.shape))
        baseq = self._backend.get_base_quat()
        inverse = baseq.copy();inverse[:,1:] *= -1
        local_delta = rotate(inverse,world[:,0]-world[:,1])
        width = np.abs(local_delta[:,1])
        target_width = (1-shape)*self._smooth_width0+shape*cfg.smooth_width_target_m
        width_cost = np.minimum(((width-target_width)/.035)**2,9.)
        height = self._backend.get_base_pos()[:,2]
        down = np.maximum(self._smooth_z0-height-.003,0)
        down_cost = np.minimum((down/.01)**2,9.)*early
        desired_height = (1-shape)*self._smooth_z0+shape*cfg.smooth_walk_height_m
        height_cost = np.minimum(((height-desired_height)/.012)**2,9.)
        world_velocity = rotate(baseq,linvel)
        vertical_cost = np.minimum((np.maximum(np.abs(world_velocity[:,2])-.10,0)/.07)**2,9.)*early
        acceleration = (linvel[:,0]-self._smooth_previous_v)/cfg.ctrl_dt
        acceleration_cost = np.minimum((np.maximum(np.abs(acceleration)-2.5,0)/2.)**2,9.)*early
        self._smooth_previous_v[:] = linvel[:,0]
        h = self._foot_height
        liftoff = (~self._smooth_air)&(h>.0015)
        self._smooth_liftoff[liftoff] = world[liftoff]
        forward = rotate(baseq,np.broadcast_to([1.,0,0],(self.num_envs,3)))
        self._smooth_axis[liftoff] = np.broadcast_to(forward[:,None,:],world.shape)[liftoff]
        self._smooth_peak = np.maximum(self._smooth_peak,h)
        landing = self._smooth_air&(h<.0007)
        advance = np.sum((world-self._smooth_liftoff)*self._smooth_axis,axis=2)
        proper = landing&(self._smooth_peak>=.012)&(advance>=.015)
        shallow = landing&(self._smooth_peak<.009)
        landing_reward = (.6*proper.sum(1)-.2*shallow.sum(1))*early
        self._smooth_landings += proper.sum(1)
        self._smooth_peak[landing] = 0
        self._smooth_air = (self._smooth_air|liftoff)&(~landing)
        single_clear = (h>.003).sum(1)==1
        swing_progress = np.minimum(h.max(1)/.025,1.)*single_clear
        double_air = (h>.003).all(1)
        # Penalize rapid site translation only for nearly flat, grounded soles;
        # normal heel/toe rolling is not treated as sliding.
        normal = rotate(quat,rotate(self._sole_delta_quat,np.broadcast_to(np.array(self._sole_normals),pos.shape)))
        flat = np.linalg.norm(normal[:,:,:2],axis=2)<.12
        grounded = h<.001
        slide_speed = np.linalg.norm(self._foot_vel[:,:,:2],axis=2)
        slide_cost = np.minimum((np.maximum(slide_speed-.05,0)/.10)**2,4.)*flat*grounded
        measured = dof_pos+self._encoder_bias
        head = measured[:,[5,6]]
        initial_head = self._newstand_bank['target'][0,[5,6]]
        head_goal = (1-shape[:,None])*initial_head+shape[:,None]*np.asarray(cfg.smooth_head_target)
        head_score = np.exp(-np.mean(((head-head_goal)/.15)**2,axis=1))
        density = (speed_delta-cfg.smooth_width_weight*width_cost
                   -cfg.smooth_down_weight*down_cost-cfg.smooth_vertical_weight*vertical_cost
                   -cfg.smooth_height_weight*height_cost
                   -cfg.smooth_acceleration_weight*acceleration_cost
                   +.6*swing_progress-1.5*double_air
                   -cfg.smooth_slide_weight*slide_cost.mean(1)+cfg.smooth_head_weight*head_score)
        added = density*cfg.ctrl_dt+landing_reward
        assert np.isfinite(added).all()
        state.info.setdefault('log',{}).update({
            'newstand/width_mm':float(width.mean()*1000),
            'newstand/start_down_mm':float(down[early].mean()*1000) if early.any() else 0.,
            'newstand/body_height_error_mm':float(np.abs(height-desired_height).mean()*1000),
            'newstand/proper_landing_fraction':float(proper.any(1).mean()),
            'newstand/shallow_landing_fraction':float(shallow.any(1).mean()),
            'newstand/smooth_density':float(density.mean()),
            'newstand/forward_velocity':float(linvel[:,0].mean()),
        })
        self._smooth_age += 1
        return reward+added.astype(reward.dtype)

    def update_state(self,state):
        result = super().update_state(state)
        q = self._backend.get_base_quat()
        upz = 1-2*(q[:,1]**2+q[:,2]**2)
        bad = (upz<np.cos(np.deg2rad(30.)))|(self._backend.get_base_pos()[:,2]<.25)
        new = bad&(~result.terminated)
        penalty = (.7*new).astype(result.reward.dtype)
        return result.replace(terminated=result.terminated|bad,reward=result.reward-penalty)

class NewStandBankReset(StagedSoleTiltReset):
    def build_reset_plan(self,env,env_ids):
        plan = super().build_reset_plan(env,env_ids)
        ids = np.asarray(env_ids,dtype=int)
        bank = env._newstand_bank
        rows = env._newstand_rng.integers(len(bank['qpos']),size=len(ids))
        plan.qpos[:] = bank['qpos'][rows]
        plan.qvel[:] = bank['qvel'][rows]
        for key in FIELDS:
            target = getattr(plan.randomization,key)
            if target is None:
                setattr(plan.randomization,key,bank[key][rows].copy())
            else:
                target[:] = bank[key][rows]
        for key in ['encoder_bias','imu_rotation','sole_angles','sole_delta_quat']:
            getattr(env,'_'+key)[ids] = bank[key][rows]
        raw = (bank['target'][rows]-env.default_angles)/env._physical_scales
        plan.info_updates['last_actions'][:] = raw
        plan.info_updates['current_actions'][:] = raw
        env._current_action_target[ids] = bank['target'][rows]
        env._previous_action_target[ids] = bank['target'][rows]
        goal = np.zeros((len(ids),13))
        goal[:,0] = env._newstand_rng.uniform(env.cfg.newstand_speed_low,env.cfg.newstand_speed_high,len(ids))
        goal[env._newstand_rng.random(len(ids))<.7,0] = .3
        plan.info_updates['commands'][:] = goal
        env._smooth_z0[ids] = bank['qpos'][rows,2]
        env._smooth_width0[ids] = bank['initial_width'][rows]
        env._smooth_previous_v[ids] = bank['linvel'][rows,0]
        env._spawn.record_episode_start(ids,plan.qpos[:,:3])
        return plan
