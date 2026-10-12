from pathlib import Path
import shutil,json,hashlib,argparse
import mujoco,numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'xduck_rigid4096_100ms_200_20260930';P=R/'project'
p=argparse.ArgumentParser();p.add_argument('--total',type=int,default=5000);args=p.parse_args();assert args.total>200
assert not P.exists()
shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'));(P/'.venv').symlink_to((B/'project/.venv').resolve());shutil.copytree(B/'physics',R/'physics')
shutil.copy2(R.parent/'xduck_gait_alignment_20260929/swing_reference.npz',R/'swing_reference.npz')
f=P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=f.read_text()
fields='''    stage_schedule_enabled: bool = False
    rollout_steps: int = 24
    reference_path: str | None = None
    reference_weight_after300: float = 0.45
    reference_height_std: float = 0.015
    reference_pitch_std: float = 0.2617993877991494
    dr_base_com_m: float = 0.00627
    dr_base_head_com_m: float = 0.00627
    dr_base_push_m_s: float = 0.4337049688440288
    dr_base_noise_level: float = 1.0
    dr_base_sole_tilt_deg: float = 3.0
    dr_base_armature_halfwidth: float = 0.1
    audit_twist: tuple[float,float,float] | None = None
'''
s=s.replace('    physical_action_scale_rad: list[float] | None = None','    physical_action_scale_rad: list[float] | None = None\n'+fields)
s=s.replace('        return SoleTiltReset()','        return StagedSoleTiltReset() if self._cfg.stage_schedule_enabled else SoleTiltReset()')
s=s.replace('    def _sample_velocity_commands(self,count):','    def _sample_velocity_commands(self,count):\n        if self._cfg.audit_twist is not None:return np.tile(self._cfg.audit_twist,(count,1))')
s=s.replace('        if self._cfg.audit_velocity is None:return super()._sample_head_commands(count)','        if self._cfg.audit_twist is not None:return np.zeros((count,4))\n        if self._cfg.audit_velocity is None:return super()._sample_head_commands(count)')
s=s.replace('        if self._cfg.audit_velocity is None:return super()._sample_body_commands(count)','        if self._cfg.audit_twist is not None:return np.zeros((count,6))\n        if self._cfg.audit_velocity is None:return super()._sample_body_commands(count)')
s=s.replace('        # Geometry is loaded only on the cold path. Flat-ground task only.','        self._dr_multiplier=1.0\n        self._reference_weight=0.0\n        self._dr_episode_multiplier=np.ones(num_envs)\n        # Geometry is loaded only on the cold path. Flat-ground task only.')
s=s.replace('        self._sole_vertices=[];body_names=[]','''        self._sole_vertices=[];body_names=[]
        self._sole_normals=[]
        self._sole_world_pitch=np.zeros((num_envs,2))
        self._reference=None
        if cfg.reference_path:
            with np.load(cfg.reference_path) as ref:self._reference={k:ref[k].copy() for k in ref.files}
            assert self._reference['height'].shape==(64,2) and self._reference['pitch'].shape==(64,2)
            assert np.all(self._reference['duration']>0)''')
s=s.replace('            self._sole_vertices.append(vertices@rot.reshape(3,3).T+m.geom_pos[g])','''            self._sole_vertices.append(vertices@rot.reshape(3,3).T+m.geom_pos[g])
            _,eig=np.linalg.eigh(np.cov(vertices.T));normal=rot.reshape(3,3)@eig[:,0]
            if (d.xmat[m.geom_bodyid[g]].reshape(3,3)@normal)[2]<0:normal=-normal
            self._sole_normals.append(normal)''')
needle='            self._foot_height[:,j]=np.maximum(0.,np.min(rotated[:,:,2],axis=1)+pos[:,j,2]-self._ground_z)'
s=s.replace(needle,needle+'''
            n=np.broadcast_to(self._sole_normals[j],(self.num_envs,3));delta=self._sole_delta_quat[:,j]
            n=n+2*np.cross(delta[:,1:],np.cross(delta[:,1:],n)+delta[:,:1]*n)
            bodyq=quat[:,j];n=n+2*np.cross(bodyq[:,1:],np.cross(bodyq[:,1:],n)+bodyq[:,:1]*n)
            w,x,y,z=self._backend.get_base_quat().T
            yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            self._sole_world_pitch[:,j]=np.arctan2(np.cos(yaw)*n[:,0]+np.sin(yaw)*n[:,1],n[:,2])''')
marker='    def apply_action(self,actions,state):'
methods='''    def _apply_stage_schedule(self):
        if not self._cfg.stage_schedule_enabled:return
        cfg=self._cfg;iteration=self.step_counter//cfg.rollout_steps
        self._cfg.reward_config.air_time_threshold_min=.1 if iteration<300 else (.14 if iteration<600 else .18)
        self._dr_multiplier=1.0 if iteration<300 else (1.5 if iteration<600 else (2.0 if iteration<900 else 2.5))
        self._reference_weight=0.0 if iteration<300 else cfg.reference_weight_after300
        m=self._dr_multiplier
        self._com_range=cfg.dr_base_com_m*m;self._head_com_range=cfg.dr_base_head_com_m*m
        cfg.domain_rand.velocity_push_xy=[-cfg.dr_base_push_m_s*m,cfg.dr_base_push_m_s*m]
        cfg.domain_rand.dof_armature_multiplier_range=[1-cfg.dr_base_armature_halfwidth*m,1+cfg.dr_base_armature_halfwidth*m]
        cfg.noise_config.level=cfg.dr_base_noise_level*m
        cfg.sole_tilt_range_deg=(-cfg.dr_base_sole_tilt_deg*m,cfg.dr_base_sole_tilt_deg*m)

    def _update_curriculum(self):
        super()._update_curriculum()
        self._apply_stage_schedule()

    def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):
        reward=super()._compute_reward(state,linvel,gyro,upvector,dof_pos)
        if self._reference_weight==0:return reward
        ref=self._reference
        if ref is None:raise ValueError('A reference is required after300')
        h=np.empty_like(self._air_time);p=np.empty_like(self._air_time)
        for side in range(2):
            phase=np.clip(self._air_time[:,side]/ref['duration'][side],0,1)
            h[:,side]=np.interp(phase,ref['phase'],ref['height'][:,side])
            p[:,side]=np.interp(phase,ref['phase'],ref['pitch'][:,side])
        h_score=np.exp(-((self._foot_height-h)/self._cfg.reference_height_std)**2)
        p_score=np.exp(-((self._sole_world_pitch-p)/self._cfg.reference_pitch_std)**2)
        cmd=state.info['commands']
        moving=(np.linalg.norm(cmd[:,:2],axis=1)+np.abs(cmd[:,2]))>.01
        forward=(cmd[:,0]>.05)&(np.abs(cmd[:,1])<.05)&(np.abs(cmd[:,2])<.1)
        supported=(~self._contact)&self._contact[:,::-1]
        # Height is direction-independent. Pitch imitation is limited to the demonstrated straight-forward gait.
        value=np.sum((.5*h_score*moving[:,None]+.5*p_score*forward[:,None])*supported,axis=1)
        reward+=(self._reference_weight*value*self._cfg.ctrl_dt).astype(reward.dtype,copy=False)
        state.info.setdefault('log',{})['audit/reference_score']=float(np.mean(value))
        return reward

'''
assert s.count(marker)==1;s=s.replace(marker,methods+marker)
s=s.replace("        log['audit/action_target_clip_fraction']=self._actual_clip_fraction", """        log['audit/action_target_clip_fraction']=self._actual_clip_fraction
        log['curriculum/iteration']=float(self.step_counter//self._cfg.rollout_steps)
        log['curriculum/air_time_threshold_ms']=self._cfg.reward_config.air_time_threshold_min*1000
        log['curriculum/dr_multiplier']=self._dr_multiplier
        log['curriculum/reference_weight']=self._reference_weight
        log['curriculum/dr_episode_min']=float(self._dr_episode_multiplier.min())
        log['curriculum/dr_episode_max']=float(self._dr_episode_multiplier.max())""")
s+='''

class StagedSoleTiltReset(SoleTiltReset):
    def build_reset_plan(self,env,env_ids):
        env._apply_stage_schedule();plan=super().build_reset_plan(env,env_ids);m=env._dr_multiplier
        ratio=1+(env._mass_inertia_scale[env_ids]-1)*m
        plan.randomization.body_mass[:,env._trunk_body_id]=env._base_body_mass[env._trunk_body_id]*ratio
        plan.randomization.body_inertia[:,env._trunk_body_id,:]=env._base_body_inertia[env._trunk_body_id,:]*ratio[:,None]
        if env.cfg.domain_rand.randomize_foot_friction:
            mu=1+(env._foot_friction[env_ids]-1)*m
            plan.randomization.geom_friction[:,env._foot_geom_ids,0]=mu[:,None]
        env._dr_episode_multiplier[env_ids]=m;plan.info_updates['dr_multiplier']=m
        return plan
'''
f.write_text(s);compile(s,str(f),'exec')
ck=B/'rigid4096_100ms_200/model_199.pt';saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['iter']==199 and saved['infos']['unilab_progress']['env_steps_per_env']==4800
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.algo.resume=True;c.algo.load_run=str(ck);c.algo.checkpoint=-1;c.algo.resume_path=None;c.algo.max_iterations=args.total-200;c.algo.run_name='rigid4096_curriculum';c.algo.save_interval=100;c.training.log_dir=str(R/c.algo.run_name)
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb')
c.env.stage_schedule_enabled=True;c.env.rollout_steps=24;c.env.reference_path=str(R/'swing_reference.npz');c.env.reference_weight_after300=.45;c.env.reference_height_std=.015;c.env.reference_pitch_std=.2617993877991494;c.env.audit_twist=None
c.env.curriculum.com_range_stages=[{'step':0,'value':.00627}];c.env.curriculum.head_com_range_stages=[{'step':0,'value':.00627}];c.env.curriculum.push_stages=[{'step':0,'value':.4337049688440288}]
OmegaConf.save(c,P/'conf/ppo/rigid4096_curriculum.yaml');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
smoke=OmegaConf.create(OmegaConf.to_container(c));smoke.algo.max_iterations=2;smoke.algo.run_name='resume_smoke4096';smoke.training.log_dir=str(R/smoke.algo.run_name);OmegaConf.save(smoke,P/'conf/ppo/resume_smoke4096.yaml')
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);assert (m.nq,m.nv,m.nu)==(21,20,14) and m.opt.timestep==.001
(R/'audit.json').write_text(json.dumps(dict(passed=True,total_updates=args.total,additional_updates=args.total-200,start_completed_updates=200,source_checkpoint=str(ck),source_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(Path(c.env.warp_compiled_model_file).read_bytes()).hexdigest(),passive_coordinates=0,mechanical_steps=20,num_envs=4096,stages=[{'iteration':0,'air_time_ms':100,'dr':1,'reference_weight':0},{'iteration':300,'air_time_ms':140,'dr':1.5,'reference_weight':.45},{'iteration':600,'air_time_ms':180,'dr':2,'reference_weight':.45},{'iteration':900,'air_time_ms':180,'dr':2.5,'reference_weight':.45}],reference_height_all_moving_directions=True,reference_pitch_straight_forward_only=True,commands=OmegaConf.to_container(c.env.commands),PPO=OmegaConf.to_container(c.algo.algorithm),physics_stage_changes_next_reset=True),indent=2))
print((R/'audit.json').read_text())
