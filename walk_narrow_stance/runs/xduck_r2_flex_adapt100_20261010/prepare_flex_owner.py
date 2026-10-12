from pathlib import Path
R=Path(__file__).resolve().parent;owner=R/'project/src/unilab/envs/locomotion/xduck_gf43x40'
s=(R/'direction_speed.py').read_text();s=s.replace('from .rigid_startcourse import RigidStartCourseEnv','from .startup_flex import StartupFlexEnv\nimport mujoco')
s=s.replace("'XDuckDirectionSpeed'","'XDuckDirectionFlex'").replace('class DirectionSpeedCfg','class DirectionFlexCfg').replace('class DirectionSpeedEnv(NewStandSmoothEnv):','class DirectionFlexEnv(StartupFlexEnv):')
pre=(owner/'new_stand_smooth.py').read_text();a=pre.index('        if cfg.newstand_bank_path is None:');b=pre.index('        super().__init__(cfg,num_envs=num_envs,backend_type=backend_type)',a);init=pre[a:b]
s=s.replace('        super().__init__(cfg,num_envs=num_envs,backend_type=backend_type)',init+'        super().__init__(cfg,num_envs=num_envs,backend_type=backend_type)\n        self._smooth_body_ids = np.array([self._backend.get_body_id(n) for n in names],np.int32)\n        assert len(self._passive_qpos_indices)==48 and cfg.flexible_dr_enabled and cfg.physics_microsteps==16',1)
s=s.replace('class DirectionReset(NewStandBankReset):','class DirectionFlexReset(NewStandBankReset):').replace('return DirectionReset()','return DirectionFlexReset()')
s=s.replace('        super()._reset_episode_history(ids)\n        self._dir_mean_v', '        super()._reset_episode_history(ids)\n        self._smooth_age[ids]=0;self._smooth_air[ids]=False;self._smooth_peak[ids]=0;self._smooth_landings[ids]=0\n        self._dir_mean_v')
s=s.replace('reward=RigidStartCourseEnv._compute_reward(self,state,linvel,gyro,upvector,dof_pos)','reward=StartupFlexEnv._compute_reward(self,state,linvel,gyro,upvector,dof_pos)')
compile(s,'direction_flex.py','exec');(owner/'direction_flex.py').write_text(s);(R/'direction_flex.py').write_text(s)
p=owner/'__init__.py';s=p.read_text();line='\nfrom .direction_flex import DirectionFlexCfg, DirectionFlexEnv\n';p.write_text(s+line if line.strip() not in s else s)
from omegaconf import OmegaConf
for name in ['PPO_FLEX100_CONFIG.yaml','TRAINING_CONFIG.yaml']:
 c=OmegaConf.load(R/name);c.training.task_name='XDuckDirectionFlex';OmegaConf.save(c,R/name)
print('FLEX_OWNER_READY',flush=True)
