from pathlib import Path
import shutil,json,hashlib
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent; B=R.parent/'xduck_rigid4096_220after900_20260930'; P=R/'project'
assert not P.exists()
shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'))
(P/'.venv').symlink_to((B/'project/.venv').resolve())
shutil.copytree(B/'physics',R/'physics');shutil.copy2(B/'swing_reference.npz',R/'swing_reference.npz')
f=P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=f.read_text()
s=s.replace('    target_walk_velocity_range:', '    speed_correction_mode: str = "off"\n    speed_correction_start_update: int = 1200\n    speed_correction_ramp_updates: int = 10\n    target_walk_velocity_range:')
s=s.replace('        self._dr_multiplier=1.0','        from .speed_history import SpeedHistory\n        self._speed_history=SpeedHistory(num_envs)\n        self._dr_multiplier=1.0',1)
s=s.replace('        self._cadence_last_touch_s[env_ids]=-1e6','        self._speed_history.reset(env_ids)\n        self._cadence_last_touch_s[env_ids]=-1e6')
s=s.replace('        if self._reference_weight==0:return reward','        reward += self._speed_correction(state,linvel).astype(reward.dtype,copy=False)\n        if self._reference_weight==0:return reward')
method='''    def _speed_correction(self,state,linvel):
        cfg=self._cfg
        if cfg.speed_correction_mode=='off':return np.zeros(self.num_envs)
        cmd=state.info['commands'][:,:3]
        mean,ready,variance,stable=self._speed_history.update(linvel,cmd,self._next_push_step)
        alpha=np.clip((self.step_counter/cfg.rollout_steps-cfg.speed_correction_start_update)/cfg.speed_correction_ramp_updates,0,1)
        moving=(np.linalg.norm(cmd[:,:2],axis=1)+np.abs(cmd[:,2]))>.01
        old=np.exp(-(np.sum((linvel[:,:2]-cmd[:,:2])**2,axis=1)+linvel[:,2]**2)/cfg.reward_config.linear_velocity_std**2)
        new=np.exp(-np.sum((mean-cmd[:,:2])**2,axis=1)/.08**2-linvel[:,2]**2/cfg.reward_config.linear_velocity_std**2)
        track=cfg.reward_config.scales['track_linear_velocity']*(new-old)*ready*moving
        penalty=.2*np.minimum(variance/.02**2,4)*stable*moving if cfg.speed_correction_mode=='mean_var' else np.zeros(self.num_envs)
        log=state.info.setdefault('log',{})
        log['speed/alpha']=float(alpha);log['speed/mean_track_delta']=float(np.mean(track));log['speed/variance_cost']=float(np.mean(penalty))
        return alpha*(track-penalty)*cfg.ctrl_dt

'''
s=s.replace('    def apply_action(self,actions,state):',method+'    def apply_action(self,actions,state):');compile(s,str(f),'exec');f.write_text(s)
shutil.copy2(R/'speed_history.py',f.parent/'speed_history.py')
ck=B/'rigid4096_220after900/model_1199.pt'
c=OmegaConf.load(B/'TRAINING_CONFIG.yaml');c.algo.load_run=str(ck);c.algo.max_iterations=30;c.algo.save_interval=30
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.reference_path=str(R/'swing_reference.npz')
for name,mode in [('control','off'),('mean_only','mean'),('mean_var','mean_var')]:
 c.algo.run_name=name;c.training.log_dir=str(R/name);c.env.speed_correction_mode=mode
 OmegaConf.save(c,P/f'conf/ppo/{name}.yaml')
 if name=='control':OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
shutil.copy2(B/'evaluate_directions.py',R/'evaluate_directions.py')
(R/'audit.json').write_text(json.dumps(dict(source=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),updates_each=30,ramp_updates=10,mean_window_s=.5,variance_window_s=1.,mean_std=.08,variance_std=.02,variance_weight=.2,seed=1),indent=2))
print('setup complete')
