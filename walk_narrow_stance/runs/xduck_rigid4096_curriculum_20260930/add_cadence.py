from pathlib import Path
import json
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';f=P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=f.read_text()
s=s.replace('    audit_twist: tuple[float,float,float] | None = None','''    audit_twist: tuple[float,float,float] | None = None
    cadence_target_cycle_s: float = 0.5632857142856981
    cadence_cycle_std_s: float = 0.06
    cadence_weight_after300: float = 0.15
    target_walk_fraction_after300: float = 0.15
    target_walk_velocity_range: tuple[float,float] = (0.15,0.20)''')
s=s.replace('        self._reference_weight=0.0','''        self._reference_weight=0.0
        self._cadence_weight=0.0
        self._cadence_last_touch_s=np.full((num_envs,2),-1e6)
        self._cadence_period_s=np.zeros((num_envs,2))
        self._cadence_previous_contact=np.zeros((num_envs,2),dtype=bool)''',1)
old='''    def _sample_velocity_commands(self,count):
        if self._cfg.audit_twist is not None:return np.tile(self._cfg.audit_twist,(count,1))
        if self._cfg.audit_velocity is None:
            return super()._sample_velocity_commands(count)
        return np.tile([self._cfg.audit_velocity,0.,0.],(count,1))'''
new='''    def _sample_velocity_commands(self,count):
        if self._cfg.audit_twist is not None:return np.tile(self._cfg.audit_twist,(count,1))
        if self._cfg.audit_velocity is not None:return np.tile([self._cfg.audit_velocity,0.,0.],(count,1))
        cmd=super()._sample_velocity_commands(count)
        if self._cfg.stage_schedule_enabled and self.step_counter//self._cfg.rollout_steps>=300:
            mask=np.random.random(count)<self._cfg.target_walk_fraction_after300
            n=int(mask.sum());lo,hi=self._cfg.target_walk_velocity_range
            cmd[mask]=np.c_[np.random.uniform(lo,hi,n),np.zeros((n,2))]
        return cmd'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('        self._reference_weight=0.0 if iteration<300 else cfg.reference_weight_after300','        self._reference_weight=0.0 if iteration<300 else cfg.reference_weight_after300\n        self._cadence_weight=0.0 if iteration<300 else cfg.cadence_weight_after300')
needle='    def _compute_reward(self,state,linvel,gyro,upvector,dof_pos):'
method='''    def _reset_episode_history(self,env_ids):
        super()._reset_episode_history(env_ids)
        self._cadence_last_touch_s[env_ids]=-1e6
        self._cadence_period_s[env_ids]=0
        self._cadence_previous_contact[env_ids]=False

    def _cadence_component(self,state):
        now=self.step_counter*self._cfg.ctrl_dt
        touchdown=self._contact&(~self._cadence_previous_contact)&(self._landed_air_time>=.08)
        old=self._cadence_last_touch_s.copy()
        elapsed=now-old
        valid_touch=touchdown&(old>-1e5)
        self._cadence_period_s[valid_touch]=elapsed[valid_touch]
        self._cadence_last_touch_s[touchdown]=now
        self._cadence_previous_contact[:]=self._contact
        cmd=state.info['commands']
        gate=(cmd[:,0]>=.10)&(cmd[:,0]<=.25)&(np.abs(cmd[:,1])<.05)&(np.abs(cmd[:,2])<.1)
        supported=self._contact.sum(axis=1)==1
        period=self._cadence_period_s
        age=now-self._cadence_last_touch_s
        target=self._cfg.cadence_target_cycle_s
        valid=(period>.2)&(period<1.)&(age<=1.5*target)
        score=np.where(valid,np.exp(-((period-target)/self._cfg.cadence_cycle_std_s)**2),0.)
        value=np.sum(score,axis=1)/np.maximum(valid.sum(axis=1),1)
        value*=gate&supported
        log=state.info.setdefault('log',{})
        log['audit/cadence_score']=float(np.mean(value))
        if valid.any():log['audit/cadence_steps_per_s']=float(np.mean(2/period[valid]))
        return value

'''
assert s.count(needle)==1;s=s.replace(needle,method+needle)
s=s.replace('        reward=super()._compute_reward(state,linvel,gyro,upvector,dof_pos)\n        if self._reference_weight==0:return reward','        cadence=self._cadence_component(state)\n        reward=super()._compute_reward(state,linvel,gyro,upvector,dof_pos)\n        if self._reference_weight==0:return reward')
s=s.replace('reward+=(self._reference_weight*value*self._cfg.ctrl_dt).astype(reward.dtype,copy=False)','reward+=((self._reference_weight*value+self._cadence_weight*cadence)*self._cfg.ctrl_dt).astype(reward.dtype,copy=False)')
s=s.replace("        log['curriculum/reference_weight']=self._reference_weight", "        log['curriculum/reference_weight']=self._reference_weight\n        log['curriculum/cadence_weight']=self._cadence_weight")
compile(s,str(f),'exec');f.write_text(s)
for p in [R/'TRAINING_CONFIG.yaml',P/'conf/ppo/rigid4096_curriculum.yaml',P/'conf/ppo/resume_smoke4096.yaml']:
 c=OmegaConf.load(p)
 for key,val in dict(cadence_target_cycle_s=.5632857142856981,cadence_cycle_std_s=.06,cadence_weight_after300=.15,target_walk_fraction_after300=.15,target_walk_velocity_range=[.15,.20]).items():c.env[key]=val
 OmegaConf.save(c,p)
a=json.loads((R/'audit.json').read_text());a.update(target_actual_forward_speed_m_s=[.15,.20],reference_steps_per_s=2/.5632857142856981,cadence_weight_after300=.15,cadence_gate='straight forward command0.10–0.25m/s with one foot in contact; not on other directions',target_velocity_training_fraction_after300=.15,broad_training_fraction_after300=.85)
(R/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps({k:a[k] for k in ['target_actual_forward_speed_m_s','reference_steps_per_s','cadence_weight_after300','target_velocity_training_fraction_after300','broad_training_fraction_after300']},indent=2))
