from pathlib import Path
import shutil,json,hashlib
import mujoco,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;B=R.parent/'xduck_speed_correction_20260930';F=R.parent/'xduck_flex4096_fromzero5000_20260929';P=R/'project'
assert not P.exists()
shutil.copytree(B/'project',P,symlinks=True,ignore=shutil.ignore_patterns('.venv','__pycache__','.git','outputs','logs'));(P/'.venv').symlink_to((B/'project/.venv').resolve())
shutil.copytree((F/'physics').resolve(),R/'physics');shutil.copy2(B/'swing_reference.npz',R/'swing_reference.npz')
f=P/'src/unilab/dr/types.py';s=f.read_text()
for field in ['jnt_stiffness','dof_damping']:
 const='RESET_TERM_'+field.upper()
 s=s.replace('RESET_TERM_KP =',f'{const} = "{field}"\nRESET_TERM_KP =',1)
 s=s.replace('    kp: np.ndarray | None = None',f'    {field}: np.ndarray | None = None\n    kp: np.ndarray | None = None',1)
 s=s.replace('            kp=payload.kp',f'            {field}=payload.{field} if self.supports_reset_term({const}) else None,\n            kp=payload.kp',1)
 s=s.replace('        if self.kp is not None:',f'        if self.{field} is not None:\n            terms.add({const})\n        if self.kp is not None:',1)
f.write_text(s)
f=P/'src/unilab/base/backend/mjwarp/xduck_gf.py';s=f.read_text().replace("'geom_quat','dof_armature'","'geom_quat','dof_armature','jnt_stiffness','dof_damping'")
s=s.replace('RESET_TERM_GEOM_POS,RESET_TERM_GEOM_QUAT,','RESET_TERM_GEOM_POS,RESET_TERM_GEOM_QUAT,RESET_TERM_JNT_STIFFNESS,RESET_TERM_DOF_DAMPING,')
s=s.replace('RESET_TERM_GEOM_QUAT,RESET_TERM_DOF_ARMATURE})','RESET_TERM_GEOM_QUAT,RESET_TERM_DOF_ARMATURE,RESET_TERM_JNT_STIFFNESS,RESET_TERM_DOF_DAMPING})');f.write_text(s)
f=P/'src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=f.read_text()
s=s.replace('    speed_correction_mode:', '    flexible_dr_enabled: bool = True\n    flexible_dr_fraction: float = 0.1\n    flexible_dr_seed: int = 20260930\n    speed_correction_mode:')
s=s.replace('cfg.physics_microsteps!=1','cfg.physics_microsteps not in (1,16)').replace('requires a compiled model and 20 x 1ms mechanics','requires compiled mechanics with microsteps 1 or 16').replace('compiled_model_file=cfg.warp_compiled_model_file,physics_microsteps=1,','compiled_model_file=cfg.warp_compiled_model_file,physics_microsteps=cfg.physics_microsteps,')
s=s.replace('        self._dr_multiplier=1.0','        self._flex_rng=np.random.default_rng(cfg.flexible_dr_seed)\n        self._dr_multiplier=1.0',1)
s=s.replace('        floor=m.geom(\'floor\').id', '''        self._flex_joint_ids=np.flatnonzero(m.jnt_stiffness>0)
        self._flex_dof_ids=m.jnt_dofadr[self._flex_joint_ids]
        self._nominal_flex_stiffness=m.jnt_stiffness.copy()
        self._nominal_flex_damping=m.dof_damping.copy()
        assert len(self._flex_joint_ids)==48
        floor=m.geom('floor').id''')
s=s.replace('cfg.sole_tilt_range_deg=(-cfg.dr_base_sole_tilt_deg*m,cfg.dr_base_sole_tilt_deg*m)','cfg.sole_tilt_range_deg=(-cfg.dr_base_sole_tilt_deg,cfg.dr_base_sole_tilt_deg)')
s=s.replace("        env._dr_episode_multiplier[env_ids]=m;plan.info_updates['dr_multiplier']=m",'''        if env.cfg.flexible_dr_enabled:
            n=len(env_ids);spread=env.cfg.flexible_dr_fraction
            k=np.broadcast_to(env._nominal_flex_stiffness,(n,len(env._nominal_flex_stiffness))).copy()
            d=np.broadcast_to(env._nominal_flex_damping,(n,len(env._nominal_flex_damping))).copy()
            k[:,env._flex_joint_ids]*=env._flex_rng.uniform(1-spread,1+spread,(n,48))
            d[:,env._flex_dof_ids]*=env._flex_rng.uniform(1-spread,1+spread,(n,48))
            # The compiled model zeroes active motor damping; only flexible DOFs vary.
            d[:,env._motor_vel_indices]=0.
            plan.randomization.jnt_stiffness=k;plan.randomization.dof_damping=d
        env._dr_episode_multiplier[env_ids]=m;plan.info_updates['dr_multiplier']=m''')
f.write_text(s)
ck=B/'mean_only/model_1229.pt';c=OmegaConf.load(B/'project/conf/ppo/mean_only.yaml')
c.algo.load_run=str(ck);c.algo.max_iterations=50;c.algo.save_interval=25;c.algo.run_name='meanonly_flex50';c.training.log_dir=str(R/c.algo.run_name)
c.env.scene.model_file=str(R/'physics/scene.xml');c.env.scene.visual_model_file=str(R/'physics/scene.xml');c.env.warp_compiled_model_file=str(R/'physics/effective_training.mjb');c.env.reference_path=str(R/'swing_reference.npz');c.env.physics_microsteps=16
c.env.flexible_dr_enabled=True;c.env.flexible_dr_fraction=.1;c.env.flexible_dr_seed=20260930;c.env.dr_base_sole_tilt_deg=3.;c.env.sole_tilt_range_deg=[-3.,3.]
# Freeze inherited scheduled reward coefficients at the source checkpoint to isolate adaptation.
for key,stages in c.env.curriculum.items():
 if key.endswith('_stages') and OmegaConf.is_list(stages):
  chosen=OmegaConf.to_container([x for x in stages if int(x.step)<=29520][-1]);chosen['step']=0;c.env.curriculum[key]=[chosen]
OmegaConf.save(c,P/'conf/ppo/meanonly_flex50.yaml');OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
sm=OmegaConf.create(OmegaConf.to_container(c));sm.algo.max_iterations=2;sm.algo.run_name='smoke4096';sm.training.log_dir=str(R/'smoke4096');OmegaConf.save(sm,P/'conf/ppo/flex_smoke.yaml')
m=mujoco.MjModel.from_binary_path(c.env.warp_compiled_model_file);assert (m.nq,m.nv,m.nu)==(69,68,14) and m.opt.timestep==.001/16
for p in (P/'src').rglob('*.py'):compile(p.read_text(),str(p),'exec')
s= (B/'evaluate_directions.py').read_text().replace('e._sole_rng=np.random.default_rng(20260930);e.reset','e._sole_rng=np.random.default_rng(20260930);e._flex_rng=np.random.default_rng(20260930);e.reset').replace('[:,1:22].copy()', '[:,1:70].copy()').replace('geom_quat=e._backend._device_model.geom_quat.numpy())','geom_quat=e._backend._device_model.geom_quat.numpy(),jnt_stiffness=e._backend._device_model.jnt_stiffness.numpy(),dof_damping=e._backend._device_model.dof_damping.numpy())')
(R/'evaluate_directions.py').write_text(s)
a=dict(source=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),source_completed=1230,additional_updates=50,final_checkpoint='meanonly_flex50/model_1279.pt',num_envs=4096,mechanical_steps=320,mechanical_dt=m.opt.timestep,passive_coordinates=48,flexible_dr_fraction=.1,sole_tilt_deg=[-3,3],other_dr_multiplier=2.5,freeze_inherited_curriculum_at_step=29520,stiffness_values=np.unique(m.jnt_stiffness[m.jnt_stiffness>0]).tolist(),damping_values=np.unique(m.dof_damping[m.dof_damping>0]).tolist())
(R/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
