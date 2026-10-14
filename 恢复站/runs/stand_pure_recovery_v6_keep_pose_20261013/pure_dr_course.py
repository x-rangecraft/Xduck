"""Scale uncertainty about nominal mechanics once, before environment creation."""
from copy import deepcopy
import math

def scaled_config(base):
 cfg=deepcopy(base)
 if not cfg.dr_curriculum_enabled:return cfg
 assert len(cfg.dr_multipliers)==4 and 0<=cfg.discovery_level<4
 m=float(cfg.dr_multipliers[cfg.discovery_level]);assert math.isfinite(m) and m>0
 dr=cfg.domain_rand
 for key in ['mass_inertia_scale_range','foot_friction_range','dof_armature_multiplier_range']:
  bounds=[1.+(float(x)-1.)*m for x in getattr(dr,key)]
  assert 0<bounds[0]<=bounds[1],(key,bounds)
  setattr(dr,key,bounds)
 cfg.noise_config.level*=m
 for key in ['com_range_stages','head_com_range_stages']:
  for stage in getattr(cfg.curriculum,key):stage['value']*=m
 return cfg

def dr_summary(cfg):
 dr=cfg.domain_rand
 return dict(enabled=cfg.dr_curriculum_enabled,level=cfg.discovery_level,
  multiplier=cfg.dr_multipliers[cfg.discovery_level] if cfg.dr_curriculum_enabled else 1.,
  **{k:list(getattr(dr,k)) for k in ['mass_inertia_scale_range','foot_friction_range','dof_armature_multiplier_range','kp_multiplier_range','kd_multiplier_range']},
  motor={k:list(getattr(dr.motor,k)) for k in ['torque_scale_range','friction_scale_range','response_time_scale_range']},
  imu_max_angle_deg=dr.imu_max_angle_deg,noise_level=cfg.noise_config.level,
  sole_tilt_range_deg=list(cfg.sole_tilt_range_deg),imu_delay_ms=list(cfg.latency.imu_delay_ms),
  joint_velocity_delay_ms=cfg.latency.joint_velocity_delay_ms,
  trunk_com_range_m=cfg.curriculum.com_range_stages[0]['value'],head_com_range_m=cfg.curriculum.head_com_range_stages[0]['value'])
