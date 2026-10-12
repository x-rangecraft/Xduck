"""Numerical reference only; integrate with the hardware API separately."""
import json
from pathlib import Path
import numpy as np
IO=json.loads(Path(__file__).with_name("policy_io.json").read_text())
DEFAULT=np.asarray(IO["default_angles_rad"],dtype=np.float32)
LOW=np.asarray(IO["target_low_rad"],dtype=np.float32)
HIGH=np.asarray(IO["target_high_rad"],dtype=np.float32)
INCREMENT=np.asarray(IO["target_increment_rad"],dtype=np.float32)
def hardware_to_policy(values):
    return np.asarray(values)[...,IO["hardware_to_policy_indices"]]
def policy_to_hardware(values):
    return np.asarray(values)[...,IO["policy_to_hardware_indices"]]
def initialize_target(measured_positions_policy_order):
    return np.clip(np.asarray(measured_positions_policy_order,dtype=np.float32),LOW,HIGH)
def build_observation(body_gyro,body_gravity_down,joint_positions,joint_velocities,previous_target,commands):
    obs=np.concatenate([body_gyro,body_gravity_down,np.asarray(joint_positions)-DEFAULT,
        joint_velocities,(np.asarray(previous_target)-DEFAULT)/IO["target_history_scale_rad"],commands],axis=-1).astype(np.float32)
    if obs.shape[-1]!=61:raise ValueError("Expected 61 observation values")
    return obs
def decode_target(previous_target,raw_actions):
    return np.clip(np.asarray(previous_target)+INCREMENT*np.tanh(np.asarray(raw_actions)),LOW,HIGH)
