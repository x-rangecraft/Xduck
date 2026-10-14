"""ONNX standing controller; caller supplies calibrated, time-aligned feedback."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

class Standing2800:
 def __init__(self,directory,previous_commanded_target):
  root=Path(directory);self.config=OmegaConf.load(root/'INFERENCE_CONFIG.yaml')
  self.session=ort.InferenceSession(str(root/self.config.model),providers=['CPUExecutionProvider'])
  self.default=np.asarray(self.config.default_angles_rad,dtype=np.float32)
  self.increment=np.asarray(self.config.target_increment_rad,dtype=np.float32)
  self.low=np.asarray(self.config.target_low_rad,dtype=np.float32);self.high=np.asarray(self.config.target_high_rad,dtype=np.float32)
  self.target=np.asarray(previous_commanded_target,dtype=np.float32).copy()
  assert self.target.shape==(14,) and np.isfinite(self.target).all()
 def step(self,gyro,projected_gravity,joint_position,joint_velocity):
  obs=np.concatenate([gyro,projected_gravity,np.asarray(joint_position)-self.default,joint_velocity,(self.target-self.default)/.12,np.zeros(13)]).astype(np.float32)
  assert obs.shape==(61,) and np.isfinite(obs).all()
  raw=self.session.run(['raw_action'],{'obs':obs[None,:]})[0][0]
  self.target=np.clip(self.target+self.increment*np.tanh(raw),self.low,self.high)
  return self.target.copy()

# Call step at 50Hz. All 14-joint vectors use INFERENCE_CONFIG joint_order.
# On handoff, initialize with the outgoing controller's LAST COMMANDED target.
# If the downstream controller further clips the command, store its actual
# commanded target back into controller.target before the next observation.
