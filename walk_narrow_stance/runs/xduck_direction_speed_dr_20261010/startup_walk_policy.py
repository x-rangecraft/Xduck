"""Experimental neural startup routing; retain the existing consumer history."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
class NeuralStartupWalk:
    def __init__(self,directory=None):
        directory=Path(directory) if directory is not None else Path(__file__).resolve().parent
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.startup=ort.InferenceSession(str(directory/'startup_distilled_20p0.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.walk=ort.InferenceSession(str(directory/'clean_r3_80.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.reset()
    def reset(self):
        # Reset only this clock when starting a new Walk activation. Keep the
        # standing consumer/motor history when constructing the 61 observations.
        self.ticks=0
    def infer(self,obs):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(1,61):raise ValueError('Expected real consumer observations shaped (1,61)')
        t=self.ticks*.02;self.ticks+=1
        if t>=1.2:return self.walk.run(None,{'obs':obs})[0]
        start=self.startup.run(None,{'obs':obs})[0]
        if t<=.6:return start
        u=np.clip((t-.6)/.6,0,1);u=u*u*u*(10+u*(-15+6*u))
        return (1-u)*start+u*self.walk.run(None,{'obs':obs})[0]
