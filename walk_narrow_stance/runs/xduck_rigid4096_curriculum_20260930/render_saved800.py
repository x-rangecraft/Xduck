from pathlib import Path
import os
os.environ['MUJOCO_GL']='egl'
import json,subprocess
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;TAG='checkpoint800_dr2';OUT=R/'videos';z=np.load(OUT/(TAG+'.npz'));report=json.loads((OUT/(TAG+'.json')).read_text());geom_pos=z['geom_pos'];geom_quat=z['geom_quat'];failure=[c['failure_s'] for c in report['cases']]
s=(R/'record_checkpoint800.py').read_text();start=s.index("m=mujoco.MjModel.from_binary_path(str(R/'physics/effective_training.mjb'))")
exec(compile(s[start:],str(R/'record_checkpoint800.py'),'exec'))
