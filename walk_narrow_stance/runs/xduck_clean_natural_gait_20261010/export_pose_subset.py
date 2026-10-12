from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parent
prefix, out = sys.argv[1:3]
ids = np.array([int(v) for v in sys.argv[3].split(',')])
fields = {'geom_pos', 'geom_quat', 'jnt_stiffness', 'dof_damping',
          'body_mass', 'body_inertia', 'body_ipos', 'geom_friction',
          'dof_armature', 'names', 'source_indices'}
for suffix in ['_trace.npz', '_stand_trace.npz']:
    z = np.load(root/(prefix+suffix))
    data = {}
    for k in z.files:
        if k == 'names':
            cases = z['qpos'].shape[1] // len(z[k])
            data[k] = z[k][ids // cases]
        elif k in fields:
            data[k] = z[k][ids]
        else:
            data[k] = z[k][:, ids]
    np.savez_compressed(root/(out+suffix), **data, original_indices=ids)
print(out)
