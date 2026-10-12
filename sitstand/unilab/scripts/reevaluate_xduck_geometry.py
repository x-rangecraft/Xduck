"""Evaluate recorded physics with physical head uprightness, no pose matching."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
from unilab.envs.locomotion.xduck_gf43x40.acceptance import GeometricAcceptance, geometric_success, upright_tilt

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--trajectory", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--stand-height",type=float,default=.3129)
p.add_argument("--sit-height",type=float,default=.2182)
a = p.parse_args()
r = np.load(a.trajectory)
m = mujoco.MjModel.from_binary_path(str(a.model))
d = mujoco.MjData(m)
head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_roll")
assert head >= 0
# Calibrate the CAD-local up axis in the level, zero-head-angle reference.
# Leg angles do not affect the head chain; this is axis calibration, not an
# acceptance requirement on the live joint angles.
d.qpos[:] = 0
d.qpos[2] = a.stand_height
d.qpos[3] = 1
mujoco.mj_fwdPosition(m, d)
head_local_up = d.xmat[head].reshape(3, 3).T @ np.array([0., 0., 1.])
cfg = GeometricAcceptance()
rows = []
for case, goal in [("stand_hold", a.stand_height), ("sit_hold", a.sit_height), ("rise", a.stand_height), ("descent", a.sit_height)]:
    for seed in (6401, 7401):
        key = f"{case}_{seed}_"
        poses = r[key + "qpos"]
        tilt = np.empty(poses.shape[:2])
        for step in range(len(poses)):
            for trial in range(poses.shape[1]):
                d.qpos[:] = poses[step, trial]
                mujoco.mj_fwdPosition(m, d)
                tilt[step, trial] = upright_tilt(d.xmat[head].reshape(3, 3), head_local_up)
        success, conditions = geometric_success(r[key + "height"] - goal, r[key + "tilt"], tilt,
                                                r[key + "base_speed"], r[key + "qvel"], cfg=cfg)
        fallen = (r[key + "height"] < .13).any(axis=0) | (r[key + "tilt"] > np.deg2rad(75)).any(axis=0)
        success &= ~fallen
        rows.append(dict(case=case, seed=seed, trials=poses.shape[1], successes=int(success.sum()),
                         falls=int(fallen.sum()), late_head_tilt_mean_deg=float(np.rad2deg(tilt[-50:]).mean()),
                         last_second_condition_violations={name: int((~mask[-50:].all(axis=0)).sum()) for name, mask in conditions.items()}))
result = dict(criterion="Geometric height/body/head uprightness and stationarity, no joint pose matching; continuous hold0.5s and success throughout final1s; original fall mask retained.",
              thresholds=asdict(cfg), head_body="head_roll", head_local_up=head_local_up.tolist(),
              reference="Zero head/neck joints, upright root; head axis calibrated with exact frozen model.",
              trajectory=str(a.trajectory),height_targets_m={'stand':a.stand_height,'sit':a.sit_height},rows=rows)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
