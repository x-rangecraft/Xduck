"""Reproduce nominal physical calibration and inspect final XL330 ONNX contracts.

This is a calibration experiment, not a learned-policy success evaluation.
The transition probe deliberately tests open-loop joint interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from unilab.envs.locomotion.microduck_dm4310.motor import Dm4310MitBatchModel, Dm4340MitConfig
from unilab.envs.locomotion.microduck_dm4310.sitstand import SIT_OVERRIDES
from unilab.envs.locomotion.microduck_dm4310.velocity import (
    CurriculumConfig,
    MicroDuckDm4310VelocityCfg,
)


def probe(start_sitting: bool, ramp_s: float | None, seed: int = 0, head_command=None) -> dict:
    np.random.seed(seed)
    cfg = MicroDuckDm4310VelocityCfg()
    path = Path(cfg.scene.model_file).with_name("scene_posture.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = cfg.sim_dt
    data = mujoco.MjData(model)
    data.qpos[:] = model.key("home").qpos
    home = data.qpos[7:].copy()
    if head_command is not None:
        home[5:9] += head_command
    sit = home.copy()
    for index, value in SIT_OVERRIDES.items():
        sit[index] = value
    start, end = (sit, home) if start_sitting else (home, sit)
    data.qpos[7:] = start
    data.qpos[2] = 0.150 if start_sitting else 0.2508
    if seed:
        data.qpos[7:] += np.random.normal(0, 0.01, 14)
        angle = np.deg2rad(np.random.uniform(-1, 1))
        data.qpos[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    mujoco.mj_forward(model, data)
    center = data.subtree_com[1]
    inertia = np.zeros((3, 3))
    for body in range(1, model.nbody):
        r = data.xipos[body] - center
        rotation = data.ximat[body].reshape(3, 3)
        inertia += rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        inertia += model.body_mass[body] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    # Replace XML position servos with the training owner's torque plant.
    model.actuator_gainprm[:] = 0
    model.actuator_gainprm[:, 0] = 1
    model.actuator_biasprm[:] = 0
    motor = Dm4310MitBatchModel(1, 14, cfg.sim_dt, cfg=Dm4340MitConfig())
    history = []
    target = start.copy()
    for step in range(round((6 + (ramp_s or 0)) / cfg.sim_dt)):
        if step % round(cfg.ctrl_dt / cfg.sim_dt) == 0:
            alpha = 0 if ramp_s is None else np.clip((data.time - 3) / ramp_s, 0, 1)
            target = start + alpha * (end - start)
        data.ctrl[:] = motor.compute(target[None], data.qpos[None, 7:], data.qvel[None, 6:])[0]
        mujoco.mj_step(model, data)
        history.append(
            [
                data.time,
                data.qpos[2],
                data.xmat[1, 8],
                np.max(np.abs(data.ctrl)),
                np.max(np.abs(data.qvel[6:])),
            ]
        )
    rows = np.asarray(history)
    final = rows[rows[:, 0] > rows[-1, 0] - 1]
    motion = rows[rows[:, 0] >= 3]
    return dict(
        head_command=head_command,
        final_qpos=data.qpos.tolist(),
        final_qvel=data.qvel.tolist(),
        final_self_pairs=sum(
            float(data.sensor(f"self_pair_{i:03d}").data[0]) > 0 for i in range(45)
        ),
        start="sit" if start_sitting else "stand",
        seed=seed,
        ramp_s=ramp_s,
        mass_kg=float(model.body_mass.sum()),
        inertia_kg_m2=inertia.tolist(),
        final_height_m=float(np.mean(final[:, 1])),
        final_min_upright=float(np.min(final[:, 2])),
        motion_min_upright=float(np.min(motion[:, 2])),
        peak_torque_nm=float(np.max(motion[:, 3])),
        peak_joint_speed_rad_s=float(np.max(motion[:, 4])),
    )


def inspect_onnx(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.zeros((1, 61), dtype=np.float32)
    obs[0, 5] = -1
    action = session.run(None, {session.get_inputs()[0].name: obs})[0]
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inputs=[{"name": x.name, "shape": x.shape} for x in session.get_inputs()],
        outputs=[{"name": x.name, "shape": x.shape} for x in session.get_outputs()],
        metadata=session.get_modelmeta().custom_metadata_map,
        neutral_input_action_finite=bool(np.isfinite(action).all()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=Path("../mjlab/models_123"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-holds", action="store_true")
    args = parser.parse_args()
    if args.stage_holds:
        results = []
        for i, stage in enumerate(CurriculumConfig().head_range_stages):
            for sitting in [False, True]:
                for head in [[0.0, 0.0, 0.0, 0.0], *itertools.product(*stage["ranges"])]:
                    result = probe(sitting, None, head_command=list(head))
                    result["head_stage_index"] = i
                    results.append(result)
            print("hold stage", i, flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        return
    result = dict(
        mujoco_version=mujoco.__version__,
        sim_dt=0.005,
        ctrl_dt=0.02,
        scope="Nominal motor plant; no transport latency or DR. Not policy success.",
        holds=[probe(sit, None, seed) for sit in (False, True) for seed in range(3)],
        open_loop_transitions=[
            probe(sit, ramp) for sit in (False, True) for ramp in (2.0, 2.9, 3.0, 4.0)
        ],
        policies=[
            inspect_onnx(args.policies / name)
            for name in ("alpha_walking.onnx", "alpha_stand.onnx", "alpha_sitstand.onnx")
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
