"""Estimate GF MIT gains from XL330 small-signal dynamics and model inertia.

The result is a starting range, not a whole-robot actuator calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from unilab.actuators.gf43x40 import GF43X40Parameters
from unilab.envs.locomotion.microduck_dm4310.velocity import POLICY_JOINT_NAMES
from unilab.envs.locomotion.microduck_xl330_official.bam import Xl330BamBatchModel

ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    ROOT.parent / "mjlab/src/mjlab_microduck/robot/microduck/scene_walk.xml",
    ROOT.parent / "Model/Duck_V1.1.0/scene.xml",
)


def plant_data(scene: Path, armature: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key("STAND").qpos
    dofs = np.array([model.joint(name).dofadr[0] for name in POLICY_JOINT_NAMES])
    model.dof_armature[dofs] = armature
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    full_mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_mass)
    inverse_mass = np.linalg.inv(full_mass)
    return {
        "scene": str(scene),
        "total_mass_kg": float(mujoco.mj_getTotalmass(model)),
        "stand_root_height_m": float(data.qpos[2]),
        "joint_effective_inertia_kg_m2": (1.0 / np.diag(inverse_mass)[dofs]).tolist(),
        "joint_locked_inertia_kg_m2": np.diag(full_mass)[dofs].tolist(),
    }


def evaluate() -> dict:
    gf = GF43X40Parameters.from_bundle()
    original = plant_data(SCENES[0], Xl330BamBatchModel.armature)
    xduck = plant_data(SCENES[1], gf.physics["armature"])
    length_ratio = xduck["stand_root_height_m"] / original["stand_root_height_m"]
    inertia_ratio = np.asarray(xduck["joint_effective_inertia_kg_m2"]) / np.asarray(
        original["joint_effective_inertia_kg_m2"]
    )
    xl_stiffness = (
        Xl330BamBatchModel.kp_fw
        * Xl330BamBatchModel.error_gain
        * 7.4
        * Xl330BamBatchModel.kt
        / Xl330BamBatchModel.resistance
    )
    xl_damping = (
        Xl330BamBatchModel.kt**2 / Xl330BamBatchModel.resistance
        + Xl330BamBatchModel.friction_viscous
    )
    gf_viscous = float(gf.physics["viscous"])
    leg_indices = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)

    def candidate(time_ratio: float) -> dict:
        kp = xl_stiffness * inertia_ratio / time_ratio**2
        kd = np.maximum(xl_damping * inertia_ratio / time_ratio - gf_viscous, 0.0)
        return {
            "per_joint_kp": kp.tolist(),
            "per_joint_kd": kd.tolist(),
            "leg_median_kp": float(np.median(kp[list(leg_indices)])),
            "leg_median_kd": float(np.median(kd[list(leg_indices)])),
            "leg_kp_p10_p90": np.quantile(kp[list(leg_indices)], [0.1, 0.9]).tolist(),
            "leg_kd_p10_p90": np.quantile(kd[list(leg_indices)], [0.1, 0.9]).tolist(),
        }

    return {
        "joint_names": POLICY_JOINT_NAMES,
        "official": original,
        "xduck": xduck,
        "length_ratio": length_ratio,
        "effective_inertia_ratio": inertia_ratio.tolist(),
        "xl330_linearized_stiffness_at_7p4v_nm_rad": xl_stiffness,
        "xl330_linearized_damping_nm_s_rad": xl_damping,
        "gf_viscous_nm_s_rad": gf_viscous,
        "same_wall_clock_bandwidth": candidate(1.0),
        "gravity_similar_time_sqrt_length": candidate(math.sqrt(length_ratio)),
        "caveat": "Floating-base joint effective inertia at STAND; no contact constraint, motor lag, force envelope, or foot-impact model.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
