"""Generate the explicit scale-adapted profile from the real, unchanged models.

All MuJoCo/asset work is offline. No model files or actuator settings are edited.
"""

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from unilab.envs.locomotion.microduck_dm4310.velocity import POLICY_JOINT_NAMES
from unilab.envs.locomotion.microduck_xl330_official.velocity import OfficialCurriculumConfig
from unilab.envs.locomotion.xduck_gf43x40.scaling import (
    matched_momentum_weight,
    xml_model_fingerprint,
)
from unilab.envs.locomotion.xduck_gf43x40.velocity import (
    _SOURCE_SCENE,
    SCALE,
    XDuckCurriculumConfig,
    XDuckGF43X40VelocityCfg,
)

ROOT = Path(__file__).resolve().parents[1]


def momentum_energy(
    path, time_scale, joint_names=POLICY_JOINT_NAMES, base_name="trunk_base", keyframe="STAND"
):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    home = model.key(keyframe).qpos.copy()
    joints = np.array([model.joint(name).qposadr[0] for name in joint_names])
    dofs = np.r_[np.arange(3, 6), [model.joint(name).dofadr[0] for name in joint_names]]
    offsets = [np.zeros(14)]
    for joint in range(14):
        for sign in [-1, 1]:
            delta = np.zeros(14)
            delta[joint] = sign * 0.08
            offsets.append(delta)
    values = []
    for delta in offsets:
        data.qpos[:] = home
        data.qpos[joints] = np.clip(
            home[joints] + delta, model.jnt_range[1:, 0] + 0.01, model.jnt_range[1:, 1] - 0.01
        )
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        matrix = np.zeros((3, model.nv))
        mujoco.mj_angmomMat(model, data, matrix, model.body(base_name).id)
        # E[||A qdot||²] for unit-covariance angular velocities; XDuck slowed by T.
        values.append(float(np.sum(np.square(matrix[:, dofs] / time_scale))))
    return float(np.mean(values)), {
        "mass_kg": float(mujoco.mj_getTotalmass(model)),
        "stand_root_height_m": float(home[2]),
        "subtree_mass_kg": {
            name: float(model.body_subtreemass[model.joint(name).bodyid[0]]) for name in joint_names
        },
        "postures": len(values),
        "mean_unit_velocity_momentum_energy": float(np.mean(values)),
        "fingerprint": xml_model_fingerprint(path),
    }


def build():
    t = math.sqrt(SCALE)
    reference = ROOT.parent / "mjlab/src/mjlab_microduck/robot/microduck/scene_walk.xml"
    robot = _SOURCE_SCENE
    er, mr = momentum_energy(reference, 1.0)
    asset = XDuckGF43X40VelocityCfg().asset
    ex, mx = momentum_energy(robot, t, asset.policy_joint_names, asset.base_name, keyframe="HOME")
    # V1.1.2 is a new URDF geometry; SCALE remains the nominal task length scale.
    weight = matched_momentum_weight(er, ex)
    curriculum = dataclasses.asdict(XDuckCurriculumConfig())
    for key, stages in curriculum.items():
        if key.endswith("_stages"):
            for stage in stages:
                stage["step"] = int(round(stage["step"] * t))
    # Angular tolerance is wall-clock based; the robot did not slow all body rates by T.
    curriculum["angular_velocity_std_stages"] = [{"step": 0, "value": math.sqrt(0.5)}]
    profile = {"defaults": ["xduck_grouped_baseline_v1", "_self_"]}
    profile["algo"] = {
        "num_steps_per_env": round(24 * t),
        "algorithm": {"gamma": 0.99 ** (1 / t), "lam": 0.95 ** (1 / t)},
    }
    profile["env"] = {
        "max_episode_seconds": math.ceil(20 * t / 0.02) * 0.02,
        "curriculum": curriculum,
    }
    profile["reward"] = {
        "action_rate_units": "physical_target_rate",
        "action_rate_time_scale": t,
        "action_rate_reference_dt": 0.02,
        "head_bias_tau_s": t,
        "angular_velocity_std": math.sqrt(0.5),
        "scale_model_fingerprint": mx["fingerprint"],
        "scales": {
            "angular_momentum": weight,
            "body_ang_vel": -0.05,
            "foot_swing_height": -0.25 * t,
        },
    }
    provenance = {
        "length_ratio": SCALE,
        "motion_time_scale": t,
        "reference": mr,
        "xduck": mx,
        "angular_momentum_weight": weight,
        "normalization_definition": "29 joint-limit-bounded perturbations around each model home; independent unit-variance root/joint angular velocities, slowed by T in XDuck. Matches mean squared centroidal momentum, not every direction/posture.",
        "models_modified": False,
        "angular_tolerance_choice": "Original wall-clock tolerance and body-angular-rate cost, not a claim of full dynamic similarity.",
        "rollout_rounding_fraction": round(24 * t) / (24 * t) - 1,
    }
    return profile, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    profile, provenance = build()
    out = ROOT / "conf/ppo/experiment/xduck_v112_scale_adapted_v1.yaml"
    data = (
        "# @package _global_\n# Generated by scripts/build_xduck_scale_profile.py. Old checkpoint replay uses the legacy profile.\n"
        + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    )
    manifest = ROOT / "docs/pretraining_audit/v1_1_2/model_normalization.json"
    if args.check:
        assert out.read_text() == data, "Scale profile is stale; regenerate it"
        assert json.loads(manifest.read_text()) == provenance
    else:
        out.write_text(data)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": str(out),
                "momentum_weight": provenance["angular_momentum_weight"],
                "reference_mass": provenance["reference"]["mass_kg"],
                "xduck_mass": provenance["xduck"]["mass_kg"],
                "rollout_steps": profile["algo"]["num_steps_per_env"],
            }
        )
    )


if __name__ == "__main__":
    main()
