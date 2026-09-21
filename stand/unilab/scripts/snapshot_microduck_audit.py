"""Materialize the effective three-task inputs for the offline physics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from unilab.envs.locomotion.microduck_dm4310.motor import Dm4310MitBatchModel, Dm4340MitConfig
from unilab.envs.locomotion.microduck_dm4310.sitstand import (
    SIT_OVERRIDES,
    SIT_Z,
    STAND_Z,
    MicroDuckDm4310SitStandCfg,
)
from unilab.envs.locomotion.microduck_dm4310.standup import MicroDuckDm4310StandUpCfg
from unilab.envs.locomotion.microduck_dm4310.velocity import (
    HEAD_COM_BODY_NAMES,
    MicroDuckDm4310VelocityCfg,
)


def snapshot():
    result = {}
    root = Path(__file__).resolve().parents[1]
    for name, cls in [
        ("velocity", MicroDuckDm4310VelocityCfg),
        ("standup", MicroDuckDm4310StandUpCfg),
        ("sitstand", MicroDuckDm4310SitStandCfg),
    ]:
        cfg = cls()
        owner = yaml.safe_load(
            (root / f"conf/ppo/task/microduck_dm4310_{name}_flat/mujoco.yaml").read_text()
        )
        if owner["reward"] != asdict(cfg.reward_config) or owner["env"]["curriculum"] != asdict(
            cfg.curriculum
        ):
            raise ValueError(f"{name}: owner YAML and config defaults differ")
        result[name] = {
            key: asdict(getattr(cfg, key))
            for key in ["commands", "curriculum", "domain_rand", "latency"]
        }
        result[name]["reward"] = asdict(cfg.reward_config)
        result[name]["algo"] = owner["algo"]
        result[name]["head_com_body_names"] = list(HEAD_COM_BODY_NAMES)
        result[name]["reset_clearance_m"] = cfg.reset_clearance_m
        result[name]["posture_reset_joint_jitter_rad"] = cfg.posture_reset_joint_jitter_rad
    motor = Dm4310MitBatchModel(1, 14, 0.005, cfg=Dm4340MitConfig())
    result["motor"] = {
        "cfg": asdict(Dm4340MitConfig()),
        "curve": [
            [float(v), float(motor.torque_speed_limit(np.full((1, 14), v))[0, 0])]
            for v in np.linspace(0, 6, 601)
        ],
    }
    result["targets"] = {"stand_z": STAND_Z, "sit_z": SIT_Z, "sit_overrides": SIT_OVERRIDES}
    paths = list((root / "src/unilab/envs/locomotion/microduck_dm4310").glob("*.py"))
    paths += list((root / "src/unilab/assets/robots/microduck_dm4310").glob("*.xml"))
    result["sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot(), indent=2) + "\n")
