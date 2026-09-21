"""Inspect actual owner reset plans at every curriculum boundary without training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import yaml

from unilab.base import registry


def audit():
    registry.ensure_registries()
    rows = []
    root = Path(__file__).resolve().parents[1]
    for kind in ["Velocity", "StandUp", "SitStand"]:
        np.random.seed(20260910)
        env = registry.make(f"MicroDuckDm4310{kind}Flat", "mujoco", num_envs=64)
        try:
            owner_name = kind.lower()
            owner = yaml.safe_load(
                (
                    root
                    / f"conf/ppo/task/microduck_dm4310_{owner_name}_flat/mujoco.yaml"
                ).read_text()
            )
            rollout_steps = int(owner["algo"]["num_steps_per_env"])
            env.init_state()
            env._cfg.curriculum.step_cap = None
            cur = asdict(env._cfg.curriculum)
            steps = sorted(
                {x["step"] for stages in cur.values() if isinstance(stages, list) for x in stages}
            )
            model = mujoco.MjModel.from_xml_path(env._cfg.scene.model_file)
            data = mujoco.MjData(model)
            floor = model.geom("floor").id
            limits = env._backend.get_joint_range()
            for step in steps:
                env.step_counter = step
                env._update_curriculum()
                plan = env._command_provider.build_reset_plan(env, np.arange(64, dtype=np.int32))
                penetration = []
                self_penetration = []
                limit_violation = []
                for q in plan.qpos:
                    data.qpos[:] = q
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    penetration.append(
                        max(
                            [
                                max(0.0, -c.dist)
                                for c in data.contact
                                if floor in (c.geom1, c.geom2)
                            ],
                            default=0.0,
                        )
                    )
                    self_penetration.append(
                        max(
                            [
                                max(0.0, -c.dist)
                                for c in data.contact
                                if floor not in (c.geom1, c.geom2)
                            ],
                            default=0.0,
                        )
                    )
                    limit_violation.append(
                        bool(np.any((q[7:] < limits[:, 0]) | (q[7:] > limits[:, 1])))
                    )
                rows.append(
                    {
                        "task": kind,
                        "iteration": step / rollout_steps,
                        "samples": 64,
                        "floor_penetration_max_m": float(np.max(penetration)),
                        "self_penetration_max_m": float(np.max(self_penetration)),
                        "fraction_self_penetrating_over_5mm": float(
                            np.mean(np.asarray(self_penetration) > 0.005)
                        ),
                        "floor_penetration_p95_m": float(np.percentile(penetration, 95)),
                        "fraction_penetrating_over_5mm": float(
                            np.mean(np.asarray(penetration) > 0.005)
                        ),
                        "fraction_joint_limit_violation": float(np.mean(limit_violation)),
                    }
                )
        finally:
            env.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(), indent=2) + "\n")
