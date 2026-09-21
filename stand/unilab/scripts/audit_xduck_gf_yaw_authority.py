"""Measure local XDuck hip-yaw action signs on the loaded GF plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "hold": (0.0, 0.0),
    "left_plus": (0.04, 0.0),
    "right_plus": (0.0, 0.04),
    "both_plus": (0.04, 0.04),
    "differential": (0.04, -0.04),
}


def evaluate(*, per_case: int = 8, steps: int = 25) -> dict:
    np.random.seed(20260917)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=xduck_gf43x40_velocity_flat/mujoco",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    env = entry.create_env(
        cfg,
        num_envs=len(CASES) * per_case,
        env_cfg_override=entry.build_ppo_env_cfg_override(cfg),
    )
    try:
        env._autoreset = False
        env.init_state()
        action = np.zeros((env.num_envs, 14), dtype=np.float32)
        masks = {}
        for index, (name, (left, right)) in enumerate(CASES.items()):
            section = slice(index * per_case, (index + 1) * per_case)
            action[section, 0] = left / env.cfg.policy_action_scale_rad
            action[section, 9] = right / env.cfg.policy_action_scale_rad
            masks[name] = section
        yaw_integral = np.zeros(env.num_envs)
        yaw_at_early = np.zeros(env.num_envs)
        falls = np.zeros(env.num_envs, dtype=bool)
        for step in range(steps):
            state = env.step(action)
            yaw = env.get_gyro()[:, 2]
            yaw_integral += yaw * env.cfg.ctrl_dt
            if step == 4:
                yaw_at_early[:] = yaw
            falls |= state.terminated
        return {
            "action_offset_rad": CASES,
            "duration_s": steps * env.cfg.ctrl_dt,
            "per_case": per_case,
            "cases": {
                name: {
                    "early_yaw_rate_rad_s": float(np.mean(yaw_at_early[section])),
                    "integrated_yaw_rad": float(np.mean(yaw_integral[section])),
                    "falls": int(np.sum(falls[section])),
                }
                for name, section in masks.items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-case", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    result = evaluate(per_case=args.per_case, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
