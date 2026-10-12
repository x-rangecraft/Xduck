"""Read-only replay audit of joint targets and 1-kHz physical joint limits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import unilab.training.xduck_imitation as imitation
from unilab.training.xduck_curriculum import ROOT, load_actor


def run(config, checkpoint, output):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    actor = load_actor(checkpoint, config)
    real_make = imitation.make_env
    report = {"checkpoint": str(checkpoint), "tolerance_rad": 1e-6, "cases": {}}
    for name, cfg, seeds, count in (
        ("nominal", imitation.stage_config(config, 0), (208021,), 16),
        ("full_dr", config, (208021, 218021, 228021, 238021), 16),
    ):
        records = []

        def make(*args, **kwargs):
            env = real_make(*args, **kwargs)
            bounds = env._backend.get_joint_range().copy()
            stats = {}
            alive = np.ones(env.num_envs, dtype=bool)
            steps = [0]
            first = np.full(env.num_envs, -1.0)
            terminal_at = np.full(env.num_envs, -1.0)
            record = {
                "stats": stats,
                "joint_names": list(env.cfg.asset.policy_joint_names),
                "joint_limits_rad": bounds.tolist(),
                "target_limits_rad": np.stack((env._target_low, env._target_high), 1).tolist(),
            }
            records.append(record)

            def observe(key, values, limits):
                values = values[alive]
                if not len(values):
                    return
                margin = np.minimum(values - limits[:, 0], limits[:, 1] - values)
                violation = margin < -1e-6
                if key not in stats:
                    stats[key] = {
                        "joint_samples": 0,
                        "outside_samples": 0,
                        "per_joint_outside": np.zeros(14, dtype=np.int64),
                        "minimum_margin_rad": np.full(14, np.inf),
                        "minimum_rad": np.full(14, np.inf),
                        "maximum_rad": np.full(14, -np.inf),
                    }
                s = stats[key]
                s["joint_samples"] += values.size
                s["outside_samples"] += int(violation.sum())
                s["per_joint_outside"] += violation.sum(0)
                s["minimum_margin_rad"] = np.minimum(s["minimum_margin_rad"], margin.min(0))
                s["minimum_rad"] = np.minimum(s["minimum_rad"], values.min(0))
                s["maximum_rad"] = np.maximum(s["maximum_rad"], values.max(0))
                if key.startswith("actual"):
                    ids = np.flatnonzero(alive)[violation.any(1)]
                    first[ids[first[ids] < 0]] = steps[0] * env.cfg.ctrl_dt

            apply = env.apply_action
            begin = env._motor.begin_substep
            terminated = env._terminated

            def action(actions, state):
                record["direction"] = "descent" if state.info["commands"][0, 0] > 0.5 else "rise"
                raw = env.default_angles + actions
                target_limits = np.stack((env._target_low, env._target_high), 1)
                observe("raw_vs_target_limits", raw, target_limits)
                observe("raw_vs_joint_limits", raw, bounds)
                target = apply(actions, state)
                observe("biased_before_clip", target + env._ankle_target_offset, target_limits)
                return target

            def motor(target, pos, vel):
                observe("motor_target", target, bounds)
                observe("actual_1khz", pos, bounds)
                return begin(target, pos, vel)

            def done(up):
                observe("actual_control_end", env.get_dof_pos(), bounds)
                flags = terminated(up)
                terminal_at[alive & flags] = (steps[0] + 1) * env.cfg.ctrl_dt
                alive[:] &= ~flags
                steps[0] += 1
                record["first_physical_violation_seconds"] = first.tolist()
                record["terminal_seconds"] = terminal_at.tolist()
                return flags

            env.apply_action = action
            env._motor.begin_substep = motor
            env._terminated = done
            return env

        imitation.make_env = make
        try:
            windows = []
            for seed in seeds:
                metrics, _ = imitation.rollout(
                    cfg, ROOT, actor=actor, count=count, segments=2, seed=seed, independent=True
                )
                windows.append(metrics)
        finally:
            imitation.make_env = real_make
        report["cases"][name] = {"windows": windows, "records": [r for r in records if r["stats"]]}
        (output / "results.json").write_text(
            json.dumps(
                report, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v
            )
        )
        print(
            name,
            {
                k: sum(r["stats"][k]["outside_samples"] for r in records if k in r["stats"])
                for k in (
                    "raw_vs_target_limits",
                    "biased_before_clip",
                    "motor_target",
                    "actual_1khz",
                    "actual_control_end",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(OmegaConf.load(args.config), args.checkpoint.resolve(), args.output.resolve())
