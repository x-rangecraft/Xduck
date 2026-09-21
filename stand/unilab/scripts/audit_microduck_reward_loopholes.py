"""Adversarial reward audit for stable-idle, spin, hop, and motor-limit loopholes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from unilab.algos.torch.bounded_gaussian import BoundedGaussianDistribution
from unilab.base import registry
from unilab.envs.locomotion.microduck_dm4310.velocity import (
    MicroDuckDm4310VelocityCfg,
    velocity_tracking_scores,
)

COMMANDS = {
    "idle": [0.0, 0.0, 0.0],
    "forward_02": [0.2, 0.0, 0.0],
    "backward_02": [-0.2, 0.0, 0.0],
    "lateral_left_02": [0.0, 0.2, 0.0],
    "lateral_right_02": [0.0, -0.2, 0.0],
    "turn_left_10": [0.0, 0.0, 1.0],
    "turn_right_10": [0.0, 0.0, -1.0],
}


def zero_action_basins(*, per_bucket: int = 32, steps: int = 500) -> dict:
    registry.ensure_registries()
    env = registry.make(
        "MicroDuckDm4310VelocityFlat", "mujoco", num_envs=per_bucket * len(COMMANDS)
    )
    try:
        env._cfg.noise_config.level = 0.0
        env._cfg.domain_rand.velocity_pushes = False
        env._mass_inertia_scale[:] = 1.0
        state = env.init_state()
        command = np.zeros((env.num_envs, 13), dtype=np.float32)
        labels = []
        for index, (name, value) in enumerate(COMMANDS.items()):
            section = slice(index * per_bucket, (index + 1) * per_bucket)
            command[section, :3] = value
            labels.extend([name] * per_bucket)
        labels = np.asarray(labels)
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), command
        )
        reward_sum = np.zeros(env.num_envs)
        falls = np.zeros(env.num_envs)
        samples = np.zeros(env.num_envs)
        for step in range(steps):
            state.info["commands"][:] = command
            env._velocity_timer[:] = 1_000_000
            env._head_timer[:] = 1_000_000
            env._body_timer[:] = 1_000_000
            state = env.step(np.zeros((env.num_envs, 14), dtype=np.float32))
            valid = ~state.terminated
            if step >= 100:
                reward_sum += state.reward / env._cfg.ctrl_dt * valid
                samples += valid
            falls += state.terminated
        result = {}
        for name in COMMANDS:
            mask = labels == name
            denominator = np.maximum(samples[mask], 1.0)
            result[name] = {
                "reward_rate_after_2s": float(np.mean(reward_sum[mask] / denominator)),
                "falls_per_10s_mean": float(np.mean(falls[mask]) * 500.0 / steps),
            }
        return result
    finally:
        env.close()


def analytic_slices() -> dict:
    cfg = MicroDuckDm4310VelocityCfg().reward_config
    errors = np.asarray([0.0, 0.15, 0.30, 0.60, 1.0, 1.3, 1.5, 2.0])
    gyro = np.zeros((len(errors), 3))
    gyro[:, 2] = errors
    _, scores = velocity_tracking_scores(
        np.zeros_like(gyro),
        gyro,
        np.zeros_like(gyro),
        cfg.linear_velocity_std,
        cfg.angular_velocity_std,
    )
    durations = np.asarray([0.10, 0.175, 0.25, 0.30, 0.45, 0.60])
    integrated = cfg.scales["air_time"] * np.maximum(
        0, np.minimum(durations, cfg.air_time_threshold_max) - cfg.air_time_threshold_min
    )
    return {
        "active_reward_scales": cfg.scales,
        "yaw_error_slice": [
            {
                "error_rad_s": float(e),
                "weighted_reward": float(cfg.scales["track_angular_velocity"] * r),
            }
            for e, r in zip(errors, scores, strict=True)
        ],
        "air_time_per_foot_continuous_integral": [
            {"flight_seconds": float(t), "reward": float(r)}
            for t, r in zip(durations, integrated, strict=True)
        ],
        "scope": "Current official-formula baseline. Double flight is not specially gated; no extra progress/composite or physical reward penalties. Motor constraints and audit metrics remain active.",
    }


def policy_envelope() -> dict:
    owner = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "conf/ppo/task/microduck_dm4310_velocity_flat/mujoco.yaml"
        ).read_text()
    )
    policy = owner["algo"]["policy"]
    generator = torch.Generator().manual_seed(20260910)
    rows = []
    for action_std in (policy["min_std"], policy["init_noise_std"], policy["max_std"]):
        distribution = BoundedGaussianDistribution(
            14,
            init_std=action_std,
            min_std=max(action_std * 0.5, 1.0e-4),
            max_std=action_std * 1.5,
            action_low=policy["action_low"],
            action_high=policy["action_high"],
        )
        distribution.update(torch.zeros((200_000, 14)))
        # Normal.sample has no generator argument; fork the global state so the
        # audit is deterministic without changing the training RNG contract.
        with torch.random.fork_rng():
            torch.random.set_rng_state(generator.get_state())
            action = distribution.sample()
            generator.set_state(torch.random.get_rng_state())
        requested_torque = torch.clamp(60.0 * action, -23.5, 23.5)
        rows.append(
            {
                "action_std_rad": action_std,
                "sample_count": int(action.numel()),
                "abs_action_p95_rad": float(torch.quantile(torch.abs(action), 0.95)),
                "requested_above_continuous_torque_fraction": float(
                    torch.mean((torch.abs(requested_torque) > 8.9).float())
                ),
                "requested_at_peak_torque_fraction": float(
                    torch.mean((torch.abs(requested_torque) >= 23.5).float())
                ),
                "action_outside_configured_bounds": int(
                    torch.count_nonzero(
                        (action < torch.as_tensor(policy["action_low"]))
                        | (action > torch.as_tensor(policy["action_high"]))
                    )
                ),
            }
        )
    return {
        "entropy_coefficient": owner["algo"]["algorithm"]["entropy_coef"],
        "std_contract_rad": {
            "minimum": policy["min_std"],
            "initial": policy["init_noise_std"],
            "maximum": policy["max_std"],
        },
        "zero_latent_action": BoundedGaussianDistribution(
            14,
            init_std=policy["init_noise_std"],
            min_std=policy["min_std"],
            max_std=policy["max_std"],
            action_low=policy["action_low"],
            action_high=policy["action_high"],
        )
        .deterministic_output(torch.zeros((1, 14)))
        .tolist()[0],
        "monte_carlo_zero_mean": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-gate", type=Path)
    args = parser.parse_args()
    result = {
        "scope": "Adversarial reward slices plus nominal zero-action basins; no learning.",
        "zero_action_basins": zero_action_basins(),
        "analytic_slices": analytic_slices(),
        "policy_envelope": policy_envelope(),
    }
    if args.legacy_gate:
        legacy = json.loads(args.legacy_gate.read_text())
        result["legacy_bad_policy"] = {
            name: {
                "reward_rate_mean": bucket["reward_rate_mean"],
                "actual_vx_vy_yaw_mean": bucket["actual_vx_vy_yaw_mean"],
                "rated_torque_exceed_fraction": bucket["rated_torque_exceed_fraction"],
                "no_load_speed_exceed_fraction": bucket["no_load_speed_exceed_fraction"],
                "airborne_fraction": bucket["airborne_fraction"],
            }
            for name, bucket in legacy["buckets"].items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
