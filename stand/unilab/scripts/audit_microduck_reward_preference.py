"""Compare standing and coordinated swing probes on the same DM4340 policy.

This is a read-only diagnostic of a fixed checkpoint. It does not train, alter
reward weights, or claim that an open-loop leg pulse is a walking controller.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rsl_rl as entry  # noqa: E402

CASES = {
    "policy_mean": (0.0, 0.0, 0.0),
    "one_left_06": (0.6, 0.6, 0.0),
    "alternate_06": (0.6, 0.6, 1.2),
    "alternate_10": (0.6, 1.0, 2.0),
    "alternate_small_08": (0.3, 0.8, 1.6),
    "alternate_medium_08": (0.4, 0.8, 1.6),
}


def swing_action(step: int, amplitude: float, swing_s: float, period_s: float) -> np.ndarray:
    """Mirrored hip/knee/ankle pulses on alternating legs."""
    action = np.zeros(14, dtype=np.float32)
    if amplitude == 0.0:
        return action
    t = step * 0.02
    if period_s == 0.0:
        if t >= swing_s:
            return action
        left = True
        phase = t / swing_s
    else:
        local = t % period_s
        left = local < period_s / 2.0
        phase = (local % (period_s / 2.0)) / swing_s
        if phase >= 1.0:
            return action
    pulse = float(np.sin(np.pi * phase) ** 2)
    if left:
        action[[2, 3, 4]] = pulse * amplitude * np.asarray([0.5, 1.0, 0.5])
    else:
        action[[11, 12, 13]] = -pulse * amplitude * np.asarray([0.5, 1.0, 0.5])
    return action


def main(checkpoint: Path, output: Path, *, num_envs: int, warmup: int, probe: int) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=microduck_dm4310_velocity_flat/mujoco",
                "training.device=cpu",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
            ],
        )
    env = entry.create_env(cfg, num_envs=num_envs)
    try:
        rl = entry._algo_config_dict(cfg)
        wrapped = entry._resolve_ppo_wrapper_cls(rl)(env, device="cpu")
        train_cfg = entry.normalize_ppo_train_cfg(rl)
        entry.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg["logger"] = "none"
        train_cfg.setdefault("runner", {})["logger"] = "none"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = entry.OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
            runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        command = np.zeros((num_envs, 13), dtype=np.float32)
        command[:, 0] = 0.2
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), command
        )
        summary = {}
        with torch.inference_mode():
            for name, (amplitude, swing_s, period_s) in CASES.items():
                np.random.seed(42)
                torch.manual_seed(42)
                wrapped.reset()
                rewards = []
                velocity = []
                contacts = []
                air_window = []
                falls = []
                torque = []
                reward_terms: dict[str, list[float]] = {}
                for step in range(warmup + probe):
                    env.state.info["commands"][:] = command
                    env.state.obs["obs"][:, -13:] = command
                    obs = wrapped.get_observations()
                    action = policy(obs).clone()
                    if step >= warmup:
                        pulse = swing_action(step - warmup, amplitude, swing_s, period_s)
                        action += torch.from_numpy(pulse)[None, :]
                    _, reward, _, _ = wrapped.step(action)
                    if step < warmup:
                        continue
                    rewards.append(reward.numpy().copy())
                    velocity.append(env.get_local_linvel()[:, 0].copy())
                    contacts.append(env._contact.sum(axis=1).copy())
                    air_window.append(
                        ((env._air_time > 0.125) & (env._air_time < 0.3)).mean(axis=1)
                    )
                    falls.append(env.state.terminated.copy())
                    torque.append(np.abs(env._motor.torque).copy())
                    for term in cfg.reward.scales:
                        key = f"Episode_Reward/{term}"
                        if key in env.state.info["log"]:
                            reward_terms.setdefault(term, []).append(
                                float(env.state.info["log"][key])
                            )
                r = np.asarray(rewards)
                v = np.asarray(velocity)
                c = np.asarray(contacts)
                a = np.asarray(air_window)
                f = np.asarray(falls)
                tau = np.asarray(torque)
                summary[name] = {
                    "mean_reward_rate": float(r.mean() / 0.02),
                    "mean_episode_reward_over_probe": float(r.sum(axis=0).mean()),
                    "mean_forward_speed_m_s": float(v.mean()),
                    "single_support_fraction": float(np.mean(c == 1)),
                    "both_airborne_fraction": float(np.mean(c == 0)),
                    "air_window_fraction": float(a.mean()),
                    "falls": int(f.sum()),
                    "torque_p99_nm": float(np.quantile(tau, 0.99)),
                    "reward_terms_rate": {
                        term: float(np.mean(values)) for term, values in reward_terms.items()
                    },
                }
                print(
                    name,
                    json.dumps(
                        {k: v for k, v in summary[name].items() if k != "reward_terms_rate"}
                    ),
                    flush=True,
                )
        result = {
            "checkpoint": str(checkpoint),
            "fixed_command_vx_m_s": 0.2,
            "num_envs": num_envs,
            "warmup_s": warmup * 0.02,
            "probe_s": probe * 0.02,
            "method": "deterministic policy with additive open-loop leg pulses; resets use the same seed but dynamic contact causes trajectories to diverge",
            "cases": summary,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--probe", type=int, default=300)
    args = parser.parse_args()
    main(
        args.checkpoint.resolve(),
        args.output.resolve(),
        num_envs=args.num_envs,
        warmup=args.warmup,
        probe=args.probe,
    )
