"""Measure command sensitivity of XDuck actor on identical physical states."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "idle": [0.0, 0.0, 0.0],
    "forward_02": [0.2, 0.0, 0.0],
    "backward_02": [-0.2, 0.0, 0.0],
    "turn_left_10": [0.0, 0.0, 1.0],
    "turn_right_10": [0.0, 0.0, -1.0],
}
LEG_IDS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def evaluate(
    checkpoint: Path, *, num_envs: int = 16, warmup: int = 100, warmup_forward: bool = False
) -> dict:
    np.random.seed(20260917)
    torch.manual_seed(20260917)
    torch.set_num_threads(1)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=xduck_gf43x40_velocity_flat/mujoco",
                "training.device=cpu",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    env = entry.create_env(
        cfg, num_envs=num_envs, env_cfg_override=entry.build_ppo_env_cfg_override(cfg)
    )
    try:
        rl_cfg = entry._algo_config_dict(cfg)
        wrapped = entry._resolve_ppo_wrapper_cls(rl_cfg)(env, device="cpu")
        train_cfg = entry.normalize_ppo_train_cfg(rl_cfg)
        entry.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg["logger"] = "none"
        train_cfg.setdefault("runner", {})["logger"] = "none"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = entry.OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
            runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        zero_commands = np.zeros((num_envs, 13), dtype=np.float32)
        if warmup_forward:
            zero_commands[:, 0] = 0.2
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), zero_commands
        )
        wrapped.reset()
        with torch.inference_mode():
            for _ in range(warmup):
                env.state.info["commands"][:] = zero_commands
                env.state.obs["obs"][:, -13:] = zero_commands
                action = policy(wrapped.get_observations())
                wrapped.step(action)
            base = wrapped.get_observations()
            outputs = {}
            for name, command in COMMANDS.items():
                observation = base.clone()
                for group in ("actor", "policy"):
                    observation[group][:, -13:] = 0.0
                    observation[group][:, -13:-10] = torch.tensor(command)
                outputs[name] = policy(observation).cpu().numpy() * env.cfg.policy_action_scale_rad
            forward = base.clone()
            for group in ("actor", "policy"):
                forward[group][:, -13:] = 0.0
                forward[group][:, -13:-10] = torch.tensor(COMMANDS["forward_02"])
            h = 0.02
            jacobian = np.zeros((14, 14), dtype=np.float64)
            for joint in range(14):
                plus = forward.clone()
                minus = forward.clone()
                for group in ("actor", "policy"):
                    plus[group][:, 34 + joint] += h
                    minus[group][:, 34 + joint] -= h
                a_plus = policy(plus).cpu().numpy() * env.cfg.policy_action_scale_rad
                a_minus = policy(minus).cpu().numpy() * env.cfg.policy_action_scale_rad
                jacobian[:, joint] = np.mean((a_plus - a_minus) / (2 * h), axis=0)
            eigenvalues = np.linalg.eigvals(jacobian)
        standing = outputs["idle"]
        return {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_index": int(checkpoint.stem.split("_")[-1]),
            "num_envs": num_envs,
            "warmup_s": warmup * env.cfg.ctrl_dt,
            "warmup_forward": warmup_forward,
            "physical_action_scale_rad": env.cfg.policy_action_scale_rad,
            "last_action_feedback": {
                "finite_difference_h_rad": h,
                "mean_jacobian": jacobian.tolist(),
                "diagonal": np.diag(jacobian).tolist(),
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "most_negative_real_eigenvalue": float(np.min(np.real(eigenvalues))),
                "eigenvalues_real_imag": [
                    [float(np.real(value)), float(np.imag(value))] for value in eigenvalues
                ],
            },
            "results": {
                name: {
                    "mean_abs_physical_action_rad": float(np.mean(np.abs(value))),
                    "leg_command_delta_rms_rad": float(
                        np.sqrt(np.mean(np.square((value - standing)[:, LEG_IDS])))
                    ),
                    "head_command_delta_rms_rad": float(
                        np.sqrt(np.mean(np.square((value - standing)[:, 5:9])))
                    ),
                    "mean_physical_action_per_joint_rad": np.mean(value, axis=0).tolist(),
                }
                for name, value in outputs.items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-forward", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.checkpoint.resolve(), warmup_forward=args.warmup_forward)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
