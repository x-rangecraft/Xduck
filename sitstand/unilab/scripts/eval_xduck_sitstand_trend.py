"""Measure commanded sit/stand outcomes for saved XDuck PPO checkpoints."""

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

from unilab.envs.locomotion.xduck_gf43x40.sitstand import LEG_IDS, SIT_LEGS

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "stand_to_stand": (0.0, 0.0),
    "stand_to_sit": (0.0, 1.0),
    "sit_to_stand": (1.0, 0.0),
    "sit_to_sit": (1.0, 1.0),
}


def evaluate(
    checkpoints: list[Path], *, num_envs: int, steps: int,
    nominal: bool = False, task: str = "xduck_gf43x40_sitstand_flat/mujoco",
) -> list[dict]:
    torch.set_num_threads(1)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"task={task}", "training.device=cpu"],
        )
    cfg.env.sit_reset_probability = 0.0
    cfg.env.sit_command_probability = 0.0
    if nominal:
        cfg.env.domain_rand.mass_inertia_scale_range = [1.0, 1.0]
        cfg.env.domain_rand.foot_friction_range = [1.0, 1.0]
        cfg.env.domain_rand.imu_max_angle_deg = 0.0
        cfg.env.domain_rand.randomize_dof_armature = False
        cfg.env.ankle_target_offset_max_deg = 0.0
        cfg.env.noise_config.level = 0.0
    env = entry.create_env(
        cfg, num_envs=num_envs, env_cfg_override=entry.build_ppo_env_cfg_override(cfg)
    )
    try:
        if nominal:
            env._com_range = 0.0
            env._head_com_range = 0.0
        rl_cfg = entry._algo_config_dict(cfg)
        wrapped = entry._resolve_ppo_wrapper_cls(rl_cfg)(env, device="cpu")
        train_cfg = entry.normalize_ppo_train_cfg(rl_cfg)
        entry.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg["logger"] = "none"
        train_cfg.setdefault("runner", {})["logger"] = "none"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = entry.OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
        rows = []
        for checkpoint in checkpoints:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.load(str(checkpoint), map_location="cpu")
            policy = runner.get_inference_policy(device="cpu")
            for name, (reset_sit, command_sit) in CASES.items():
                np.random.seed(20260928)
                torch.manual_seed(20260928)
                env.seed_observation_noise(20260928)
                env.cfg.sit_reset_probability = reset_sit
                env.cfg.sit_command_probability = command_sit
                if nominal:
                    env._com_range = 0.0
                    env._head_com_range = 0.0
                wrapped.reset()
                assert np.all(env.state.info["steps"] == 0)
                assert np.all(env.state.info["commands"][:, 0] == command_sit)
                initial_height = float(np.mean(env._backend.get_base_pos()[:, 2]))
                target_height = (
                    env.cfg.reward_config.sit_height
                    if command_sit
                    else env.cfg.reward_config.stand_height
                )
                target_legs = SIT_LEGS if command_sit else env.default_angles[LEG_IDS]
                tail_height = []
                tail_upright = []
                tail_pose_error = []
                tail_success = []
                with torch.inference_mode():
                    for step in range(steps):
                        action = policy(wrapped.get_observations())
                        _, _, done, _ = wrapped.step(action)
                        if bool(done.any()):
                            raise RuntimeError("Evaluation crossed an episode boundary")
                        assert np.all(env.state.info["commands"][:, 0] == command_sit)
                        if step < steps - 50:
                            continue
                        height = env._backend.get_base_pos()[:, 2].copy()
                        quat = env._backend.get_base_quat()
                        upright = 1.0 - 2.0 * (
                            np.square(quat[:, 1]) + np.square(quat[:, 2])
                        )
                        pose_error = np.mean(
                            np.abs(env.get_dof_pos()[:, LEG_IDS] - target_legs), axis=1
                        )
                        success = (
                            (np.abs(height - target_height) < 0.025)
                            & (upright > 0.9)
                            & (pose_error < 0.25)
                        )
                        tail_height.append(height)
                        tail_upright.append(upright)
                        tail_pose_error.append(pose_error)
                        tail_success.append(success)
                rows.append(
                    {
                        "checkpoint": checkpoint.name,
                        "case": name,
                        "initial_height_m": initial_height,
                        "last_second_height_m": float(np.mean(tail_height)),
                        "last_second_upright": float(np.mean(tail_upright)),
                        "last_second_pose_error_rad": float(np.mean(tail_pose_error)),
                        "held_success_rate": float(
                            np.mean(np.mean(tail_success, axis=0) >= 0.8)
                        ),
                    }
                )
        return rows
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--nominal", action="store_true", help="disable task DR and observation noise")
    parser.add_argument("--task", default="xduck_gf43x40_sitstand_flat/mujoco")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 50:
        parser.error("--num-envs must be positive and --steps must be at least 50")
    rows = evaluate(args.checkpoints, num_envs=args.num_envs, steps=args.steps, nominal=args.nominal, task=args.task)
    result = {"num_envs_per_case": args.num_envs, "steps_per_case": args.steps, "nominal": args.nominal, "task": args.task, "rows": rows}
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
