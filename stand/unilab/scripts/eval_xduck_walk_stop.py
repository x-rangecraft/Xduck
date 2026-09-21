"""Live walk -> stop policy switch without resetting physics or motor state."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import train_rsl_rl as E
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.envs.locomotion.xduck_gf43x40.handoff import WalkHandoffBank
from unilab.envs.locomotion.xduck_gf43x40.stop import XDuckWalkStopCfg
from unilab.envs.locomotion.xduck_gf43x40.velocity import XDuckGF43X40VelocityCfg

ROOT = Path(__file__).resolve().parents[1]


def load_policy(path, wrapper, cfg):
    saved = json.loads((path.parent / "run_config.json").read_text())["config"]
    train_cfg = E.normalize_ppo_train_cfg(saved["algo"])
    E.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    runner = E.OnPolicyRunner(wrapper, train_cfg, log_dir=None, device="cpu")
    runner.load(str(path), map_location="cpu")
    return runner.get_inference_policy(device="cpu"), saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-checkpoint", type=Path, required=True)
    parser.add_argument("--stop-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--walk-seconds", type=float, default=5.0)
    parser.add_argument("--stop-seconds", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pushes",
        action="store_true",
        help="Explicit robustness test; default videos have no pushes",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    E.ensure_registries()
    stop_saved = json.loads((args.stop_checkpoint.parent / "run_config.json").read_text())["config"]
    walk_saved = json.loads((args.walk_checkpoint.parent / "run_config.json").read_text())["config"]
    for key in ["control_config", "policy_action_scale_rad", "ctrl_dt", "sim_dt"]:
        if stop_saved["env"].get(key) != walk_saved["env"].get(key):
            raise ValueError(f"Walk/stop action contract differs: {key}")
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=xduck_gf43x40_velocity_flat/mujoco", "training.device=cpu"],
        )
    o = E.build_ppo_env_cfg_override(cfg)
    allowed = {f.name for f in dataclasses.fields(XDuckGF43X40VelocityCfg)}
    o.update({k: v for k, v in stop_saved["env"].items() if k in allowed})
    o["domain_rand"]["velocity_pushes"] = args.pushes
    o["domain_rand"]["push_robots"] = False
    o["max_episode_seconds"] = args.walk_seconds + args.stop_seconds + 2
    env = E.create_env(cfg, num_envs=args.num_envs, env_cfg_override=o)
    try:
        criterion = XDuckWalkStopCfg()
        for f in dataclasses.fields(criterion):
            if f.name.startswith("settled_") and f.name in stop_saved["env"]:
                setattr(criterion, f.name, stop_saved["env"][f.name])
        bank_path = stop_saved["env"].get("handoff_bank", criterion.handoff_bank)
        WalkHandoffBank.load(bank_path).validate_contract(env.cfg, env.default_angles)
        wrapper = E._resolve_ppo_wrapper_cls(walk_saved["algo"])(env, device="cpu")
        walk, _ = load_policy(args.walk_checkpoint, wrapper, cfg)
        stop, _ = load_policy(args.stop_checkpoint, wrapper, cfg)
        command = np.zeros((args.num_envs, 13), np.float32)
        command[:, 0] = args.speed
        env._maybe_resample_commands = lambda s: s.info["commands"].__setitem__(
            slice(None), command
        )
        dt = env.cfg.ctrl_dt
        switch = int(round(args.walk_seconds / dt))
        steps = switch + int(round(args.stop_seconds / dt))
        held = np.zeros(args.num_envs)
        latency = np.full(args.num_envs, np.nan)
        failed = np.zeros(args.num_envs, bool)
        qtrace = []
        vtrace = []
        ctrace = []
        settled_trace = []
        speed_trace = []
        walk_speed_trace = []
        walk_failed = np.zeros(args.num_envs, bool)
        with torch.inference_mode():
            for step in range(steps):
                if step == switch:
                    command[:] = 0
                env.state.info["commands"][:] = command
                env.state.obs["obs"][:, -13:] = command
                action = (walk if step < switch else stop)(wrapper.get_observations())
                _, _, done, _ = wrapper.step(action)
                if step < switch:
                    walk_speed_trace.append(np.linalg.norm(env.get_local_linvel()[:, :2], axis=1))
                    walk_failed |= done.cpu().numpy().astype(bool)
                if step >= switch:
                    failed |= done.cpu().numpy().astype(bool)
                    upright = -env._projected_gravity()[:, 2]
                    failed |= (upright < np.cos(np.deg2rad(criterion.failure_tilt_deg))) | (
                        env._backend.get_base_pos()[:, 2]
                        < env._init_qpos[2] * criterion.minimum_root_height_ratio
                    )
                    settled = (
                        (
                            np.linalg.norm(env.get_local_linvel(), axis=1)
                            < criterion.settled_linear_speed
                        )
                        & (np.linalg.norm(env.get_gyro(), axis=1) < criterion.settled_angular_speed)
                        & (
                            np.sqrt(np.mean(env.get_dof_vel() ** 2, axis=1))
                            < criterion.settled_joint_speed_rms
                        )
                        & env._contact.all(axis=1)
                        & (upright > np.cos(np.deg2rad(15)))
                        & (~failed)
                    )
                    held = np.where(settled, held + dt, 0)
                    hit = (held >= criterion.settled_hold_seconds) & np.isnan(latency)
                    latency[hit] = (step - switch + 1) * dt
                    settled_trace.append(settled.copy())
                    speed_trace.append(np.linalg.norm(env.get_local_linvel()[:, :2], axis=1))
                qtrace.append(
                    np.r_[
                        env._backend.get_base_pos()[0],
                        env._backend.get_base_quat()[0],
                        env.get_dof_pos()[0],
                    ]
                )
                vtrace.append(env.get_local_linvel()[0].copy())
                ctrace.append(command[0, :3].copy())
        pre_switch_speed = np.mean(np.array(walk_speed_trace)[-round(1 / dt) :], axis=0)
        eligible = (pre_switch_speed > criterion.settled_linear_speed) & (~walk_failed)
        report = {
            "pre_switch_planar_speed_per_env": pre_switch_speed.tolist(),
            "eligible_walking_handoffs": int(eligible.sum()),
            "walk_checkpoint": str(args.walk_checkpoint.resolve()),
            "stop_checkpoint": str(args.stop_checkpoint.resolve()),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "pushes": args.pushes,
            "switch_s": switch * dt,
            "stop_observation_s": args.stop_seconds,
            "settled_success_envs": int(
                np.count_nonzero((held >= criterion.settled_hold_seconds) & ~failed & eligible)
            ),
            "failed_envs": int(failed.sum()),
            "first_sustained_stop_latency_s": [None if np.isnan(x) else float(x) for x in latency],
            "last_second_planar_speed_per_env": np.mean(
                np.array(speed_trace)[-round(1 / dt) :], axis=0
            ).tolist(),
            "no_reset_at_switch": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            qpos=np.array(qtrace),
            local_velocity=np.array(vtrace),
            command_trace=np.array(ctrace),
            dt=dt,
            model_file=env.cfg.scene.model_file,
            checkpoint=str(args.stop_checkpoint),
            switch_s=switch * dt,
        )
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
