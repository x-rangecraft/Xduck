"""Collect upright walking reset states using an existing XDuck checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import train_rsl_rl as E
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.envs.locomotion.xduck_gf43x40.handoff import WalkHandoffBank, canonicalize_handoffs
from unilab.envs.locomotion.xduck_gf43x40.scaling import xml_model_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/xduck_stop/walk_handoffs.npz")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}; choose a new bank path")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    E.ensure_registries()
    saved = json.loads((args.checkpoint.parent / "run_config.json").read_text())["config"]
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=xduck_gf43x40_velocity_flat/mujoco", "training.device=cpu"],
        )
    o = E.build_ppo_env_cfg_override(cfg)
    o.update(saved["env"])
    o["reward_config"] = saved["reward"]
    o["reward_config"]["scale_model_fingerprint"] = None
    o["control_config"] = saved["env"]["control_config"]
    o["policy_action_scale_rad"] = saved["env"]["policy_action_scale_rad"]
    o["communication"] = saved["env"]["communication"]
    for key in ("scene", "reference_keyframe", "action_offset_low", "action_offset_high"):
        if key in saved["env"]:
            o[key] = saved["env"][key]
    o["domain_rand"]["velocity_pushes"] = False
    o["domain_rand"]["push_robots"] = False
    o["max_episode_seconds"] = args.steps * .02 + 2
    cfg.env.noise_config = OmegaConf.create(o["noise_config"])
    env = E.create_env(cfg, num_envs=16, env_cfg_override=o)
    try:
        env._encoder_bias.fill(0.0)
        rl = saved["algo"]
        wrapper = E._resolve_ppo_wrapper_cls(rl)(env, device="cpu")
        train_cfg = E.normalize_ppo_train_cfg(rl)
        E.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        runner = E.OnPolicyRunner(wrapper, train_cfg, log_dir=None, device="cpu")
        runner.load(str(args.checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        commands = np.zeros((16, 13), np.float32)
        commands[:, :3] = np.repeat(
            np.array(
                [
                    [0.05, 0, 0],
                    [0.1, 0, 0],
                    [0.2, 0, 0],
                    [0.3, 0, 0],
                    [0.2, 0, 0.2],
                    [0.2, 0, -0.2],
                    [0.15, 0.03, 0],
                    [0.15, -0.03, 0],
                ]
            ),
            2,
            axis=0,
        )
        env._maybe_resample_commands = lambda state: state.info["commands"].__setitem__(
            slice(None), commands
        )
        qs, vs, actions = [], [], []
        with torch.inference_mode():
            for step in range(args.steps):
                env.state.info["commands"][:] = commands
                env.state.obs["obs"][:, -13:] = commands
                a = policy(wrapper.get_observations())
                _, _, done, _ = wrapper.step(a)
                if step < 100 or step % 5:
                    continue
                q = np.c_[
                    env._backend.get_base_pos(), env._backend.get_base_quat(), env.get_dof_pos()
                ]
                v = np.c_[env._backend.get_base_lin_vel(), env.get_gyro(), env.get_dof_vel()]
                upright = -env._projected_gravity()[:, 2]
                speed = np.linalg.norm(env.get_local_linvel()[:, :2], axis=1)
                good = (
                    (~done.cpu().numpy().astype(bool))
                    & (upright > np.cos(np.deg2rad(25)))
                    & (speed > 0.02)
                    & (speed < 0.7)
                    & (q[:, 2] > 0.23)
                )
                if good.any():
                    qs.append(q[good].copy())
                    vs.append(v[good].copy())
                    actions.append(a.cpu().numpy()[good].copy())
        if not qs:
            raise RuntimeError("No valid upright walking states collected")
        q, v = canonicalize_handoffs(np.concatenate(qs), np.concatenate(vs))
        action = np.concatenate(actions)
        walking_count = len(q)
        stand_q = np.tile(env._init_qpos, (walking_count, 1))
        stand_q[:, :2] = 0
        env._reset_clearance.lift_to_clearance(stand_q, env.cfg.reset_clearance_m)
        q = np.concatenate((q, stand_q))
        v = np.concatenate((v, np.zeros((walking_count,20))))
        action = np.concatenate((action,np.zeros((walking_count,14))))
        meta = {
            "version": 1,
            "walking_samples": walking_count,
            "standing_samples": walking_count,
            "model_fingerprint": xml_model_fingerprint(env.cfg.scene.model_file),
            "joint_names": list(env.cfg.asset.policy_joint_names),
            "default_angles": env.default_angles.tolist(),
            "action_scale_rad": env.cfg.policy_action_scale_rad,
            "joint_kp": list(env.cfg.control_config.joint_kp),
            "joint_kd": list(env.cfg.control_config.joint_kd),
            "ctrl_dt": env.cfg.ctrl_dt,
            "sim_dt": env.cfg.sim_dt,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "seed": args.seed,
            "commands": commands.tolist(),
            "note": "Mechanical qpos/qvel and previous action only. Reset actuator/filter/transport memory; validate live policy switching separately.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, qpos=q, qvel=v, actions=action, metadata=json.dumps(meta))
        bank = WalkHandoffBank.load(args.output)
        bank.validate_contract(env.cfg, env.default_angles)
        print(
            json.dumps(
                {
                    "bank": str(args.output),
                    "samples": len(q),
                    "speed_range": np.linalg.norm(v[:, :2], axis=1).min().item(),
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
