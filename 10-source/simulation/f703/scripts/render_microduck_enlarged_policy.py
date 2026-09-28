#!/usr/bin/env python3

"""Render a fixed-command video for an enlarged Micro Duck ONNX policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT_DIR / "conf" / "ppo"
sys.path.append(str(ROOT_DIR))

from scripts.evaluate_microduck_enlarged_walk import (  # noqa: E402
    Command,
    _fixed_observation,
    _infer,
)
from scripts.evaluate_microduck_roller import _roller_obs  # noqa: E402
from unilab.base import registry  # noqa: E402
from unilab.envs.locomotion import (  # noqa: E402, F401
    microduck as _microduck_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_104 as _microduck_enlarged_104_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_106 as _microduck_enlarged_106_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_107 as _microduck_enlarged_107_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_110 as _microduck_enlarged_110_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_111 as _microduck_enlarged_111_registry,
)
from unilab.envs.locomotion import (
    microduck_enlarged_mod1 as _microduck_enlarged_mod1_registry,
)

TASKS = {
    "v112-walk": (
        "microduck_enlarged_112_walk_flat/mujoco",
        "MicroDuckEnlarged112WalkFlat",
    ),
    "v112-stand": (
        "microduck_enlarged_112_stand_flat/mujoco",
        "MicroDuckEnlarged112StandFlat",
    ),
    "original-roller": (
        "microduck_roller_flat/mujoco",
        "MicroDuckRollerFlat",
    ),
    "stand": ("microduck_enlarged_stand_flat/mujoco", "MicroDuckEnlargedStandFlat"),
    "walk": ("microduck_enlarged_walk_flat/mujoco", "MicroDuckEnlargedWalkFlat"),
    "v104-stand": (
        "microduck_enlarged_104_stand_flat/mujoco",
        "MicroDuckEnlarged104StandFlat",
    ),
    "v104-walk": (
        "microduck_enlarged_104_walk_flat/mujoco",
        "MicroDuckEnlarged104WalkFlat",
    ),
    "v106-walk": (
        "microduck_enlarged_106_walk_flat/mujoco",
        "MicroDuckEnlarged106WalkFlat",
    ),
    "v106-stand": (
        "microduck_enlarged_106_stand_flat/mujoco",
        "MicroDuckEnlarged106StandFlat",
    ),
    "v107-walk": (
        "microduck_enlarged_107_walk_flat/mujoco",
        "MicroDuckEnlarged107WalkFlat",
    ),
    "v107-stand": (
        "microduck_enlarged_107_stand_flat/mujoco",
        "MicroDuckEnlarged107StandFlat",
    ),
    "v110-walk": (
        "microduck_enlarged_110_walk_flat/mujoco",
        "MicroDuckEnlarged110WalkFlat",
    ),
    "v110-stand": (
        "microduck_enlarged_110_stand_flat/mujoco",
        "MicroDuckEnlarged110StandFlat",
    ),
    "v111-walk": (
        "microduck_enlarged_111_walk_flat/mujoco",
        "MicroDuckEnlarged111WalkFlat",
    ),
    "v111-stand": (
        "microduck_enlarged_111_stand_flat/mujoco",
        "MicroDuckEnlarged111StandFlat",
    ),
    "mod1-roller": (
        "microduck_enlarged_mod1_roller_flat/mujoco",
        "MicroDuckEnlargedMod1RollerFlat",
    ),
    "mod1-crouch": (
        "microduck_enlarged_mod1_roller_crouch_flat/mujoco",
        "MicroDuckEnlargedMod1RollerCrouchFlat",
    ),
    "mod1-2knee-stand": (
        "microduck_enlarged_mod1_2knee_stand_flat/mujoco",
        "MicroDuckEnlargedMod1TwoKneeStandFlat",
    ),
    "mod1-2knee-walk": (
        "microduck_enlarged_mod1_2knee_walk_flat/mujoco",
        "MicroDuckEnlargedMod1TwoKneeWalkFlat",
    ),
}


def _make_env(task: str, *, kp: float | None = None, kd: float | None = None):
    task_override, task_name = TASKS[task]
    overrides = [f"task={task_override}"]
    if kp is not None:
        overrides.append(f"env.control_config.Kp={kp}")
    if kd is not None:
        overrides.append(f"env.control_config.Kd={kd}")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config_mlx", overrides=overrides)
    env_override = OmegaConf.to_container(cfg.env, resolve=True)
    if not isinstance(env_override, dict):
        raise TypeError("resolved environment config must be a mapping")
    env_override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    registry.ensure_registries()
    return registry.make(
        task_name,
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=env_override,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--action-multiplier", type=float, default=1.0)
    parser.add_argument("--camera-distance", type=float, default=1.4)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--camera-azimuth", type=float, default=115.0)
    args = parser.parse_args()

    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    if args.action_multiplier <= 0.0:
        parser.error("--action-multiplier must be positive")
    if args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if args.vx is None:
        vx = (
            0.0
            if args.task
            in {
                "stand",
                "v104-stand",
                "v106-stand",
                "v107-stand",
                "v110-stand",
                "v111-stand",
                "v112-stand",
                "mod1-2knee-stand",
            }
            else (0.6 if args.task in {"original-roller", "mod1-roller"} else 0.3)
        )
    else:
        vx = args.vx
    command = Command("video", (vx, args.vy, args.wz))

    np.random.seed(args.seed)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    env = _make_env(args.task, kp=args.kp, kd=args.kd)
    env.set_autoreset(False)
    try:
        # ``init_state`` performs the complete reset and retains all matching
        # reset-time DR info.  A second unmanaged reset would desynchronize the
        # physical model from per-episode motor/gain samples.
        state = env.init_state()
        if env.state is None:
            raise RuntimeError("enlarged Micro Duck environment did not retain reset state")
        if args.task in {"original-roller", "mod1-roller"}:
            initial_obs = _roller_obs(state, vx)
        elif args.task == "mod1-crouch":
            initial_obs = np.asarray(state.obs["obs"], dtype=np.float32)
        else:
            initial_obs = _fixed_observation(state, command)

        def _step(obs: np.ndarray) -> np.ndarray:
            action = np.nan_to_num(
                _infer(session, obs) * args.action_multiplier,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            next_state = env.step(action)
            if args.task in {"original-roller", "mod1-roller"}:
                return _roller_obs(next_state, vx)
            if args.task == "mod1-crouch":
                return np.asarray(next_state.obs["obs"], dtype=np.float32)
            return _fixed_observation(next_state, command)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = env.run_playback_mode(
            play_render_mode="record",
            play_steps=int(round(args.seconds / float(env.cfg.ctrl_dt))),
            output_video=args.output,
            initialize=lambda: initial_obs,
            step=_step,
            camera_kwargs={
                "cam_distance": args.camera_distance,
                "cam_elevation": args.camera_elevation,
                "cam_azimuth": args.camera_azimuth,
                "cam_tracking": True,
                "cam_tracking_env_idx": 0,
                "cam_tracking_extra_envs": 0,
            },
        )
        if output is None:
            raise RuntimeError("renderer did not produce a video")
        print(f"Video: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
