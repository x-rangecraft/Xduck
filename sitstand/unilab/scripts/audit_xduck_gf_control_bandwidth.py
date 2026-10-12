"""Small-signal MicroDuck knee tracking at gait and Nyquist frequencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES_HZ = (1.0, 5.0, 10.0, 25.0)
AMPLITUDE_RAD = 0.05


def evaluate(
    *,
    task: str = "xduck_gf43x40_velocity_flat/mujoco",
    kp: float = 60.0,
    kd: float = 2.0,
    steps: int = 250,
    warmup: int = 50,
) -> dict:
    np.random.seed(42)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={task}",
                f"env.control_config.Kp={kp}",
                f"env.control_config.Kd={kd}",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    env = entry.create_env(
        cfg,
        num_envs=len(FREQUENCIES_HZ),
        env_cfg_override=entry.build_ppo_env_cfg_override(cfg),
    )
    try:
        env._autoreset = False
        env.init_state()
        targets, positions, torques = [], [], []
        falls = np.zeros(len(FREQUENCIES_HZ), dtype=int)
        already_fallen = np.zeros(len(FREQUENCIES_HZ), dtype=bool)
        for step in range(steps):
            t = step * env.cfg.ctrl_dt
            physical = np.array(
                [
                    AMPLITUDE_RAD * np.sin(2.0 * np.pi * f * t)
                    if f < 25.0
                    else AMPLITUDE_RAD * (-1.0) ** step
                    for f in FREQUENCIES_HZ
                ]
            )
            action = np.zeros((len(FREQUENCIES_HZ), 14), dtype=np.float32)
            action[:, 3] = physical / float(getattr(env.cfg, "policy_action_scale_rad", 1.0))
            state = env.step(action)
            falls += (state.terminated & ~already_fallen).astype(int)
            already_fallen |= state.terminated
            if step >= warmup:
                targets.append(physical)
                positions.append(env.get_dof_pos()[:, 3] - env.default_angles[3])
                torques.append(env._motor.torque[:, 3].copy())
        targets = np.asarray(targets)
        positions = np.asarray(positions)
        torques = np.asarray(torques)
        result = {}
        sample_index = np.arange(warmup, steps)
        for index, f in enumerate(FREQUENCIES_HZ):
            reference = (
                np.sin(2.0 * np.pi * f * sample_index * env.cfg.ctrl_dt)
                if f < 25.0
                else (-1.0) ** sample_index
            )
            quadrature = (
                np.cos(2.0 * np.pi * f * sample_index * env.cfg.ctrl_dt)
                if f < 25.0
                else np.zeros_like(reference)
            )
            design = np.column_stack((reference, quadrature, np.ones(len(reference))))
            coefficients = np.linalg.lstsq(design, positions[:, index], rcond=None)[0]
            achieved = float(np.hypot(coefficients[0], coefficients[1]))
            result[str(f)] = {
                "target_amplitude_rad": AMPLITUDE_RAD,
                "joint_amplitude_rad": achieved if not falls[index] else None,
                "joint_over_target_amplitude": (
                    achieved / AMPLITUDE_RAD if not falls[index] else None
                ),
                "torque_p99_nm": float(np.quantile(np.abs(torques[:, index]), 0.99)),
                "falls": int(falls[index]),
            }
        return {
            "task": task,
            "configured_kp": kp,
            "configured_kd": kd,
            "joint": "left_knee",
            "ctrl_dt_s": env.cfg.ctrl_dt,
            "frequencies_hz": result,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="xduck_gf43x40_velocity_flat/mujoco")
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    result = evaluate(task=args.task, kp=args.kp, kd=args.kd, steps=args.steps, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
