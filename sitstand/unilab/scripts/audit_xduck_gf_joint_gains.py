"""Isolated per-joint GF gain perturbations on the whole XDuck plant.

Diagnostic only: one joint receives a small sine target in each environment;
the other thirteen joints keep the common standing gains. No training or
deployment configuration is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

from unilab.envs.locomotion.microduck_dm4310.velocity import POLICY_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES_HZ = (1.0, 5.0)
AMPLITUDE_RAD = 0.05


def evaluate(*, probe_kp: float, probe_kd: float, steps: int = 150, warmup: int = 50) -> dict:
    if steps <= warmup or probe_kp < 0 or probe_kd < 0:
        raise ValueError("steps > warmup and nonnegative probe gains are required")
    np.random.seed(20260918)
    entry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=xduck_gf43x40_velocity_flat/mujoco",
                "env.control_config.Kp=70.0",
                "env.control_config.Kd=3.0",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    count = len(POLICY_JOINT_NAMES) * len(FREQUENCIES_HZ)
    env = entry.create_env(
        cfg, num_envs=count, env_cfg_override=entry.build_ppo_env_cfg_override(cfg)
    )
    try:
        env._autoreset = False
        env.init_state()
        for joint_index in range(len(POLICY_JOINT_NAMES)):
            for frequency_index in range(len(FREQUENCIES_HZ)):
                row = joint_index * len(FREQUENCIES_HZ) + frequency_index
                env._motor.kp[row, joint_index] = probe_kp
                env._motor.kd[row, joint_index] = probe_kd
        action = np.zeros((count, len(POLICY_JOINT_NAMES)), dtype=np.float32)
        positions = []
        torques = []
        falls = np.zeros(count, dtype=bool)
        for step in range(steps):
            action.fill(0.0)
            t = step * env.cfg.ctrl_dt
            for joint_index in range(len(POLICY_JOINT_NAMES)):
                for frequency_index, frequency in enumerate(FREQUENCIES_HZ):
                    row = joint_index * len(FREQUENCIES_HZ) + frequency_index
                    offset = AMPLITUDE_RAD * np.sin(2.0 * np.pi * frequency * t)
                    action[row, joint_index] = offset / env.cfg.policy_action_scale_rad
            state = env.step(action)
            falls |= state.terminated
            if step >= warmup:
                positions.append(env.get_dof_pos() - env.default_angles)
                torques.append(np.abs(env._motor.torque).copy())
        position_history = np.asarray(positions)
        torque_history = np.asarray(torques)
        sample_steps = np.arange(warmup, steps)
        rows = {}
        for joint_index, joint_name in enumerate(POLICY_JOINT_NAMES):
            rows[joint_name] = {}
            for frequency_index, frequency in enumerate(FREQUENCIES_HZ):
                row = joint_index * len(FREQUENCIES_HZ) + frequency_index
                phase = 2.0 * np.pi * frequency * sample_steps * env.cfg.ctrl_dt
                design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
                coefficients = np.linalg.lstsq(
                    design, position_history[:, row, joint_index], rcond=None
                )[0]
                achieved = float(np.hypot(coefficients[0], coefficients[1]))
                rows[joint_name][str(frequency)] = {
                    "falls": bool(falls[row]),
                    "joint_over_target_amplitude": achieved / AMPLITUDE_RAD
                    if not falls[row]
                    else None,
                    "phase_lag_rad": float(-np.arctan2(coefficients[1], coefficients[0]))
                    if not falls[row]
                    else None,
                    "torque_p99_nm": float(np.quantile(torque_history[:, row, joint_index], 0.99)),
                }
        return {
            "probe_kp": probe_kp,
            "probe_kd": probe_kd,
            "other_joint_kp": 70.0,
            "other_joint_kd": 3.0,
            "target_amplitude_rad": AMPLITUDE_RAD,
            "steps": steps,
            "warmup": warmup,
            "joints": rows,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-kp", type=float, required=True)
    parser.add_argument("--probe-kd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(probe_kp=args.probe_kp, probe_kd=args.probe_kd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
