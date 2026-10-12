"""Diagnostic 2x2: swap robot geometry and motor under one XDuck task contract.

The off-diagonal combinations are deliberately non-deployable. They isolate
plant response; they are not evidence that a learned policy will transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import train_rsl_rl as entry
from hydra import compose, initialize_config_dir

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.microduck_dm4310.velocity import POLICY_JOINT_NAMES
from unilab.envs.locomotion.microduck_xl330_official.bam import Xl330BamBatchModel
from unilab.envs.locomotion.microduck_xl330_official.velocity import (
    MicroDuckXl330OfficialVelocityEnv,
)
from unilab.envs.locomotion.xduck_gf43x40.velocity import (
    XDuckGF43X40VelocityCfg,
    XDuckGF43X40VelocityEnv,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCENE = ROOT.parent / "mjlab/src/mjlab_microduck/robot/microduck/scene_walk.xml"


class XDuckWithXL330Motor(XDuckGF43X40VelocityEnv):
    """Use official XL330 plant on either scene with the GF task/reward."""

    def _create_backend(self, cfg, num_envs: int, backend_type: str):
        return MicroDuckXl330OfficialVelocityEnv._create_backend(self, cfg, num_envs, backend_type)

    def _build_motor(self, cfg, num_envs: int):
        del cfg
        return Xl330BamBatchModel(
            num_envs,
            self._backend.get_joint_dof_indices(POLICY_JOINT_NAMES),
            self._backend.model.nv,
        )

    def _physics_step_action_delay(self, backend, ctrl: np.ndarray) -> np.ndarray:
        return MicroDuckXl330OfficialVelocityEnv._physics_step_action_delay(self, backend, ctrl)


def make_cfg(scene: str, *, kp: float, kd: float) -> XDuckGF43X40VelocityCfg:
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        owner = compose(
            config_name="config",
            overrides=[
                "task=xduck_gf43x40_velocity_flat/mujoco",
                "env.noise_config.level=0",
                "env.domain_rand.velocity_pushes=false",
                "env.domain_rand.randomize_foot_friction=false",
                "env.domain_rand.randomize_joint_friction=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    cfg = XDuckGF43X40VelocityCfg()
    apply_cfg_overrides(cfg, entry.build_ppo_env_cfg_override(owner))
    cfg.control_config.Kp = kp
    cfg.control_config.Kd = kd
    if scene == "official":
        cfg.scene.model_file = str(OFFICIAL_SCENE)
        cfg.scene.visual_model_file = str(OFFICIAL_SCENE)
        cfg.reward_config.foot_target_height = 0.02
    cfg.validate()
    return cfg


def probe(
    scene: str,
    motor: str,
    *,
    num_envs: int,
    steps: int,
    kp: float,
    kd: float,
    gf_command_delay_ms: float,
) -> dict:
    np.random.seed(20260917)
    cfg = make_cfg(scene, kp=kp, kd=kd)
    cls = XDuckGF43X40VelocityEnv if motor == "gf43x40" else XDuckWithXL330Motor
    env = cls(cfg, num_envs=num_envs, backend_type="mujoco")
    try:
        if motor == "gf43x40":
            env._motor.communication.command_delay_s = gf_command_delay_ms / 1000.0
        env._autoreset = False
        state = env.init_state()
        command = np.zeros((num_envs, 13), dtype=np.float32)
        command[:, 0] = 0.2
        env._maybe_resample_commands = lambda current: current.info["commands"].__setitem__(
            slice(None), command
        )
        state.info["commands"][:] = command
        state.obs["obs"][:, -13:] = command
        action = np.zeros((num_envs, 14), dtype=np.float32)
        first_fall = np.full(num_envs, np.nan)
        heights = []
        torques = []
        rewards = []
        speeds = []
        for step in range(steps):
            state = env.step(action)
            new_fall = state.terminated & np.isnan(first_fall)
            first_fall[new_fall] = (step + 1) * cfg.ctrl_dt
            heights.append(env._backend.get_base_pos()[:, 2].copy())
            torques.append(np.abs(env._motor.torque).copy())
            valid = np.isnan(first_fall)
            if step >= 50 and np.any(valid):
                rewards.extend((state.reward[valid] / cfg.ctrl_dt).tolist())
                speeds.extend(env.get_local_linvel()[valid, 0].tolist())
        return {
            "scene": scene,
            "motor": motor,
            "configured_kp": kp,
            "configured_kd": kd,
            "gf_command_delay_ms": gf_command_delay_ms if motor == "gf43x40" else None,
            "num_envs": num_envs,
            "duration_s": steps * cfg.ctrl_dt,
            "falls": int(np.sum(~np.isnan(first_fall))),
            "first_fall_median_s": float(np.nanmedian(first_fall))
            if np.any(~np.isnan(first_fall))
            else None,
            "height_last_median_m": float(np.median(heights[-1])),
            "torque_p99_nm": float(np.quantile(torques, 0.99)),
            "torque_peak_nm": float(np.max(torques)),
            "fixed_forward_command_m_s": 0.2,
            "reward_rate_after_1s_before_fall": float(np.mean(rewards)) if rewards else None,
            "actual_vx_after_1s_before_fall": float(np.mean(speeds)) if speeds else None,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--scene", choices=("official", "xduck"))
    parser.add_argument("--motor", choices=("xl330", "gf43x40"))
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--gf-command-delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    entry.ensure_registries()
    result = {
        "description": "Same XDuck task/action/reward; model x motor zero-action support",
        "cases": [
            probe(
                scene,
                motor,
                num_envs=args.num_envs,
                steps=args.steps,
                kp=args.kp,
                kd=args.kd,
                gf_command_delay_ms=args.gf_command_delay_ms,
            )
            for scene in ("official", "xduck")
            for motor in ("xl330", "gf43x40")
            if (args.scene is None or scene == args.scene)
            and (args.motor is None or motor == args.motor)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
