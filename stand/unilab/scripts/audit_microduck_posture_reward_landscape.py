"""Evaluate actual stand-up and sit/stand rewards at adversarial state basins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.envs.locomotion.microduck_dm4310.sitstand import SIT_OVERRIDES, SIT_Z, STAND_Z


def _poses(env) -> dict[str, np.ndarray]:
    home = env._backend.get_keyframe_qpos("home").copy()
    home[2] = STAND_Z
    sit = home.copy()
    sit[2] = SIT_Z
    for index, value in SIT_OVERRIDES.items():
        sit[7 + index] = value
    prone = home.copy()
    prone[2] = 0.10
    prone[3:7] = [2.0**-0.5, 0.0, 2.0**-0.5, 0.0]
    supine = home.copy()
    supine[2] = 0.10
    supine[3:7] = [2.0**-0.5, 0.0, -(2.0**-0.5), 0.0]
    side = home.copy()
    side[2] = 0.10
    side[3:7] = [2.0**-0.5, 2.0**-0.5, 0.0, 0.0]
    limit = home.copy()
    limit[7:] = env._joint_limits[:, 1] - 1.0e-4
    poses = {
        "standing": home,
        "sitting": sit,
        "prone": prone,
        "supine": supine,
        "side": side,
        "upper_joint_limits": limit,
    }
    batch = np.stack(list(poses.values()))
    env._reset_clearance.lift_to_clearance(batch, 0.001)
    return dict(zip(poses, batch, strict=True))


def _evaluate_state(env, qpos: np.ndarray, *, command_flag: float, yaw_rate: float = 0.0) -> dict:
    ids = np.arange(env.num_envs, dtype=np.int32)
    qpos_batch = np.broadcast_to(qpos, (env.num_envs, len(qpos))).copy()
    qvel = np.zeros((env.num_envs, env._backend.get_init_qvel().size))
    qvel[:, 5] = yaw_rate
    env._backend.set_state(ids, qpos_batch, qvel)
    env._reset_episode_history(ids)
    if hasattr(env, "_posture_alpha"):
        env._posture_alpha[:] = command_flag
    state = env.state
    state.info["steps"][:] = 10
    state.info["commands"][:] = 0.0
    state.info["commands"][:, 0] = command_flag
    state.info["current_actions"][:] = 0.0
    state.info["last_actions"][:] = 0.0
    env._velocity_timer[:] = 1_000_000
    env._head_timer[:] = 1_000_000
    env._body_timer[:] = 1_000_000
    state = env.update_state(state)
    return {
        "reward_rate": float(np.mean(state.reward) / env._cfg.ctrl_dt),
        "terminated_fraction": float(np.mean(state.terminated)),
        "terms": dict(state.info["log"]),
    }


def audit_task(kind: str) -> dict:
    registry.ensure_registries()
    env = registry.make(f"MicroDuckDm4310{kind}Flat", "mujoco", num_envs=4)
    try:
        env.init_state()
        poses = _poses(env)
        output = {}
        if kind == "StandUp":
            for name, pose in poses.items():
                output[name] = _evaluate_state(env, pose, command_flag=0.0)
            output["standing_yaw_spin_1_3"] = _evaluate_state(
                env, poses["standing"], command_flag=0.0, yaw_rate=1.3
            )
            good = output["standing"]["reward_rate"]
            checks = {
                name: good > output[name]["reward_rate"] + 1.0
                for name in (
                    "sitting",
                    "prone",
                    "supine",
                    "side",
                    "upper_joint_limits",
                    "standing_yaw_spin_1_3",
                )
            }
        else:
            for command_name, flag in (("stand_command", 0.0), ("sit_command", 1.0)):
                for pose_name, pose in poses.items():
                    output[f"{command_name}/{pose_name}"] = _evaluate_state(
                        env, pose, command_flag=flag
                    )
                output[f"{command_name}/target_yaw_spin_1_3"] = _evaluate_state(
                    env,
                    poses["standing" if flag == 0.0 else "sitting"],
                    command_flag=flag,
                    yaw_rate=1.3,
                )
            checks = {}
            for command_name, target in (("stand_command", "standing"), ("sit_command", "sitting")):
                good = output[f"{command_name}/{target}"]["reward_rate"]
                for bad in ("prone", "supine", "side", "upper_joint_limits"):
                    checks[f"{command_name}/{bad}"] = (
                        good > output[f"{command_name}/{bad}"]["reward_rate"] + 1.0
                    )
                checks[f"{command_name}/wrong_posture"] = (
                    good
                    > output[
                        f"{command_name}/{('sitting' if target == 'standing' else 'standing')}"
                    ]["reward_rate"]
                    + 1.0
                )
                checks[f"{command_name}/yaw_spin"] = (
                    good > output[f"{command_name}/target_yaw_spin_1_3"]["reward_rate"] + 1.0
                )
        return {"states": output, "checks": checks, "passed": all(checks.values())}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"standup": audit_task("StandUp"), "sitstand": audit_task("SitStand")}
    result["passed"] = all(task["passed"] for task in result.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
