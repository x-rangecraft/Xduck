"""Cold-path physics and contract gate for the all-DM4340 MicroDuck asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.locomotion.microduck_dm4310.motor import Dm4310MitBatchModel, Dm4340MitConfig

RATED_TORQUE = 8.9
MAX_MECHANICAL_TORQUE = 23.5
EXPECTED_MASS = 10.352
EXPECTED_ACTIONS = 14


def validate(
    duration: float, video: Path | None = None, fps: int = 30
) -> dict[str, float | int | bool]:
    path = ASSETS_ROOT_PATH / "robots" / "microduck_dm4310" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    home = model.key("home")
    data.qpos[:] = home.qpos
    data.ctrl[:] = 0.0
    target = home.qpos[7:].copy()
    motor = Dm4310MitBatchModel(1, model.nu, model.opt.timestep, cfg=Dm4340MitConfig())
    mujoco.mj_forward(model, data)

    initial_z = float(data.qpos[2])
    max_abs_torque = 0.0
    max_abs_torque_by_actuator = np.zeros(model.nu, dtype=np.float64)
    rated_exceed_steps = 0
    samples = max(1, int(duration / model.opt.timestep))
    frames: list[np.ndarray] = []
    renderer = None
    camera = None
    if video is not None:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.20)
        camera.distance = 1.15
        camera.azimuth = 145.0
        camera.elevation = -12.0
    target_frames = max(1, round(duration * fps))
    for _ in range(samples):
        data.ctrl[:] = motor.compute(target[None], data.qpos[None, 7:], data.qvel[None, 6:])[0]
        mujoco.mj_step(model, data)
        torque = np.abs(data.actuator_force)
        max_abs_torque = max(max_abs_torque, float(np.max(torque)))
        max_abs_torque_by_actuator = np.maximum(max_abs_torque_by_actuator, torque)
        rated_exceed_steps += int(np.any(torque > RATED_TORQUE))
        if (
            renderer is not None
            and camera is not None
            and len(frames) < target_frames
            and data.time + 1e-12 >= len(frames) / fps
        ):
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()
    if video is not None:
        import mediapy as media

        video.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(video, frames, fps=fps)

    up = np.empty(9)
    mujoco.mju_quat2Mat(up, data.qpos[3:7])
    up_z = float(up.reshape(3, 3)[2, 2])
    total_mass = float(np.sum(model.body_mass))
    measured_limits = np.full(model.nu, MAX_MECHANICAL_TORQUE, dtype=np.float64)
    result: dict[str, float | int | bool] = {
        "mass_kg": total_mass,
        "num_actions": int(model.nu),
        "initial_z_m": initial_z,
        "final_z_m": float(data.qpos[2]),
        "up_z": up_z,
        "max_abs_torque_nm": max_abs_torque,
        "max_neck_pitch_torque_nm": float(
            max_abs_torque_by_actuator[model.actuator("neck_pitch").id]
        ),
        "max_hip_roll_torque_nm": float(
            max(
                max_abs_torque_by_actuator[model.actuator("left_hip_roll").id],
                max_abs_torque_by_actuator[model.actuator("right_hip_roll").id],
            )
        ),
        "rated_torque_exceed_fraction": rated_exceed_steps / samples,
        "finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
    }
    result["contract_pass"] = bool(
        abs(total_mass - EXPECTED_MASS) <= 1e-4
        and model.nu == EXPECTED_ACTIONS
        and result["finite"]
        and np.all(max_abs_torque_by_actuator <= measured_limits + 1e-5)
        and np.allclose(model.actuator_forcerange[:, 1], measured_limits)
        and np.allclose(model.actuator_forcerange[:, 0], -measured_limits)
    )
    result["static_hold_pass"] = bool(
        result["contract_pass"] and up_z >= 0.95 and float(data.qpos[2]) >= initial_z * 0.85
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    result = validate(args.duration, video=args.video, fps=args.fps)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if result["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
