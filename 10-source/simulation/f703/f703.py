"""Portable entry point for the frozen F703 milestone; no device operations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    import yaml

    import unilab

    assert platform.system() == "Darwin" and platform.machine() == "arm64", (
        "Use Apple Silicon macOS."
    )
    assert Path(unilab.__file__).resolve().is_relative_to(ROOT / "src"), (
        "Wrong UniLab installation."
    )
    identity = json.loads((ROOT / "evidence/source_identity.json").read_text())
    for relative, expected in identity["frozen_training_sources_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    original = json.loads((ROOT / "checkpoints/f703/run_config.json").read_text())["config"]
    for mode, checkpoint in [("reproduce", "parent502"), ("continue", "f703")]:
        actual = yaml.safe_load((ROOT / f"configs/f703_{mode}.yaml").read_text())
        expected = copy.deepcopy(original)
        expected["algo"]["load_run"] = f"checkpoints/{checkpoint}/model_200.safetensors"
        expected["training"]["log_root"] = f"outputs/{mode}"
        assert actual == expected, f"Frozen config drift: {mode}"
    assert sha(ROOT / "checkpoints/f703/model_200.safetensors") == identity["f703_sha256"]
    for relative, expected in identity["payload_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    differences = []
    for name in (
        "numpy",
        "mlx",
        "mlx-metal",
        "mujoco-uni",
        "onnxruntime",
        "numba",
        "scipy",
        "torch",
    ):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        expected = identity["source_runtime"]["packages"][name]
        if actual != expected:
            differences.append(f"{name}: current={actual}, reference={expected}")
    if sys.version_info[:2] != (3, 13):
        differences.append(f"Python: current={sys.version.split()[0]}, reference=3.13.13")
    print("PASS: F703 sources, model, assets and training configuration.", flush=True)
    if differences:
        print(
            "Reference-version differences (advisory, no automatic installation):\n  "
            + "\n  ".join(differences),
            flush=True,
        )
    else:
        print("Runtime versions match the recorded reference.", flush=True)


def train(args):
    verify()
    command = [
        sys.executable,
        str(ROOT / "scripts/train_mlx_ppo.py"),
        "--config-path",
        str(ROOT / "configs"),
        "--config-name",
        f"f703_{args.mode}",
    ]
    if args.iterations is not None:
        if args.iterations < 1:
            raise ValueError("iterations must be positive")
        command.append(f"algo.max_iterations={args.iterations}")
    if args.output:
        command.append(f"training.log_root={Path(args.output).resolve()}")
    subprocess.run(command, cwd=ROOT, check=True)


def fixed_observation(state, command):
    import numpy as np

    count = state.obs["obs"].shape[0]
    twist = np.broadcast_to(command, (count, 3)).astype(np.float32, copy=True)
    state.info["commands"] = twist.copy()
    state.info["head_commands"] = np.zeros((count, 4), np.float32)
    state.info["body_commands"] = np.zeros((count, 6), np.float32)
    obs = np.asarray(state.obs["obs"], np.float32).copy()
    assert obs.shape == (count, 61)
    obs[:, 48:51] = twist
    obs[:, 51:61] = 0
    return obs


def evaluate(args):
    verify()
    import numpy as np
    import onnxruntime as ort

    from unilab.base import registry
    from unilab.utils.rotation import np_yaw_from_quat

    cfg = json.loads((ROOT / "checkpoints/f703/run_config.json").read_text())["config"]
    ec = copy.deepcopy(cfg["env"])
    ec["reward_config"] = copy.deepcopy(cfg["reward"])
    ec["identified_dynamics"]["response_time_constant_range_s"] = [args.response_time] * 2
    ec.update(
        max_episode_seconds=19.0,
        command_resample_interval_s=0.0,
        fixed_remote_sequence=False,
        fixed_remote_sequence_fraction=0.0,
    )
    model = ROOT / "policy/V112_F703_Forward_web_actions_huiyo.onnx"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def infer(obs):
        return session.run(["actions"], {"obs": obs})[0]

    if args.checkpoint:
        import mlx.core as mx

        model = Path(args.checkpoint).resolve()
        weights = {k: np.asarray(v) for k, v in mx.load(str(model)).items()}
        layers = sorted(
            int(k.split(".")[2])
            for k in weights
            if k.startswith("actor.layers.") and k.endswith(".weight")
        )
        assert len(layers) == 4

        def infer(obs):
            value = obs
            for i in layers:
                value = (
                    value @ weights[f"actor.layers.{i}.weight"].T
                    + weights[f"actor.layers.{i}.bias"]
                )
                if i != layers[-1]:
                    value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
            return value.astype(np.float32)

    out = (
        Path(args.output).resolve()
        if args.output
        else ROOT / "outputs" / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True, exist_ok=False)
    registry.ensure_registries()
    np.random.seed(args.seed)
    env = registry.make(
        cfg["training"]["task_name"], sim_backend="mujoco", num_envs=8, env_cfg_override=ec
    )
    env.set_autoreset(False)
    records, current = [], 0

    def command(step):
        return np.array([0.3 if 250 <= step < 650 else 0, 0, 0], np.float32)

    try:
        state = env.init_state()
        assert np.isclose(env.cfg.ctrl_dt, 0.02)

        def advance(obs):
            nonlocal state, current
            actions = infer(np.asarray(obs, np.float32))
            assert actions.shape == (8, 14) and np.isfinite(actions).all()
            state = env.step(actions)
            current += 1
            if state.terminated.any():
                raise RuntimeError(f"Terminated at step {current}: {state.terminated.tolist()}")
            records.append(
                {
                    "t_s": current * 0.02,
                    "tilt_deg": np.rad2deg(
                        np.arccos(np.clip(-env._projected_gravity()[:, 2], -1, 1))
                    ).tolist(),
                    "velocity_body_xyz_m_s": env.get_local_linvel().tolist(),
                    "yaw_rad": np_yaw_from_quat(env._backend.get_base_quat()).tolist(),
                }
            )
            return fixed_observation(state, command(current))

        if args.video:
            env.run_playback_mode(
                play_render_mode="record",
                play_steps=900,
                output_video=out / "walk_stop.mp4",
                initialize=lambda: fixed_observation(state, command(0)),
                step=advance,
                camera_kwargs={
                    "cam_distance": 0.95,
                    "cam_elevation": -15.0,
                    "cam_azimuth": -80.0,
                    "cam_tracking": True,
                    "cam_tracking_env_idx": 0,
                    "cam_tracking_extra_envs": 0,
                },
            )
        else:
            obs = fixed_observation(state, command(0))
            for _ in range(900):
                obs = advance(obs)
        velocities = np.asarray([r["velocity_body_xyz_m_s"] for r in records])
        tilts = np.asarray([r["tilt_deg"] for r in records])
        forward = velocities[600:650, :, 0].mean(axis=0)
        stopped = np.linalg.norm(velocities[850:900, :, :2], axis=2).mean(axis=0)
        passed = bool(np.all(forward >= 0.18) and np.all(stopped < 0.02))
        result = {
            "passed": passed,
            "scope": "8-env 18s simulation; not heading or hardware qualification",
            "seed": args.seed,
            "response_time_s": args.response_time,
            "model_sha256": sha(model),
            "frames_per_env": current,
            "max_tilt_deg_by_env": tilts.max(axis=0).tolist(),
            "forward_last_second_vx_m_s": forward.tolist(),
            "stop_last_second_xy_speed_m_s": stopped.tolist(),
            "records": records,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
        print(f"Output: {out}")
        if not passed:
            raise RuntimeError("Forward/stop behavior check failed; see result.json")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("check")
    p = subs.add_parser("train")
    p.add_argument("--mode", choices=["reproduce", "continue"], default="continue")
    p.add_argument("--iterations", type=int)
    p.add_argument("--output")
    p = subs.add_parser("evaluate")
    p.add_argument("--checkpoint")
    p.add_argument("--seed", type=int, default=1601)
    p.add_argument("--response-time", type=float, choices=[0.001, 0.005, 0.01], default=0.01)
    p.add_argument("--video", action="store_true")
    p.add_argument("--output")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "check":
        verify()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
