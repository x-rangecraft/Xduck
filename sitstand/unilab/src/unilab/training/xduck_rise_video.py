"""Record neural-policy sit/stand timing and render actual simulation poses."""

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

import unilab.training.xduck_imitation as imitation
from unilab.training.xduck_curriculum import ROOT, load_actor


def run(output, direction="rise", render_video=True):
    if direction not in ("rise", "descent"):
        raise ValueError("Direction must be rise or descent")
    sitting = direction == "descent"
    target_height = 0.2182 if sitting else 0.3129
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "training_runs/xduck_rise_com_continue100_20260929/com_1/train/model_267.pt"
    cfg = OmegaConf.load(ROOT / "training_runs/xduck_feedback_repair_v4_20260929/final_config.yaml")
    actor = load_actor(checkpoint, cfg)
    torch.set_num_threads(4)
    original_make = imitation.make_env
    model = None
    joint_names = None
    clips = []
    report = {
        "checkpoint": str(checkpoint),
        "direction": direction,
        "playback_speed": 1,
        "video_selection": "First seed, environment index 0, without success filtering",
        "cases": {},
    }
    for label, current, seeds in (
        ("NOMINAL", imitation.stage_config(cfg, 0), (208021,)),
        ("FULL DR", cfg, (208021, 218021)),
    ):
        traces = []

        def make(*args, **kwargs):
            nonlocal model, joint_names
            env = original_make(*args, **kwargs)
            if model is None:
                model = mujoco.MjModel.from_xml_path(env._backend.get_scene_model_file())
                joint_names = list(env.cfg.asset.policy_joint_names)
            rows = []
            traces.append(rows)
            terminated = env._terminated

            def done(up):
                flags = terminated(up)
                if (env.state.info["commands"][0, 0] > 0.5) == sitting:
                    pos = env._backend.get_base_pos().copy()
                    quat = env._backend.get_base_quat().copy()
                    q = env.get_dof_pos().copy()
                    speed = np.linalg.norm(env._backend.get_base_lin_vel(), axis=1)
                    tilt = np.degrees(
                        np.arccos(np.clip(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1))
                    )
                    stable = (
                        (abs(pos[:, 2] - target_height) < 0.01)
                        & (tilt < 5)
                        & (speed < 0.03)
                        & (
                            np.sqrt(np.mean((q - env._refs[direction].nodes[-1]) ** 2, axis=1))
                            < 0.08
                        )
                    )
                    rows.append(
                        {
                            "base_pos": pos,
                            "quat": quat,
                            "joints": q,
                            "speed": speed,
                            "tilt": tilt,
                            "stable": stable,
                            "done": flags.copy(),
                        }
                    )
                return flags

            env._terminated = done
            return env

        imitation.make_env = make
        windows = []
        try:
            for seed in seeds:
                metrics, _ = imitation.rollout(
                    current, ROOT, actor=actor, count=16, segments=2, seed=seed, independent=True
                )
                windows.append(metrics)
        finally:
            imitation.make_env = original_make
        batches = [{k: np.stack([x[k] for x in rows]) for k in rows[0]} for rows in traces if rows]
        times = []
        for bi, batch in enumerate(batches):
            np.savez_compressed(output / f"{label.replace(' ', '_')}_{bi}.npz", **batch)
            for e in range(16):
                failed = batch["done"][:, e].any()
                success = not failed and batch["stable"][-25:, e].all()
                height = batch["base_pos"][:, e, 2]
                reached = np.flatnonzero(abs(height - target_height) < 0.01)
                stable_end = None
                for t in range(24, len(height)):
                    if (
                        batch["stable"][t - 24 : t + 1, e].all()
                        and not batch["done"][: t + 1, e].any()
                    ):
                        stable_end = (t + 1) * 0.02
                        break
                times.append(
                    {
                        "seed": seeds[bi],
                        "environment": e,
                        "success": bool(success),
                        "first_height_seconds": float((reached[0] + 1) * 0.02)
                        if len(reached)
                        else None,
                        "stable_confirmation_seconds": stable_end,
                    }
                )
        report["cases"][label] = {"windows": windows, "timings": times}
        clips.append((label, batches[0]))
    (output / "timing.json").write_text(json.dumps(report, indent=2))
    if render_video:
        render(output, model, joint_names, clips, report)


def render_saved(output):
    from unilab.base.backend.mujoco.xml import materialize_scene_fragments
    from unilab.envs.locomotion.xduck_gf43x40.direct import XDuckSitStandDirectCfg

    cfg = XDuckSitStandDirectCfg()
    scene = materialize_scene_fragments(
        cfg.scene.model_file, fragment_files=cfg.scene.fragment_files
    )
    try:
        model = mujoco.MjModel.from_xml_path(scene)
    finally:
        Path(scene).unlink(missing_ok=True)
    clips = [
        (label, dict(np.load(output / f"{label.replace(' ', '_')}_0.npz")))
        for label in ("NOMINAL", "FULL DR")
    ]
    render(
        output,
        model,
        cfg.asset.policy_joint_names,
        clips,
        json.loads((output / "timing.json").read_text()),
    )


def render(output, model, joint_names, clips, report):
    direction = report.get("direction", "rise")
    action_label = "SIT DOWN" if direction == "descent" else "STAND UP"
    model.mat_rgba[:, 3] = 1
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 1.5, -8
    addresses = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ]
    video = output / f"latest_model_{direction}.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "960x640",
        "-pix_fmt",
        "rgb24",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(video),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for label, batch in clips:
            terminal = np.flatnonzero(batch["done"][:, 0])
            for frame in range(0, 200, 2):
                index = min(frame, int(terminal[0])) if len(terminal) else frame
                data.qpos[:3] = batch["base_pos"][index, 0]
                data.qpos[3:7] = batch["quat"][index, 0]
                data.qpos[addresses] = batch["joints"][index, 0]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.30]
                w, x, y, z = data.qpos[3:7]
                camera.azimuth = 90 + np.degrees(
                    np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                )
                renderer.update_scene(data, camera, scene_option=option)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 960, 86), fill=(18, 23, 31))
                draw.text(
                    (18, 9),
                    f"LATEST PPO model_267 | {action_label} | {label} | 1x REAL TIME",
                    fill="white",
                    font=ImageFont.load_default(size=20),
                )
                draw.text(
                    (18, 39),
                    f"Command t={(frame + 1) * 0.02:.2f}s | height={batch['base_pos'][index, 0, 2]:.3f}m | tilt={batch['tilt'][index, 0]:.1f}deg | speed={batch['speed'][index, 0]:.3f}m/s",
                    fill="white",
                    font=ImageFont.load_default(size=17),
                )
                state = (
                    "TERMINATED - FRAME HELD"
                    if len(terminal) and frame >= terminal[0]
                    else (
                        "Within target posture thresholds"
                        if batch["stable"][index, 0]
                        else "Moving / settling"
                    )
                )
                draw.text(
                    (18, 64),
                    f"seed 208021, env 0 | {state}",
                    fill=(255, 210, 100),
                    font=ImageFont.load_default(size=15),
                )
                encoder.stdin.write(np.asarray(im).tobytes())
                if frame == 100:
                    im.save(output / f"preview_{label.replace(' ', '_')}.png")
    finally:
        encoder.stdin.close()
        encoder.wait()
        renderer.close()
    if encoder.returncode:
        raise RuntimeError("Video encoder failed")
    print(video, flush=True)
    for label, case in report["cases"].items():
        successes = [r for r in case["timings"] if r["success"]]
        print(
            label,
            "success",
            len(successes),
            "timing",
            {
                k: [
                    float(np.min([r[k] for r in successes])),
                    float(np.median([r[k] for r in successes])),
                    float(np.max([r[k] for r in successes])),
                ]
                for k in ("first_height_seconds", "stable_confirmation_seconds")
            },
            flush=True,
        )


if __name__ == "__main__":
    run(ROOT / "training_runs/xduck_latest_rise_video_20260929")
