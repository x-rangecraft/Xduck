"""Evaluate fixed-command trials through the unchanged registered Xduck owner."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-mode", choices=("curriculum", "fixed_zero", "pitch_only", "neck_only"), default="curriculum")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    sys.path.insert(0, str(args.project))
    from unilab.training import BackendAdapter, create_env, ensure_registries
    from unilab.envs.locomotion.xduck_gf43x40.pure import CASE_NAMES

    torch.set_num_threads(2)
    ensure_registries()
    config = OmegaConf.create(json.loads((args.checkpoint.parent / "run_config.json").read_text())["config"])
    geometric_head = bool(config.env.get("geometric_head_enabled", False))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    progress = saved["infos"]["unilab_progress"]["env_steps_per_env"]
    actor_cfg = OmegaConf.to_container(config.algo.actor, resolve=True)
    actor_cfg.pop("class_name")
    actor = MLPModel(TensorDict({"actor": torch.zeros(16, 61)}, batch_size=[16]),
                     {"actor": ["actor"]}, "actor", 14, **actor_cfg)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    rows, trajectories = [], {}
    for seed in (6401, 7401):
        for case, label in enumerate(CASE_NAMES):
            np.random.seed(seed + case * 100)
            torch.manual_seed(seed + case * 100)
            cfg = copy.deepcopy(config)
            cfg.env.evaluation_case = case
            cfg.env.posture_dwell_seconds = [100., 100.]
            cfg.env.max_episode_seconds = 6.1
            cfg.env.curriculum.step_cap = progress
            if args.head_mode == "fixed_zero":
                for item in cfg.env.curriculum.head_range_stages:
                    item.ranges = [[0., 0.] for _ in range(4)]
            elif args.head_mode == "pitch_only":
                # Preserve pitch sampling and RNG draw counts; zero yaw/roll only.
                for item in cfg.env.curriculum.head_range_stages:
                    item.ranges[2] = [0., 0.]
                    item.ranges[3] = [0., 0.]
            elif args.head_mode == "neck_only":
                for item in cfg.env.curriculum.head_range_stages:
                    for joint in (1, 2, 3):
                        item.ranges[joint] = [0., 0.]
            env = create_env(cfg, num_envs=16,
                env_cfg_override=BackendAdapter(cfg, root_dir=args.project).build_task_env_cfg_override())
            try:
                env.step_counter = progress
                env._update_curriculum()
                env.init_state()
                env.set_autoreset(False)
                failed = np.zeros(16, dtype=bool)
                traces = {k: [] for k in ("height", "qvel", "action", "pose_error", "success", "clip", "command_effect", "qpos", "target", "head_error", "height_error", "tilt", "base_speed", "commands")}
                if geometric_head:
                    traces["head_tilt"] = []
                for _ in range(300):
                    obs = TensorDict({"actor": torch.from_numpy(env.state.obs["obs"].copy())}, batch_size=[16])
                    with torch.inference_mode():
                        action = actor(obs).numpy()
                        alternate = obs.clone()
                        alternate["actor"][:, 48] = 1 - alternate["actor"][:, 48]
                        other = actor(alternate).numpy()
                    state = env.step(action)
                    failed |= env._failed | state.terminated | state.truncated
                    scale = env.cfg.control_config.action_scale * env.cfg.policy_action_scale_rad
                    incremental = cfg.env.pure_policy_revision == 6
                    target = env._decoded_raw_target.copy() if incremental else env.default_angles + action * scale
                    command_effect = (np.tanh(action) - np.tanh(other)) * env._increments if incremental else (action - other) * scale
                    if incremental and cfg.env.get('head_absolute_position', False):
                        command_effect[:, 5:9] = (action[:, 5:9] - other[:, 5:9]) * cfg.env.head_absolute_scale_rad
                    qpos = env._backend.get_physics_state()[:, 1:22]
                    head_goal = env.default_angles[5:9] + state.info["commands"][:, 3:7]
                    values = dict(qpos=qpos, target=env._current_action_target.copy(), head_error=qpos[:, 12:16] - head_goal,
                        height=env._backend.get_base_pos()[:, 2], qvel=env.get_dof_vel(),
                        action=action, pose_error=env.last_pose_error, success=env.last_success,
                        height_error=env.last_height_error, tilt=env.last_tilt, base_speed=env.last_speed,
                        commands=state.info["commands"],
                        clip=(target < env._target_low) | (target > env._target_high),
                        command_effect=np.sqrt(np.mean(command_effect ** 2, axis=1)))
                    if geometric_head:
                        values["head_tilt"] = env._geometric_head_score()[0]
                    for key, value in values.items():
                        traces[key].append(value.copy())
                data = {key: np.asarray(value) for key, value in traces.items()}
                # Owner success already requires 0.5 s of the original strict hold criteria.
                # Additionally require it throughout the last second and remember every fall.
                owner_sustained = data["success"][-50:].all(axis=0) & ~failed
                head_sustained = ((data["head_tilt"][-50:] < np.deg2rad(10)).all(axis=0)
                                  if geometric_head else (np.abs(data["head_error"][-50:]) < .15).all(axis=(0, 2)))
                sustained = owner_sustained & head_sustained
                row = dict(seed=seed, case=label, trials=16, successes=int(sustained.sum()), owner_successes=int(owner_sustained.sum()),
                    failures=int(failed.sum()), late_height_mean=float(data["height"][-150:].mean()),
                    late_joint_speed_rms=float(np.sqrt(np.mean(data["qvel"][-150:] ** 2))),
                    clip_fraction=float(data["clip"].mean()),
                    late_action_delta_rms=float(np.sqrt(np.mean(np.diff(data["action"][-150:], axis=0) ** 2))),
                    late_pose_error_rad=float(data["pose_error"][-150:].mean()),
                    late_head_error_rms_rad=float(np.sqrt(np.mean(data["head_error"][-150:] ** 2))),
                    late_target_delta_rms_rad=float(np.sqrt(np.mean(np.diff(data["target"][-150:], axis=0) ** 2))),
                    target_excursion_rms_rad=float(np.sqrt(np.mean(np.ptp(data["target"], axis=0) ** 2))),
                    command_effect_rms=float(data["command_effect"][-150:].mean()))
                rows.append(row)
                # Independent blockers may overlap; do not sum these counts.
                conditions = {
                    "height": np.abs(data["height_error"][-50:]) < .015,
                    "tilt": data["tilt"][-50:] < np.deg2rad(10),
                    "base_speed": data["base_speed"][-50:] < .04,
                    "pose": data["pose_error"][-50:] < .1,
                    "joint_speed": np.max(np.abs(data["qvel"][-50:]), axis=2) < .5,
                    "head": (np.abs(data["head_error"][-50:]) < .15).all(axis=2),
                }
                if geometric_head:
                    conditions.pop("pose")
                    conditions["head"] = data["head_tilt"][-50:] < np.deg2rad(10)
                    row["late_head_tilt_mean_deg"] = float(np.rad2deg(data["head_tilt"][-50:]).mean())
                row["success_criterion"] = "geometric_head_no_joint_pose" if geometric_head else "legacy_joint_pose_and_head_tracking"
                # Report movement quality separately from endpoint success.
                # Full-trajectory peaks include the original noisy resets.
                angles = data["qpos"][:, :, 12:16]
                excursion = np.ptp(angles, axis=0)
                peak_body = data["tilt"].max(axis=0)
                row["transition_head_joint_excursion_median_deg"] = np.rad2deg(np.median(excursion, axis=0)).tolist()
                row["transition_body_peak_tilt_median_deg"] = float(np.rad2deg(np.median(peak_body)))
                row["transition_body_peak_tilt_max_deg"] = float(np.rad2deg(peak_body.max()))
                row["transition_head_joint_peak_speed_rad_s"] = np.abs(data["qvel"][:, :, 5:9]).max(axis=(0, 1)).tolist()
                if geometric_head:
                    row["transition_head_peak_tilt_median_deg"] = float(np.rad2deg(np.median(data["head_tilt"].max(axis=0))))
                row["last_second_condition_violations"] = {
                    name: int((~mask.all(axis=0)).sum()) for name, mask in conditions.items()
                }
                for key, value in data.items():
                    trajectories[f"{label}_{seed}_{key}"] = value
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        env_steps_per_env=progress, deterministic=True,
        command_effect_units=("legs: radians per control-tick increment; head: radians of absolute target"
                              if config.env.get('head_absolute_position', False) else
                              "radians per control-tick target increment" if config.env.pure_policy_revision == 6 else "radians of absolute target"),
        head_mode=args.head_mode,
        conditions="Original physical/reset/observation DR retained; fixed sit/stand command; curriculum frozen at checkpoint. Head mode: " + args.head_mode,
        criterion=("Geometric height/body/head uprightness and stationarity; no joint pose requirement; minimum0.5s hold and success throughout final1s; cumulative fall mask retained."
                   if geometric_head else "Original owner success for entire final second; cumulative original owner fall mask; all four head joint errors <0.15 rad for entire final second."), rows=rows), indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"), **trajectories)


if __name__ == "__main__":
    main()
