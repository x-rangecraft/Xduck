"""Independent evaluation and bounded, validation-gated pure PPO stages."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.envs.locomotion.xduck_gf43x40.pure import CASE_NAMES
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.rsl_rl_reset import RslRlVecEnvWrapper

ROOT = Path(__file__).resolve().parents[3]
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def require_pure_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    provenance = json.loads((checkpoint.parent / "run_config.json").read_text())
    if provenance["config"]["training"]["task_name"] not in (
        "XDuckGF43X40SitStandPureFlat", "XDuckGF43X40SitStandMicroduckPPOFlat",
        "XDuckGF43X40SitStandContinuousPPOFlat",
        "XDuckGF43X40SitStandDiscoveryPilotFlat",
    ):
        raise ValueError("Requires a pure PPO checkpoint; 62D alone is not compatibility")
    revision = provenance["config"]["env"].get("pure_policy_revision")
    expected_actor = {
        2: "unilab.training.xduck_pure_actor:PureActor",
        3: "rsl_rl.models.MLPModel",
        4: "unilab.training.xduck_pure_actor:ContinuousTargetActor",
        5: "rsl_rl.models.MLPModel",
    }.get(revision)
    actual_actor = provenance["config"]["algo"].get("actor", {}).get("class_name")
    if expected_actor is None or actual_actor != expected_actor:
        raise ValueError("Pure policy history/actor revision mismatch; retrain from scratch")
    is_continuous = provenance["config"]["training"]["task_name"] == "XDuckGF43X40SitStandContinuousPPOFlat"
    if is_continuous != (revision == 4):
        raise ValueError("Executed-target memory requires the revision 4 task")
    is_pilot = provenance["config"]["training"]["task_name"] == "XDuckGF43X40SitStandDiscoveryPilotFlat"
    if is_pilot != (revision == 5):
        raise ValueError("Discovery pilot requires the revision 5 task")
    return checkpoint


def base_config():
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["task=xduck_gf43x40_sitstand_pure_flat/mujoco"]
        )
    if cfg.reward.progress_discount != cfg.algo.algorithm.gamma:
        raise ValueError("Potential shaping and PPO must use the same discount")
    return cfg


def stage_config(template, level):
    cfg = copy.deepcopy(template)
    cfg.env.dr_level = level
    # Improve smoothness only after nominal skill passed; no early rise tax.
    cfg.reward.scales.target_rate = -0.02 if level == 0 else -0.05
    cfg.reward.scales.torque_rate = 0.0 if level == 0 else -0.005
    cfg.reward.scales.rise_speed = 0.0 if level < 0.5 else -0.5
    return cfg


def evaluate(
    cfg, checkpoint, *, seeds=(6401, 7401), num_envs=16, continuous=False, cases=(0, 1, 2, 3)
):
    """No automatic resets; cumulative failure masks survive all action segments."""
    checkpoint = require_pure_checkpoint(checkpoint)
    ensure_registries()
    windows = []
    for seed in seeds:
        for case in cases if not continuous else (0,):
            np.random.seed(seed + case * 100)
            torch.manual_seed(seed + case * 100)
            local = copy.deepcopy(cfg)
            local.env.nominal_env_fraction = 0.0
            local.env.evaluation_case = case
            local.env.posture_dwell_seconds = [100.0, 100.0]
            local.env.max_episode_seconds = 30.0 if continuous else 6.1
            local.env.adaptive_chunk_size = False
            env = create_env(
                local,
                num_envs=num_envs,
                env_cfg_override=BackendAdapter(local, root_dir=ROOT).build_task_env_cfg_override(),
            )
            try:
                wrapped = RslRlVecEnvWrapper(env, device="cpu")
                env.set_autoreset(False)
                train_cfg = normalize_ppo_train_cfg(
                    OmegaConf.to_container(local.algo, resolve=True)
                )
                train_cfg["runner"] = {"logger": "none"}
                runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
                runner.load(str(checkpoint), map_location="cpu")
                policy = runner.get_inference_policy(device="cpu")
                failed = np.zeros(num_envs, dtype=bool)
                segments = (
                    [
                        (0.0, 3.0, "stand_hold"),
                        (1.0, 6.0, "descent"),
                        (0.0, 6.0, "rise"),
                        (1.0, 6.0, "descent"),
                        (0.0, 6.0, "rise"),
                    ]
                    if continuous
                    else [(float(case in (1, 3)), 6.0, CASE_NAMES[case])]
                )
                for command, seconds, label in segments:
                    clipped = []
                    env.state.info["commands"][:, 0] = command
                    env.state.obs["obs"][:, 48] = command
                    env.state.obs["critic"][:, 48] = command
                    env._completion_paid[:] = False
                    env._goal_hold_steps[:] = 0
                    peak_descent = np.zeros(num_envs)
                    peak_acceleration = np.zeros(num_envs)
                    prior_vz = env._backend.get_base_lin_vel()[:, 2].copy()
                    total_return = np.zeros(num_envs)
                    late_joint_speed_squared = []
                    late_height = []
                    steps = round(seconds / env.cfg.ctrl_dt)
                    for step in range(steps):
                        with torch.inference_mode():
                            action = policy(wrapped.get_observations()).cpu().numpy()
                        state = env.step(action)
                        if not all(np.isfinite(obs).all() for obs in state.obs.values()):
                            raise FloatingPointError("Nonfinite evaluation observation")
                        if not np.isfinite(env.get_dof_pos()).all():
                            raise FloatingPointError("Nonfinite evaluation physical state")
                        was_alive = ~failed
                        # Official-style training allows fallen robots to keep
                        # acting; evaluation must still remember physical falls.
                        failed |= state.terminated | state.truncated | env._failed
                        total_return += np.where(was_alive, state.reward, 0)
                        vz = env._backend.get_base_lin_vel()[:, 2]
                        peak_descent = np.maximum(peak_descent, -vz)
                        peak_acceleration = np.maximum(
                            peak_acceleration, np.abs(vz - prior_vz) / env.cfg.ctrl_dt
                        )
                        prior_vz = vz.copy()
                        clipped.append(state.info["log"]["pure/target_clip_fraction"])
                        if step >= max(0, steps - round(3.0 / env.cfg.ctrl_dt)):
                            late_joint_speed_squared.append(float(np.mean(env.get_dof_vel() ** 2)))
                            late_height.append(float(env._backend.get_base_pos()[:, 2].mean()))
                    success = env.last_success & ~failed
                    windows.append(
                        {
                            "seed": seed,
                            "case": label,
                            "continuous": continuous,
                            "trials": num_envs,
                            "successes": int(success.sum()),
                            "rate": float(success.mean()),
                            "failures": int(failed.sum()),
                            "max_descent_speed": float(peak_descent.max()),
                            "max_vertical_acceleration": float(peak_acceleration.max()),
                            "clip_fraction": float(np.mean(clipped)),
                            "mean_return": float(total_return.mean()),
                            "late_joint_speed_rms": float(np.sqrt(np.mean(late_joint_speed_squared))),
                            "late_height_mean": float(np.mean(late_height)),
                            "height_error_mean": float(np.mean(np.abs(env.last_height_error))),
                            "height_error_max": float(np.max(np.abs(env.last_height_error))),
                            "tilt_deg_max": float(np.rad2deg(env.last_tilt).max()),
                        }
                    )
            finally:
                env.close()
    return windows


def passes(windows, *, required_cases=None):
    # Operational starting gate, not a population confidence claim.
    required = (
        {"stand_hold", "rise", "descent"}
        if windows and all(w.get("continuous", False) for w in windows)
        else set(CASE_NAMES)
    )
    seeds = {w["seed"] for w in windows}
    if required_cases is not None:
        required = set(required_cases)
    return (
        len(seeds) >= 2
        and all(required <= {w["case"] for w in windows if w["seed"] == seed} for seed in seeds)
        and all(
            w["trials"] >= 16
            and w["rate"] >= 0.9
            and w["clip_fraction"] <= 0.02
            and w["max_descent_speed"] <= 0.16
            and w["max_vertical_acceleration"] <= 20.0
            for w in windows
        )
    )


def train_stages(output, *, blocks=1, iterations=100, num_envs=256, source=None, start_level=0.0):
    """Use the ordinary runner in subprocesses; failed gates stop advancement."""
    template = base_config()
    checkpoint = require_pure_checkpoint(source) if source else None
    if start_level > 0 and checkpoint is None:
        raise ValueError("A nonzero start stage requires an accepted pure checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "checkpoint": str(checkpoint) if checkpoint else None}
    try:
        stages = [(level, continuous) for level in LEVELS for continuous in (False, True)]
        for level, continuous_stage in stages:
            if level < start_level:
                continue
            cfg = stage_config(template, level)
            cfg.env.case_weights = [0.1, 0.1, 0.4, 0.4]
            if continuous_stage:
                cfg.env.max_episode_seconds = 30.0
                cfg.env.posture_dwell_seconds = [4.0, 6.0]
            if checkpoint is not None and level > 0 and not continuous_stage:
                # Check the preceding stage before entering a harder domain.
                previous_cfg = stage_config(
                    template, level if continuous_stage else max(0.0, level - 0.25)
                )
                prerequisite = evaluate(previous_cfg, checkpoint)
                prerequisite_cycles = evaluate(previous_cfg, checkpoint, continuous=True)
                if not passes(prerequisite) or not passes(prerequisite_cycles):
                    status.update(
                        status="prerequisite_failed",
                        level=level,
                        evaluation=prerequisite,
                        continuous=prerequisite_cycles,
                    )
                    break
            accepted = False
            for block in range(blocks):
                phase_name = f"cycles_dr_{level:g}" if continuous_stage else f"dr_{level:g}"
                folder = output / phase_name / f"block_{block:03d}"
                folder.mkdir(parents=True)
                cfg.algo.num_envs = num_envs
                cfg.algo.max_iterations = iterations
                cfg.algo.save_interval = min(25, iterations)
                cfg.training.log_dir = str(folder / "train")
                cfg.algo.load_run = str(checkpoint) if checkpoint else "-1"
                OmegaConf.save(cfg, folder / "config.yaml")
                with (folder / "console.log").open("w") as log:
                    subprocess.run(
                        [
                            "uv",
                            "run",
                            "--offline",
                            "--no-project",
                            "--python",
                            str(ROOT / ".venv/bin/python"),
                            "python",
                            "-u",
                            "-B",
                            str(ROOT / "scripts/train_rsl_rl.py"),
                            "--config-path",
                            str(folder),
                            "--config-name",
                            "config",
                        ],
                        cwd=ROOT,
                        env=os.environ.copy(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                models = list((folder / "train").glob("model_*.pt"))
                checkpoint = max(models, key=lambda p: int(p.stem.split("_")[-1]))
                cases = (0, 1, 2, 3)
                required = CASE_NAMES
                windows = evaluate(cfg, checkpoint, cases=cases)
                nominal = copy.deepcopy(cfg)
                nominal.env.dr_level = 0.0
                nominal_windows = (
                    windows if level == 0 else evaluate(nominal, checkpoint, cases=cases)
                )
                accepted = passes(windows, required_cases=required) and passes(
                    nominal_windows, required_cases=required
                )
                if accepted and continuous_stage:
                    cycles = evaluate(cfg, checkpoint, continuous=True)
                    nominal_cycles = (
                        cycles if level == 0 else evaluate(nominal, checkpoint, continuous=True)
                    )
                    accepted = passes(cycles) and passes(nominal_cycles)
                else:
                    cycles = []
                    nominal_cycles = []
                receipt = {
                    "level": level,
                    "phase": phase_name,
                    "checkpoint": str(checkpoint),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "accepted": accepted,
                    "windows": windows,
                    "nominal": nominal_windows,
                    "continuous": cycles,
                    "nominal_continuous": nominal_cycles,
                }
                (folder / "evaluation.json").write_text(json.dumps(receipt, indent=2))
                status.update(checkpoint=str(checkpoint), level=level, accepted=accepted)
                (output / "status.json").write_text(json.dumps(status, indent=2))
                if accepted:
                    break
            if not accepted:
                status["status"] = "gate_failed"
                break
        else:
            status["status"] = "simulation_gate_passed_hardware_unverified"
    except Exception as error:
        status.update(status="error", error=str(error))
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dr-level", type=float, choices=LEVELS, default=0.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.evaluate_only:
        if args.checkpoint is None:
            parser.error("--evaluate-only requires --checkpoint")
        args.output.mkdir(parents=True, exist_ok=False)
        cfg = stage_config(base_config(), args.dr_level)
        windows = evaluate(cfg, args.checkpoint)
        (args.output / "evaluation.json").write_text(
            json.dumps({"windows": windows, "accepted": passes(windows)}, indent=2)
        )
    else:
        if min(args.blocks, args.iterations, args.num_envs) < 1:
            parser.error("budgets must be positive")
        train_stages(
            args.output.resolve(),
            blocks=args.blocks,
            iterations=args.iterations,
            num_envs=args.num_envs,
            source=args.checkpoint,
            start_level=args.dr_level,
        )


if __name__ == "__main__":
    main()
