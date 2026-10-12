"""Bounded feedback ablations, isolated from production curriculum acceptance."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from unilab.training.xduck_curriculum import ROOT, load_actor
from unilab.training.xduck_imitation import rollout, stage_config


def ankle_config(cfg, degrees):
    result = stage_config(cfg, 0.0)
    result.env.ankle_target_offset_max_deg = degrees
    return result


def evaluate(cfg, checkpoint, seeds, count):
    actor = load_actor(checkpoint, cfg)
    output = {}
    for mode in ("full", "no_feedback"):
        policy = copy.deepcopy(actor)
        if mode == "no_feedback":
            with torch.no_grad():
                for parameter in policy.mlp.parameters():
                    parameter.zero_()
        output[mode] = []
        for seed in seeds:
            metrics, _ = rollout(
                cfg, ROOT, actor=policy, count=count, segments=2, seed=seed, independent=True
            )
            output[mode].append(metrics)
    return output


def run_experiment(
    source,
    output,
    iterations=16,
    num_envs=256,
    ankle_degrees=1.5,
    arms=("rehearsal", "no_rehearsal", "no_rehearsal_lr"),
    exploration_lr=1e-5,
):
    torch.set_num_threads(4)
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = OmegaConf.load(source / "final_config.yaml")
    base.algo.algorithm.demonstration_path = str(source / "demonstrations.npz")
    checkpoint = source / "imitation.pt"
    cfg = ankle_config(base, ankle_degrees)
    report = {
        "status": "running",
        "source": str(checkpoint),
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ankle_degrees": ankle_degrees,
        "iterations_per_arm": iterations,
        "num_envs": num_envs,
        "steps_per_iteration": 64,
        "training_seed": 921,
        "evaluation_seeds": [18021, 28021],
        "nominal_evaluation_seed": 38021,
        "exploration_lr": exploration_lr,
        "arms": {},
        "scope": "diagnostic only; no production checkpoint promotion",
    }

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2))

    save()
    # Same initialization, rollout length and random seed in all three arms.
    for name, rehearsal, lr in (
        ("rehearsal", 8, 6.25e-7),
        ("no_rehearsal", 0, 6.25e-7),
        ("no_rehearsal_lr", 0, exploration_lr),
    ):
        if name not in arms:
            continue
        arm = output / name
        arm.mkdir()
        current = copy.deepcopy(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Fresh optimizer state in EVERY arm; actor and critic initialization identical.
        for key in ("optimizer_state_dict", "rehearsal_optimizer_state_dict"):
            if key in state:
                state[key]["state"] = {}
                for group in state[key]["param_groups"]:
                    group["lr"] = lr if key == "optimizer_state_dict" else 2.5e-6
        state["adaptive_learning_rate"] = lr
        torch.save(state, arm / "initial.pt")
        current.algo.num_envs = num_envs
        current.algo.num_steps_per_env = 64
        current.algo.max_iterations = iterations
        current.algo.save_interval = max(1, iterations // 2)
        current.algo.seed = 921
        current.algo.load_run = str(arm / "initial.pt")
        current.algo.algorithm.rehearsal_steps = rehearsal
        current.algo.algorithm.learning_rate = lr
        current.algo.algorithm.learning_rate_ceiling = lr
        current.algo.algorithm.diagnostics_path = str(arm / "feedback.jsonl")
        current.training.device = "cpu"
        current.training.no_play = True
        current.training.log_dir = str(arm / "train")
        OmegaConf.save(current, arm / "config.yaml")
        report["active_arm"] = name
        save()
        print(f"Training {name}: {iterations} updates", flush=True)
        with (arm / "console.log").open("w") as log:
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
                    str(arm),
                    "--config-name",
                    "config",
                ],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((arm / "train/run_summary.json").read_text())
        trained = Path(summary["last_checkpoint"])
        print(f"Evaluating {name}", flush=True)
        report["arms"][name] = {
            "checkpoint": str(trained),
            "ankle": evaluate(cfg, trained, (18021, 28021), 16),
            "nominal": evaluate(stage_config(base, 0), trained, (38021,), 16),
        }
        save()
    report["status"] = "completed"
    report.pop("active_arm", None)
    save()
    print(f"Completed: {output / 'results.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--ankle-degrees", type=float, default=1.5)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("rehearsal", "no_rehearsal", "no_rehearsal_lr"),
        default=("rehearsal", "no_rehearsal", "no_rehearsal_lr"),
    )
    parser.add_argument("--exploration-lr", type=float, default=1e-5)
    args = parser.parse_args()
    if (
        args.iterations < 1
        or args.num_envs < 4
        or not 0 < args.ankle_degrees <= 3
        or not 0 < args.exploration_lr <= 1e-3
    ):
        parser.error("positive iterations, at least 4 envs, ankle degrees in (0,3] required")
    if args.output.exists():
        parser.error("output already exists; use a fresh directory")
    try:
        run_experiment(
            args.source,
            args.output,
            args.iterations,
            args.num_envs,
            args.ankle_degrees,
            args.arms,
            args.exploration_lr,
        )
    except Exception as error:
        report_path = args.output / "results.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.update(status="failed", error=str(error))
            report_path.write_text(json.dumps(report, indent=2))
        raise
