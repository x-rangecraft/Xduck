"""Bounded rise-only PPO blocks with COM-only curriculum acceptance."""

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from unilab.training.xduck_curriculum import ROOT, load_actor
from unilab.training.xduck_imitation import rollout, stage_config
from unilab.training.xduck_pretrain import TensorPolicy


def passes(windows):
    """Require two paired windows, both directions, without hidden regressions."""
    return len(windows) >= 2 and all(
        new["initial_domains"] == old["initial_domains"]
        and all(
            new[d]["trials"] >= 16
            and new[d]["rate"] >= 0.9
            and new[d]["successes"] >= old[d]["successes"]
            for d in ("rise", "descent")
        )
        for old, new in windows
    )


def run(
    output,
    source,
    iterations=64,
    start_com=0.25,
    checkpoint_override=None,
    accepted_checkpoint=None,
):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    template = OmegaConf.load(source / "final_config.yaml")
    selected = json.loads((source / "recovery_evaluation.json").read_text())["selected_checkpoint"]
    initial_path = checkpoint_override or Path(selected)
    initial = torch.load(initial_path, weights_only=False, map_location="cpu")
    # Refit the critic before updating the actor under the new reset distribution.
    initial["recovery_updates"] = 0
    checkpoint = output / "initial.pt"
    torch.save(initial, checkpoint)
    status = {
        "status": "starting",
        "pid": os.getpid(),
        "selected_checkpoint": str(accepted_checkpoint or selected),
        "initial_checkpoint": str(initial_path),
        "start_com": start_com,
        "explicit_gate_override": checkpoint_override is not None,
    }

    def save():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / "status.json")

    save()
    try:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            if fraction < start_com:
                continue
            stage = output / f"com_{fraction:g}"
            stage.mkdir()
            evaluation_cfg = copy.deepcopy(template)
            for name in ("com_range_stages", "head_com_range_stages"):
                for item in evaluation_cfg.env.curriculum[name]:
                    item.value *= fraction
            cfg = copy.deepcopy(evaluation_cfg)
            cfg.env.sit_reset_probability = 1.0
            cfg.env.sit_command_probability = 0.0
            cfg.env.max_episode_seconds = 4.0
            cfg.algo.algorithm.class_name = "unilab.training.xduck_ppo:RecoveryPPO"
            cfg.algo.algorithm.critic_warmup_updates = 4
            cfg.algo.algorithm.critic_steps = 16
            cfg.algo.algorithm.critic_learning_rate = 3e-4
            cfg.algo.algorithm.disable_finite_checks = False
            cfg.algo.algorithm.diagnostics_path = str(stage / "feedback.jsonl")
            cfg.algo.load_run = str(checkpoint)
            cfg.algo.max_iterations = iterations
            cfg.algo.num_envs = 64
            cfg.algo.save_interval = 8
            cfg.algo.seed = 159021 + int(100 * fraction)
            cfg.training.device = "cpu"
            cfg.training.no_play = True
            cfg.training.log_dir = str(stage / "train")
            OmegaConf.save(cfg, stage / "config.yaml")
            status.update(status="training", com_fraction=fraction, stage_directory=str(stage))
            save()
            with (stage / "console.log").open("w") as log:
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
                        str(stage),
                        "--config-name",
                        "config",
                    ],
                    cwd=ROOT,
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            summary = json.loads((stage / "train/run_summary.json").read_text())
            candidate = Path(summary["last_checkpoint"])
            status.update(status="evaluating", candidate_checkpoint=str(candidate))
            save()
            previous_actor = load_actor(checkpoint, template)
            candidate_actor = load_actor(candidate, template)
            comparisons = {}
            for name, test_cfg in (
                ("nominal", stage_config(template, 0)),
                ("com_stage", evaluation_cfg),
            ):
                windows = []
                for seed in (208021, 218021):
                    pair = []
                    for actor in (previous_actor, candidate_actor):
                        metrics, _ = rollout(
                            test_cfg,
                            ROOT,
                            actor=actor,
                            count=16,
                            segments=2,
                            seed=seed,
                            independent=True,
                        )
                        pair.append(metrics)
                    windows.append(pair)
                comparisons[name] = {"windows": windows, "passed": passes(windows)}
                (stage / "evaluation.json").write_text(json.dumps(comparisons, indent=2))
            if not all(v["passed"] for v in comparisons.values()):
                status.update(status="stopped_gate_failed", failed_stage=fraction)
                save()
                return
            checkpoint = candidate
            status["selected_checkpoint"] = str(checkpoint)
            exported = torch.jit.trace(TensorPolicy(candidate_actor), torch.zeros(1, 62))
            torch.jit.save(exported, str(stage / "actor_accepted.pt"))
            save()
        status["status"] = "curriculum_passed"
        save()
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--start-com", type=float, choices=(0.25, 0.5, 0.75, 1.0), default=0.25)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--accepted-checkpoint", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    run(
        args.output.resolve(),
        args.source.resolve(),
        args.iterations,
        args.start_com,
        args.checkpoint.resolve() if args.checkpoint else None,
        args.accepted_checkpoint.resolve() if args.accepted_checkpoint else None,
    )
