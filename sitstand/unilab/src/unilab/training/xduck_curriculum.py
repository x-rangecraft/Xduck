"""Run bounded PPO blocks and promote DR only on closed-loop success evidence."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel

from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.xduck_imitation import (
    DR_LEVELS,
    SuccessCurriculum,
    actor_batch,
    no_regression,
    rollout,
    stage_config,
)
from unilab.training.xduck_policy import CorrectiveTimingActor, TimingActor

ROOT = Path(__file__).resolve().parents[3]


def load_actor(checkpoint, cfg):
    model_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))["actor"]
    name = model_cfg.pop("class_name")
    model_type = (
        CorrectiveTimingActor
        if name.endswith("CorrectiveTimingActor")
        else TimingActor
        if name.endswith("TimingActor")
        else MLPModel
    )
    actor = model_type(
        actor_batch(torch.zeros(1, 62)), {"actor": ["actor"]}, "actor", 14, **model_cfg
    )
    actor.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    )
    return actor.eval()


def evaluate_pair(cfg, previous_checkpoint, checkpoint, stage, completed, run, block_dir):
    actor = load_actor(checkpoint, cfg)
    previous = load_actor(previous_checkpoint, cfg)
    comparisons = {}
    regression_ok = True
    # Fixed seeds expose drift; a fresh paired holdout prevents selecting
    # solely against one set of perturbations. Evaluate undiluted DR.
    for label, fraction, seed, independent in (
        ("nominal_fixed", 0.0, 401, False),
        ("dr_fixed", DR_LEVELS[stage], 1085, False),
        ("dr_independent", DR_LEVELS[stage], 2085, True),
        ("dr_holdout", DR_LEVELS[stage], 5000 + completed, False),
    ):
        evaluation_cfg = stage_config(cfg, fraction)
        baseline, _ = rollout(
            evaluation_cfg,
            ROOT,
            actor=previous,
            count=16,
            segments=4,
            seed=seed,
            independent=independent,
        )
        candidate, _ = rollout(
            evaluation_cfg,
            ROOT,
            actor=actor,
            count=16,
            segments=4,
            seed=seed,
            independent=independent,
        )
        passed = no_regression(candidate, baseline)
        if label == "nominal_fixed":
            passed &= all(candidate[k]["rate"] >= 0.9 for k in ("rise", "descent"))
        comparisons[label] = {
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "passed": passed,
        }
        regression_ok &= passed
        progress = {
            "status": "evaluating",
            "completed_iterations": completed,
            "candidate_checkpoint": str(checkpoint),
            "checkpoint": str(previous_checkpoint),
            "last_window": label,
            "paired_comparisons": comparisons,
        }
        (run / "pipeline_status.json").write_text(json.dumps(progress, indent=2))
        (block_dir / "evaluation_progress.json").write_text(json.dumps(progress, indent=2))
    return comparisons, regression_ok


def damp_checkpoint(previous, candidate, output, fraction):
    """Backtrack a proposed policy update; inference remains a standalone actor.

    Keep critic learning and optimizer moments, as in projected optimization.
    Explicitly reduce both optimizers' step sizes for subsequent proposals.
    """
    if not 0 < fraction < 1:
        raise ValueError("Backtracking fraction must be between zero and one")
    base = torch.load(previous, map_location="cpu", weights_only=False)
    state = torch.load(candidate, map_location="cpu", weights_only=False)
    for name, value in state["actor_state_dict"].items():
        old = base["actor_state_dict"][name]
        if name.startswith("timing_head.") or name == "frequencies":
            if not torch.equal(value, old):
                raise ValueError("Frozen nominal timing changed")
        elif torch.is_floating_point(value):
            state["actor_state_dict"][name] = old + fraction * (value - old)
        elif not torch.equal(value, old):
            raise ValueError("Actor nonfloating state changed")
    for key in ("optimizer_state_dict", "rehearsal_optimizer_state_dict"):
        if key in state:
            for group in state[key]["param_groups"]:
                group["lr"] *= fraction
    state["adaptive_learning_rate"] = state["optimizer_state_dict"]["param_groups"][0]["lr"]
    state.setdefault("infos", {})["policy_backtracking"] = {
        "previous": str(previous),
        "proposal": str(candidate),
        "fraction": fraction,
    }
    torch.save(state, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--block", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0 or args.block <= 0:
        parser.error("iterations and block must be positive")
    torch.set_num_threads(4)
    run = args.run.resolve()
    lock_handle = (run / ".pipeline.lock").open("w")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = OmegaConf.load(run / "final_config.yaml")
    accepted = json.loads((run / "student_evaluation.json").read_text())
    if not all(
        accepted[k]["rate"] >= 0.9 and accepted[k]["trials"] >= 32 for k in ("rise", "descent")
    ):
        raise RuntimeError("Imitation has not passed independent closed-loop evaluation")
    checkpoint = run / "imitation.pt"
    approval = json.loads((run / "pretraining_accepted.json").read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != approval["checkpoint_sha256"]:
        raise RuntimeError("Imitation checkpoint differs from evaluated policy")
    # The accepted initialization already has two nominal closed-loop windows.
    # Do not spend PPO updates relearning an already passed nominal stage.
    windows = approval.get("windows", [])
    nominal_accepted = len(windows) >= 2 and all(
        w[k]["trials"] >= 32 and w[k]["rate"] >= 0.9 for w in windows for k in ("rise", "descent")
    )
    gate = SuccessCurriculum(stage=1 if nominal_accepted else 0)
    completed = 0
    report_path = run / "curriculum_state.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved["status"] == "curriculum_passed":
            return
        if saved["status"].startswith("stopped_"):
            raise RuntimeError("Previous policy collapsed; correct it before resuming")
        gate = SuccessCurriculum(**saved["gate"])
        completed = saved["completed_iterations"]
        checkpoint = Path(saved["checkpoint"])
    while completed < args.iterations:
        current = stage_config(cfg, DR_LEVELS[gate.stage])
        amount = min(args.block, args.iterations - completed)
        block_dir = run / f"ppo_{completed:05d}_dr{gate.stage}"
        if block_dir.exists():
            raise RuntimeError(f"Existing block {block_dir}; inspect before restarting")
        previous_checkpoint = checkpoint
        current.algo.seed = 1 + completed
        current.algo.load_run = str(checkpoint)
        current.algo.max_iterations = amount
        current.algo.save_interval = min(100, amount)
        current.training.device = "cpu"
        current.training.no_play = True
        current.training.log_dir = str(block_dir)
        config_file = run / "active_stage.yaml"
        OmegaConf.save(current, config_file)
        (run / "pipeline_status.json").write_text(
            json.dumps(
                {
                    "status": "training",
                    "completed_iterations": completed,
                    "training_block_iterations": amount,
                    "dr_fraction": DR_LEVELS[gate.stage],
                    "checkpoint": str(checkpoint),
                    "block_directory": str(block_dir),
                },
                indent=2,
            )
        )
        with (run / f"ppo_{completed:05d}.log").open("w") as log:
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
                    str(run),
                    "--config-name",
                    "active_stage",
                ],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((block_dir / "run_summary.json").read_text())
        checkpoint = Path(summary["last_checkpoint"])
        completed += summary["run_completed_iterations"]
        comparisons, regression_ok = evaluate_pair(
            cfg, previous_checkpoint, checkpoint, gate.stage, completed, run, block_dir
        )
        proposal = checkpoint
        backtracking = [
            {
                "fraction": 1.0,
                "checkpoint": str(proposal),
                "passed": regression_ok,
                "comparisons": comparisons,
            }
        ]
        if not regression_ok:
            for fraction in (0.25, 0.0625):
                checkpoint = block_dir / f"model_backtracked_{fraction}.pt"
                damp_checkpoint(previous_checkpoint, proposal, checkpoint, fraction)
                comparisons, regression_ok = evaluate_pair(
                    cfg, previous_checkpoint, checkpoint, gate.stage, completed, run, block_dir
                )
                backtracking.append(
                    {
                        "fraction": fraction,
                        "checkpoint": str(checkpoint),
                        "passed": regression_ok,
                        "comparisons": comparisons,
                    }
                )
                if regression_ok:
                    break
        metrics = comparisons["dr_holdout"]["candidate"]
        before = gate.stage
        promotion_ready = regression_ok and all(
            v["candidate"][k]["rate"] >= gate.threshold
            for v in comparisons.values()
            for k in ("rise", "descent")
        )
        promotion_ready &= all(
            v["candidate"]["cycles"]["rate"] >= gate.threshold
            for v in comparisons.values()
            if "cycles" in v["candidate"]
        )
        if not promotion_ready:
            gate.streak = 0
        promoted = gate.observe(metrics) if promotion_ready else False
        report = {
            "status": "running",
            "gate": asdict(gate),
            "completed_iterations": completed,
            "checkpoint": str(checkpoint),
            "evaluated_stage": before,
            "promoted": promoted,
            "metrics": metrics,
            "paired_comparisons": comparisons,
            "candidate_checkpoint": str(checkpoint),
            "regression_passed": regression_ok,
            "backtracking": backtracking,
        }
        if not regression_ok:
            report["status"] = "stopped_regression_rolled_back"
            report["checkpoint"] = str(previous_checkpoint)
        elif gate.stage == len(DR_LEVELS) - 1 and gate.streak >= 2:
            report["status"] = "curriculum_passed"
        elif min(metrics[k]["rate"] for k in ("rise", "descent")) < 0.25:
            report["status"] = "stopped_success_collapse"
        elif completed >= args.iterations:
            report["status"] = "iteration_limit_reached"
        (block_dir / "closed_loop_evaluation.json").write_text(json.dumps(report, indent=2))
        report_path.write_text(json.dumps(report, indent=2))
        (run / "pipeline_status.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if report["status"] == "curriculum_passed" or not regression_ok:
            break
        # Avoid continuing thousands of updates after catastrophic forgetting.
        if min(metrics[k]["rate"] for k in ("rise", "descent")) < 0.25:
            raise RuntimeError("Closed-loop success collapsed; stopped for correction")


if __name__ == "__main__":
    main()
