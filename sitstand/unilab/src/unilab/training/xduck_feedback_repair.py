"""Supervised feedback initialization and measured closed-loop acceptance."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.training.xduck_curriculum import load_actor
from unilab.training.xduck_feedback_experiment import ankle_config, evaluate
from unilab.training.xduck_policy import ProprioceptiveFeedback
from unilab.training.xduck_pretrain import TensorPolicy


def fit_correction(actor, trace_dir, robust_traces=None):
    """Fit clean and optional noisy calibration seeds, excluding evaluation seeds."""
    chunks = []
    files = [
        trace_dir / f"{mode}_{index}.npz"
        for mode in ("learned", "zero", "oracle_0", "oracle_1.0")
        for index in (1, 2)
    ]
    if robust_traces is not None:
        files += [
            robust_traces / f"{mode}_{index}.npz"
            for mode in ("zero", "oracle_0")
            for index in (1, 2, 4, 5)
        ]
    for file in files:
        with np.load(file) as d:
            alive = np.concatenate(
                (np.ones_like(d["done"][:1]), ~np.maximum.accumulate(d["done"], axis=0)[:-1]),
                axis=0,
            )
            age = d["obs"][:, :, 61] * 4
            mask = alive & (age >= 0.2)
            chunks.append((d["obs"][mask], d["bias"][mask][:, [4, 13]]))
    obs = np.concatenate([x for x, _ in chunks])
    targets = np.concatenate([y for _, y in chunks])
    features = ProprioceptiveFeedback.features(torch.from_numpy(obs)).numpy().astype(np.float64)
    fits = {}
    with torch.no_grad():
        for cmd in (0, 1):
            rows = obs[:, 48] == cmd
            x, y = features[rows], targets[rows].astype(np.float64)
            mean, scale = x.mean(0), x.std(0).clip(1e-6)
            x = np.c_[(x - mean) / scale, np.ones(len(x))]
            weights = np.linalg.solve(x.T @ x + 1e-2 * np.eye(x.shape[1]), x.T @ y)
            actor.mlp.feature_mean[cmd].copy_(torch.tensor(mean))
            actor.mlp.feature_scale[cmd].copy_(torch.tensor(scale))
            actor.mlp.estimator[cmd].copy_(torch.tensor(weights))
            fits[str(cmd)] = {
                "samples": len(x),
                "train_rmse_deg": float(np.degrees(np.sqrt(np.mean((x @ weights - y) ** 2)))),
            }
    return fits


def prepare(source, traces, output, robust_traces=None):
    torch.set_num_threads(4)
    source, traces, output = source.resolve(), traces.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(source / "final_config.yaml")
    state = torch.load(source / "imitation.pt", weights_only=False, map_location="cpu")
    cfg.algo.actor.class_name = "unilab.training.xduck_policy:CorrectiveTimingActor"
    # Convert the old MLP into the residual submodule; its initial output is zero.
    state["actor_state_dict"] = {
        ("mlp.residual." + k[4:] if k.startswith("mlp.") else k): v
        for k, v in state["actor_state_dict"].items()
    }
    from unilab.training.rsl_rl import normalize_ppo_train_cfg
    from unilab.training.xduck_imitation import actor_batch
    from unilab.training.xduck_policy import CorrectiveTimingActor

    model_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))["actor"]
    model_cfg.pop("class_name")
    actor = CorrectiveTimingActor(
        actor_batch(torch.zeros(1, 62)), {"actor": ["actor"]}, "actor", 14, **model_cfg
    )
    missing, unexpected = actor.load_state_dict(state["actor_state_dict"], strict=False)
    assert (
        set(missing)
        == {
            "mlp.feature_mean",
            "mlp.feature_scale",
            "mlp.estimator",
            "mlp.ankle_projection",
            "mlp.prior_weights",
            "mlp.prior_frequencies",
        }
        and not unexpected
    )
    actor.mlp.prior_weights.copy_(actor.timing_head.weight)
    fit = fit_correction(actor, traces, robust_traces)
    state["actor_state_dict"] = actor.state_dict()
    # Structural conversion is a fresh optimizer start, not an optimizer resume.
    state.pop("optimizer_state_dict", None)
    state.pop("rehearsal_optimizer_state_dict", None)
    cfg.algo.algorithm.rehearsal_steps = 0
    cfg.algo.algorithm.learning_rate = 1e-5
    cfg.algo.algorithm.learning_rate_ceiling = 1e-5
    cfg.algo.algorithm.lam = 0.99
    cfg.algo.num_steps_per_env = 256
    cfg.env.posture_dwell_seconds = [4.0, 4.5]
    cfg.algo.algorithm.demonstration_path = str(source / "demonstrations.npz")
    checkpoint = output / "feedback_init.pt"
    torch.save(state, checkpoint)
    OmegaConf.save(cfg, output / "final_config.yaml")
    report = {
        "fit": fit,
        "checkpoint": str(checkpoint),
        "windows": {},
        "robust_training_traces": str(robust_traces) if robust_traces else None,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2))
    for name, test_cfg in [
        ("nominal", ankle_config(cfg, 0)),
        ("ankle15", ankle_config(cfg, 1.5)),
        ("ankle3", ankle_config(cfg, 3)),
    ]:
        print("Evaluating", name, flush=True)
        report["windows"][name] = evaluate(test_cfg, checkpoint, (48021, 58021), 16)
        (output / "results.json").write_text(json.dumps(report, indent=2))
        print(
            {
                mode: {d: sum(w[d]["successes"] for w in windows) for d in ["rise", "descent"]}
                for mode, windows in report["windows"][name].items()
            },
            flush=True,
        )
    actor = load_actor(checkpoint, cfg)
    exported = torch.jit.trace(TensorPolicy(actor), torch.zeros(1, 62))
    torch.jit.save(exported, str(output / "actor_feedback.pt"))
    report["status"] = "evaluated"
    (output / "results.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robust-traces", type=Path)
    parser.add_argument("--train-iterations", type=int, default=0)
    parser.add_argument("--training-domain", choices=("ankle3", "full"), default="full")
    parser.add_argument("--num-envs", type=int, default=128)
    args = parser.parse_args()
    if args.train_iterations < 0:
        parser.error("train-iterations must be nonnegative")
    if args.num_envs < 4:
        parser.error("num-envs must be at least 4")
    prepare(args.source, args.traces, args.output, args.robust_traces)
    if args.train_iterations:
        run_recovery(
            args.output.resolve(), args.train_iterations, args.training_domain, args.num_envs
        )


def prepare_recovery_training(run, iterations=12, count=128, domain="full"):
    """Materialize a fresh optimizer after the supervised architecture conversion."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from unilab.training.rsl_rl import normalize_ppo_train_cfg

    if domain not in ("ankle3", "full"):
        raise ValueError("Unknown recovery training domain")
    cfg = OmegaConf.load(run / "final_config.yaml")
    checkpoint = run / "feedback_init.pt"
    actor = load_actor(checkpoint, cfg)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    critic_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))["critic"]
    critic_cfg.pop("class_name")
    critic = MLPModel(
        TensorDict({"critic": torch.zeros(1, 77)}, batch_size=[1]),
        {"critic": ["critic"]},
        "critic",
        1,
        **critic_cfg,
    )
    critic.load_state_dict(state["critic_state_dict"])
    state["optimizer_state_dict"] = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-5
    ).state_dict()
    state["adaptive_learning_rate"] = 1e-5
    state["iter"] = -1
    state["infos"] = {
        "unilab_progress": {"version": 1, "env_steps_per_env": 0, "next_iteration": 0}
    }
    initial = run / "recovery_start.pt"
    torch.save(state, initial)
    if domain == "ankle3":
        cfg = ankle_config(cfg, 3)
    cfg.algo.algorithm.class_name = "unilab.training.xduck_ppo:RecoveryPPO"
    cfg.algo.algorithm.critic_warmup_updates = 4
    cfg.algo.algorithm.critic_steps = 16
    cfg.algo.algorithm.critic_learning_rate = 3e-4
    cfg.algo.algorithm.diagnostics_path = str(run / "recovery_feedback.jsonl")
    cfg.algo.load_run = str(initial)
    cfg.algo.num_envs = count
    cfg.algo.max_iterations = iterations
    cfg.algo.save_interval = 4
    cfg.algo.seed = 64921
    cfg.training.device = "cpu"
    cfg.training.no_play = True
    cfg.training.log_dir = str(run / "recovery_train")
    OmegaConf.save(cfg, run / "recovery_config.yaml")
    return cfg


def evaluate_recovery(run, domain="ankle3"):
    cfg = OmegaConf.load(run / "final_config.yaml")
    summary = json.loads((run / "recovery_train/run_summary.json").read_text())
    candidate = Path(summary["last_checkpoint"])
    report = {"candidate": str(candidate), "windows": {}}
    for name, test_cfg in (("nominal", ankle_config(cfg, 0)), ("ankle3", ankle_config(cfg, 3))):
        report["windows"][name] = evaluate(test_cfg, candidate, (48021, 58021), 16)
        (run / "recovery_evaluation.json").write_text(json.dumps(report, indent=2))
    # A finite candidate comparison selects an artifact; it never interrupts
    # an ongoing learning block over a single failed trial.
    initial = json.loads((run / "results.json").read_text())
    passed = all(
        new[d]["successes"] >= old[d]["successes"]
        and new[d]["rate"] >= 0.9
        and new["initial_domains"] == old["initial_domains"]
        for name in ("nominal", "ankle3")
        for new, old in zip(report["windows"][name]["full"], initial["windows"][name]["full"])
        for d in ("rise", "descent")
    )
    report["candidate_accepted"] = passed
    if domain == "full":
        seeds = (108021, 118021, 128021, 138021)
        reference = evaluate(cfg, run / "feedback_init.pt", seeds, 16)
        candidate_result = evaluate(cfg, candidate, seeds, 16)
        report["strong_dr"] = {"reference": reference, "candidate": candidate_result}
        passed = passed and all(
            sum(w[d]["successes"] for w in candidate_result["full"])
            >= sum(w[d]["successes"] for w in reference["full"])
            for d in ("rise", "descent")
        )
        passed = passed and all(
            new["initial_domains"] == old["initial_domains"]
            for new, old in zip(candidate_result["full"], reference["full"])
        )
        report["candidate_accepted"] = passed
    report["selected_checkpoint"] = str(candidate if passed else run / "feedback_init.pt")
    report["scope"] = (
        "Full DR non-regression comparison; acceptance does not imply 90% task success"
        if domain == "full"
        else "nominal and ankle-only simulation; full DR needs separate acceptance"
    )
    selected_actor = load_actor(Path(report["selected_checkpoint"]), cfg)
    exported = torch.jit.trace(TensorPolicy(selected_actor), torch.zeros(1, 62))
    torch.jit.save(exported, str(run / "actor_selected.pt"))
    report["selected_export"] = str(run / "actor_selected.pt")
    (run / "recovery_evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def run_recovery(run, iterations, domain="full", count=128):
    from unilab.training.xduck_curriculum import ROOT

    initial = json.loads((run / "results.json").read_text())
    if not all(
        w[d]["rate"] >= 0.9
        for name in ("nominal", "ankle3")
        for w in initial["windows"][name]["full"]
        for d in ("rise", "descent")
    ):
        raise RuntimeError("Feedback initialization failed closed-loop acceptance")
    if (run / "recovery_train").exists():
        raise ValueError("Recovery training already exists; inspect before resuming")
    prepare_recovery_training(run, iterations, count=count, domain=domain)
    with (run / "recovery_console.log").open("w") as log:
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
                "recovery_config",
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return evaluate_recovery(run, domain)
