"""Owner of the accepted demonstration-to-neural-policy initialization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.rsl_rl_reset import RslRlVecEnvWrapper
from unilab.training.xduck_imitation import actor_batch, make_env, rollout, stage_config

ROOT = Path(__file__).resolve().parents[3]


class TensorPolicy(torch.nn.Module):
    """Standalone tensor inference; contains no environment or reference reader."""

    def __init__(self, actor):
        super().__init__()
        self.feedback = actor.mlp
        self.timing_head = actor.timing_head
        self.register_buffer("frequencies", actor.frequencies.clone())

    def forward(self, raw):
        phase = raw[:, 61:62].clamp(0, 1)
        command = raw[:, 48:49].clamp(0, 1)
        basis = torch.cos(torch.pi * phase * self.frequencies[None, :])
        features = torch.cat((basis * (1 - command), basis * command), dim=1)
        return self.feedback(raw) + self.timing_head(features)


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pretraining_accepted.json").exists():
        raise ValueError("Already accepted; use a new output directory to regenerate")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=xduck_gf43x40_sitstand_direct_flat/mujoco",
                "algo.empirical_normalization=false",
                "algo.policy.init_noise_std=0.005",
                "algo.algorithm.learning_rate=0.00001",
            ],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
    train_cfg["actor"]["class_name"] = "unilab.training.xduck_policy:TimingActor"
    cfg.algo.actor, cfg.algo.critic = train_cfg["actor"], train_cfg["critic"]
    OmegaConf.save(cfg, output / "final_config.yaml")
    nominal = stage_config(cfg, 0)
    OmegaConf.save(nominal, output / "stage0_config.yaml")
    teacher_metrics, data = rollout(nominal, ROOT, collect=True, seed=7)
    (output / "teacher_evaluation.json").write_text(json.dumps(teacher_metrics, indent=2))
    if not all(teacher_metrics[k]["rate"] >= 0.9 for k in ("rise", "descent")):
        raise RuntimeError("Teacher failed nominal closed-loop gate")
    np.savez_compressed(output / "demonstrations.npz", **data)
    cfg.env.nominal_env_fraction = 0.25
    cfg.algo.algorithm.class_name = "unilab.training.xduck_ppo:RecoveryPPO"
    cfg.algo.algorithm.demonstration_path = str(output / "demonstrations.npz")
    # The frozen timing head already retains the nominal motion. Regressing
    # feedback back to the demonstration after every PPO update erases recovery.
    cfg.algo.algorithm.rehearsal_steps = 0
    cfg.algo.algorithm.rehearsal_batch_size = 1024
    cfg.algo.algorithm.rehearsal_lr = 2.5e-6
    cfg.algo.algorithm.learning_rate = 1e-5
    cfg.algo.algorithm.learning_rate_ceiling = 1e-5
    cfg.algo.algorithm.critic_warmup_updates = 4
    cfg.algo.algorithm.critic_steps = 16
    cfg.algo.algorithm.critic_learning_rate = 3e-4
    cfg.algo.algorithm.lam = 0.99
    cfg.algo.num_steps_per_env = 256
    cfg.env.posture_dwell_seconds = [4.0, 4.5]
    cfg.algo.algorithm.schedule = "fixed"
    cfg.algo.algorithm.entropy_coef = 0.0
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
    OmegaConf.save(cfg, output / "final_config.yaml")
    env = make_env(nominal, ROOT, 2)
    try:
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(output / "runner"), device="cpu")
        runner.logger.init_logging_writer()
        actor = runner.alg.actor
        _, ids = np.unique(data["obs"][:, [48, 61]], axis=0, return_index=True)
        x, y = torch.from_numpy(data["obs"][ids]), torch.from_numpy(data["actions"][ids])
        basis = actor.time_features(x).double()
        solution = torch.linalg.lstsq(basis, y.double(), rcond=1e-8, driver="gelsd").solution
        with torch.no_grad():
            actor.timing_head.weight.copy_(solution.T.float())
            error = actor(actor_batch(torch.from_numpy(data["obs"]))) - torch.from_numpy(
                data["actions"]
            )
        fit = {
            "rmse_rad": error.square().mean().sqrt().item(),
            "peak_rad": error.abs().max().item(),
            "samples": len(data["obs"]),
        }
        print("Supervised fit", fit, flush=True)
        if fit["peak_rad"] >= 0.001:
            raise RuntimeError("Neural timing approximation exceeded error budget")
        runner.current_learning_iteration = -1
        runner.save(
            str(output / "imitation.pt"),
            infos={
                "unilab_progress": {"version": 1, "env_steps_per_env": 0, "next_iteration": 0},
                "imitation_fit": fit,
            },
        )
        (output / "imitation_fit.json").write_text(json.dumps(fit, indent=2))
        windows = []
        for seed in (401, 501):
            metrics, _ = rollout(nominal, ROOT, actor=actor, count=16, segments=4, seed=seed)
            (output / f"student_evaluation_{seed}.json").write_text(json.dumps(metrics, indent=2))
            windows.append(metrics)
            print("Neural closed loop", seed, metrics, flush=True)
        if not all(m[k]["rate"] >= 0.9 for m in windows for k in ("rise", "descent")):
            raise RuntimeError(
                "Student failed nominal closed-loop gate; PPO not authorized by gate"
            )
        (output / "student_evaluation.json").write_text(json.dumps(windows[-1], indent=2))
        exported = torch.jit.trace(TensorPolicy(actor).eval(), torch.zeros(1, 62))
        torch.jit.save(exported, str(output / "actor_nominal.pt"))
        check = torch.from_numpy(data["obs"][:64])
        torch.testing.assert_close(exported(check), actor(actor_batch(check)), atol=1e-6, rtol=1e-5)
        record = {
            "checkpoint_sha256": hashlib.sha256((output / "imitation.pt").read_bytes()).hexdigest(),
            "windows": windows,
            "fit": fit,
            "scope": "nominal simulation only; not strong DR acceptance",
        }
        (output / "pretraining_accepted.json").write_text(json.dumps(record, indent=2))
        runner.logger.stop_logging_writer()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
