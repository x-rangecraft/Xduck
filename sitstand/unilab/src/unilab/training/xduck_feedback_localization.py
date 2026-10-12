"""Causal localization of XDuck feedback failures; simulation diagnostics only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import unilab.training.xduck_imitation as imitation
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.xduck_curriculum import ROOT, load_actor
from unilab.training.xduck_feedback_experiment import ankle_config


class ProbePolicy(torch.nn.Module):
    def __init__(self, actor, critic, template, mode):
        super().__init__()
        self.actor, self.critic = actor, critic
        self.template = template
        self.mode = mode
        self.env = None
        self.records = []
        self.pending = None

    def forward(self, batch):
        raw = batch["actor"]
        feed = self.actor.mlp(raw)
        timing = self.actor.timing_head(self.actor.time_features(raw))
        cmd = (raw[:, 48] > 0.5).long()
        phase = (raw[:, 61] * 200).round().long().clamp(0, 200)
        nominal = self.template[cmd, phase].clone()
        nominal[:, 48] = raw[:, 48]
        nominal[:, 61] = raw[:, 61]
        phase_feed = self.actor.mlp(nominal)
        bias = torch.as_tensor(self.env._ankle_target_offset.copy(), dtype=torch.float32)
        if self.mode == "zero":
            correction = torch.zeros_like(feed)
        elif self.mode == "phase_only":
            correction = phase_feed
        elif self.mode == "scale10":
            correction = 10 * feed
        elif self.mode == "inverted":
            correction = -feed
        elif self.mode.startswith("oracle"):
            delay = float(self.mode.split("_")[1])
            correction = -bias * (raw[:, 61:62] * 4 >= delay)
        else:
            correction = feed
        critic_obs = torch.as_tensor(self.env.state.obs["critic"].copy())
        value = self.critic(TensorDict({"critic": critic_obs}, batch_size=[len(raw)]))
        self.pending = {
            "obs": raw.numpy().copy(),
            "critic_obs": critic_obs.numpy(),
            "feedback": feed.numpy(),
            "phase_feedback": phase_feed.numpy(),
            "correction": correction.numpy(),
            "bias": bias.numpy(),
            "value": value.numpy().reshape(-1),
        }
        return timing + correction


def run(cfg, checkpoint, output, modes, seeds, count):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    actor = load_actor(checkpoint, cfg)
    critic_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))["critic"]
    critic_cfg.pop("class_name")
    critic = MLPModel(
        TensorDict({"critic": torch.zeros(1, 77)}, batch_size=[1]),
        {"critic": ["critic"]},
        "critic",
        1,
        **critic_cfg,
    )
    critic.load_state_dict(
        torch.load(checkpoint, weights_only=False, map_location="cpu")["critic_state_dict"]
    )
    critic.eval()
    with np.load(cfg.algo.algorithm.demonstration_path) as data:
        demo = data["obs"]
    template = np.empty((2, 201, 62), np.float32)
    for cmd in (0, 1):
        rows = demo[demo[:, 48] == cmd]
        for step in range(201):
            distances = abs(rows[:, 61] - step / 200)
            template[cmd, step] = rows[distances <= distances.min() + 1e-7].mean(axis=0)
    real_make = imitation.make_env
    report = {
        "checkpoint": str(checkpoint),
        "modes": {},
        "seeds": seeds,
        "count": count,
        "oracle_scope": "Privileged exact ankle cancellation for diagnosis, not deployable policy",
    }
    OmegaConf.save(cfg, output / "config.yaml")
    try:
        for mode in modes:
            policy = ProbePolicy(actor, critic, torch.tensor(template), mode)

            def make(*args, **kwargs):
                env = real_make(*args, **kwargs)
                policy.env = env
                segment = []
                policy.records.append(segment)
                terminal = env._terminated
                original_step = env.step

                def terminated(up):
                    flags = terminal(up)
                    if policy.pending is not None:
                        q = env._backend.get_base_quat()
                        policy.pending.update(
                            tilt=np.degrees(
                                np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
                            ),
                            height=env._backend.get_base_pos()[:, 2].copy(),
                            velocity=env._backend.get_base_lin_vel().copy(),
                            q=env.get_dof_pos().copy(),
                            trunk=env._backend.get_sensor_data("reference_trunk_floor")[
                                :, 0
                            ].copy(),
                        )
                    return flags

                def step(action):
                    result = original_step(action)
                    if policy.pending is not None:
                        policy.pending.update(
                            reward=result.reward.copy(),
                            done=(result.terminated | result.truncated).copy(),
                        )
                        segment.append(policy.pending)
                        policy.pending = None
                    return result

                env._terminated = terminated
                env.step = step
                return env

            imitation.make_env = make
            windows = []
            for seed in seeds:
                result, _ = imitation.rollout(
                    cfg, ROOT, actor=policy, count=count, segments=2, seed=seed, independent=True
                )
                windows.append(result)
            files = []
            for i, segment in enumerate(policy.records):
                if not segment:
                    continue
                file = output / f"{mode}_{i}.npz"
                np.savez_compressed(
                    file, **{k: np.stack([row[k] for row in segment]) for k in segment[0]}
                )
                files.append(file.name)
            summary = {
                direction: {
                    "successes": sum(w[direction]["successes"] for w in windows),
                    "trials": sum(w[direction]["trials"] for w in windows),
                }
                for direction in ("rise", "descent")
            }
            report["modes"][mode] = {"summary": summary, "windows": windows, "traces": files}
            (output / "results.json").write_text(json.dumps(report, indent=2))
            print(mode, summary, flush=True)
    finally:
        imitation.make_env = real_make


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[18021, 28021])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "learned",
            "zero",
            "phase_only",
            "scale10",
            "inverted",
            "oracle_0",
            "oracle_1.0",
            "oracle_1.5",
            "oracle_2.0",
        ],
    )
    args = parser.parse_args()
    cfg = ankle_config(OmegaConf.load(args.config), 1.5)
    run(cfg, args.checkpoint, args.output, args.modes, args.seeds, args.count)


def training_signal_survey(cfg, checkpoint, output, steps=512, count=128):
    """Frozen-policy survey of natural training commands; no optimizer updates."""
    torch.set_num_threads(4)
    np.random.seed(4921)
    torch.manual_seed(4921)
    actor = load_actor(checkpoint, cfg)
    critic_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))["critic"]
    critic_cfg.pop("class_name")
    critic = MLPModel(
        TensorDict({"critic": torch.zeros(1, 77)}, batch_size=[1]),
        {"critic": ["critic"]},
        "critic",
        1,
        **critic_cfg,
    )
    critic.load_state_dict(
        torch.load(checkpoint, weights_only=False, map_location="cpu")["critic_state_dict"]
    )
    env = imitation.make_env(cfg, ROOT, count)
    rows = []
    try:
        env.init_state()
        for _ in range(steps):
            obs = env.state.obs["obs"].copy()
            cobs = torch.as_tensor(env.state.obs["critic"].copy())
            bias = env._ankle_target_offset.copy()
            with torch.no_grad():
                batch = imitation.actor_batch(obs)
                mean = actor(batch)
                action = actor(batch, stochastic_output=True)
                value = critic(TensorDict({"critic": cobs}, batch_size=[count])).numpy().reshape(-1)
            state = env.step(action.numpy())
            rows.append(
                {
                    "obs": obs,
                    "bias": bias,
                    "epsilon": (action - mean).numpy(),
                    "value": value,
                    "reward": state.reward.copy(),
                    "done": (state.terminated | state.truncated).copy(),
                }
            )
        with torch.no_grad():
            final_value = (
                critic(
                    TensorDict(
                        {"critic": torch.as_tensor(env.state.obs["critic"].copy())},
                        batch_size=[count],
                    )
                )
                .numpy()
                .reshape(-1)
            )
        data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        next_values = np.concatenate([data["value"][1:], final_value[None]])
        delta = data["reward"] + 0.99 * next_values * ~data["done"] - data["value"]
        advantage = np.zeros_like(delta)
        for start in range(0, steps, 64):
            gae = np.zeros(count)
            for t in range(min(start + 64, steps) - 1, start - 1, -1):
                gae = delta[t] + 0.99 * 0.95 * ~data["done"][t] * gae
                advantage[t] = gae
        data.update(delta=delta, advantage=advantage, next_value=next_values)
        np.savez_compressed(output, **data)
        print("Saved frozen training distribution survey", output, flush=True)
    finally:
        env.close()
