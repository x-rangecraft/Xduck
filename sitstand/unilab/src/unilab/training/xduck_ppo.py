"""PPO followed by nominal demonstration rehearsal, with resumable state."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from unilab.training.checked_ppo import FiniteCheckedPPO, check_optimizer_tensors
from unilab.training.xduck_imitation import actor_batch


class RehearsalPPO(FiniteCheckedPPO):
    """Explicit auxiliary updates; rollout collection remains the RSL-RL lifecycle."""

    def __init__(
        self,
        *args,
        demonstration_path,
        rehearsal_steps=8,
        rehearsal_batch_size=1024,
        rehearsal_lr=1e-5,
        learning_rate_ceiling=1e-5,
        diagnostics_path=None,
        **kwargs,
    ):
        if rehearsal_steps < 0 or rehearsal_batch_size < 1 or rehearsal_lr <= 0:
            raise ValueError("Rehearsal steps must be nonnegative; batch size and LR positive")
        if not np.isfinite(learning_rate_ceiling) or learning_rate_ceiling <= 0:
            raise ValueError("Learning rate ceiling must be positive and finite")
        self.learning_rate_ceiling = learning_rate_ceiling
        super().__init__(*args, **kwargs)
        self._cap_learning_rate()
        self.optimizer.register_step_pre_hook(lambda opt, args, kwargs: self._cap_learning_rate())
        with open(demonstration_path, "rb") as stream:
            self.demonstration_sha256 = hashlib.sha256(stream.read()).hexdigest()
        with np.load(demonstration_path) as data:
            self.demo_obs = torch.tensor(data["obs"], device=self.device)
            self.demo_actions = torch.tensor(data["actions"], device=self.device)
        if self.demo_obs.shape[1:] != (62,) or self.demo_actions.shape != (len(self.demo_obs), 14):
            raise ValueError("Invalid XDuck demonstration dimensions")
        if not torch.isfinite(self.demo_obs).all() or not torch.isfinite(self.demo_actions).all():
            raise ValueError("Nonfinite demonstrations")
        self.rehearsal_steps = rehearsal_steps
        self.rehearsal_batch_size = rehearsal_batch_size
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path else None
        self._probe_rows = []
        self._actor_grad_norms = []
        if self.diagnostics_path:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            self.optimizer.register_step_pre_hook(self._record_actor_gradient)
        # Only mean feedback is rehearsed; learned stochastic exploration is PPO's responsibility.
        self.rehearsal_optimizer = torch.optim.Adam(self.actor.mlp.parameters(), lr=rehearsal_lr)
        self.rehearsal_optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: check_optimizer_tensors(opt, gradients=True)
        )
        self.rehearsal_optimizer.register_step_post_hook(
            lambda opt, args, kwargs: check_optimizer_tensors(opt, gradients=False)
        )

    def _cap_learning_rate(self):
        # A deliberate run-level safety cap also applies after checkpoint resume.
        self.learning_rate = min(self.learning_rate, self.learning_rate_ceiling)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _record_actor_gradient(self, optimizer, args, kwargs):
        norms = [
            p.grad.detach().square().sum()
            for p in self.actor.mlp.parameters()
            if p.grad is not None
        ]
        self._actor_grad_norms.append(float(torch.stack(norms).sum().sqrt()) if norms else 0.0)

    def act(self, obs):
        if self.diagnostics_path and len(self._probe_rows) < 128:
            raw = torch.cat([obs[key] for key in self.actor.obs_groups], dim=-1)
            # Span the batch so the probe includes both nominal and DR cohorts.
            stride = max(1, len(raw) // 32)
            self._probe_rows.append(raw[::stride][:32].detach().clone())
        return super().act(obs)

    @torch.no_grad()
    def _feedback(self, rows):
        batch = actor_batch(rows)
        return self.actor.mlp(self.actor.get_latent(batch)).detach().clone()

    def update(self):
        probes = {}
        if self.diagnostics_path:
            probes["nominal"] = self.demo_obs[:: max(1, len(self.demo_obs) // 1024)][:1024]
            if self._probe_rows:
                probes["rollout"] = torch.cat(self._probe_rows)
        before = {k: self._feedback(v) for k, v in probes.items()}
        self._actor_grad_norms.clear()
        metrics = super().update()
        after_ppo = {k: self._feedback(v) for k, v in probes.items()}
        losses = []
        for _ in range(self.rehearsal_steps):
            indices = torch.randint(
                len(self.demo_obs), (self.rehearsal_batch_size,), device=self.device
            )
            mean = self.actor(actor_batch(self.demo_obs[indices]))
            loss = ((mean - self.demo_actions[indices]) / 0.01).square().mean()
            self.rehearsal_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(), self.max_grad_norm)
            self.rehearsal_optimizer.step()
            losses.append(float(loss.detach()))
        metrics["nominal_rehearsal_scaled_mse"] = float(np.mean(losses)) if losses else 0.0
        if self.diagnostics_path:
            record = {
                "learning_rate": self.learning_rate,
                "rehearsal_steps": self.rehearsal_steps,
                "actor_grad_norm_mean": float(np.mean(self._actor_grad_norms))
                if self._actor_grad_norms
                else 0.0,
                "probes": {},
            }
            for name, rows in probes.items():
                final = self._feedback(rows)
                delta = after_ppo[name] - before[name]
                undo = final - after_ppo[name]
                values = {
                    "before_rms_rad": float(before[name].square().mean().sqrt()),
                    "after_ppo_rms_rad": float(after_ppo[name].square().mean().sqrt()),
                    "after_rehearsal_rms_rad": float(final.square().mean().sqrt()),
                    "ppo_delta_rms_rad": float(delta.square().mean().sqrt()),
                    "rehearsal_delta_rms_rad": float(undo.square().mean().sqrt()),
                    "rehearsal_opposition": float(
                        -(delta * undo).sum() / delta.square().sum().clamp_min(1e-20)
                    ),
                }
                record["probes"][name] = values
                metrics.update({f"feedback_{name}_{k}": v for k, v in values.items()})
            with self.diagnostics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        self._probe_rows.clear()
        return metrics

    def save(self):
        state = super().save()
        state["rehearsal_optimizer_state_dict"] = self.rehearsal_optimizer.state_dict()
        state["demonstration_sha256"] = self.demonstration_sha256
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if (
            loaded_dict.get("demonstration_sha256", self.demonstration_sha256)
            != self.demonstration_sha256
        ):
            raise ValueError("Resume demonstration data differs from checkpoint")
        result = super().load(loaded_dict, load_cfg, strict)
        self._cap_learning_rate()
        if (
            load_cfg is None or load_cfg.get("optimizer", False)
        ) and "rehearsal_optimizer_state_dict" in loaded_dict:
            self.rehearsal_optimizer.load_state_dict(loaded_dict["rehearsal_optimizer_state_dict"])
        return result


class RecoveryPPO(RehearsalPPO):
    """Critic-only warmup, followed by PPO with additional value fitting."""

    def __init__(
        self, *args, critic_warmup_updates=4, critic_steps=16, critic_learning_rate=3e-4, **kwargs
    ):
        if critic_warmup_updates < 0 or critic_steps < 1 or critic_learning_rate <= 0:
            raise ValueError("Invalid critic recovery configuration")
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu:
            raise ValueError("RecoveryPPO currently requires single-process training")
        self.critic_warmup_updates = critic_warmup_updates
        self.critic_steps = critic_steps
        self.recovery_updates = 0
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.value_optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: check_optimizer_tensors(opt, gradients=True)
        )
        self.value_optimizer.register_step_post_hook(
            lambda opt, args, kwargs: check_optimizer_tensors(opt, gradients=False)
        )

    def update(self):
        observations = self.storage.observations.flatten(0, 1)
        targets = self.storage.returns.flatten(0, 1).detach().clone()
        warming = self.recovery_updates < self.critic_warmup_updates
        if warming:
            metrics = {
                "value": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "nominal_rehearsal_scaled_mse": 0.0,
            }
            self.storage.clear()
            self._probe_rows.clear()
        else:
            metrics = super().update()
        losses = []
        for _ in range(self.critic_steps):
            indices = torch.randint(len(targets), (min(2048, len(targets)),), device=self.device)
            loss = (self.critic(observations[indices]) - targets[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite critic recovery loss")
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            losses.append(float(loss.detach()))
        metrics["critic_recovery_mse"] = float(np.mean(losses))
        metrics["critic_warmup"] = float(warming)
        self.recovery_updates += 1
        return metrics

    def save(self):
        state = super().save()
        state["value_optimizer_state_dict"] = self.value_optimizer.state_dict()
        state["recovery_updates"] = self.recovery_updates
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            if "value_optimizer_state_dict" in loaded_dict:
                self.value_optimizer.load_state_dict(loaded_dict["value_optimizer_state_dict"])
            self.recovery_updates = loaded_dict.get("recovery_updates", 0)
        return result
