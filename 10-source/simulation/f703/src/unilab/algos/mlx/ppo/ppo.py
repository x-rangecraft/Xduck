"""PPO trainer implemented with MLX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from unilab.algos.mlx.common import RolloutBuffer, diag_gaussian_entropy, diag_gaussian_log_prob

from .model import MLPActorCritic


@dataclass
class PPOConfig:
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    learning_rate: float = 3e-4
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    log_ratio_clip: float = 20.0
    schedule: str = "fixed"
    desired_kl: float = 0.01
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    normalize_advantage_per_mini_batch: bool = False
    adaptive_kl_beta: float = 0.9
    adaptive_lr_decay: float = 1.5
    adaptive_lr_growth: float = 1.2
    adaptive_lr_update_interval: int = 1
    target_kl_stop: float | None = None
    metrics_interval: int = 8
    finite_check_interval: int = 8
    enable_compile: bool = False
    warmup_strict_iters: int = 0
    warmup_metrics_interval: int = 1
    warmup_finite_check_interval: int = 1
    disable_finite_checks: bool = False
    symmetry_cfg: dict | None = None
    # Optional surgical actor adaptation.  The critic remains fully trainable,
    # while the actor may change only selected columns of its first layer.
    # This preserves the exact warm-start policy whenever those inputs are zero.
    actor_train_input_columns: tuple[int, ...] | list[int] | None = None
    # Training-only consistency in physical action units. No inference transform.
    policy_consistency: dict | None = None


class PPOTrainer:
    """PPO update logic for `MLPActorCritic` and `RolloutBuffer`."""

    def __init__(self, model: MLPActorCritic, cfg: PPOConfig, symmetry: Any = None) -> None:
        self.model = model
        self.cfg = cfg
        self.symmetry = symmetry
        symmetry_cfg = cfg.symmetry_cfg or {}
        if symmetry_cfg.get("use_data_augmentation", False):
            raise ValueError(
                "MLX PPO currently supports mirror loss only, not PPO data augmentation"
            )
        self.mirror_loss_coeff = (
            float(symmetry_cfg.get("mirror_loss_coeff", 0.0))
            if symmetry_cfg.get("use_mirror_loss", False)
            else 0.0
        )
        if not math.isfinite(self.mirror_loss_coeff) or self.mirror_loss_coeff < 0:
            raise ValueError("mirror_loss_coeff must be finite and non-negative")
        if self.mirror_loss_coeff and symmetry is None:
            raise ValueError("Mirror loss requires an env-owned symmetry adapter")
        self._dtype = getattr(model, "dtype", mx.float32)
        consistency = cfg.policy_consistency or {}
        unknown = set(consistency) - {"weight", "obs_noise_half_width", "action_scale"}
        if unknown:
            raise ValueError(f"Unknown policy_consistency fields: {sorted(unknown)}")
        self.consistency_weight = float(consistency.get("weight", 0.0))
        if not math.isfinite(self.consistency_weight) or self.consistency_weight < 0:
            raise ValueError("policy_consistency weight must be finite and non-negative")
        self._consistency_noise = None
        self._consistency_scale = None
        if self.consistency_weight:
            noise = consistency.get("obs_noise_half_width", [])
            scale = consistency.get("action_scale", [])
            if len(noise) != model.obs_dim or len(scale) != model.action_dim:
                raise ValueError("policy_consistency dimensions must match actor obs/action")
            if not all(math.isfinite(v) and v >= 0 for v in noise) or not any(noise):
                raise ValueError("policy_consistency noise must be finite, non-negative, nonzero")
            if not all(math.isfinite(v) and v > 0 for v in scale):
                raise ValueError("policy_consistency action_scale must be finite and positive")
            self._consistency_noise = mx.array(noise, dtype=self._dtype)
            self._consistency_scale = mx.array(scale, dtype=self._dtype)
        self._actor_anchor_obs = None
        self._actor_anchor_actions = None
        self._actor_anchor_weight = 0.0
        columns = cfg.actor_train_input_columns
        self.actor_train_input_columns = None if columns is None else tuple(columns)
        if self.actor_train_input_columns is not None:
            if not self.actor_train_input_columns:
                raise ValueError("actor_train_input_columns cannot be empty")
            if len(set(self.actor_train_input_columns)) != len(self.actor_train_input_columns):
                raise ValueError("actor_train_input_columns must be unique")
            if min(self.actor_train_input_columns) < 0 or max(self.actor_train_input_columns) >= int(
                model.obs_dim
            ):
                raise ValueError("actor_train_input_columns are outside the actor observation")
        self.learning_rate = float(cfg.learning_rate)
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)
        self.loss_and_grad = nn.value_and_grad(model, self._loss_fn)
        self.compiled_loss_and_grad = self.loss_and_grad
        if self.cfg.enable_compile and hasattr(mx, "compile"):
            try:
                self.compiled_loss_and_grad = mx.compile(self.loss_and_grad)
            except Exception:
                self.compiled_loss_and_grad = self.loss_and_grad
        self._kl_ema: float | None = None

    def set_actor_anchor(self, obs, actions, weight: float) -> None:
        """Preserve deterministic actions on a small, fixed safety set."""
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("actor anchor weight must be finite and positive")
        if obs.ndim != 2 or actions.ndim != 2 or obs.shape[0] != actions.shape[0]:
            raise ValueError("actor anchor observations/actions must be aligned matrices")
        if obs.shape[1] != self.model.obs_dim or actions.shape[1] != self.model.action_dim:
            raise ValueError("actor anchor dimensions do not match the policy")
        self._actor_anchor_obs = mx.stop_gradient(obs.astype(self._dtype))
        self._actor_anchor_actions = mx.stop_gradient(actions.astype(self._dtype))
        self._actor_anchor_weight = float(weight)

    @staticmethod
    def _tree_leaves(tree: Any) -> list[Any]:
        flat_tree = cast(list[tuple[str, Any]], tree_flatten(tree))
        return [leaf for _, leaf in flat_tree]

    @staticmethod
    def _all_finite(tree) -> bool:
        leaves = PPOTrainer._tree_leaves(tree)
        if not leaves:
            return True
        checks = [mx.all(mx.isfinite(leaf)) for leaf in leaves]  # type: ignore[arg-type]
        mx.eval(*checks)
        return all(bool(c.item()) for c in checks)

    def _clip_grads(self, grads):
        """Global gradient clipping similar to rsl-rl max_grad_norm."""
        if self.cfg.max_grad_norm <= 0.0:
            return grads
        leaves = self._tree_leaves(grads)
        if not leaves:
            return grads
        sq_norm = mx.array(0.0, dtype=self._dtype)
        for leaf in leaves:
            sq_norm = sq_norm + mx.sum(leaf * leaf)
        global_norm = mx.sqrt(sq_norm + 1e-12)
        clip_coef = mx.minimum(1.0, self.cfg.max_grad_norm / (global_norm + 1e-6))
        return tree_map(lambda g: g * clip_coef, grads)

    def _mask_actor_grads(self, grads):
        """Freeze the actor except selected first-layer input columns."""
        if self.actor_train_input_columns is None:
            return grads
        masked = dict(grads)
        actor = tree_map(mx.zeros_like, grads["actor"])
        layers = list(actor["layers"])
        first = dict(layers[0])
        source = grads["actor"]["layers"][0]["weight"]
        ids = mx.array(self.actor_train_input_columns)
        column_mask = mx.any(mx.arange(source.shape[1])[:, None] == ids[None, :], axis=1)
        first["weight"] = source * column_mask[None, :]
        layers[0] = first
        actor["layers"] = layers
        masked["actor"] = actor
        for name in ("std", "log_std"):
            if name in masked:
                masked[name] = mx.zeros_like(masked[name])
        return masked

    def _loss_fn(self, model: MLPActorCritic, batch: Dict[str, mx.array]) -> mx.array:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_values = batch["old_values"]

        if self.cfg.normalize_advantage_per_mini_batch:
            advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        mean, sigma, log_std = model.distribution_params(obs)
        values = model.value(batch.get("critic", obs))
        log_probs = diag_gaussian_log_prob(actions, mean, log_std)
        entropy = mx.mean(diag_gaussian_entropy(log_std))

        log_ratio = mx.clip(
            log_probs - old_log_probs, -self.cfg.log_ratio_clip, self.cfg.log_ratio_clip
        )
        ratio = mx.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))

        if self.cfg.use_clipped_value_loss:
            value_pred_clipped = old_values + mx.clip(
                values - old_values, -self.cfg.clip_param, self.cfg.clip_param
            )
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = mx.mean(mx.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = mx.mean((returns - values) ** 2)

        loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
        if self.mirror_loss_coeff:
            loss = loss + self.mirror_loss_coeff * self._mirror_loss(model, obs)
        if self.consistency_weight:
            loss = loss + self.consistency_weight * self._consistency_loss(model, batch, mean)
        if self._actor_anchor_obs is not None:
            anchor_prediction = model.policy(self._actor_anchor_obs)
            anchor_loss = mx.mean(mx.square(anchor_prediction - self._actor_anchor_actions))
            loss = loss + self._actor_anchor_weight * anchor_loss
        return loss

    def _consistency_loss(self, model, batch, mean=None):
        """Same-state perturbation pair; never compare shuffled temporal states."""
        if not self.consistency_weight:
            return mx.array(0.0, dtype=self._dtype)
        obs = mx.stop_gradient(batch["obs"])
        delta = mx.stop_gradient(batch["consistency_delta"])
        if mean is None:
            mean = model.policy(obs)
        difference = (model.policy(obs + delta) - mean) * self._consistency_scale
        return mx.mean(mx.square(difference))

    def _mirror_loss(self, model: MLPActorCritic, obs: mx.array) -> mx.array:
        # Mirror raw observations before normalization. Match upstream RSL-RL:
        # gradients flow only through the prediction for the mirrored input.
        obs = mx.stop_gradient(obs)
        target = mx.stop_gradient(self.symmetry.mirror_action(model.policy(obs)))
        prediction = model.policy(self.symmetry.mirror_obs(obs))
        return mx.mean(mx.square(prediction - target))

    def _metrics(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_values = batch["old_values"]
        old_mu = batch["old_mu"]
        old_sigma = batch["old_sigma"]

        if self.cfg.normalize_advantage_per_mini_batch:
            advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        mean, sigma, log_std = self.model.distribution_params(obs)
        values = self.model.value(batch.get("critic", obs))
        log_probs = diag_gaussian_log_prob(actions, mean, log_std)
        entropy = mx.mean(diag_gaussian_entropy(log_std))
        sigma = mx.maximum(sigma, 1e-5)
        old_sigma_safe = mx.maximum(old_sigma, 1e-5)

        log_ratio = mx.clip(
            log_probs - old_log_probs, -self.cfg.log_ratio_clip, self.cfg.log_ratio_clip
        )
        ratio = mx.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))
        clip_fraction = mx.mean((mx.abs(ratio - 1.0) > self.cfg.clip_param).astype(self._dtype))

        if self.cfg.use_clipped_value_loss:
            value_pred_clipped = old_values + mx.clip(
                values - old_values, -self.cfg.clip_param, self.cfg.clip_param
            )
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = mx.mean(mx.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = mx.mean((returns - values) ** 2)

        ratio_mean = mx.mean(ratio)
        ratio_max = mx.max(ratio)
        std_mean = mx.mean(sigma)
        adv_std = mx.std(advantages)
        returns_var = mx.var(returns)
        explained_variance = 1.0 - mx.var(returns - values) / (returns_var + 1e-8)

        # Match rsl-rl style analytic KL for adaptive LR.
        kl = mx.sum(
            mx.log(sigma / old_sigma_safe)
            + (old_sigma_safe**2 + (old_mu - mean) ** 2) / (2.0 * sigma**2)
            - 0.5,
            axis=-1,
        )
        kl_mean = mx.mean(kl)
        symmetry_loss = (
            self._mirror_loss(self.model, obs) if self.mirror_loss_coeff else mx.array(0.0)
        )
        consistency_loss = self._consistency_loss(self.model, batch)
        mx.eval(
            policy_loss,
            value_loss,
            entropy,
            kl_mean,
            clip_fraction,
            ratio_mean,
            ratio_max,
            std_mean,
            adv_std,
            explained_variance,
            symmetry_loss,
            consistency_loss,
        )
        return {
            "surrogate": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(kl_mean.item()),
            "clip_fraction": float(clip_fraction.item()),
            "ratio_mean": float(ratio_mean.item()),
            "ratio_max": float(ratio_max.item()),
            "std_mean": float(std_mean.item()),
            "adv_std": float(adv_std.item()),
            "value_explained_variance": float(explained_variance.item()),
            "symmetry": float(symmetry_loss.item()),
            "policy_consistency_rad2": float(consistency_loss.item()),
        }

    def update(self, buffer: RolloutBuffer, iteration: int = -1) -> Dict[str, float]:
        agg = {
            "surrogate": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": 0.0,
            "ratio_max": 0.0,
            "std_mean": 0.0,
            "adv_std": 0.0,
            "value_explained_variance": 0.0,
            "symmetry": 0.0,
            "policy_consistency_rad2": 0.0,
        }
        updates = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grads = 0
        rolled_back_updates = 0
        skipped_nonfinite_metrics = 0
        early_stopped_kl = 0
        last_metrics: Dict[str, float] | None = None
        in_warmup = (iteration >= 0) and (iteration < int(self.cfg.warmup_strict_iters))
        metrics_interval = (
            max(1, int(self.cfg.warmup_metrics_interval))
            if in_warmup
            else max(1, int(self.cfg.metrics_interval))
        )
        finite_check_interval = (
            max(1, int(self.cfg.warmup_finite_check_interval))
            if in_warmup
            else max(1, int(self.cfg.finite_check_interval))
        )
        target_dtype = self._dtype
        for batch_idx, batch in enumerate(
            buffer.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        ):
            # Mixed precision: cast batch to model dtype (e.g. float32) when buffer is float16.
            batch = tree_map(
                lambda x: (
                    x.astype(target_dtype)
                    if hasattr(x, "astype") and getattr(x, "dtype", None) != target_dtype
                    else x
                ),
                batch,
            )
            do_full_checks = batch_idx % finite_check_interval == 0
            if self.consistency_weight:
                # Sample outside autodiff/compile; metrics reuse the same pair.
                batch["consistency_delta"] = mx.random.uniform(
                    low=-1.0, high=1.0, shape=batch["obs"].shape, dtype=target_dtype
                ) * self._consistency_noise
            if self.cfg.disable_finite_checks:
                do_full_checks = False
            do_metrics = (batch_idx % metrics_interval == 0) or (last_metrics is None)

            try:
                loss, grads = self.compiled_loss_and_grad(self.model, batch)
            except Exception:
                # Fallback: some MLX versions do not support compiling this closure shape.
                self.compiled_loss_and_grad = self.loss_and_grad
                loss, grads = self.loss_and_grad(self.model, batch)
            if do_full_checks and (not mx.all(mx.isfinite(loss)).item()):
                skipped_nonfinite_loss += 1
                continue
            if do_full_checks and (not self._all_finite(grads)):
                skipped_nonfinite_grads += 1
                continue
            grads = self._mask_actor_grads(grads)
            grads = self._clip_grads(grads)
            if do_full_checks and (not self._all_finite(grads)):
                skipped_nonfinite_grads += 1
                continue
            self.optimizer.update(self.model, grads)
            mx.eval(loss, self.model.parameters(), self.optimizer.state)
            if do_full_checks and (not self._all_finite(self.model.parameters())):
                skipped_nonfinite_grads += 1
                continue

            if do_metrics:
                metrics = self._metrics(batch)
                if not all(math.isfinite(v) for v in metrics.values()):
                    skipped_nonfinite_metrics += 1
                    continue
                last_metrics = metrics
            else:
                metrics = (
                    last_metrics
                    if last_metrics is not None
                    else {
                        "surrogate": 0.0,
                        "value": 0.0,
                        "entropy": 0.0,
                        "approx_kl": 0.0,
                        "clip_fraction": 0.0,
                        "ratio_mean": 1.0,
                        "ratio_max": 1.0,
                        "std_mean": 0.0,
                        "adv_std": 0.0,
                        "value_explained_variance": 0.0,
                        "symmetry": 0.0,
                        "policy_consistency_rad2": 0.0,
                    }
                )

            # The optimizer step above has already been applied, so account for
            # it before an optional KL early stop.  Previously the break came
            # first and reported zero updates even though parameters changed.
            for key in agg:
                agg[key] += metrics[key]
            updates += 1

            if (
                self.cfg.target_kl_stop is not None
                and metrics["approx_kl"] > self.cfg.target_kl_stop
            ):
                early_stopped_kl += 1
                break

            if do_metrics and self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                kl = metrics["approx_kl"]
                if self._kl_ema is None:
                    self._kl_ema = kl
                else:
                    beta = min(max(self.cfg.adaptive_kl_beta, 0.0), 0.999)
                    self._kl_ema = beta * self._kl_ema + (1.0 - beta) * kl
                if (updates + 1) % max(1, int(self.cfg.adaptive_lr_update_interval)) == 0:
                    kl_for_lr = self._kl_ema
                    if kl_for_lr > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(
                            self.cfg.min_learning_rate,
                            self.learning_rate / max(self.cfg.adaptive_lr_decay, 1.01),
                        )
                    elif 0.0 < kl_for_lr < self.cfg.desired_kl / 2.0:
                        self.learning_rate = min(
                            self.cfg.max_learning_rate,
                            self.learning_rate * max(self.cfg.adaptive_lr_growth, 1.0),
                        )
                    self.optimizer.learning_rate = mx.array(self.learning_rate, dtype=self._dtype)

        if updates == 0:
            return {
                **agg,
                "learning_rate": self.learning_rate,
                "updates_applied": 0.0,
                "skipped_nonfinite_loss": float(skipped_nonfinite_loss),
                "skipped_nonfinite_grads": float(skipped_nonfinite_grads),
                "rolled_back_updates": float(rolled_back_updates),
                "skipped_nonfinite_metrics": float(skipped_nonfinite_metrics),
                "early_stopped_kl": float(early_stopped_kl),
            }
        out = {key: value / updates for key, value in agg.items()}
        out["learning_rate"] = self.learning_rate
        out["updates_applied"] = float(updates)
        out["skipped_nonfinite_loss"] = float(skipped_nonfinite_loss)
        out["skipped_nonfinite_grads"] = float(skipped_nonfinite_grads)
        out["rolled_back_updates"] = float(rolled_back_updates)
        out["skipped_nonfinite_metrics"] = float(skipped_nonfinite_metrics)
        out["early_stopped_kl"] = float(early_stopped_kl)
        return out
