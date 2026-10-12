"""Opt-in finite-value checks for the current RSL-RL PPO runtime."""

import math

import torch
from rsl_rl.models import MLPModel
from uni_rl.algos.rsl_rl_ppo import FinalObservationAwarePPO


def check_optimizer_tensors(optimizer, *, gradients):
    values = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            values.append(torch.isfinite(parameter).all())
            if gradients and parameter.grad is not None:
                values.append(torch.isfinite(parameter.grad).all())
    if values and not bool(torch.stack(values).all()):
        raise FloatingPointError("PPO parameter/gradient is nonfinite; stopping update")


class FiniteCheckedPPO(FinalObservationAwarePPO):
    """Keep UniRL timeout handling and PPO updates, adding active numeric guards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer.register_step_pre_hook(
            lambda optimizer, args, kwargs: check_optimizer_tensors(optimizer, gradients=True)
        )
        self.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: check_optimizer_tensors(optimizer, gradients=False)
        )

    def save(self):
        state = super().save()
        state["adaptive_learning_rate"] = float(self.learning_rate)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            rates = {float(g["lr"]) for g in self.optimizer.param_groups}
            if len(rates) != 1:
                raise ValueError("PPO resume requires a shared optimizer learning rate")
            rate = float(loaded_dict.get("adaptive_learning_rate", rates.pop()))
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid checkpoint learning rate")
            self.learning_rate = rate
            for group in self.optimizer.param_groups:
                group["lr"] = rate
        return result

    def _supports_compiled_update_path(self):
        # UniRL's fused path calls actor.mlp directly and would omit a custom
        # actor's additional branches. Keep its ordinary forward-based path.
        if type(self.actor) is not MLPModel:
            return False
        return super()._supports_compiled_update_path()

    def update(self):
        metrics = super().update()
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("PPO loss metric is nonfinite; stopping training")
        return metrics
