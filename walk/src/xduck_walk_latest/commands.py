"""XDuck walking command tensor in the checkpoint's thirteen-value order."""

from dataclasses import dataclass

import numpy as np
from unilab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class XDuckWalkCommandCfg(CommandTermCfg):
    velocity_low: tuple[float, float, float] = (-0.5782732918, -0.4337049688, -0.6917144639)
    velocity_high: tuple[float, float, float] = (0.5782732918, 0.4337049688, 0.6917144639)
    head_low: tuple[float, float, float, float] = (-0.05, -0.05, -0.07, -0.015)
    head_high: tuple[float, float, float, float] = (0.05, 0.05, 0.07, 0.015)
    body_low: tuple[float, float, float, float, float, float] = (-0.005, -0.005, -0.005, -0.05, -0.05, -0.05)
    body_high: tuple[float, float, float, float, float, float] = (0.005, 0.005, 0.005, 0.05, 0.05, 0.05)
    head_resampling_time_range: tuple[float, float] = (2.89136645896, 7.22841614740)
    body_resampling_time_range: tuple[float, float] = (2.89136645896, 7.22841614740)
    rel_forward_envs: float = 0.2
    rel_turn_in_place_envs: float = 0.15
    rel_standing_envs: float = 0.02
    forward_min_speed: float = 0.43370496884

    def build(self, env):
        return XDuckWalkCommand(self, env)


class XDuckWalkCommand(CommandTerm):
    """Resample velocity, head and body on their separate legacy timers."""

    cfg: XDuckWalkCommandCfg

    def __init__(self, cfg: XDuckWalkCommandCfg, env):
        super().__init__(cfg, env)
        self._command = np.zeros((env.num_envs, 13), dtype=np.float32)
        self._head_time = np.zeros(env.num_envs, dtype=np.float32)
        self._body_time = np.zeros(env.num_envs, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _update_metrics(self, env_ids=None) -> None:
        pass

    def _resample_command(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        rng = self._env.rng
        cfg = self.cfg
        velocity = rng.uniform(cfg.velocity_low, cfg.velocity_high, (n, 3))
        forward = rng.uniform(size=n) < cfg.rel_forward_envs
        standing = rng.uniform(size=n) < cfg.rel_standing_envs
        turn = rng.uniform(size=n) < cfg.rel_turn_in_place_envs
        velocity[forward, 1:] = 0.0
        velocity[forward, 0] = np.maximum(np.abs(velocity[forward, 0]), cfg.forward_min_speed)
        velocity[turn, :2] = 0.0
        nturn = int(np.count_nonzero(turn))
        if nturn:
            turn_limit = min(abs(cfg.velocity_low[2]), abs(cfg.velocity_high[2]))
            sign = np.where(rng.uniform(size=nturn) < 0.5, -1.0, 1.0)
            velocity[turn, 2] = sign * rng.uniform(min(0.4 * 1.445683, turn_limit), turn_limit, nturn)
        velocity[standing] = 0.0
        self._command[env_ids, :3] = velocity
        # CommandTerm.reset() clears command_counter before its first resample.
        initial = env_ids[self.command_counter[env_ids] == 0]
        if len(initial):
            self._sample_head(initial)
            self._sample_body(initial)

    def _sample_head(self, ids):
        self._command[ids, 3:7] = self._env.rng.uniform(
            self.cfg.head_low, self.cfg.head_high, (len(ids), 4)
        )
        self._head_time[ids] = self._env.rng.uniform(
            *self.cfg.head_resampling_time_range, size=len(ids)
        )

    def _sample_body(self, ids):
        self._command[ids, 7:] = self._env.rng.uniform(
            self.cfg.body_low, self.cfg.body_high, (len(ids), 6)
        )
        self._body_time[ids] = self._env.rng.uniform(
            *self.cfg.body_resampling_time_range, size=len(ids)
        )

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        ids = np.arange(self.num_envs, dtype=np.int32) if env_ids is None else env_ids
        # On a reset compute(dt=0) the command was just sampled above.
        if env_ids is not None:
            return
        self._head_time[ids] -= self._env.step_dt
        self._body_time[ids] -= self._env.step_dt
        head_due = ids[self._head_time[ids] <= 0]
        body_due = ids[self._body_time[ids] <= 0]
        if len(head_due):
            self._sample_head(head_due)
        if len(body_due):
            self._sample_body(body_due)
