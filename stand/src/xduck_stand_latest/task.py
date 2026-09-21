"""Manager-Based GF43X40 stand / walk-to-stop task over the shared walk plant."""

from __future__ import annotations

import json
from pathlib import Path

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.managers.event_manager import EventTermCfg
from unilab.managers.metrics_manager import MetricsTermCfg
from unilab.managers.reward_manager import RewardTermCfg
from unilab.managers.termination_manager import TerminationTermCfg
from xduck_walk_latest import rewards as walk_rewards
from xduck_walk_latest.task import make_walk_env, walk_cfg

from .commands import XDuckStandCommandCfg
from .terms import HandoffReset, SettledState, handoff_action_rate_l2, stop_failure

_TASK_NAME = "XDuckGF43X40WalkStopFlat"
_ASSETS = Path(__file__).resolve().parent / "assets"
_BANK = _ASSETS / "handoff/mixed70stand30walk.npz"
_REWARD = _ASSETS / "selected_reward.json"


def _stop_rewards() -> dict[str, RewardTermCfg]:
    saved = json.loads(_REWARD.read_text())
    result: dict[str, RewardTermCfg] = {}
    for name, weight in saved["scales"].items():
        if name == "action_rate_l2":
            func = handoff_action_rate_l2
        else:
            func = walk_rewards.REWARD_FUNCTIONS[name]
        params = {}
        if name == "track_linear_velocity":
            params["std"] = float(saved["linear_velocity_std"])
        elif name == "track_angular_velocity":
            params["std"] = float(saved["angular_velocity_std"])
        elif name == "upright":
            params["std"] = float(saved["upright_std"])
        result[name] = RewardTermCfg(func=func, weight=float(weight), params=params)
    return result


def stand_cfg() -> ManagerBasedRlEnvCfg:
    """Derive stop semantics from the same GF plant and policy I/O as walking."""
    cfg = walk_cfg()
    cfg.max_episode_seconds = 12.0
    cfg.commands["walk"] = XDuckStandCommandCfg(resampling_time_range=(6.0, 12.0))
    cfg.events = dict(cfg.events)
    for axis in ("x", "y"):
        cfg.events["velocity_push"].params["velocity_range"][axis] = (-0.2, 0.2)
    cfg.events["handoff_reset"] = EventTermCfg(
        func=HandoffReset, mode="reset", params={"bank_path": str(_BANK)}
    )
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations["stop_failure"] = TerminationTermCfg(
        func=stop_failure,
        params={"max_tilt_deg": 55.0, "min_height_ratio": 0.6},
    )
    cfg.rewards = _stop_rewards()
    cfg.metrics = {
        "stop_settled": MetricsTermCfg(
            func=SettledState,
            params={
                "linear_speed": 0.03,
                "angular_speed": 0.2,
                "joint_speed_rms": 0.15,
                "hold_seconds": 1.0,
                "contact_threshold": 0.1,
            },
            reduce="last",
        )
    }
    return cfg


def make_stand_env(
    cfg: ManagerBasedRlEnvCfg | None = None,
    num_envs: int = 1,
    backend_type: str = "mujoco",
):
    return make_walk_env(
        cfg=stand_cfg() if cfg is None else cfg,
        num_envs=num_envs,
        backend_type=backend_type,
    )


registry.register_env_config(_TASK_NAME, ManagerBasedRlEnvCfg)
registry.register_env(_TASK_NAME, make_stand_env, sim_backend="mujoco")
