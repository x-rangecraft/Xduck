"""Thin PPO CLI that adds XDuck walk/stand registry and Hydra owner paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from unilab.cli import package_root as unilab_package_root

_CONF = Path(__file__).resolve().parents[2] / "conf/ppo"
_PACKAGES = ("xduck_stand_latest.tasks",)


def _run(mode: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"xduck-stand-{mode}")
    parser.add_argument("--load-run", default=None) if mode == "eval" else None
    args, overrides = parser.parse_known_args(argv)
    existing = [
        p.strip()
        for p in os.environ.get("UNILAB_EXTRA_REGISTRY_PACKAGES", "").split(",")
        if p.strip()
    ]
    os.environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] = ",".join(
        dict.fromkeys([*existing, *_PACKAGES])
    )
    os.environ["XDUCK_STAND_BANK"] = str(
        Path(__file__).resolve().parent / "assets/handoff/mixed70stand30walk.npz"
    )
    script = unilab_package_root() / "scripts/train_rsl_rl.py"
    command = [
        sys.executable,
        str(script),
        "--config-path",
        str(_CONF),
        "task=xduck_gf43x40_walk_stop_flat/mujoco",
    ]
    if mode == "eval":
        command.append("training.play_only=true")
        if args.load_run is not None:
            command.append(f"algo.load_run={args.load_run}")
    command.extend(overrides)
    return subprocess.run(command, check=False).returncode


def train_main(argv: Sequence[str] | None = None) -> int:
    return _run("train", argv)


def eval_main(argv: Sequence[str] | None = None) -> int:
    return _run("eval", argv)
