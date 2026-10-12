"""Evaluate each completed 100-iteration XDuck GF checkpoint during training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from eval_xduck_gf_checkpoint import evaluate


def checkpoint_index(path: Path) -> int | None:
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def write_trend(check_dir: Path) -> None:
    rows = []
    for path in check_dir.glob("model_*_eval.json"):
        result = json.loads(path.read_text())
        forward = result["buckets"]["forward_02"]
        idle = result["buckets"]["idle"]
        rows.append(
            {
                "checkpoint": result["checkpoint_index"],
                "forward_vx_m_s": forward["vx"],
                "forward_yaw_rad_s": forward["yaw_rate"],
                "forward_single_support": forward["single_support"],
                "forward_air_window": forward["air_window"],
                "forward_contact_changes_per_s": forward["contact_change"],
                "forward_reward_rate": forward["reward_rate"],
                "forward_torque_p99_nm": forward["torque_p99_nm"],
                "forward_action_clip": forward["action_target_clip"],
                "idle_vx_m_s": idle["vx"],
                "idle_yaw_rad_s": idle["yaw_rate"],
                "idle_reward_rate": idle["reward_rate"],
                "forward_falls": forward["falls"],
                "idle_falls": idle["falls"],
            }
        )
    rows.sort(key=lambda row: row["checkpoint"])
    if not rows:
        return
    path = check_dir / "trend.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def process_available(run_dir: Path, *, min_age_s: float = 5.0) -> list[int]:
    check_dir = run_dir / "checks"
    check_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    now = time.time()
    for checkpoint in sorted(
        run_dir.glob("model_*.pt"), key=lambda path: checkpoint_index(path) or -1
    ):
        index = checkpoint_index(checkpoint)
        if index is None or index <= 0 or (index % 100 != 0 and index != 999):
            continue
        output = check_dir / f"model_{index:04d}_eval.json"
        if output.exists() or checkpoint.stat().st_size < 1_000_000:
            continue
        if now - checkpoint.stat().st_mtime < min_age_s:
            continue
        result = evaluate(checkpoint, per_bucket=8, steps=400, warmup=100)
        output.write_text(json.dumps(result, indent=2) + "\n")
        completed.append(index)
        print(f"evaluated model_{index}.pt -> {output}", flush=True)
    if completed:
        write_trend(check_dir)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    while True:
        try:
            process_available(run_dir)
        except Exception as exc:
            print(f"checkpoint evaluation failed; will retry: {exc!r}", flush=True)
        if args.once or (run_dir / "checks/model_0999_eval.json").exists():
            break
        if args.train_pid is not None:
            try:
                os.kill(args.train_pid, 0)
            except ProcessLookupError:
                print("training process ended before the final checkpoint", flush=True)
                break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
