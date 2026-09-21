"""Cold-path conversion of pinned upstream curricula to a 1024-env sample budget."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "mjlab"
REVISION = "d424a0c899f6b33cbd3daeb279913134349c0b63"
SOURCE = "src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py"
PROFILE = ROOT / "conf/ppo/experiment/xduck_official4096_sample_aligned_1024_v1.yaml"
EVIDENCE = ROOT / "docs/pretraining_audit/curriculum4096_to1024"
# (upstream term, stages field, value field, XDuck field, physical-unit factor)
MAPPING = [
    ("action_rate_weight", "weight_stages", "weight", "action_rate_stages", 1.0),
    ("standing_envs", "standing_stages", "rel_standing_envs", "standing_stages", 1.0),
    ("head_pose_range", "range_stages", "ranges", "head_range_stages", 1.0),
    ("head_pose_bias_weight", "weight_stages", "weight", "head_bias_stages", 1.0),
    ("com_range", "range_stages", "range", "com_range_stages", 2.09),
    ("head_com_range", "range_stages", "range", "head_com_range_stages", 2.09),
]


def source_at_commit(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "show", f"{REVISION}:{path}"], text=True
    )


def literal(node):
    if isinstance(node, ast.Name) and node.id == "NUM_STEPS_PER_ENV":
        return 24
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return literal(node.left) * literal(node.right)
    raise ValueError(f"Unsupported upstream literal: {ast.dump(node)}")


def extract_terms(source):
    terms = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript) and ast.unparse(target.value) == "cfg.curriculum"
        ):
            continue
        name = literal(target.slice)
        if isinstance(node.value, ast.Call):
            terms[name] = {
                kw.arg: (literal(kw.value) if kw.arg == "params" else ast.unparse(kw.value))
                for kw in node.value.keywords
            }
    return terms


def build():
    source = source_at_commit(SOURCE)
    mdp = source_at_commit("src/mjlab_microduck/tasks/mdp.py")
    terms = extract_terms(source)
    comparisons = {}
    for node in ast.walk(ast.parse(mdp)):
        if isinstance(node, ast.FunctionDef):
            for compare in ast.walk(node):
                if (
                    isinstance(compare, ast.Compare)
                    and ast.unparse(compare.left) == "env.common_step_counter"
                ):
                    if len(compare.ops) == 1 and isinstance(compare.ops[0], (ast.Gt, ast.GtE)):
                        comparisons[node.name] = isinstance(compare.ops[0], ast.Gt)
    curriculum = {"step_cap": None}
    rows = []
    for name, stages_key, value_key, target, unit_scale in MAPPING:
        term = terms[name]
        strict = comparisons[term["func"].split(".")[-1]]
        converted = []
        for stage in term["params"][stages_key]:
            anchor = stage["step"]
            # Preserve upstream's earliest eligible integer step, not reset timing.
            eligible = anchor + int(strict) if anchor else 0
            step = eligible * 4
            value = stage[value_key]
            if value_key != "ranges":
                value *= unit_scale
            converted.append({"step": step, "ranges" if value_key == "ranges" else "value": value})
            rows.append(
                {
                    "term": name,
                    "target": target,
                    "official_anchor_step": anchor,
                    "official_anchor_iteration": anchor / 24,
                    "official_strict_gt": strict,
                    "official_earliest_step": eligible,
                    "target_step": step,
                    "target_nominal_iteration": anchor / 24 * 4,
                    "sample_budget": eligible * 4096,
                    "official_value": stage[value_key],
                    "target_value": value,
                }
            )
        curriculum[target] = converted
    profile = {
        "defaults": ["xduck_grouped_baseline_v1", "_self_"],
        "algo": {"num_envs": 1024, "num_steps_per_env": 24},
        "env": {
            "curriculum": curriculum,
            "communication": {
                "jitter_mode": "measured_20260918",
                "command_jitter_ms": [0, 0],
                "feedback_jitter_ms": [0, 0],
                "feedback_delay_ms": 0.0,
                "jitter_seed": 0,
            },
        },
    }
    text = (
        "# @package _global_\n# Generated by scripts/build_xduck_curriculum_profile.py.\n# Same aggregate samples as upstream 4096 x 24; amplitudes do not scale with env count.\n"
        + OmegaConf.to_yaml(OmegaConf.create(profile))
    )
    evidence = {
        "upstream_revision": REVISION,
        "upstream_url": f"https://github.com/pollen-robotics/microduck_rl/blob/{REVISION}/{SOURCE}",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "mdp_sha256": hashlib.sha256(mdp.encode()).hexdigest(),
        "reference_num_envs": 4096,
        "target_num_envs": 1024,
        "rollout_steps": 24,
        "alignment": "aggregate samples at earliest eligible stage; not wall time, PPO optimizer parity or reset dispatch parity",
        "rows": rows,
        "fixed_body_pose": terms["body_pose_range"]["params"]["range_stages"],
    }
    return text, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, evidence = build()
    if args.check:
        assert PROFILE.read_text() == text, "Profile differs from pinned upstream conversion"
    else:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        PROFILE.write_text(text)
        (EVIDENCE / "mapping.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps({"profile": str(PROFILE), "stages": len(evidence["rows"]), "check": args.check})
    )


if __name__ == "__main__":
    main()
