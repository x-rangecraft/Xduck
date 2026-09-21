"""Export every planned curriculum boundary and empirical reward budgets to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEDULE_REWARDS = {
    "action_rate_stages": "action_rate_l2",
    "head_bias_stages": "head_pose_bias",
    "torque_rate_stages": "joint_torque_rate_l2",
    "arrival_damping_stages": "arrival_damping",
    "body_pose_weight_stages": "body_pose_tracking",
    "sharp_height_stages": "height_stand_sharp",
    "sharp_upright_stages": "upright_sharp",
    "composite_stages": "standing_composite",
    "posture_composite_stages": "posture_composite",
    "descent_speed_stages": "descent_speed",
    "rise_speed_stages": "rise_speed",
}


def export(directory):
    snapshot_path = directory / "aligned_snapshot.json"
    calculations_path = directory / "aligned_physics_curriculum.json"
    if not snapshot_path.exists():
        snapshot_path = directory / "after.json"
    if not calculations_path.exists():
        calculations_path = directory / "after_calculations.json"
    evidence_root = directory if (directory / "reference_rollouts").exists() else directory.parent
    snapshot = json.loads(snapshot_path.read_text())
    calculations = json.loads(calculations_path.read_text())
    reference = json.loads((evidence_root / "reference_rollouts/summary.json").read_text())["runs"]
    slew = json.loads((evidence_root / "torque_slew_counterfactual.json").read_text())["cases"]
    rows = []
    for task, phases in calculations["curricula"].items():
        source_name = {"velocity": "walking", "standup": "stand", "sitstand": "sitstand"}[task]
        runs = [r for r in reference if r["task"] == source_name]
        tau_runs = [r for r in slew if r["case"].startswith(source_name + "_")]
        action = float(np.mean([r["action_delta_l2_mean"] for r in runs]))
        momentum = float(np.mean([r["new_kinematic_L2_mean"] for r in runs]))
        old_momentum = float(np.mean([r["old_L2_mean"] for r in runs]))
        torque = float(np.mean([r["new_normalized_torque_delta_l2"] for r in tau_runs]))
        for phase in phases:
            parameters = phase["parameters"]
            scales = snapshot[task]["reward"]["scales"].copy()
            for key, name in SCHEDULE_REWARDS.items():
                if key in parameters:
                    scales[name] = parameters[key]
            h = phase["head_corner_audit"]
            amp = np.max(np.abs(parameters["head_range_stages"]), axis=1)
            changes = {}
            for name, delta in phase["increments"].items():
                before = np.asarray(delta["before"])
                after = np.asarray(delta["after"])
                mask = np.abs(before) > 1e-12
                changes[name] = {
                    **delta,
                    "max_magnitude_ratio_where_nonzero": float(
                        np.max(np.abs(after[mask] / before[mask]))
                    )
                    if np.any(mask)
                    else None,
                    "activates_zero_component": bool(np.any((~mask) & (np.abs(after) > 1e-12))),
                }
            sigma = 0.5
            expected_head = np.mean(
                [
                    math.sqrt(math.pi) * sigma * math.erf(a / sigma) / (2 * a) if a > 0 else 1.0
                    for a in amp
                ]
            )
            rows.append(
                {
                    "task": task,
                    "iteration": phase["iteration"],
                    "env_step": phase["env_step"],
                    "automatic_promotion_enabled": phase["automatic_promotion_enabled"],
                    "inactive_inherited_parameters": json.dumps(
                        ["standing_stages"] if task == "sitstand" else []
                    ),
                    "effective_changed_parameters": json.dumps(
                        [
                            key
                            for key in changes
                            if not (task == "sitstand" and key == "standing_stages")
                        ]
                    ),
                    "effective_zero_twist_fraction": parameters["standing_stages"]
                    if task != "sitstand"
                    else None,
                    "effective_forward_bucket_fraction": (1 - parameters["standing_stages"])
                    * snapshot[task]["commands"]["rel_forward_envs"]
                    * (1 - snapshot[task]["commands"]["rel_turn_in_place_envs"])
                    if task != "sitstand"
                    else None,
                    "effective_turn_bucket_fraction": (1 - parameters["standing_stages"])
                    * snapshot[task]["commands"]["rel_turn_in_place_envs"]
                    if task != "sitstand"
                    else None,
                    "samples_per_rank": phase["transitions_per_rank"],
                    "equivalent_iterations_at_4096_envs": phase["transitions_per_rank"]
                    / (4096 * 24),
                    "head_abs_command_rad": json.dumps(amp.tolist()),
                    "head_worst_resample_jump_rad": float(2 * max(amp)),
                    "com_uncertainty_bound_mm": 1000 * h["com_uncertainty_bound_m"],
                    "minimum_head_grid_margin_mm": 1000 * h["minimum_com_margin_after_dr_m"],
                    "head_grid_hard_limit_failures": h["hard_limit_failures"],
                    "static_rated_torque_utilization": h["max_feasible_rated_utilization"],
                    "maximum_push_vector_m_s": phase["push_max_vector_m_s"],
                    "push_kinetic_energy_j": 0.5
                    * calculations["new_model"]["mass_kg"]
                    * phase["push_max_vector_m_s"] ** 2,
                    "capture_point_shift_mm": 1000 * phase["capture_point_shift_m"],
                    "push_requires_support_adaptation": phase["requires_stepping_under_worst_push"],
                    "reference_action_rate_cost": scales["action_rate_l2"] * action,
                    "reference_lifted_momentum_cost": scales["angular_momentum"]
                    * momentum
                    / 0.25**2,
                    "original_reference_momentum_cost": -0.02 * old_momentum,
                    "reference_frozen_motor_slew_cost": scales.get("joint_torque_rate_l2", 0)
                    * torque,
                    "head_tracking_expected_score_if_ignored": expected_head,
                    "head_bias_expected_cost_if_ignored": -scales.get("head_pose_bias", 0)
                    * float(np.mean(amp / 2)),
                    "all_effective_reward_weights": json.dumps(scales, sort_keys=True),
                    "all_parameters": json.dumps(parameters, sort_keys=True),
                    "increments_and_growth_ratios": json.dumps(changes, sort_keys=True),
                }
            )
    with (directory / "curriculum_all_stages.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "curriculum_reward_budgets.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} planned boundaries exported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
