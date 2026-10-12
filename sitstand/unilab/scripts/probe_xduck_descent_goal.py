"""Diagnostic reachability of the existing SIT goal; never used for PPO."""
import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.motor_path import ContinuousMotorPath
from unilab.envs.locomotion.xduck_gf43x40.sitstand import LEG_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ensure_registries()
    archive = json.loads((root / "src/unilab/envs/locomotion/xduck_gf43x40/sitstand_reference_v4.json").read_text())["descent"]
    path = ContinuousMotorPath(archive["nodes"], archive["durations"])
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in (6401, 7401):
        np.random.seed(seed + 300)
        with initialize_config_dir(config_dir=str(root / "conf/ppo"), version_base=None):
            cfg = compose(config_name="config", overrides=["task=xduck_gf43x40_sitstand_incremental/mjwarp"])
        cfg.env.warp_compiled_model_file = str(args.physics_model)
        cfg.env.evaluation_case = 3
        cfg.env.max_episode_seconds = 8.1
        cfg.env.posture_dwell_seconds = [100., 100.]
        cfg.env.curriculum.step_cap = 36000
        for item in cfg.env.curriculum.head_range_stages:
            item.ranges = [[0., 0.]] * 4
        env = create_env(cfg, num_envs=16, env_cfg_override=BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override())
        try:
            env.step_counter = 36000
            env._update_curriculum()
            env.init_state()
            env.set_autoreset(False)
            goal = env.default_angles.copy()
            goal[LEG_IDS] = env._sit_legs
            np.testing.assert_allclose(archive["nodes"][-1], goal, atol=1e-6)
            failed = np.zeros(16, dtype=bool)
            trace = {key: [] for key in ("success", "pose_error", "qpos", "qvel", "target", "height", "head_error")}
            for step in range(400):
                desired = path.sample(np.full(16, max(0., step * .02 - 1.5)))
                action = np.arctanh(np.clip((desired - env._current_action_target) / env._increments, -.999999, .999999))
                env.step(action)
                failed |= env._failed
                values = dict(success=env.last_success, pose_error=env.last_pose_error,
                              qpos=env._backend.get_physics_state()[:, 1:22], qvel=env.get_dof_vel(),
                              target=env._current_action_target, height=env._backend.get_base_pos()[:, 2],
                              head_error=env.get_dof_pos()[:, 5:9] - goal[5:9])
                for key, value in values.items():
                    trace[key].append(value.copy())
            data = {key: np.asarray(value) for key, value in trace.items()}
            head_ok = (np.abs(data["head_error"][-50:]) < .15).all(axis=(0, 2))
            success = data["success"][-50:].all(axis=0) & head_ok & ~failed
            row = dict(seed=seed, trials=16, successes=int(success.sum()), falls=int(failed.sum()),
                       late_pose_error=float(data["pose_error"][-50:].mean()),
                       late_height=float(data["height"][-50:].mean()),
                       late_tracking_rms=float(np.sqrt(np.mean((data["qpos"][-50:, :, 7:] - data["target"][-50:]) ** 2))),
                       target_matches_goal=bool(np.allclose(data["target"][-50:], goal, atol=1e-6)))
            np.savez_compressed(args.output / f"seed_{seed}.npz", **data)
            rows.append(row)
            (args.output / "results.json").write_text(json.dumps(dict(
                conditions="Fixed diagnostic motor path to exact existing SIT goal, unchanged GF/decoder/limits; original physical/reset/noise DR; 1500-stage curriculum; no PPO or policy training; 8 s.", rows=rows), indent=2))
            print(json.dumps(row), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
