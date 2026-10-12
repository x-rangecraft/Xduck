"""Physical decoder checks, not policy-learning acceptance."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from hydra import compose, initialize_config_dir
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.envs.locomotion.xduck_gf43x40.motor_path import ContinuousMotorPath


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physics-model", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    ensure_registries()
    with initialize_config_dir(config_dir=str(root / "conf/ppo"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=xduck_gf43x40_sitstand_incremental/mjwarp"])
    cfg.env.warp_compiled_model_file = str(a.physics_model)
    archive = json.loads((root / "src/unilab/envs/locomotion/xduck_gf43x40/sitstand_reference_v4.json").read_text())
    rows = []
    for case, mode in [(2, "path"), (3, "path"), (0, "zero"), (1, "zero"), (0, "random"), (1, "random")]:
        np.random.seed(6601)
        c = copy.deepcopy(cfg)
        c.env.evaluation_case = case
        c.env.max_episode_seconds = 8.1
        c.env.posture_dwell_seconds = [100., 100.]
        for stage in c.env.curriculum.head_range_stages:
            stage.ranges = [[0., 0.]] * 4
        env = create_env(c, num_envs=8, env_cfg_override=BackendAdapter(c, root_dir=root).build_task_env_cfg_override())
        try:
            env.init_state()
            env.set_autoreset(False)
            assert env.state.obs["obs"].shape == (8, 61)
            path = ContinuousMotorPath(archive["rise" if case == 2 else "descent"]["nodes"], archive["rise" if case == 2 else "descent"]["durations"]) if mode == "path" else None
            failed = np.zeros(8, dtype=bool)
            trace = {k: [] for k in ("qpos", "height", "qvel", "success", "target_delta", "desired_target_error")}
            for step in range(400):
                previous = env._current_action_target.copy()
                desired = previous
                if mode == "path":
                    desired = path.sample(np.full(8, max(0., step * .02 - 1.5)))
                    action = np.arctanh(np.clip((desired - previous) / env._increments, -.999999, .999999))
                elif mode == "random":
                    action = np.random.normal(0, .25, (8, 14))
                else:
                    action = np.zeros((8, 14))
                env.step(action)
                failed |= env._failed
                delta = env._current_action_target - previous
                assert (np.abs(delta) <= env._increments + 1e-8).all()
                values = dict(qpos=env._backend.get_physics_state()[:, 1:22],
                    height=env._backend.get_base_pos()[:, 2], qvel=env.get_dof_vel(),
                    success=env.last_success, target_delta=delta,
                    desired_target_error=env._current_action_target - desired)
                for k, v in values.items():
                    trace[k].append(v.copy())
            data = {k: np.asarray(v) for k, v in trace.items()}
            row = dict(case=case, mode=mode, trials=8,
                successes=int((data["success"][-50:].all(axis=0) & ~failed).sum()), falls=int(failed.sum()),
                late_height=float(data["height"][-100:].mean()),
                late_joint_speed_rms=float(np.sqrt(np.mean(data["qvel"][-100:] ** 2))),
                target_delta_rms=float(np.sqrt(np.mean(data["target_delta"] ** 2))),
                max_desired_target_error=float(np.abs(data["desired_target_error"]).max()))
            a.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.output / f"case_{case}_{mode}.npz", **data)
            rows.append(row)
            (a.output / "results.json").write_text(json.dumps(dict(conditions="GF unchanged; original physical/reset randomization; seed6601; fixed command/head; no autoreset; 8s", rows=rows), indent=2))
            print(json.dumps(row), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
