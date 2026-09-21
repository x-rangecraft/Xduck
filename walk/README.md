# XDuck GF43X40 walking on UniLab 1.3.1

This source project replaces the old in-tree walk environment with a Manager-Based task package. It pins the company UniLab commit `447e5ce` (1.3.1) and uses its published `unisim-core 1.7.3` MuJoCo backend. The motor calibration, 1 ms control model, 20 ms host command period, V1.1.2 scene, 14 joint order, 61D actor / 76D critic layouts, and the selected 1975 bounded Gaussian checkpoint are carried over.

## Use

From this directory, with access to the private company repository:

```bash
uv sync
uv run python -m unilab.scripts.train_rsl_rl --config-path "$(pwd)/conf/ppo" \
  training.device=cpu training.no_play=true
```

Use `algo.num_envs`, `algo.max_iterations`, and `training.device` to set the training size. To load the selected checkpoint, add:

```text
algo.load_run=<absolute path to checkpoints/model_1975_strong_bounded.pt>
```

The package registers `XDuckGF43X40VelocityFlatLatest` through the `unilab.tasks` entry point. The Hydra owner is `conf/ppo/task/xduck_gf43x40_walk_latest/mujoco.yaml`; its reward terms live at root `reward`, as required by UniLab 1.3.1. The programmatic owner is `xduck_walk_latest.task.walk_cfg()` and `make_walk_env(cfg, num_envs, backend_type="mujoco")`. Stand can reuse `xduck_walk_latest.actions.GF43X40ActionCfg` and `xduck_walk_latest.assets.scene_asset_path()`.

## Migration status

- The action term calls the frozen GF43X40 motor on every MuJoCo physics substep through the public Manager-Based `ActionTerm` and `SimBackend.set_pre_step_control` contracts. It uses `q + torque` with unit-gain position actuators. Native force sensor and motor torque agree in a focused test.
- The task XML moves the keyframe to a task fragment and materializes the old cold-path actuator force range, broad control range, 0.2543 armature, and zero damping/friction in the model. V1.1.2 meshes are included here.
- The saved 1975 actor and critic state dictionaries load strictly in the current RSL model. A one-iteration resume through the current training entrypoint also succeeds.
- Selected reward names and weights, velocity/head/body command shape, foot-friction/armature reset randomization, GF motor/gain multipliers, interval velocity pushes, and actor observation noise/delay are represented.

**This is not training-trajectory parity.** The latest mjbatch callback provides current joint state but does not refresh the authored foot contact/position/velocity sensors after each 1 ms substep. The old backend copied all sensors each substep. This package samples foot history at the 20 ms control boundary; air time, swing height, both-airborne and foot rewards therefore differ. Critic foot clearance uses flat-ground height instead of the old raycast, and the exact reset/command curriculum, head/body resampling phase, IMU orientation randomization and some legacy reward details have not been numerically matched. The original scene uses `implicitfast`; the factory sets public `refresh_pre_step_body_state=False` to avoid mjbatch's Euler-only sensor-copyout path while the GF action reads only fresh joint state. Any claim that the 1975 checkpoint has the same gait quality needs a controlled rollout comparison.

Run focused validation from this directory:

```bash
uv run pytest -q tests
```

The tests cover frozen GF motor traces, strict checkpoint model loading, observation field positions, 1 ms native torque transmission, and a two-environment Manager-Based smoke. Generated training logs are ignored by Git.
