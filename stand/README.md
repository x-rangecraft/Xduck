# Stand: XDuck V1.1.2 + GF43X40-10

Selected simulation policy: `checkpoints/model_599_stand_push02.pt`. Its `run_config.json` records the final stand/stop configuration, including the ±0.2 m/s per-axis push curriculum. This actor uses an ordinary Gaussian deterministic mean and its own observation normalizer. `policy/xc_standing.onnx` embeds that normalizer. The external reference, joint order and 0.12 rad action scale are in `policy/xc_policy.py`.

## Layout and verification

- `unilab/` — frozen UniLab training code with the current PPO-only `uni_rl` integration, GF actuator calibration and walk-to-stop owner YAML. Its `data/xduck_stop/mixed70stand30walk.npz` is the saved handoff bank for this final run.
- `Model/1.1.2/mjcf/` — exact V1.1.2 simulation scene and meshes.
- `checkpoints/` — stand checkpoint and original run configuration. `verify_checkpoint.py` rebases historical absolute paths to this folder before loading the environment.
- `policy/` — derived ONNX and shared policy adapter.
- `evidence/` — original validation report, manifest, and higher-push stress test.

From `stand/unilab`:

```bash
uv sync --locked --extra mujoco
uv run python ../verify_checkpoint.py
uv run train --algo ppo --task xduck_gf43x40_walk_stop_flat --sim mujoco \
  env.handoff_bank=data/xduck_stop/mixed70stand30walk.npz \
  algo.num_envs=8 algo.num_steps_per_env=8 algo.max_iterations=1 \
  algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1 \
  training.logger=tensorboard training.no_play=true
```

The short command is a fresh training smoke, not a continuation of checkpoint 599. The frozen checkpoint was loaded and stepped with its saved PPO config in the packaged environment: actor 61D, action 14D. For a walk-to-stand transition test, use the adjacent `../walk/checkpoints/model_1975_strong_bounded.pt` as the walk policy and preserve the simulation, motor, communication and action history across the switch. The walk actor has a different bounded output and normalizer.

## Evidence and limits

At ±0.2 m/s pushes per XY axis, 96 tested 60-second trajectories showed zero falls; push cases finished strictly settled in 20/24 trials per group. Stronger ±0.3 and ±0.4 stress tests produced failures, so the verified zero-fall range remains ±0.2. Some original walk eligibility checks measured motion magnitude rather than net forward motion; the subsequent stress report applies a signed-forward filter. See `evidence/REPORT.md` and `evidence/STRESS_PUSH.md`.
