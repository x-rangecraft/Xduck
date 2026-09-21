# Walk: XDuck V1.1.2 + GF43X40-10

Selected simulation policy: `checkpoints/model_1975_strong_bounded.pt`. Its saved `run_config.json` includes the strong domain randomization settings, reward, policy architecture and `UnitSlopeBoundedGaussianDistribution`; use it with this checkpoint. `policy/xc_walking.onnx` embeds the observation normalizer and bounded deterministic output. The external action mapping is in `policy/xc_policy.py` (reference `GAIT_REFERENCE`, scale 0.12 rad, then clip to target limits). The policy uses the native V1.1.2 joint direction: do not invert it again.

## Layout and verification

- `unilab/` — frozen UniLab training code with the current PPO-only `uni_rl` integration, including the GF calibrated actuator and the velocity owner YAML.
- `Model/1.1.2/mjcf/` — exact V1.1.2 simulation scene and meshes. The task's `gait_reference_v112.xml` lives under `unilab/src/unilab/envs/locomotion/xduck_gf43x40/`.
- `checkpoints/` — PyTorch checkpoint and original saved run configuration. Its historical source paths are absolute and are rebased by `verify_checkpoint.py`; do not execute those historical paths as commands.
- `policy/` — ONNX export and the shared Walk/Stand policy adapter.
- `evidence/` — original validation report and manifest.

From `walk/unilab`:

```bash
uv sync --locked --extra mujoco
uv run python ../verify_checkpoint.py
uv run train --algo ppo --task xduck_gf43x40_velocity_flat --sim mujoco \
  algo.num_envs=8 algo.num_steps_per_env=8 algo.max_iterations=1 \
  algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1 \
  training.logger=tensorboard training.no_play=true
```

The short command is an installation smoke, not a continuation of checkpoint 1975. The frozen checkpoint was loaded and stepped with its saved PPO config in the packaged environment: actor 61D, action 14D. A fresh machine still needs its own `uv sync` and validation.

## Evidence and limits

The selected 1975 policy had no observed fall or joint-limit violation across 84 independent-seed, 30-second strong-DR rollouts with 352 push events. Forward-command mean speed varied from 0.102 to 0.425 m/s and some development cases barely moved or went backward. The accompanying videos with pushes disabled are not the strong-DR acceptance run. Full scope is in `evidence/REPORT.md`.
