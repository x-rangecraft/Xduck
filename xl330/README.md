# XL330 official velocity on UniLab 1.3.1

This directory is a standalone task package for the remaining XL330 walking project. It registers `MicroduckXl330OfficialVelocityFlat` through UniLab's external task registry, and uses the UniLab 1.3.1 Manager-Based runtime, `unisim-core` MuJoCo backend, and `uni_rl` PPO runner. It preserves the 14-action, 61-policy-observation and 76-critic-observation contract and uses an XL330 BAM voltage servo `ActionTerm` with physics-substep feedback. Task configuration is in `src/microduck_rl_unilab/conf/ppo/task/microduck_xl330_velocity_flat/`.

From this directory:

```bash
uv sync
uv run xl330-train --algo ppo --task microduck_xl330_velocity_flat --sim mujoco training.no_play=true
```

Small smoke:

```bash
uv run xl330-train --algo ppo --task microduck_xl330_velocity_flat --sim mujoco algo.num_envs=2 algo.num_steps_per_env=2 algo.max_iterations=1 training.no_play=true
```

Evaluation requires a new compatible checkpoint:

```bash
uv run xl330-eval --algo ppo --task microduck_xl330_velocity_flat --sim mujoco --load-run -1
```

The MuJoCo scene changes `implicitfast` to `Euler`: the current `mjbatch` execution path requires Euler. Solver, cone, gravity and 200 Hz physics / 50 Hz control settings remain declared. This physics change and the newer `unisim` backend can change trajectories; retrain and compare before deployment. Legacy XL330 checkpoint weights may have matching 61D/76D/14D tensor shapes, but older runner metadata, normalization, actuator dynamics, and simulator integration have not been certified compatible. Do not resume an old checkpoint directly without an explicit conversion and side-by-side rollout check.

The BAM action is adapted from the Apache-2.0 `microduck_rl_unilab` implementation. Its torque-domain friction and finite-difference external-torque estimate are approximations where the public `SimBackend` cannot expose the old MuJoCo internals. Meshes and XML are bundled for a self-contained build. Third-party robot asset license is in `assets/robots/microduck/LICENSE.pollen-robotics.txt`.

## Xduck XL330 parity audit

The migration was checked against `Xduck/unilab/src/unilab/envs/locomotion/microduck_xl330_official/velocity.py` and its inherited DM4310 task, rather than only the upstream MicroDuck example.

| Contract | Current status |
|---|---|
| Actor 61D and action 14D | Same ordered channels and raw policy action; BAM position target keeps encoder bias and 3–6 physics-step delay. |
| Critic 76D | Restored Xduck order: base linear velocity first; twist, feet, head and body commands in their original positions. Contact force channels are raw 3D `netforce` values. |
| Rewards | All 16 initial weights match Xduck's `OfficialRewardConfig`. Angular momentum now retains the old `0.25` reference before applying its `-0.02` weight. Reward sum is multiplied by the 0.02 s control step. |
| Commands | Velocity ranges, 3–8 s resampling, 20% forward bucket, 15% turn bucket and 2% initial standing bucket match. Turn takes precedence over standing. |
| Reset | Same 14-joint STAND pose under the `home` keyframe, absolute root height 0.12–0.13 m, XY ±0.5 m, yaw ±π, zero root velocity. |
| Domain randomization | IMU/encoder noise, CoM range, foot friction, armature, voltage, BAM friction and pushes are configured. Mass/inertia factor is sampled once and reapplied across resets by UniLab's event term. |

**Remaining physical differences:**

- Xduck's old task used `scene_walk.xml` with position actuators and a private per-substep callback that read MuJoCo force components and wrote dynamic friction into solver fields. The new scene uses torque `<motor>` actuators and a public `ActionTerm`. The BAM electrical constants and firmware law match, but dynamic friction uses the documented torque-domain approximation and external load uses finite differences. Exact torque and trajectory parity is unproven.
- The old critic foot height and foot rewards sampled foot sites with a terrain-clearance query and tracked contact/peak height every **5 ms physics step**. The Manager-Based common terms use foot-body world height and update contact/air-time/peak history every **20 ms control step**. The public UniSim interface does not expose the same site-clearance query; these terms may diverge during swing and landing.
- The old foot-slip term used `framelinvel` at each foot site. The common term uses foot-body link velocity. The old self-collision signal was a subtree `found` sensor; this port reinstates that sensor for the reward.
- The required MuJoCo `mjbatch` execution path uses **Euler** instead of the old `implicitfast` integration. This changes contact and motor transients even with otherwise matching parameters.
- No old XL330 checkpoint was resumed or accepted as deployment-equivalent. Shape equality alone does not prove policy or normalizer compatibility. Train a fresh policy, then run side-by-side rollouts before deployment.

Validation with this project's locked company UniLab 1.3.1 / UniSim 1.7.3 environment: 22 tests passed, including a 2-env MuJoCo reset/step and command-precedence test; Ruff passed; one PPO update completed and wrote a new `model_0.pt`. Long-run convergence and sim-to-real behavior remain unverified.
