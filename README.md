# XDuck GF43X40-10 simulation

This `simulation` branch contains two separate, reproducible MuJoCo projects for the XDuck V1.1.2 robot:

- [walk](walk/) — the selected forward/velocity policy, checkpoint 1975.
- [stand](stand/) — the selected stand and walk-to-stop policy, checkpoint 599.

Each directory carries the frozen task/backend source used for that candidate, the V1.1.2 MJCF and meshes, GF actuator calibration, exact checkpoint and saved run configuration, an ONNX export, and a one-step verification script. The shared `xc_policy.py` maps the 61D observation and 14D raw action for the two exported policies. The two projects use different actor distributions and normalizers; do not exchange their `run_config.json` files.

This is the earlier UniLab/MuJoCo simulation runtime with the PPO entrypoint using `unilab-rl==1.3.0`. It is not the full Manager-Based / UniSim migration of UniLab 1.3.1. The former DM4310 Velocity/SitStand/StandUp task registrations, assets, logs, and checkpoints are not included.

Both candidates have finite simulation evidence only. The walk policy still has velocity-consistency failures; the stand policy's zero-fall result applies to its tested ±0.2 m/s per-axis push range. See each directory's `evidence/REPORT.md` before using a policy on hardware.
