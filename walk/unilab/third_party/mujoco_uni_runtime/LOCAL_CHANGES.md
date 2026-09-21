# Local UniLab changes

This directory vendors `mujoco-uni-runtime` 0.3.1 and is selected through
`[tool.uv.sources]` in the root `pyproject.toml`.

Local extension:

- Adds `dof_frictionloss` to the native reset-randomization field registry.
- Supports full-field and indexed get/set plus per-env reset payloads.
- Keeps `needs_refresh=false`; MuJoCo reads `dof_frictionloss` directly during
  constraint/friction evaluation and no `mj_setConst` refresh is required.
- Adds runtime and MicroDuck integration tests proving per-env isolation,
  per-reset resampling and the configured 0.9..1.1 multiplier bounds.

If upstream gains the same field, remove this vendor source after verifying the
upstream field name, shape `(nv,)`, reset semantics and tests are equivalent.
