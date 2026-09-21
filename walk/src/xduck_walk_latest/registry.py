"""UniLab third-party task registry hook.

The Hydra owner YAML remains a separate integration requirement for `train`.
Direct `make_walk_env()` is available for smoke and controlled evaluation.
"""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .task import make_walk_env

registry.register_env_config("XDuckGF43X40VelocityFlatLatest", ManagerBasedRlEnvCfg)
registry.register_env("XDuckGF43X40VelocityFlatLatest", make_walk_env, sim_backend="mujoco")
