"""Xduck XL330 velocity task on UniLab 1.3.1 Manager-Based runtime."""

from pathlib import Path

from unilab.base import registry
from unilab.envs import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
)

from microduck_rl_unilab.assets import ensure_microduck_assets

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _asset_path(path: str) -> str:
    source = Path(path)
    return str(source if source.is_absolute() else (_PROJECT_ROOT / source).resolve())


def make_microduck_velocity_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve bundled robot assets before constructing the generic runtime."""
    ensure_microduck_assets()
    assert cfg.scene is not None
    cfg.scene.model_file = _asset_path(cfg.scene.model_file)
    cfg.scene.fragment_files = [_asset_path(item) for item in cfg.scene.fragment_files]
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("MicroduckXl330OfficialVelocityFlat", ManagerBasedRlEnvCfg)
registry.register_env(
    "MicroduckXl330OfficialVelocityFlat",
    make_microduck_velocity_env,
    sim_backend="mujoco",
)

__all__ = ["make_microduck_velocity_env"]
