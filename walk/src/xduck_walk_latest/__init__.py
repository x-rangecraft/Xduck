"""XDuck V1.1.2 walking task integration for Manager-Based UniLab."""

from .actions import GF43X40ActionCfg
from .assets import scene_asset_path

__all__ = ["GF43X40ActionCfg", "scene_asset_path"]

__unilab_registry_modules__ = ("xduck_walk_latest.registry",)
