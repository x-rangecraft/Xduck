"""Prevent V112 training profiles or accepted mechanics from silently drifting."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.microduck_enlarged_112.tasks import (
    ASSET_ROOT,
    MicroDuckEnlarged112StandFlatCfg,
    MicroDuckEnlarged112WalkFlatCfg,
)

ROOT = Path(__file__).resolve().parents[4]
ACCEPTED_ROBOT_SHA256 = "88e7801536b1fcdd3853d24f7bec81ad09a93d0612ae5b6a4a617ba01467cdec"
ACCEPTED_SCENE_SHA256 = "49b6470c23351d39aa0234f4094abc8452c7f262944ca9628357a6136f0c0dac"


def test_accepted_model_and_scene_fingerprints():
    # Intentional release pins: update only after explicit mechanical acceptance.
    robot = ASSET_ROOT / "microduck_enlarged_112.xml"
    scene = ASSET_ROOT / "scene_flat.xml"
    assert hashlib.sha256(robot.read_bytes()).hexdigest() == ACCEPTED_ROBOT_SHA256
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == ACCEPTED_SCENE_SHA256
    assert ET.parse(scene).getroot().find("include").get("file") == robot.name


def test_driver_home_reference_matches_standing_height():
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
    home = model.key("home")
    expected = [0, 0, 0.384, 0, -0.384, 0, 0, 0, 0, 0, 0, -0.384, 0, 0.384]
    np.testing.assert_allclose(home.qpos[7:], expected, atol=1e-12)
    np.testing.assert_allclose(home.ctrl, expected, atol=1e-12)
    np.testing.assert_array_equal(home.qvel, 0)
    np.testing.assert_allclose(home.qpos[3:7], [1, 0, 0, 0])
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose(
            config_name="config_mlx", overrides=["task=microduck_enlarged_112_stand_flat/mujoco"]
        )
    assert cfg.reward.base_height_target == pytest.approx(home.qpos[2], abs=1e-12)


@pytest.mark.parametrize("config_name", ["config", "config_mlx"])
def test_all_v112_training_profiles_resolve_to_accepted_scene(config_name):
    config_dir = ROOT / "conf" / "ppo"
    profiles = sorted(config_dir.glob("task/microduck_enlarged_112_*_flat/*.yaml"))
    assert profiles
    owners = {
        "MicroDuckEnlarged112StandFlat": MicroDuckEnlarged112StandFlatCfg,
        "MicroDuckEnlarged112WalkFlat": MicroDuckEnlarged112WalkFlatCfg,
    }
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for profile in profiles:
            task = profile.relative_to(config_dir / "task").with_suffix("").as_posix()
            cfg = compose(config_name=config_name, overrides=[f"task={task}"])
            assert cfg.training.task_name in owners, task
            assert cfg.training.sim_backend == "mujoco", task
            env_cfg = owners[cfg.training.task_name]()
            apply_cfg_overrides(env_cfg, OmegaConf.to_container(cfg.env, resolve=True))
            assert Path(env_cfg.scene.model_file).resolve() == ASSET_ROOT / "scene_flat.xml", task
            assert not env_cfg.scene.fragment_files, task
