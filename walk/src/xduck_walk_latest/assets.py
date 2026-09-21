"""Cold-path task scene lookup; the adjacent Model directory owns meshes."""

from pathlib import Path


def scene_asset_path() -> str:
    path = Path(__file__).resolve().parents[2] / "assets" / "scene.xml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def sensor_fragment_path() -> str:
    return str(Path(scene_asset_path()).with_name("sensors.xml"))


def gait_fragment_path() -> str:
    return str(Path(scene_asset_path()).with_name("gait_reference_v112.xml"))
