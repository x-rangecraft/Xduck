"""Import a duck MJCF release while preserving UniLab-owned contracts."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH

ROBOT_DIR = ASSETS_ROOT_PATH / "robots" / "microduck_dm4310"
ROBOT_FILES = ("microduck_dm4310.xml", "microduck_dm4310_allcollisions.xml")
SCENE_FILES = ("scene_flat.xml", "scene_posture.xml")
KEY_NAME_MAP = {"init": "INIT", "home": "STAND", "SIT": "SIT", "FOLD": "FOLD"}


def _parser() -> ET.XMLParser:
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def _parse(path: Path) -> ET.ElementTree:
    return ET.parse(path, parser=_parser())


def _find_default(root: ET.Element, class_name: str) -> ET.Element:
    node = root.find(f".//default[@class='{class_name}']")
    if node is None:
        raise ValueError(f"missing default class {class_name!r}")
    return node


def _replace_node(parent: ET.Element, old: ET.Element, new: ET.Element) -> None:
    parent.insert(list(parent).index(old), new)
    parent.remove(old)


def _replace_unilab_contracts(source_root: ET.Element, current_root: ET.Element) -> None:
    source_sensor = source_root.find("sensor")
    current_sensor = current_root.find("sensor")
    if source_sensor is None or current_sensor is None:
        raise ValueError("robot XML must contain a sensor block")
    _replace_node(source_root, source_sensor, copy.deepcopy(current_sensor))

    source_dm = source_root.find(".//default[@class='dm4310']")
    if source_dm is not None:
        current_dm = _find_default(current_root, "dm4310")
        for parent in source_root.findall(".//default"):
            if source_dm in list(parent):
                _replace_node(parent, source_dm, copy.deepcopy(current_dm))
                break
        else:
            raise ValueError("dm4310 default has no parent")

    # V1.0.5+ names the all-4340 release class ``gf4340``. Normalize that source
    # spelling, then preserve UniLab's calibrated all-DM4340 controller default.
    source_dm4340 = source_root.find(".//default[@class='dm4340']")
    if source_dm4340 is None:
        source_dm4340 = source_root.find(".//default[@class='gf4340']")
    if source_dm4340 is None:
        raise ValueError("robot XML must contain a dm4340 or gf4340 default")
    current_dm4340 = _find_default(current_root, "dm4340")
    for parent in source_root.findall(".//default"):
        if source_dm4340 in list(parent):
            _replace_node(parent, source_dm4340, copy.deepcopy(current_dm4340))
            break
    else:
        raise ValueError("dm4340/gf4340 default has no parent")
    for joint in source_root.findall(".//joint"):
        if joint.attrib.get("class") in {"dm4310", "dm4340", "gf4340"}:
            joint.set("class", "dm4340")
    actuator = source_root.find("actuator")
    if actuator is not None:
        for node in actuator:
            if node.attrib.get("class") in {"dm4310", "dm4340", "gf4340"}:
                node.set("class", "dm4340")
                node.tag = "motor"
                for attr in ("kp", "kv", "dampratio", "inheritrange"):
                    node.attrib.pop(attr, None)


def _collision_geoms(root: ET.Element) -> list[tuple[tuple[str, str, int], ET.Element]]:
    """Return collision geoms keyed by body, mesh, and local occurrence.

    Release V1.1.0 removes the jaw collision, so positional naming would shift
    every collision after it.  The structural key preserves stable UniLab names
    for surviving geoms while still allowing a release to add or remove parts.
    """
    collisions: list[tuple[tuple[str, str, int], ET.Element]] = []
    for body in root.findall(".//body"):
        occurrences: dict[str, int] = {}
        for geom in body.findall("geom"):
            if geom.attrib.get("class") not in {"collision", "self_collision_only"}:
                continue
            mesh = geom.attrib.get("mesh", "")
            occurrence = occurrences.get(mesh, 0)
            occurrences[mesh] = occurrence + 1
            collisions.append(((body.attrib.get("name", ""), mesh, occurrence), geom))
    return collisions


def _build_allcollisions(source_root: ET.Element, current_root: ET.Element) -> ET.Element:
    root = copy.deepcopy(source_root)
    _replace_unilab_contracts(root, current_root)
    current_names = {
        key: geom.attrib["name"]
        for key, geom in _collision_geoms(current_root)
        if "name" in geom.attrib
    }
    used_names = set(current_names.values())
    next_index = 0
    for key, geom in _collision_geoms(root):
        name = current_names.get(key) or geom.attrib.get("name")
        if name is None:
            while f"robot_collision_{next_index}" in used_names:
                next_index += 1
            name = f"robot_collision_{next_index}"
            next_index += 1
        geom.set("name", name)
        used_names.add(name)

    required = {"trunk_self_collision", "left_foot_collision", "right_foot_collision"}
    actual = {geom.attrib["name"] for _, geom in _collision_geoms(root)}
    if missing := required - actual:
        raise ValueError(f"release removed required collision geoms: {sorted(missing)}")
    return root


def _build_training_robot(allcollisions_root: ET.Element) -> ET.Element:
    root = copy.deepcopy(allcollisions_root)
    leg_index = 0
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.attrib.get("class") not in {"collision", "self_collision_only"}:
                continue
            name = geom.attrib.get("name")
            mesh = geom.attrib.get("mesh")
            if name == "trunk_self_collision" or name in {
                "left_foot_collision",
                "right_foot_collision",
            }:
                continue
            if mesh == "leg":
                geom.set("class", "self_collision_only")
                geom.set(
                    "name",
                    "left_leg_self_collision" if leg_index == 0 else "right_leg_self_collision",
                )
                leg_index += 1
            else:
                body.remove(geom)
    if leg_index != 2:
        raise ValueError(f"expected two leg collision geoms, got {leg_index}")
    return root


def _write_robot(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def _source_release(source_dir: Path) -> str:
    match = re.search(r"v(\d+\.\d+\.\d+)", source_dir.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot determine duck release from directory name: {source_dir.name}")
    return f"V{match.group(1)}"


def _replace_scene_keys(scene_path: Path, source_scene: ET.Element, source_release: str) -> None:
    tree = _parse(scene_path)
    root = tree.getroot()
    source_keys = {node.attrib["name"]: node for node in source_scene.findall("./keyframe/key")}
    for target in root.findall("./keyframe/key"):
        source_name = KEY_NAME_MAP[target.attrib["name"]]
        source = source_keys[source_name]
        target.set("qpos", source.attrib["qpos"])
        target.set("ctrl", " ".join("0" for _ in source.attrib["ctrl"].split()))

    source_description = None
    for node in source_scene:
        if node.tag is ET.Comment and node.text and "uniform" in node.text:
            source_description = re.sub(
                r"\s+v\d+\.\d+\.\d+\s*$", "", node.text.strip(), flags=re.IGNORECASE
            )
            break
    if source_description is None:
        source_description = "uniform s=2.0900 yaw+z=3.2mm"
    marker = f"{source_description}; source duck {source_release}"
    comments = [node for node in root if node.tag is ET.Comment]
    if comments:
        comments[0].text = f" {marker} "
    else:
        root.insert(0, ET.Comment(f" {marker} "))
    ET.indent(root, space="    ")
    tree.write(scene_path, encoding="unicode")


def import_model(source_dir: Path) -> None:
    source_robot_path = source_dir / "robot_allcollisions.xml"
    source_scene_path = source_dir / "scene.xml"
    if not source_robot_path.is_file() or not source_scene_path.is_file():
        raise FileNotFoundError(f"{source_dir} is not a duck MJCF model directory")

    current_all = _parse(ROBOT_DIR / "microduck_dm4310_allcollisions.xml").getroot()
    source_root = _parse(source_robot_path).getroot()
    allcollisions = _build_allcollisions(source_root, current_all)
    training = _build_training_robot(allcollisions)
    _write_robot(ROBOT_DIR / ROBOT_FILES[0], training)
    _write_robot(ROBOT_DIR / ROBOT_FILES[1], allcollisions)

    source_scene = _parse(source_scene_path).getroot()
    source_release = _source_release(source_dir)
    for scene_name in SCENE_FILES:
        _replace_scene_keys(ROBOT_DIR / scene_name, source_scene, source_release)

    source_assets = source_dir / "assets"
    for asset in source_assets.iterdir():
        if asset.is_file():
            shutil.copy2(asset, ROBOT_DIR / "assets" / asset.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dir", type=Path, help="directory containing a duck release scene/XML/assets"
    )
    args = parser.parse_args()
    import_model(args.source_dir.resolve())
    print(f"imported MicroDuck model from {args.source_dir}")


if __name__ == "__main__":
    main()
