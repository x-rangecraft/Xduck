"""Regenerate explicit all-pairs robot self-contact sensors for posture tasks."""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH


def main() -> None:
    robot_path = ASSETS_ROOT_PATH / "robots/microduck_dm4310/microduck_dm4310_allcollisions.xml"
    scene_path = ASSETS_ROOT_PATH / "robots/microduck_dm4310/scene_posture.xml"
    robot_root = ET.parse(robot_path).getroot()
    collision_names = [
        geom.attrib["name"]
        for geom in robot_root.findall(".//geom")
        if geom.attrib.get("class") in {"collision", "self_collision_only"}
        and geom.attrib.get("name")
    ]
    required = {"trunk_self_collision", "left_foot_collision", "right_foot_collision"}
    if missing := required - set(collision_names):
        raise ValueError(f"missing required robot collision geoms: {sorted(missing)}")
    head_collision_names = [
        geom.attrib["name"]
        for geom in robot_root.findall(".//geom")
        if geom.attrib.get("class") in {"collision", "self_collision_only"}
        and geom.attrib.get("mesh") in {"top_head_shell", "jaw", "bottom_head_shell"}
    ]
    if not head_collision_names:
        raise ValueError("expected at least one named head collision geom")

    tree = ET.parse(scene_path)
    root = tree.getroot()
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for node in list(sensor):
        name = node.attrib.get("name", "")
        if name.startswith(("head_ground_force_", "head_ground_found_", "self_pair_")):
            sensor.remove(node)
    for collision_name in head_collision_names:
        suffix = collision_name.removeprefix("robot_collision_")
        for data in ("force", "found"):
            ET.SubElement(
                sensor,
                "contact",
                name=f"head_ground_{data}_{suffix}",
                geom1="floor",
                geom2=collision_name,
                data=data,
                num="1",
                reduce="netforce" if data == "force" else "none",
            )
    for index, (left, right) in enumerate(itertools.combinations(collision_names, 2)):
        ET.SubElement(
            sensor,
            "contact",
            name=f"self_pair_{index:03d}",
            geom1=left,
            geom2=right,
            data="found",
            num="1",
            reduce="none",
        )
    ET.indent(tree, space="    ")
    tree.write(scene_path, encoding="unicode")
    pair_count = len(collision_names) * (len(collision_names) - 1) // 2
    print(f"wrote {scene_path} with {pair_count} self-contact pair sensors")


if __name__ == "__main__":
    main()
