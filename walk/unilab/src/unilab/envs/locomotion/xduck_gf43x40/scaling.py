"""Unit conversions for the optional XDuck scale-adapted task profile."""

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def xml_model_fingerprint(path):
    """Fingerprint the source XML and includes, only on the cold path."""
    root = Path(path).resolve()
    pending = [root]
    sources = {}
    while pending:
        item = pending.pop()
        if item in sources:
            continue
        data = item.read_bytes()
        sources[item] = data
        for node in ET.fromstring(data).iter("include"):
            pending.append((item.parent / node.attrib["file"]).resolve())
    digest = hashlib.sha256()
    for item in sorted(sources, key=lambda p: os.path.relpath(p, root.parent)):
        digest.update(os.path.relpath(item, root.parent).encode() + b"\0")
        digest.update(sources[item] + b"\0")
    return digest.hexdigest()


def normalized_target_rate_l2(current, previous, *, ctrl_dt, reference_dt, time_scale):
    """Squared target-angle change over one reference-time control interval.

    Targets are radians after clipping and action latency selection. Policy
    actions/history remain unchanged. Slowing a trajectory by time_scale
    preserves this cost (up to finite-difference discretization).
    """
    for name, value in (
        ("ctrl_dt", ctrl_dt),
        ("reference_dt", reference_dt),
        ("time_scale", time_scale),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    delta = (current - previous) * (time_scale * reference_dt / ctrl_dt)
    return np.sum(np.square(delta), axis=-1)


def matched_momentum_weight(reference_energy, robot_energy, reference_weight=-0.02):
    """Match expected squared momentum in a declared matched-motion ensemble.

    Energies must be calculated from the actual compiled mass distributions.
    This calibrates a scalar reward, not the robot dynamics or every posture.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (reference_energy, robot_energy)):
        raise ValueError("momentum energies must be positive and finite")
    if not math.isfinite(reference_weight) or reference_weight >= 0:
        raise ValueError("reference momentum weight must be negative and finite")
    return reference_weight * reference_energy / robot_energy
