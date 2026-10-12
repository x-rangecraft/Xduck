"""Cold-path motion checks for publishing a coordinated reference."""

import hashlib
import json

import numpy as np


def motion_quality(data, direction):
    """Measure actual nominal motion, excluding initial contact settling."""
    trace = np.asarray(data["trace"], dtype=float)
    hold = round(data["durations"][0] / data["sample_dt"])
    trace = trace[hold:]
    sign = 1 if direction == "rise" else -1
    arrival = np.flatnonzero(abs(trace[:, 14] - trace[-1, 14]) < 0.005)[0]
    knees = trace[:, [3, 12]]
    return {
        "reverse_height_m": float(np.maximum(-sign * np.diff(trace[:, 14]), 0).sum()),
        "neck_at_height_arrival_deg": float(np.degrees(abs(trace[arrival, 5]))),
        "knee_extra_travel_deg": float(
            np.degrees((abs(np.diff(knees, axis=0)).sum(axis=0) - abs(knees[-1] - knees[0])).max())
        ),
    }


def require_coordinated_motion(data, direction):
    """Reject endpoint-only success with a second adjustment or height rebound."""
    quality = motion_quality(data, direction)
    if (
        quality["reverse_height_m"] > 0.005
        or quality["neck_at_height_arrival_deg"] > 10
        or quality["knee_extra_travel_deg"] > 1
    ):
        raise ValueError(f"{direction} reference still has secondary motion: {quality}")
    return quality


# User accepted this exact candidate on 2026-09-29 after viewing its side video.
# This exception concerns motion appearance only; model/contact/gain checks remain.
# Provenance and remaining rebound: docs/audits/xduck_sitstand_reference_v4_20260929.md
USER_ACCEPTED_REFERENCE_SHA256 = "05f0423eecdf7bd268e399f62fdce84d5d57b7ff62b45b65a3e7f924742dd116"


def require_reference_motion(data):
    """Allow the exact user-selected motion, or require strict motion quality."""
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest == USER_ACCEPTED_REFERENCE_SHA256:
        return {key: motion_quality(data[key], key) for key in ("rise", "descent")}
    return {key: require_coordinated_motion(data[key], key) for key in ("rise", "descent")}
