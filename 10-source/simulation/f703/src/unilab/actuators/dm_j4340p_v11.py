"""24 V DM-J4340P-2EC V1.1 output-shaft limits.

The vendor manual publishes 12/40 N m rated/peak torque and 36/56 rpm
rated/no-load speed.  It does not publish a controller-ready four-quadrant
map, so the simulation uses a conservative linear motoring envelope from
40 N m at stall to zero at 56 rpm.  Braking torque remains available up to
the peak value so an overspeed joint can decelerate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DMJ4340PV11Spec:
    voltage_v: float = 24.0
    rated_torque_nm: float = 12.0
    peak_torque_nm: float = 40.0
    rated_speed_rpm: float = 36.0
    no_load_speed_rpm: float = 56.0
    gear_ratio: float = 40.0
    mass_kg: float = 0.362
    diameter_m: float = 0.057
    length_m: float = 0.0565


DM_J4340P_V11_24V = DMJ4340PV11Spec()


def clip_dm_j4340p_v11_torque(
    requested_torque_nm: np.ndarray,
    output_speed_rad_s: np.ndarray,
    spec: DMJ4340PV11Spec = DM_J4340P_V11_24V,
) -> np.ndarray:
    """Clip requested output torque to the conservative 24 V envelope."""

    requested = np.asarray(requested_torque_nm)
    speed = np.asarray(output_speed_rad_s)
    if requested.shape != speed.shape:
        raise ValueError("requested torque and output speed must share shape")
    speed_rpm = np.abs(speed) * 60.0 / (2.0 * np.pi)
    motoring_limit = spec.peak_torque_nm * np.clip(
        1.0 - speed_rpm / spec.no_load_speed_rpm,
        0.0,
        1.0,
    )
    is_motoring = requested * speed >= 0.0
    magnitude_limit = np.where(is_motoring, motoring_limit, spec.peak_torque_nm)
    return np.clip(requested, -magnitude_limit, magnitude_limit)
