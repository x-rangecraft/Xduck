"""Physical sit/stand acceptance without a prescribed joint configuration."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeometricAcceptance:
    height_error_m: float = .015
    body_tilt_deg: float = 10.
    head_tilt_deg: float = 10.
    base_speed_m_s: float = .04
    joint_speed_rad_s: float = .5
    minimum_hold_s: float = .5
    final_window_s: float = 1.


def upright_tilt(rotation, local_up):
    """Tilt of the physical up axis; yaw is unrestricted."""
    world_up = np.einsum("...ij,j->...i", rotation, local_up)
    return np.arccos(np.clip(world_up[..., 2], -1., 1.))


def geometric_success(height_error, body_tilt, head_tilt, base_speed, joint_velocity,
                      dt=.02, cfg=GeometricAcceptance()):
    conditions = dict(
        height=np.abs(height_error) < cfg.height_error_m,
        body_upright=body_tilt < np.deg2rad(cfg.body_tilt_deg),
        head_upright=head_tilt < np.deg2rad(cfg.head_tilt_deg),
        base_stationary=base_speed < cfg.base_speed_m_s,
        joints_stationary=np.max(np.abs(joint_velocity), axis=-1) < cfg.joint_speed_rad_s,
    )
    stable = np.logical_and.reduce(list(conditions.values()))
    count = np.zeros(stable.shape[1], dtype=int)
    history = []
    for frame in stable:
        count = np.where(frame, count + 1, 0)
        history.append(count * dt >= cfg.minimum_hold_s)
    window = round(cfg.final_window_s / dt)
    fallen = ((body_tilt > np.deg2rad(75))).any(axis=0)
    success = np.asarray(history)[-window:].all(axis=0) & ~fallen
    return success, conditions


def quiet_curriculum_gate(baseline, candidate):
    """First tightening requires retained success and improved full motion."""
    import math
    checks = {}
    for case in ("descent", "rise"):
        old, new = baseline[case], candidate[case]
        checks[case + "_success_retained"] = new["successes"] >= math.ceil(.9 * old["successes"])
        checks[case + "_no_additional_falls"] = new["falls"] <= old["falls"]
        checks[case + "_body_tilt_not_worse"] = new["body_peak_deg"] <= old["body_peak_deg"]
    checks["descent_neck_excursion_improved"] = candidate["descent"]["neck_excursion_deg"] <= .85 * baseline["descent"]["neck_excursion_deg"]
    checks["rise_body_tilt_improved"] = candidate["rise"]["body_peak_deg"] <= .9 * baseline["rise"]["body_peak_deg"]
    return all(checks.values()), checks
