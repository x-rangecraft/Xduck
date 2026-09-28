"""Physical positive/negative witnesses for the equivalent COM correction.

Driver HOME was read via robot.calibration(action=get) on 2026-09-23.
The negative witness is the mean measured angle after 3 s in experiment
20260923_195437 (result.tar SHA256 d0db1a747621f3a1b2b562bbf2ed0356
142a988dfd282558bf5630e20063b103). The user confirmed this pose falls backward.
These are fixed regression inputs, not runtime defaults or hardware commands.
"""

import mujoco
import numpy as np
import pytest

from unilab.envs.locomotion.microduck_enlarged_112.tasks import ASSET_ROOT

DRIVER_HOME = np.array([0, 0, 0.384, 0, -0.384, 0, 0, 0, 0, 0, 0, -0.384, 0, 0.384], float)
# The COM correction was defined in this historical pose. Keep its reference
# fixed when the task HOME changes; otherwise undoing it rotates the correction.
CALIBRATION_HOME = np.array(
    [
        0,
        0,
        0.59484481673125933,
        0.005,
        -0.58984848993636585,
        -0.25,
        -0.3490658503988659,
        0,
        0,
        0,
        0,
        -0.59484481673125933,
        -0.005,
        0.58984848993636607,
    ],
    float,
)
FINAL_MEASURED = np.array(
    [
        0.000014,
        -0.001931,
        0.582236,
        0.002023,
        -0.588503,
        -0.248244,
        -0.340570,
        0.000667,
        0,
        -0.000187,
        0.007488,
        -0.601533,
        0.005555,
        0.589743,
    ]
)
UPPER_BODIES = (
    "base_link",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "control-box",
    "Battery",
    "IMU",
    "power-strip",
)
FOOT_GEOMS = (
    "left_foot_collision",
    "right_foot_collision",
    "left_ankle_collision",
    "right_ankle_collision",
)


def load_model(*, original=False):
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
    if original:
        # Undo the documented translation for an A/B witness. No source writes.
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[7:] = CALIBRATION_HOME
        mujoco.mj_forward(model, data)
        for name in UPPER_BODIES:
            index = model.body(name).id
            model.body_ipos[index] += data.xmat[index].reshape(3, 3).T @ [0.035, 0, 0]
        mujoco.mj_setConst(model, data)
    return model


def run_pose(model, target, kp, kd, seconds=60.0):
    """Finite-torque PD, free base, original contacts; no state reset in rollout.

    This isolates mechanical stability. It is not the fitted motor/latency model;
    the retained acceptance run additionally exercises that production chain.
    """
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1:3] = [-kp, -kd]
    data = mujoco.MjData(model)
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = target
    data.ctrl[:] = target
    mujoco.mj_forward(model, data)
    bottom = np.inf
    for name in FOOT_GEOMS:
        geom = model.geom(name).id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        bottom = min(bottom, world[:, 2].min())
    data.qpos[2] = -bottom + 0.0001
    mujoco.mj_forward(model, data)
    initial_height = float(data.qpos[2])
    initial_com = data.subtree_com[model.body("base_link").id].copy()
    maximum_tilt = 0.0
    fall = None
    trace = []
    for step in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        # mj_step leaves derived positions at the beginning of the last step.
        # Refresh before measuring contacts, tilt and rendering recorded qpos.
        mujoco.mj_forward(model, data)
        rotation = data.xmat[model.body("base_link").id].reshape(3, 3)
        tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        pitch = float(np.rad2deg(np.arctan2(rotation[0, 2], rotation[2, 2])))
        maximum_tilt = max(maximum_tilt, tilt)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        if step % 10 == 0:
            trace.append(np.r_[data.time, data.qpos, tilt, pitch])
        if tilt >= 45 or data.qpos[2] < initial_height - 0.12:
            fall = {"time_s": float(data.time), "pitch_deg": pitch}
            break
    floor = model.geom("floor").id
    support = set()
    for contact_id, contact in enumerate(data.contact):
        if floor not in contact.geom:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        if force[0] > 0.01:
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            support.add(model.geom(other).name)
    return {
        "duration_s": float(data.time),
        "fall": fall,
        "max_tilt_deg": maximum_tilt,
        "initial_com_m": initial_com.tolist(),
        "support_geoms": sorted(support),
        "base_xy_displacement_m": float(np.linalg.norm(data.qpos[:2])),
        "max_final_joint_error_rad": float(np.max(np.abs(data.qpos[7:] - target))),
    }, np.asarray(trace)


@pytest.mark.parametrize("kp,kd", [(30, 2), (200, 4), (250, 5), (500, 5)])
def test_driver_initialization_stands_on_feet_for_60_seconds(kp, kd):
    result, _ = run_pose(load_model(), DRIVER_HOME, kp, kd)
    assert result["fall"] is None
    assert result["duration_s"] == pytest.approx(60)
    assert result["max_tilt_deg"] < 5
    assert result["base_xy_displacement_m"] < 0.03
    support = result["support_geoms"]
    assert support and set(support) <= set(FOOT_GEOMS)
    assert any(name.startswith("left_") for name in support)
    assert any(name.startswith("right_") for name in support)


@pytest.mark.parametrize("kp,kd", [(30, 2), (250, 5), (500, 5)])
def test_measured_failed_pose_falls_backward_only_after_com_correction(kp, kd):
    before, _ = run_pose(load_model(original=True), FINAL_MEASURED, kp, kd, 10)
    after, _ = run_pose(load_model(), FINAL_MEASURED, kp, kd, 10)
    assert before["fall"] is None
    assert before["max_tilt_deg"] < 5
    assert after["fall"] is not None
    assert after["fall"]["pitch_deg"] < -40
    assert after["fall"]["time_s"] < 3


def test_calibration_pose_com_shift_preserves_mass_and_centroidal_inertia():
    models = [load_model(original=True), load_model()]
    com = []
    for model in models:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[7:] = CALIBRATION_HOME
        mujoco.mj_forward(model, data)
        com.append(data.subtree_com[model.body("base_link").id].copy())
        assert model.body_mass.sum() == pytest.approx(8.8933)
    np.testing.assert_array_equal(models[0].body_inertia, models[1].body_inertia)
    np.testing.assert_allclose(np.subtract(com[1], com[0]), [-0.015022769950412, 0, 0], atol=1e-12)
