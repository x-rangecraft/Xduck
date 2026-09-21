"""Short deterministic XL330 BAM/MuJoCo versus UniLab batch physics probe.

Run from the UniLab project with ``uv run scripts/audit_microduck_xl330_step_parity.py``.
This is a diagnostic, not a training result. It uses the sibling MJLab's BAM
pure-Python package as the reference and disables randomized command latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from unilab.base import registry


def _load_bam(workspace: Path):
    # Append, not prepend: the UniLab Python 3.13 environment must retain its
    # own binary NumPy/MuJoCo wheels. BAM itself is pure Python.
    import numpy  # noqa: F401

    site_packages = next((workspace / "mjlab/.venv/lib").glob("python*/site-packages"))
    sys.path.append(str(site_packages))
    from bam.model import load_model
    from bam.mujoco import MujocoController

    return load_model, MujocoController


def probe(steps: int) -> dict[str, float | int]:
    workspace = Path(__file__).resolve().parents[2]
    load_model, controller_cls = _load_bam(workspace)
    registry.ensure_registries()
    env = registry.make("MicroDuckXl330OfficialVelocityFlat", "mujoco", num_envs=1)
    try:
        env.init_state()
        backend = env._backend
        # Replace the stochastic reset state with one reproducible STAND pose.
        qpos = backend.get_keyframe_qpos("STAND").astype(np.float64)
        qpos[2] = 0.125
        qvel = np.zeros(backend.model.nv, dtype=np.float64)
        backend.set_state(np.asarray([0], dtype=np.int32), qpos[None], qvel[None])
        source_model = backend.get_playback_model(0)
        with tempfile.TemporaryDirectory(prefix="xl330-parity-") as tmp:
            model_path = Path(tmp) / "reference.mjb"
            mujoco.mj_saveModel(source_model, str(model_path))
            model = mujoco.MjModel.from_binary_path(str(model_path))
        data = mujoco.MjData(model)

        # The official BAM controller drives unit-gain motor actuators. UniLab
        # feeds q+tau into unit-gain position actuators to apply the same tau.
        model.actuator_gainprm[:] = 0.0
        model.actuator_gainprm[:, 0] = 1.0
        model.actuator_biasprm[:] = 0.0
        model.actuator_biastype[:] = int(mujoco.mjtBias.mjBIAS_NONE)
        full_state = backend.get_physics_state()[0].astype(np.float64)

        bam = load_model(motor_name="xl330", model="m6")
        bam.actuator.kp = 200.0
        bam.actuator.vin = 7.35
        names = [model.actuator(i).name for i in range(model.nu)]
        reference = controller_cls(
            bam, names, model, data, vin_drop_gain=0.0, vin_min=6.0
        )
        # MujocoController.__init__ calls mj_setConst, which resets data state.
        # Inject the shared state only after controller construction.
        mujoco.mj_setState(model, data, full_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        reference.reset(data.qpos)
        initial_qpos_error = float(
            np.max(np.abs(full_state[1 : 1 + model.nq] - data.qpos))
        )

        motor = env._motor
        motor.vin[:] = 7.35
        motor.vin_drop_gain[:] = 0.0
        motor.friction_scale[:] = 1.0
        motor.delayed_target = lambda target: target
        backend._dof_force_components[0, 0] = data.qfrc_bias
        backend._dof_force_components[0, 1] = data.qfrc_constraint
        backend._dof_force_components[0, 2] = data.qfrc_actuator
        backend._dof_force_components[0, 3] = 0.0

        max_qpos = max_qvel = max_torque = 0.0
        contact_mismatches = 0
        floor_id = model.geom("floor").id
        foot_ids = [model.geom("left_foot_collision").id, model.geom("right_foot_collision").id]
        first_step = {}
        for step in range(steps):
            phase = 0.05 * np.sin(0.5 * step + np.arange(model.nu))
            target = env.default_angles + phase
            reference.q_target = target.copy()
            reference.update()
            expected_torque = data.ctrl.copy()
            mujoco.mj_step(model, data)

            backend.step(target[None, :], nsteps=1)
            native_state = backend.get_physics_state()[0]
            qpos = native_state[1 : 1 + model.nq]
            qvel = native_state[1 + model.nq : 1 + model.nq + model.nv]
            max_qpos = max(max_qpos, float(np.max(np.abs(qpos - data.qpos))))
            max_qvel = max(max_qvel, float(np.max(np.abs(qvel - data.qvel))))
            max_torque = max(
                max_torque, float(np.max(np.abs(motor.torque[0] - expected_torque)))
            )
            contacts = {
                frozenset((int(c.geom1), int(c.geom2)))
                for c in data.contact[: data.ncon]
            }
            expected_contact = np.asarray(
                [frozenset((floor_id, foot_id)) in contacts for foot_id in foot_ids]
            )
            actual_contact = np.asarray(
                [
                    bool(backend.get_sensor_data(name)[0, 0])
                    for name in ("left_foot_found", "right_foot_found")
                ]
            )
            contact_mismatches += int(np.count_nonzero(expected_contact != actual_contact))
            if step == 0:
                first_step = {
                    "qpos_abs_error": float(np.max(np.abs(qpos - data.qpos))),
                    "qvel_abs_error": float(np.max(np.abs(qvel - data.qvel))),
                    "torque_abs_error_nm": float(
                        np.max(np.abs(motor.torque[0] - expected_torque))
                    ),
                    "reference_torque_first_joint": float(expected_torque[0]),
                    "unilab_torque_first_joint": float(motor.torque[0, 0]),
                }
        return {
            "physics_substeps": steps,
            "initial_qpos_abs_error": initial_qpos_error,
            "first_step": first_step,
            "max_qpos_abs_error": max_qpos,
            "max_qvel_abs_error": max_qvel,
            "max_torque_abs_error_nm": max_torque,
            "foot_contact_mismatches": contact_mismatches,
        }
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(probe(args.steps), indent=2))
