"""Experimental Xduck GF backend over the exact compiled Warp scene.

This module remains fail-closed until device motor and DR pathways are
validated for the task. GF/CAN remains on the host at 1 kHz; physics is CUDA.
It is not selected by the shipped CPU training configuration.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from unilab.base.scene import SceneCfg
from unilab.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_GEOM_FRICTION,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

from .backend import MjwarpBackend, _cuda_graph_eligibility, _suspend_gc


class XDuckGF43X40WarpBackend(MjwarpBackend):
    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        compiled_model_file: str,
        physics_microsteps: int,
        base_name: str,
        push_body_name: str | None = None,
    ):
        self._xduck_microsteps = int(physics_microsteps)
        fields = [
            "body_ipos",
            "body_mass",
            "body_inertia",
            "geom_friction",
            "dof_armature",
            "body_subtreemass",
            "body_invweight0",
            "dof_invweight0",
            "actuator_acc0",
        ]
        super().__init__(
            scene,
            num_envs,
            sim_dt,
            base_name=base_name,
            push_body_name=None,
            nconmax=256,
            njmax=512,
            compiled_model_file=compiled_model_file,
            model_batch_sizes={name: num_envs for name in fields},
            physics_microsteps=physics_microsteps,
        )
        self._xduck_push_body_id = (
            None if push_body_name is None else self.get_body_id(push_body_name)
        )
        self._xduck_motor_callback = None
        self._body_pose_names = [self._cpu_model.body(i).name for i in range(self._cpu_model.nbody)]

    def _initialize_cuda_graphs(self, device):
        # Mass/CoM/armature DR must update the corresponding derived constants
        # per world; sharing world zero's inverse weights corrupts contact DR.
        mean = self._device_model.stat.meaninertia
        self._device_model.stat.meaninertia = self._warp.array(
            np.repeat(mean.numpy()[:1], self._num_envs), dtype=mean.dtype, device=device
        )
        self._disable_cuda_graphs("GF uses a separate 1-ms mechanics graph")
        self._xduck_motor_graph = None
        self._xduck_reset_forward_graph = None
        self._xduck_reset_graph = None
        eligible, reason = _cuda_graph_eligibility(self._warp, device)
        if not eligible:
            self._xduck_graph_reason = reason
            return
        try:
            with _suspend_gc(), self._warp.ScopedDevice(device):
                with self._warp.ScopedCapture() as capture:
                    for _ in range(self._xduck_microsteps):
                        self._mujoco_warp.step(self._device_model, self._device_data)
                # Reset must reuse captured collision scratch just like step.
                # Eager forward alongside a live step graph duplicates large
                # convex collision buffers at episode boundaries.
                with self._warp.ScopedCapture() as forward_capture:
                    self._mujoco_warp.forward(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as reset_capture:
                    self._mujoco_warp.reset_data(
                        self._device_model, self._device_data, reset=self._reset_mask_device
                    )
            self._xduck_motor_graph = capture.graph
            self._xduck_reset_forward_graph = forward_capture.graph
            self._xduck_reset_graph = reset_capture.graph
            self._xduck_graph_reason = None
        except Exception as exc:
            self._xduck_graph_reason = f"{type(exc).__name__}: {exc}"

    def _execute_device_forward(self):
        if self._xduck_reset_forward_graph is None:
            return super()._execute_device_forward()
        self._warp.capture_launch(self._xduck_reset_forward_graph)

    def _execute_device_reset(self):
        if self._xduck_reset_graph is None:
            return super()._execute_device_reset()
        self._warp.capture_launch(self._xduck_reset_graph)

    def set_pre_step_control(self, fn):
        # Registration is separate from execution; step() refuses to advance
        # until the original calibrated host GF controller has been installed.
        self._xduck_motor_callback = fn

    def step(self, ctrl: np.ndarray, nsteps: int = 1):
        if self._xduck_motor_callback is None:
            raise RuntimeError("GF GPU motor callback must be registered before stepping")
        command = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self._nu)
        if command.shape != expected:
            raise ValueError(f"GF GPU control expected {expected}, got {command.shape}")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError("nsteps must be positive")
        start = time.monotonic()
        phases = {
            "motor_host_ms": 0.0,
            "control_upload_ms": 0.0,
            "warp_enqueue_ms": 0.0,
            "cuda_sync_ms": 0.0,
            "state_download_ms": 0.0,
        }
        for _ in range(int(nsteps)):
            # Host transport/motor remains bit-for-bit the calibrated GF owner.
            # The device advances the configured mechanics microsteps.
            applied = np.asarray(self._xduck_motor_callback(self, command), dtype=np.float32)
            now = time.monotonic()
            phases["motor_host_ms"] += (now - start) * 1000.0
            start = now
            if applied.shape != expected or not np.isfinite(applied).all():
                raise ValueError("GF GPU motor returned nonfinite or misshaped control")
            self._device_data.ctrl.assign(applied)
            now = time.monotonic()
            phases["control_upload_ms"] += (now - start) * 1000.0
            start = now
            if self._xduck_motor_graph is None:
                for _ in range(self._xduck_microsteps):
                    self._mujoco_warp.step(self._device_model, self._device_data)
            else:
                self._warp.capture_launch(self._xduck_motor_graph)
            now = time.monotonic()
            phases["warp_enqueue_ms"] += (now - start) * 1000.0
            start = now
            self._synchronize()
            now = time.monotonic()
            phases["cuda_sync_ms"] += (now - start) * 1000.0
            start = now
            self._time_cache += self._sim_dt
            self._refresh_host_cache()
            now = time.monotonic()
            phases["state_download_ms"] += (now - start) * 1000.0
            start = now
        self._last_step_breakdown = phases
        return {"timing": phases}

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {
                    RESET_TERM_BODY_IPOS,
                    RESET_TERM_BODY_MASS,
                    RESET_TERM_BODY_INERTIA,
                    RESET_TERM_GEOM_FRICTION,
                    RESET_TERM_DOF_ARMATURE,
                }
            ),
            supports_interval_body_velocity_delta=True,
        )

    def get_body_inertia(self):
        return np.asarray(self._cpu_model.body_inertia, dtype=np.float32).copy()

    def get_dof_frictionloss(self):
        return np.asarray(self._cpu_model.dof_frictionloss, dtype=np.float32).copy()

    def _refresh_host_cache(self):
        super()._refresh_host_cache()
        # cvel is a Warp spatial vector: angular xyz then linear xyz.
        self._body_cvel_cache = np.asarray(self._device_data.cvel.numpy(), dtype=np.float32)
        self._site_xpos_cache = np.asarray(self._device_data.site_xpos.numpy(), dtype=np.float32)

    def get_body_ang_vel_w(self, body_ids: np.ndarray):
        ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        return self._body_cvel_cache[:, ids, :3]

    def sample_site_clearance(
        self, site_ids: np.ndarray, *, radius: float = 0.04, max_distance: float = 1.0
    ):
        # This exact frozen task has one group-0 geom: the z=0 floor plane.
        # The CPU owner rays downward at 3 offsets; all three hit that plane
        # at the same height, or return clipped site z when below it.
        if (
            self._cpu_model.ngeom == 0
            or int(self._cpu_model.geom_group[0]) != 0
            or any(int(self._cpu_model.geom_group[i]) == 0 for i in range(1, self._cpu_model.ngeom))
        ):
            raise NotImplementedError("GPU site clearance requires the audited flat floor only")
        ids = np.asarray(site_ids, dtype=np.int32).reshape(-1)
        if ids.size == 0 or np.any(ids < 0) or np.any(ids >= self._cpu_model.nsite):
            raise ValueError("site_ids must select existing sites")
        return np.clip(self._site_xpos_cache[:, ids, 2], 0.0, max_distance)

    def enable_dof_force_cache(self, enabled: bool = True):
        # The GF current proxy uses calibrated output torque, not the BAM
        # solver-force observer. Preserve the parent lifecycle contract here;
        # a force-component read still fails until it has a parity-tested owner.
        self._xduck_force_cache_enabled = bool(enabled)

    def get_dof_force_components(self):
        raise NotImplementedError("GF GPU force-component export has not been validated")

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ):
        rows = self._validate_rows(env_indices)
        if randomization is not None and not randomization.is_empty():
            unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
                randomization.requested_terms()
            )
            if unsupported:
                raise NotImplementedError(f"Unsupported GF Warp reset DR: {sorted(unsupported)}")
            indices = torch.as_tensor(rows, device="cuda:0", dtype=torch.long)
            for name in ("body_ipos", "body_mass", "body_inertia", "geom_friction", "dof_armature"):
                value = getattr(randomization, name)
                if value is None:
                    continue
                target = self._warp.to_torch(getattr(self._device_model, name))
                expected = (len(rows), *target.shape[1:])
                if np.shape(value) != expected:
                    raise ValueError(f"{name} reset DR expected {expected}, got {np.shape(value)}")
                tensor = torch.as_tensor(value, device="cuda:0", dtype=target.dtype)
                target.index_copy_(0, indices, tensor)
            self._mujoco_warp.set_const(self._device_model, self._device_data)
        return super().set_state(rows, qpos, qvel, randomization=None)

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan):
        if plan.is_empty():
            return
        if plan.push_perturbation_limit is not None or plan.body_force is not None:
            raise NotImplementedError("GF Warp force pushes are not validated")
        if plan.body_linear_velocity_delta is None:
            return
        ids = np.asarray(plan.body_ids, dtype=np.int32).reshape(-1)
        if ids.shape != (1,) or int(ids[0]) != self._xduck_push_body_id:
            raise NotImplementedError("GF Warp velocity push supports only the free base body")
        value = np.asarray(plan.body_linear_velocity_delta, dtype=np.float32)
        expected = (self._num_envs, 1, 3)
        if value.shape != expected:
            raise ValueError(f"Expected velocity push shape {expected}, got {value.shape}")
        delta = torch.as_tensor(value[:, 0, :], device="cuda:0")
        self._warp.to_torch(self._device_data.qvel)[:, :3].add_(delta)
        self._qvel_cache[:, :3] += value[:, 0, :]

    def get_body_pos_w(self, body_ids: np.ndarray):
        return np.stack(
            [
                self.get_sensor_data("track_pos_w_" + self._body_pose_names[int(i)])
                for i in body_ids
            ],
            axis=1,
        )

    def get_body_quat_w(self, body_ids: np.ndarray):
        return np.stack(
            [
                self.get_sensor_data("track_quat_w_" + self._body_pose_names[int(i)])
                for i in body_ids
            ],
            axis=1,
        )
