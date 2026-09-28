"""Opt-in V112 dry-friction hypothesis, solved with the full MuJoCo dynamics.

At zero velocity the signed resisting torque lies in [-Fc_minus, Fc_plus].
Use a symmetric solver constraint of radius (Fc_plus + Fc_minus)/2 and an
explicit bias (Fc_plus - Fc_minus)/2 to retain the existing directional losses.
The static and sliding thresholds are equal: there is no new fitted parameter.
"""

import numpy as np


def install_dry_friction(model, dof_indices, positive_nm, negative_nm):
    """Cold-path installation, before worker models are materialized."""
    model.dof_frictionloss[dof_indices] = 0.5 * (positive_nm + negative_nm)
    # Velocity constraints remain soft numerically. High impedance limits creep;
    # this is a solver setting, not a measured motor stiffness/time constant.
    model.dof_solimp[dof_indices] = [0.9999, 0.9999, 0.001, 0.5, 2.0]


def explicit_dry_friction_loss(velocity, positive_nm, negative_nm, b_positive, b_negative):
    """Loss outside the native symmetric constraint; do not add tanh Coulomb."""
    bias = 0.5 * (positive_nm - negative_nm)
    viscous = np.where(velocity >= 0, b_positive, b_negative)
    return bias + viscous * velocity


class V112DryFriction:
    """V112-only switch; historical/default tasks retain their original plant."""

    def __init__(self, cfg, *args, **kwargs):
        self._dry_friction_enabled = bool(cfg.dry_friction_enabled)
        if self._dry_friction_enabled:
            fit, dr = cfg.identified_dynamics, cfg.full_motor_dr
            if not fit.enabled or not fit.asymmetric_friction or not dr.enabled:
                raise ValueError(
                    "dry friction requires identified asymmetric friction and full_motor_dr"
                )
            # SimBackend currently has no reset-time frictionloss setter. Freeze
            # only Coulomb coefficients in this candidate, never silently ignore DR.
            for name in ("coulomb_positive_scale_range", "coulomb_negative_scale_range"):
                if list(getattr(dr, name)) != [1.0, 1.0]:
                    raise ValueError(f"dry friction requires full_motor_dr.{name}=[1,1]")
            self._dry_positive_nm = float(dr.coulomb_positive_nominal_nm)
            self._dry_negative_nm = float(dr.coulomb_negative_nominal_nm)
            if (
                not np.isfinite([self._dry_positive_nm, self._dry_negative_nm]).all()
                or min(self._dry_positive_nm, self._dry_negative_nm) <= 0
            ):
                raise ValueError("dry-friction thresholds must be finite and positive")
        super().__init__(cfg, *args, **kwargs)

    def _apply_identified_model_parameters(self, backend):
        super()._apply_identified_model_parameters(backend)
        if self._dry_friction_enabled:
            install_dry_friction(
                backend.model,
                backend.get_joint_dof_indices(self.JOINT_NAMES),
                self._dry_positive_nm,
                self._dry_negative_nm,
            )

    def _identified_motor_control(self, **kwargs):
        # Advance the established gain/response state exactly once, retaining its
        # diagnostics. The candidate replaces the old explicit friction output.
        target = super()._identified_motor_control(**kwargs)
        if not self._dry_friction_enabled:
            return target
        q, v, kp, kd = (kwargs[name] for name in ("qpos", "qvel", "kp", "kd"))
        loss = explicit_dry_friction_loss(
            v,
            self._dry_positive_nm,
            self._dry_negative_nm,
            self._identified_viscous_positive_nm_s_rad,
            self._identified_viscous_negative_nm_s_rad,
        )
        # Envelope bounds motor production. Passive friction can still oppose an
        # externally back-driven joint, including above the motoring speed limit.
        drive = np.clip(
            self._identified_torque_state, kwargs["torque_lower"], kwargs["torque_upper"]
        )
        actuator_torque = drive - loss
        if self._state is not None:
            info = self._state.info
            # Native constraint torque is not available through SimBackend's
            # current public telemetry; do not label actuator torque as net torque.
            info.pop("motor_fit_friction_nm", None)
            info.pop("motor_fit_net_torque_nm", None)
            info["motor_dry_friction_actuator_torque_nm"] = actuator_torque.copy()
            info["motor_dry_friction_explicit_loss_nm"] = loss.copy()
            info["motor_dry_friction_constraint_bound_nm"] = np.full_like(
                v, 0.5 * (self._dry_positive_nm + self._dry_negative_nm)
            )
        return q + (actuator_torque + kd * v) / kp
