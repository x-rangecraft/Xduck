"""XDuck reference-residual sit/stand task, separate from archived PPO recipes.

References are motor-target paths accepted by replay on this model. Measured
nominal motion supplies coherent joint, height and attitude reward targets.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import unilab.actuators.gf43x40 as gf43x40
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.base import LocomotionBaseEnv
from unilab.utils.rotation import np_quat_apply_inverse

from .motor_path import ContinuousMotorPath
from .reference_quality import require_reference_motion
from .scaling import xml_model_fingerprint
from .sitstand import (
    LEG_IDS,
    SitStandAlignedCurriculumConfig,
    SitStandAlignedRewardConfig,
    SitStandDRProvider,
    XDuckSitStandAlignedCfg,
    XDuckSitStandEnv,
    _sitstand_scene,
)
from .velocity import XDuckGF43X40VelocityEnv

REFERENCE_FILE = Path(__file__).with_name("sitstand_reference_v4.json")
EXTRA_OBS = 35


def reference_scene():
    scene = _sitstand_scene()
    scene.fragment_files.append(str(Path(__file__).with_name("reference_sensors.xml")))
    return scene


def reference_fingerprint(scene):
    """Cold-path identity covers XML, mesh bytes, task fragments and motor calibration."""
    digest = hashlib.sha256(xml_model_fingerprint(scene.model_file).encode())
    files = sorted((Path(scene.model_file).parent / "assets").glob("*"))
    files += [Path(p) for p in scene.fragment_files]
    files += [Path(gf43x40.__file__).with_name("calibration.json")]
    for p in files:
        if p.is_file():
            digest.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return digest.hexdigest()


class MotorReference:
    def __init__(self, data):
        style = data.get("interpolation", "segment_smoothstep")
        if style not in ("segment_smoothstep", "continuous_quintic"):
            raise ValueError("unknown reference interpolation")
        self.path = (
            ContinuousMotorPath(data["nodes"], data["durations"])
            if style == "continuous_quintic"
            else None
        )
        self.nodes = np.asarray(data["nodes"], dtype=np.float64)
        self.knots = np.r_[0.0, np.cumsum(data["durations"])]
        self.trace = np.asarray(data["trace"], dtype=np.float64)
        self.dt = float(data["sample_dt"])
        self.duration = float(self.knots[-1])
        self.end = float((len(self.trace) - 1) * self.dt)
        if self.nodes.shape != (len(self.knots), 14) or self.trace.shape[1] != 19:
            raise ValueError("invalid reference dimensions: q14 / height1 / up3 / vz1")
        if not np.all(np.diff(self.knots) > 0) or self.end < self.duration + 1.0:
            raise ValueError("reference needs positive segment times and one second of settling")
        if not np.all(np.isfinite(self.nodes)) or not np.all(np.isfinite(self.trace)):
            raise ValueError("nonfinite reference")

    def sample(self, elapsed):
        elapsed = np.asarray(elapsed)
        q = np.broadcast_to(self.nodes[0], (len(elapsed), 14)).copy()
        for j, duration in enumerate(np.diff(self.knots)):
            phase = np.clip((elapsed - self.knots[j]) / duration, 0, 1)
            smooth = phase * phase * (3 - 2 * phase)
            q += smooth[:, None] * (self.nodes[j + 1] - self.nodes[j])
        if self.path is not None:
            q = self.path.sample(elapsed)
        index = np.clip(elapsed / self.dt, 0, len(self.trace) - 1)
        lo = index.astype(np.intp)
        hi = np.minimum(lo + 1, len(self.trace) - 1)
        dynamics = self.trace[lo] + (self.trace[hi] - self.trace[lo]) * (index - lo)[:, None]
        return q, dynamics


@dataclass
class ReferenceRewardConfig(SitStandAlignedRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            **{k: 0.0 for k in SitStandAlignedRewardConfig().scales},
            "reference_tracking": 8.0,
            "reference_hold": 2.0,
            "residual_rate": -0.02,
            "reference_velocity_error": -2.0,
            "reference_fall": -10.0,
            "reference_contact": -20.0,
        }
    )


def reference_curriculum():
    """Fixed final DR: no claim that multiplying iteration milestones aligns PPO."""
    cfg = SitStandAlignedCurriculumConfig()
    for name in ("com_range_stages", "head_com_range_stages"):
        setattr(cfg, name, [{"step": 0, "value": getattr(cfg, name)[-1]["value"]}])
    cfg.head_range_stages = [{"step": 0, "ranges": [[0.0, 0.0]] * 4}]
    for name in ("action_rate_stages", "rise_speed_stages", "descent_speed_stages"):
        setattr(cfg, name, [{"step": 0, "value": 0.0}])
    return cfg


@registry.envcfg("XDuckGF43X40SitStandReferenceFlat")
@dataclass
class XDuckSitStandReferenceCfg(XDuckSitStandAlignedCfg):
    scene: SceneCfg = field(default_factory=reference_scene)
    reference_file: str = str(REFERENCE_FILE)
    motor_envelope_mode: str = "continuous_conservative"
    whole_body_mass_dr: bool = True
    policy_action_scale_rad: float = 0.08
    curriculum: SitStandAlignedCurriculumConfig = field(default_factory=reference_curriculum)
    residual_bound_rad: float = 0.08
    residual_rate_rad_s: float = 0.5
    max_episode_seconds: float = 30.0
    posture_dwell_seconds: tuple[float, float] = (4.0, 6.0)
    reward_config: ReferenceRewardConfig = field(default_factory=ReferenceRewardConfig)


class ReferenceDRProvider(SitStandDRProvider):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        # The parent chose the actual physical start, independently of command.
        env._reset_reference(env_ids, env._posture_alpha[env_ids] > 0.5)
        return plan


@registry.env("XDuckGF43X40SitStandReferenceFlat", sim_backend="mujoco")
class XDuckSitStandReferenceEnv(XDuckSitStandEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        data = json.loads(Path(cfg.reference_file).read_text())
        if not data.get("head_and_trunk_contact_free", False):
            raise ValueError("reference has not passed forbidden-contact acceptance")
        if data.get("version", 0) >= 4:
            require_reference_motion(data)
        if data["model_fingerprint"] != reference_fingerprint(cfg.scene):
            raise ValueError("reference model mismatch; replay and regenerate after model edits")
        if cfg.control_config.joint_kp != data["kp"] or cfg.control_config.joint_kd != data["kd"]:
            raise ValueError("reference requires the validated user joint gains")
        if cfg.motor_envelope_mode != "continuous_conservative":
            raise ValueError("reference requires the continuous conservative motor envelope")
        if not 0 < cfg.residual_bound_rad <= 0.08 or not 0 < cfg.residual_rate_rad_s <= 0.5:
            raise ValueError("residual bounds exceed the reviewed control contract")
        self._refs = {key: MotorReference(data[key]) for key in ("rise", "descent")}
        if cfg.posture_dwell_seconds[0] <= max(r.end for r in self._refs.values()):
            raise ValueError("command dwell must include complete motion and settling")
        self._ref_direction = np.zeros(num_envs, dtype=np.int8)
        self._ref_posture = np.zeros(num_envs, dtype=np.int8)
        self._ref_elapsed = np.zeros(num_envs)
        self._ref_phase = np.zeros(num_envs)
        self._ref_target = np.zeros((num_envs, 14))
        self._ref_dynamics = np.zeros((num_envs, 19))
        self._residual = np.zeros((num_envs, 14))
        self._residual_velocity = np.zeros((num_envs, 14))
        self._target_clipped = np.zeros((num_envs, 14), dtype=bool)
        self._fall_steps = np.zeros(num_envs, dtype=np.int32)
        super().__init__(cfg, num_envs, backend_type)
        for ref in self._refs.values():
            if np.any(ref.nodes < self._target_low) or np.any(ref.nodes > self._target_high):
                raise ValueError("reference motor targets outside model limits")
        expected_sit = self.default_angles.copy()
        expected_sit[LEG_IDS] = self._sit_legs
        for name, start, end in [
            ("rise", expected_sit, self.default_angles),
            ("descent", self.default_angles, expected_sit),
        ]:
            if not np.allclose(self._refs[name].nodes[[0, -1]], [start, end], atol=1e-6):
                raise ValueError("reference endpoints differ from configured sit/stand")

    def _make_dr_provider(self):
        return ReferenceDRProvider()

    def _reset_reference(self, ids, sitting):
        self._ref_direction[ids] = 0
        self._ref_elapsed[ids] = 0
        self._ref_phase[ids] = 0
        self._ref_posture[ids] = sitting
        self._residual[ids] = 0
        self._residual_velocity[ids] = 0
        self._fall_steps[ids] = 0
        self._target_clipped[ids] = False
        for sit, name in [(False, "rise"), (True, "descent")]:
            subset = np.asarray(ids)[sitting == sit]
            if len(subset):
                ref = self._refs[name]
                self._ref_target[subset], self._ref_dynamics[subset] = ref.sample(
                    np.full(len(subset), ref.end)
                )

    @property
    def obs_groups_spec(self):
        return {k: v + EXTRA_OBS for k, v in super().obs_groups_spec.items()}

    def _compute_obs(self, *args, env_ids=None, **kwargs):
        obs = super()._compute_obs(*args, env_ids=env_ids, **kwargs)
        ids = slice(None) if env_ids is None else env_ids
        extra = np.c_[
            self._ref_direction[ids],
            self._ref_phase[ids],
            self._ref_posture[ids],
            self._ref_target[ids] - self.default_angles,
            self._residual[ids],
            self._ref_dynamics[ids, 14:18],
        ]
        return {k: np.concatenate([v, extra.astype(v.dtype)], axis=1) for k, v in obs.items()}

    def _maybe_resample_commands(self, state):
        # Advance reference with the motor target before physics; do not run
        # the legacy linear height/joint interpolation after physics.
        XDuckGF43X40VelocityEnv._maybe_resample_commands(self, state)

    def _advance_training_reference(self, state):
        desired = (state.info["commands"][:, 0] > 0.5).astype(np.int8)
        start = (self._ref_direction == 0) & (desired != self._ref_posture)
        self._ref_direction[start] = np.where(desired[start], 1, -1)
        self._ref_elapsed[start] = 0
        for direction, name in [(-1, "rise"), (1, "descent")]:
            ids = np.flatnonzero(self._ref_direction == direction)
            if not len(ids):
                continue
            ref = self._refs[name]
            self._ref_elapsed[ids] += self.cfg.ctrl_dt
            self._ref_target[ids], self._ref_dynamics[ids] = ref.sample(self._ref_elapsed[ids])
            self._ref_phase[ids] = np.clip(self._ref_elapsed[ids] / ref.duration, 0, 1)
            done = ids[self._ref_elapsed[ids] >= ref.end]
            self._ref_direction[done] = 0
            self._ref_posture[done] = direction == 1
    def apply_action(self, actions, state):
        LocomotionBaseEnv.apply_action(self, actions, state)
        self._advance_training_reference(state)
        request = self.cfg.residual_bound_rad * np.tanh(actions)
        delta = np.clip(
            request - self._residual,
            -self.cfg.residual_rate_rad_s * self.cfg.ctrl_dt,
            self.cfg.residual_rate_rad_s * self.cfg.ctrl_dt,
        )
        self._residual += delta
        self._residual_velocity[:] = delta / self.cfg.ctrl_dt
        raw = self._ref_target + self._residual
        target = np.clip(raw, self._target_low, self._target_high)
        self._target_clipped[:] = np.abs(raw - target) > 1e-8
        self._previous_action_target[:] = self._current_action_target
        self._current_action_target[:] = target
        return target

    def _physics_step_action_delay(self, backend, ctrl):
        biased = ctrl + self._ankle_target_offset
        self._target_clipped[:] |= (biased < self._target_low) | (biased > self._target_high)
        return super()._physics_step_action_delay(backend, ctrl)

    def _forbidden_contact(self):
        return (self._backend.get_sensor_data("reference_head_floor")[:, 0] > 0) | (
            self._backend.get_sensor_data("reference_trunk_floor")[:, 0] > 0
        )

    def _terminated(self, upvector):
        bad = (upvector[:, 2] < np.cos(np.deg2rad(65))) | (
            self._backend.get_base_pos()[:, 2] < 0.13
        )
        self._fall_steps[:] = np.where(bad, self._fall_steps + 1, 0)
        return (self._fall_steps >= 10) | self._forbidden_contact()

    def _tracking_score(self, joint_error, height_error, orientation_error):
        return np.exp(-joint_error - np.square(height_error / 0.02) - orientation_error)

    def _vertical_velocity_cost(self, error):
        return np.square(np.clip(error / 0.1, -10, 10))

    def _compute_reward(self, state, linvel, gyro, upvector, dof_pos):
        del linvel, gyro, upvector
        ref = self._ref_dynamics
        height = self._backend.get_base_pos()[:, 2]
        vel = self._backend.get_base_lin_vel()
        up = np_quat_apply_inverse(
            self._backend.get_base_quat(), np.tile([0.0, 0.0, 1.0], (self.num_envs, 1))
        )
        joint_error = np.mean(np.square((dof_pos - ref[:, :14]) / 0.15), axis=1)
        height_error = height - ref[:, 14]
        orientation_error = np.sum(np.square((up - ref[:, 15:18]) / 0.15), axis=1)
        tracking = self._tracking_score(joint_error, height_error, orientation_error)
        hold = (
            (self._ref_direction == 0)
            & (np.abs(height_error) < 0.01)
            & (joint_error < (0.08 / 0.15) ** 2)
        )
        hold &= (up[:, 2] > np.cos(np.deg2rad(5))) & (np.linalg.norm(vel, axis=1) < 0.03)
        terms = {
            "reference_tracking": tracking,
            "reference_hold": hold.astype(float),
            "residual_rate": np.mean(np.square(self._residual_velocity / 0.5), axis=1),
            "reference_velocity_error": self._vertical_velocity_cost(vel[:, 2] - ref[:, 18]),
            "reference_fall": (self._fall_steps > 0).astype(float),
            "reference_contact": self._forbidden_contact().astype(float),
        }
        reward = np.zeros(self.num_envs, dtype=np.float32)
        for name, value in terms.items():
            weighted = self.cfg.reward_config.scales[name] * value
            reward += weighted * self.cfg.ctrl_dt
            self._episode_reward_sums[name] += weighted * self.cfg.ctrl_dt
        log = state.info.setdefault("log", {})
        log.update(
            {
                "sitstand/reference_tracking": float(tracking.mean()),
                "sitstand/target_clip_fraction": float(self._target_clipped.mean()),
                "sitstand/height_error_m": float(np.abs(height_error).mean()),
                "sitstand/fresh_samples": self.step_counter * self.num_envs,
            }
        )
        for posture, label in [(0, "stand"), (1, "sit")]:
            mask = (self._ref_direction == 0) & (self._ref_posture == posture)
            if mask.any():
                log[f"sitstand/{label}_hold_success"] = float(hold[mask].mean())
        for direction, label in [(-1, "rise"), (1, "descent")]:
            mask = self._ref_direction == direction
            if mask.any():
                log[f"sitstand/{label}_tracking"] = float(tracking[mask].mean())
        return reward
