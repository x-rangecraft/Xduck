"""C085/leg25 mixed roulade demonstration reset data and provider; no alternate robot entry."""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.envs.locomotion.microduck_enlarged_106.roulade import (
    MicroDuckEnlarged106RouladeDomainRandomizationProvider,
)

MOTOR_HISTORY = (
    "_roulade_motor_torque",
    "_peak_time",
    "_peak_run",
    "_peak_longest",
    "_positive_work",
    "_negative_work",
    "_torque_peak",
)


@dataclass
class DemonstrationResetConfig:
    dataset_file: str | None = None
    episodes: list[int] = field(default_factory=lambda: [5, 15, 17])
    frame_min: int = 340
    frame_max: int = 550
    sampling: str = "random"
    episode_weights: list[float] | None = None


class DemonstrationResetBank:
    def __init__(self, cfg, env_cfg):
        self.data = None
        self.cursor = 0
        self.sampling = cfg.sampling
        self.probabilities = None
        if cfg.dataset_file is None:
            return
        if cfg.sampling not in ("random", "cycle"):
            raise ValueError("demonstration sampling must be random or cycle")
        if env_cfg.motor_map != "all_x40" or env_cfg.episode_peak_budget_s != 1.0:
            raise ValueError("s1193 resets require all_x40 and the original 1s peak budget")
        if env_cfg.ctrl_dt != 0.02 or env_cfg.motor_constraints.hard_limit_margin != 0.98:
            raise ValueError("s1193 resets require the original control dt and torque margin")
        for name in (
            "randomize_body_mass",
            "random_com",
            "randomize_ground_friction",
            "randomize_dof_armature",
            "randomize_kp",
            "randomize_kd",
        ):
            if getattr(env_cfg.domain_rand, name):
                raise ValueError("demonstration history requires unrandomized source dynamics")
        if env_cfg.control_config.simulate_action_latency:
            raise ValueError("s1193 reset bank does not contain an action latency queue")
        root = Path(__file__).resolve().parents[5]
        path = Path(cfg.dataset_file)
        if not path.is_absolute():
            path = root / path
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        # Historical rollout evidence predates the experiment rename. Keep the
        # immutable dataset compatible while exposing only the new runtime name.
        if "_v107_torque" in data and "_roulade_motor_torque" not in data:
            data["_roulade_motor_torque"] = data.pop("_v107_torque")
        scene = Path(env_cfg.scene.model_file)
        if not scene.is_absolute():
            scene = root / scene
        include = ET.parse(scene).getroot().find("include")
        if include is None or not include.get("file"):
            raise ValueError("demonstration scene must include exactly one robot entry")
        robot = scene.parent / str(include.get("file"))
        for name, asset in [
            ("scene_sha256", scene),
            ("robot_sha256", robot),
        ]:
            if str(data[name]) != hashlib.sha256(asset.read_bytes()).hexdigest():
                raise ValueError(f"demonstration {name} does not match the selected asset")
        if not np.isclose(float(data["motor_response_s"]), env_cfg.motor_response_s):
            raise ValueError("demonstration motor response mismatch")
        if float(data["action_scale"]) != env_cfg.control_config.action_scale:
            raise ValueError("demonstration action scale mismatch")
        n = len(data["episode"])
        expected = {
            "qpos": (n, 21),
            "qvel": (n, 20),
            "obs": (n, 61),
            **{key: (n, 14) for key in MOTOR_HISTORY},
        }
        for key, shape in expected.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid demonstration array: {key}")
        if np.any(data["_peak_time"] < 0) or np.any(data["_peak_time"] > 1 + 1e-8):
            raise ValueError("invalid demonstration peak history")
        if not np.allclose(np.linalg.norm(data["qpos"][:, 3:7], axis=1), 1, atol=1e-5):
            raise ValueError("invalid demonstration quaternion")
        if not np.array_equal(data["info__current_actions"], data["obs"][:, 34:48]):
            raise ValueError("demonstration previous-action alignment mismatch")
        mask = np.isin(data["episode"], cfg.episodes)
        mask &= (data["frame"] >= cfg.frame_min) & (data["frame"] <= cfg.frame_max)
        self.eligible = np.flatnonzero(mask)
        if len(self.eligible) == 0:
            raise ValueError("demonstration reset selection is empty")
        if cfg.episode_weights is not None:
            weights = np.asarray(cfg.episode_weights, dtype=float)
            if cfg.sampling != "random":
                raise ValueError("weighted demonstration reset sampling must be random")
            if weights.shape != (len(cfg.episodes),) or not np.isfinite(weights).all():
                raise ValueError("demonstration episode weights must match configured episodes")
            if np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("demonstration episode weights require a positive sum")
            row_weights = np.zeros(len(self.eligible), dtype=float)
            eligible_episodes = data["episode"][self.eligible]
            for episode, weight in zip(cfg.episodes, weights, strict=True):
                count = np.count_nonzero(eligible_episodes == episode)
                if count == 0 and weight > 0:
                    raise ValueError("weighted demonstration episode has no eligible rows")
                if count:
                    row_weights[eligible_episodes == episode] = weight / count
            self.probabilities = row_weights / row_weights.sum()
        self.data = data

    def sample(self, count):
        if self.sampling == "random":
            return np.random.choice(self.eligible, count, p=self.probabilities)
        ids = self.eligible[(self.cursor + np.arange(count)) % len(self.eligible)]
        self.cursor += count
        return ids


class DemonstrationResetProvider(MicroDuckEnlarged106RouladeDomainRandomizationProvider):
    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        bank = env._demonstration_bank
        if bank.data is not None:
            selected = bank.sample(len(env_ids))
            data = bank.data
            plan.qpos = data["qpos"][selected].copy()
            plan.qvel = data["qvel"][selected].copy()
            for key, value in data.items():
                if key.startswith("info__"):
                    plan.info_updates[key[6:]] = value[selected].copy()
            plan.info_updates["demonstration_reset_index"] = selected
            plan.info_updates["demonstration_episode"] = data["episode"][selected].copy()
            plan.info_updates["demonstration_frame"] = data["frame"][selected].copy()
            env._spawn.record_episode_start(env_ids, plan.qpos[:, :3])
        # Policy phase must integrate through flight/contact transitions.  Its
        # reset value is aligned with the restored task frontier when present.
        plan.info_updates["policy_roll_accum"] = np.asarray(
            plan.info_updates["roulade_accum"]
        ).copy()
        plan.info_updates["policy_roll_frontier"] = np.asarray(
            plan.info_updates["roulade_max"]
        ).copy()
        return plan
