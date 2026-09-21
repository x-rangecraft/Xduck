"""Load the frozen GF checkpoint and step its packaged MuJoCo environment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "unilab" / "src"))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from unilab.base import registry  # noqa: E402
from unilab.training.rsl_rl import normalize_ppo_train_cfg  # noqa: E402


def main() -> None:
    role = ROOT.name
    if role == "walk":
        task = "XDuckGF43X40VelocityFlat"
        checkpoint = ROOT / "checkpoints/model_1975_strong_bounded.pt"
    elif role == "stand":
        task = "XDuckGF43X40WalkStopFlat"
        checkpoint = ROOT / "checkpoints/model_599_stand_push02.pt"
    else:
        raise ValueError(f"Unexpected project directory: {role}")

    saved = json.loads((ROOT / "checkpoints/run_config.json").read_text())["config"]
    override = dict(saved["env"])
    override["scene"] = {
        "model_file": str(ROOT / "Model/1.1.2/mjcf/scene.xml"),
        "visual_model_file": str(ROOT / "Model/1.1.2/mjcf/scene.xml"),
        "fragment_files": [
            str(ROOT / "unilab/src/unilab/envs/locomotion/xduck_gf43x40/sensors.xml"),
            str(ROOT / "unilab/src/unilab/envs/locomotion/xduck_gf43x40/gait_reference_v112.xml"),
        ],
    }
    override["reward_config"] = saved["reward"]
    if role == "stand":
        override["handoff_bank"] = str(
            ROOT / "unilab/data/xduck_stop/mixed70stand30walk.npz"
        )

    registry.ensure_registries()
    env = registry.make(task, "mujoco", env_cfg_override=override, num_envs=1)
    try:
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        train_cfg = normalize_ppo_train_cfg(saved["algo"])
        train_cfg.setdefault("runner", {})["logger"] = "none"
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
        runner.load(str(checkpoint), map_location="cpu")
        with torch.inference_mode():
            observation, _ = wrapped.reset()
            action = runner.get_inference_policy(device="cpu")(observation)
            wrapped.step(action)
        print(f"{role}: {task}, actor={wrapped.num_obs}, action={tuple(action.shape)}, step=ok")
    finally:
        env.close()


if __name__ == "__main__":
    main()
