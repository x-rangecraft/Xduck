"""UniLab reset contract adapter for the installed UniRL PPO runtime.

Keep learner/runtime ownership in UniRL; scatter full-reset state in the
workspace so reinstalling the dependency cannot reintroduce stale episodes.
"""

from typing import Any

import numpy as np
from tensordict import TensorDict
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper as UniRlVecEnvWrapper


class RslRlVecEnvWrapper(UniRlVecEnvWrapper):
    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        if self.env.state is None:
            self.env.init_state()

        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, info = self.env.reset(env_indices)
        # NpEnv.reset is the low-level physical reset also used by automatic
        # resets. Its caller owns scattering the returned observation/info.
        # A VecEnv reset must do that too, or get_observations/step still read
        # the previous episode's commands, actions and timeout counter.
        state = self.env.state
        state.obs = obs_out
        state.info = {**getattr(state, "info", {}), **info}
        state.info["steps"] = np.zeros(self.num_envs, dtype=np.uint32)
        for name in ("terminated", "truncated"):
            flags = getattr(state, name, None)
            if flags is not None:
                flags.fill(False)
        state.final_observation = None
        state.info.pop("final_observation", None)
        state.info.pop("_final_observation", None)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out, state.info), state.info
