"""Micro Duck actor/action reflection, matching upstream tasks/symmetry.py.

Only mirror consistency loss is supported: privileged critic inputs are not
mirrored or used as synthetic PPO samples. Tables follow the canonical 14-joint
and 61D actor order, with left/right pitch joints using opposite signs.
"""

from __future__ import annotations

JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
JOINT_SIGN = (-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1)
OBS_PERM = (
    tuple(range(6))
    + tuple(6 + j for j in JOINT_PERM)
    + tuple(20 + j for j in JOINT_PERM)
    + tuple(34 + j for j in JOINT_PERM)
    + tuple(range(48, 61))
)
OBS_SIGN = (
    (-1, 1, -1, 1, -1, 1) + JOINT_SIGN * 3 + (1, -1, -1) + (1, 1, -1, -1) + (1, -1, 1, -1, 1, -1)
)


class MicroDuckSymmetryAugmentation:
    batch_multiplier = 2

    def __init__(self, *, device: str):
        if device == "mlx":
            import mlx.core as mx

            self._obs_perm = mx.array(OBS_PERM, dtype=mx.int32)
            self._obs_sign = mx.array(OBS_SIGN, dtype=mx.float32)
            self._act_perm = mx.array(JOINT_PERM, dtype=mx.int32)
            self._act_sign = mx.array(JOINT_SIGN, dtype=mx.float32)
            self._cat = lambda xs: mx.concatenate(xs, axis=0)
        else:
            import torch

            self._obs_perm = torch.tensor(OBS_PERM, device=device, dtype=torch.long)
            self._obs_sign = torch.tensor(OBS_SIGN, device=device, dtype=torch.float32)
            self._act_perm = torch.tensor(JOINT_PERM, device=device, dtype=torch.long)
            self._act_sign = torch.tensor(JOINT_SIGN, device=device, dtype=torch.float32)
            self._cat = lambda xs: torch.cat(xs, dim=0)

    def mirror_obs(self, obs, *, obs_group: str = "obs"):
        if obs_group != "obs" or obs.shape[-1] != 61:
            raise ValueError("Micro Duck mirror loss requires the 61D actor observation")
        return obs[..., self._obs_perm] * self._obs_sign

    def mirror_action(self, actions):
        if actions.shape[-1] != 14:
            raise ValueError("Micro Duck symmetry requires 14D actions")
        return actions[..., self._act_perm] * self._act_sign

    def augment_obs(self, obs, *, obs_group: str = "obs"):
        return self._cat([obs, self.mirror_obs(obs, obs_group=obs_group)])

    def augment_obs_and_actions(self, obs, actions, *, obs_group: str = "obs"):
        return self.augment_obs(obs, obs_group=obs_group), self._cat(
            [actions, self.mirror_action(actions)]
        )


def microduck_symmetry(env, obs=None, actions=None):
    """RSL-RL boundary adapter for the same mirror-only recipe."""
    from tensordict import TensorDict

    sample = obs["policy"] if obs is not None else actions
    adapter = MicroDuckSymmetryAugmentation(device=str(sample.device))
    aug_obs = None
    if obs is not None:
        aug_obs = TensorDict(
            {
                key: adapter.augment_obs(value) if key == "policy" else adapter._cat([value, value])
                for key, value in obs.items()
            },
            batch_size=[sample.shape[0] * 2],
            device=sample.device,
        )
    aug_actions = (
        None if actions is None else adapter._cat([actions, adapter.mirror_action(actions)])
    )
    return aug_obs, aug_actions
