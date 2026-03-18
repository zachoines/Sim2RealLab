"""Expert demonstration replay buffer loaded from HDF5.

HDF5 format (from ``collect_demos.py``):
    episode_XXXX/obs      (T, obs_dim)
    episode_XXXX/actions   (T, act_dim)
    episode_XXXX/rewards   (T,)  — optional, used for return-weighted filtering
"""

from __future__ import annotations

import h5py
import numpy as np
import torch


class ExpertDemoBuffer:
    """Replay buffer of expert demonstrations loaded from HDF5.

    Supports optional per-transition weighting by normalized episode return
    when reward data is available in the HDF5 file.
    """

    def __init__(
        self,
        path: str,
        device: str = "cuda",
        min_return_percentile: float = 0.0,
        action_noise_std: float = 0.05,
    ):
        """Load expert demonstrations from HDF5.

        Args:
            path: Path to HDF5 demo file.
            device: Torch device.
            min_return_percentile: Drop episodes below this return percentile
                (0.0 = keep all, 0.25 = drop bottom 25%). Only applies when
                reward data is available.
            action_noise_std: Std of Gaussian noise added to demo actions during
                sampling. Regularizes the BC target to prevent overfitting to
                exact gamepad inputs.
        """
        self.device = device
        obs_list = []
        act_list = []
        episode_returns = []
        has_rewards = False

        with h5py.File(path, "r") as f:
            num_eps = f.attrs["num_episodes"]
            print(f"[DemoBuffer] Loading {num_eps} expert episodes from {path}")
            for i in range(num_eps):
                grp = f[f"episode_{i:04d}"]
                obs_list.append(np.array(grp["obs"]))
                act_list.append(np.array(grp["actions"]))
                if "rewards" in grp:
                    has_rewards = True
                    rewards = np.array(grp["rewards"])
                    episode_returns.append(float(rewards.sum()))

        # Filter by return percentile if rewards are available
        if has_rewards and min_return_percentile > 0.0 and len(episode_returns) > 1:
            threshold = np.percentile(episode_returns, min_return_percentile * 100)
            keep = [i for i, r in enumerate(episode_returns) if r >= threshold]
            obs_list = [obs_list[i] for i in keep]
            act_list = [act_list[i] for i in keep]
            episode_returns = [episode_returns[i] for i in keep]
            print(f"[DemoBuffer] Filtered to {len(keep)}/{num_eps} episodes "
                  f"(return >= {threshold:.1f}, percentile {min_return_percentile:.0%})")

        self.obs = torch.from_numpy(np.concatenate(obs_list, axis=0)).float().to(device)
        self.actions = torch.from_numpy(np.concatenate(act_list, axis=0)).float().to(device)
        self.num_samples = self.obs.shape[0]
        self.has_rewards = has_rewards
        self.action_noise_std = action_noise_std

        print(f"[DemoBuffer] Loaded {self.num_samples} expert transitions "
              f"(obs: {self.obs.shape}, actions: {self.actions.shape})")
        if has_rewards:
            print(f"[DemoBuffer] Episode returns: min={min(episode_returns):.1f}, "
                  f"max={max(episode_returns):.1f}, mean={np.mean(episode_returns):.1f}")

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (obs, expert_actions) with action noise."""
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        actions = self.actions[indices]
        if self.action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.action_noise_std
        return self.obs[indices], actions
