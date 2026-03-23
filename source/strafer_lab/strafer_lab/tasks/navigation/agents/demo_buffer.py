"""Expert demonstration replay buffer loaded from HDF5.

Accepts either a single ``.h5`` file or a directory containing multiple
``.h5`` files (loaded and concatenated in sorted order). This allows
incremental demo collection across sessions.

HDF5 format (from ``collect_demos.py``):
    episode_XXXX/obs      (T, obs_dim)
    episode_XXXX/actions   (T, act_dim)
    episode_XXXX/rewards   (T,)  — optional, used for return-weighted filtering
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


class ExpertDemoBuffer:
    """Replay buffer of expert demonstrations loaded from HDF5.

    Supports optional per-transition weighting by normalized episode return
    when reward data is available in the HDF5 file.

    ``path`` may be a single ``.h5`` file or a directory.  When a directory is
    given, all ``.h5`` files inside it are loaded and concatenated (sorted by
    filename).  This lets you collect demos incrementally across sessions and
    point the training script at the folder.
    """

    def __init__(
        self,
        path: str,
        device: str = "cuda",
        min_return_percentile: float = 0.0,
        action_noise_std: float = 0.05,
        expected_obs_dim: int | None = None,
    ):
        """Load expert demonstrations from HDF5 file(s).

        Args:
            path: Path to a single HDF5 demo file **or** a directory
                containing one or more ``.h5`` files.
            device: Torch device.
            min_return_percentile: Drop episodes below this return percentile
                (0.0 = keep all, 0.25 = drop bottom 25%). Only applies when
                reward data is available.
            action_noise_std: Std of Gaussian noise added to demo actions during
                sampling. Regularizes the BC target to prevent overfitting to
                exact gamepad inputs.
            expected_obs_dim: If set, assert that loaded obs matches this dim.
                Helps catch mismatches between demo files and training env
                (e.g., NoCam demos with a Depth variant).
        """
        self.device = device

        # Resolve file list
        p = Path(path)
        if p.is_dir():
            h5_files = sorted(p.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(
                    f"No .h5 files found in directory: {path}"
                )
            print(f"[DemoBuffer] Found {len(h5_files)} demo file(s) in {path}")
        elif p.is_file():
            h5_files = [p]
        else:
            raise FileNotFoundError(f"Demo path does not exist: {path}")

        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        episode_returns: list[float] = []
        has_rewards = False
        total_episodes = 0

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                num_eps = f.attrs["num_episodes"]
                print(f"[DemoBuffer]   {h5_path.name}: {num_eps} episodes")
                for i in range(num_eps):
                    grp = f[f"episode_{i:04d}"]
                    obs_list.append(np.array(grp["obs"]))
                    act_list.append(np.array(grp["actions"]))
                    if "rewards" in grp:
                        has_rewards = True
                        rewards = np.array(grp["rewards"])
                        episode_returns.append(float(rewards.sum()))
                total_episodes += num_eps

        print(f"[DemoBuffer] Total: {total_episodes} episodes "
              f"from {len(h5_files)} file(s)")

        # Filter by return percentile if rewards are available
        if has_rewards and min_return_percentile > 0.0 and len(episode_returns) > 1:
            threshold = np.percentile(episode_returns, min_return_percentile * 100)
            keep = [i for i, r in enumerate(episode_returns) if r >= threshold]
            obs_list = [obs_list[i] for i in keep]
            act_list = [act_list[i] for i in keep]
            episode_returns = [episode_returns[i] for i in keep]
            print(f"[DemoBuffer] Filtered to {len(keep)}/{total_episodes} episodes "
                  f"(return >= {threshold:.1f}, percentile {min_return_percentile:.0%})")

        self.obs = torch.from_numpy(np.concatenate(obs_list, axis=0)).float().to(device)
        self.actions = torch.from_numpy(np.concatenate(act_list, axis=0)).float().to(device)
        self.num_samples = self.obs.shape[0]
        self.has_rewards = has_rewards
        self.action_noise_std = action_noise_std

        if expected_obs_dim is not None and self.obs.shape[1] != expected_obs_dim:
            raise ValueError(
                f"Demo obs_dim={self.obs.shape[1]} but expected "
                f"{expected_obs_dim}. Recollect demos with a matching env "
                f"variant (use Depth variant for {expected_obs_dim}-dim obs)."
            )

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
