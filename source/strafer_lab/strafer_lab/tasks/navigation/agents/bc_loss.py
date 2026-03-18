"""Demo Augmented Policy Gradient (DAPG) for PPO training.

**NOTE:** This module is kept for backward compatibility. New code should use
the modular auxiliary loss architecture in ``strafer_ppo.py`` + ``aux_dapg.py``:

    from strafer_lab.tasks.navigation.agents.strafer_ppo import (
        install_strafer_ppo, register_auxiliary,
    )
    from strafer_lab.tasks.navigation.agents.aux_dapg import DAPGAuxiliary

    install_strafer_ppo()
    register_auxiliary(DAPGAuxiliary(demo_path="demos.h5"))

The ``register_dapg_loss()`` / ``register_bc_loss()`` functions below still
work but use their own independent monkey-patch of PPO.update(). They cannot
be combined with the new ``install_strafer_ppo()`` architecture.

``ExpertDemoBuffer`` is shared between this module and the new auxiliary
modules — it is the canonical demo loading class.

HDF5 format (from ``collect_demos.py``):
    episode_XXXX/obs      (T, obs_dim)
    episode_XXXX/actions   (T, act_dim)
    episode_XXXX/rewards   (T,)  — optional, used for return-weighted filtering
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn


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
            print(f"[DAPG] Loading {num_eps} expert episodes from {path}")
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
            print(f"[DAPG] Filtered to {len(keep)}/{num_eps} episodes "
                  f"(return >= {threshold:.1f}, percentile {min_return_percentile:.0%})")

        self.obs = torch.from_numpy(np.concatenate(obs_list, axis=0)).float().to(device)
        self.actions = torch.from_numpy(np.concatenate(act_list, axis=0)).float().to(device)
        self.num_samples = self.obs.shape[0]
        self.has_rewards = has_rewards
        self.action_noise_std = action_noise_std

        print(f"[DAPG] Loaded {self.num_samples} expert transitions "
              f"(obs: {self.obs.shape}, actions: {self.actions.shape})")
        if has_rewards:
            print(f"[DAPG] Episode returns: min={min(episode_returns):.1f}, "
                  f"max={max(episode_returns):.1f}, mean={np.mean(episode_returns):.1f}")

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (obs, expert_actions) with action noise."""
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        actions = self.actions[indices]
        if self.action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.action_noise_std
        return self.obs[indices], actions


# ---------------------------------------------------------------------------
# Monkey-patching PPO.update()
# ---------------------------------------------------------------------------

_DAPG_STATE: dict = {}


def register_dapg_loss(
    demo_path: str,
    bc_weight: float = 0.03,
    bc_decay_steps: int = 3000,
    bc_batch_size: int = 128,
    min_return_percentile: float = 0.0,
    action_noise_std: float = 0.05,
    device: str = "cuda",
) -> None:
    """Monkey-patch RSL-RL PPO to add DAPG demo-augmented loss.

    Must be called BEFORE creating the OnPolicyRunner.

    The DAPG term is added inside the PPO mini-batch loop as:

        L_dapg = λ * mean(-log π(a_demo | s_demo))

    where λ decays linearly from ``bc_weight`` to 0 over ``bc_decay_steps``.

    Args:
        demo_path: Path to HDF5 file of expert demonstrations.
        bc_weight: Initial DAPG loss weight (decays linearly to 0).
        bc_decay_steps: Number of PPO update calls over which to decay weight.
        bc_batch_size: Expert transitions to sample per PPO mini-batch.
        min_return_percentile: Drop demo episodes below this return percentile.
        action_noise_std: Std of Gaussian noise added to demo actions during
            sampling. Regularizes BC target to prevent overfitting.
        device: Torch device for the demo buffer.
    """
    from rsl_rl.algorithms.ppo import PPO

    # Load demos
    buffer = ExpertDemoBuffer(demo_path, device=device,
                              min_return_percentile=min_return_percentile,
                              action_noise_std=action_noise_std)

    # Store state
    _DAPG_STATE["buffer"] = buffer
    _DAPG_STATE["initial_weight"] = bc_weight
    _DAPG_STATE["decay_steps"] = bc_decay_steps
    _DAPG_STATE["batch_size"] = bc_batch_size
    _DAPG_STATE["update_count"] = 0

    # Save original update
    _original_update = PPO.update

    def _update_with_dapg(self) -> dict[str, float]:
        """Wrapped PPO.update() with DAPG demo-augmented loss."""
        buf = _DAPG_STATE["buffer"]
        step = _DAPG_STATE["update_count"]
        decay = _DAPG_STATE["decay_steps"]
        w0 = _DAPG_STATE["initial_weight"]

        # Linear decay: λ = w0 * max(0, 1 - step/decay)
        bc_lambda = w0 * max(0.0, 1.0 - step / decay) if decay > 0 else w0
        _DAPG_STATE["update_count"] = step + 1

        if bc_lambda <= 0.0:
            # DAPG phase complete — run original PPO
            return _original_update(self)

        # ---- Replicate PPO.update() with DAPG injection ----
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_dapg_loss = 0

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            # Per-mini-batch advantage normalization
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Symmetric augmentation (preserve existing RSL-RL functionality)
            num_aug = 1
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Forward pass on RL rollout batch
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive learning rate (KL divergence)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ---- DAPG: demo-augmented BC loss ----
            # Sample expert transitions and compute log-prob under current policy.
            # Demo samples are independent (no temporal ordering), so we must
            # temporarily clear RNN hidden state to avoid batch-size mismatch,
            # then restore it for the next RL mini-batch.
            bc_obs, bc_actions = buf.sample(_DAPG_STATE["batch_size"])
            bc_obs_dict = {"policy": bc_obs}

            if self.policy.is_recurrent:
                saved_h_a = self.policy.memory_a.hidden_state
                saved_h_c = self.policy.memory_c.hidden_state
                self.policy.memory_a.reset()
                self.policy.memory_c.reset()

            self.policy.act(bc_obs_dict)
            bc_log_prob = self.policy.get_actions_log_prob(bc_actions)

            if self.policy.is_recurrent:
                self.policy.memory_a.reset(hidden_state=saved_h_a)
                self.policy.memory_c.reset(hidden_state=saved_h_c)
            # NLL: maximize log-prob of expert actions
            dapg_loss = -bc_log_prob.mean()

            # Combined loss: PPO + value + entropy + DAPG
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + bc_lambda * dapg_loss
            )

            # Symmetry loss (preserve existing RSL-RL functionality)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"]
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Backward pass — single combined gradient
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Accumulate metrics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_dapg_loss += dapg_loss.item()

        # Average over all mini-batch updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_dapg_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "dapg_nll": mean_dapg_loss,
            "dapg_weight": bc_lambda,
        }
        if self.rnd:
            loss_dict["rnd"] = 0.0  # tracked but not averaged here
        if self.symmetry:
            loss_dict["symmetry"] = 0.0

        return loss_dict

    PPO.update = _update_with_dapg
    print(f"[DAPG] Registered demo-augmented policy gradient: "
          f"weight={bc_weight}, decay_steps={bc_decay_steps}, "
          f"batch_size={bc_batch_size}, "
          f"action_noise_std={action_noise_std}, "
          f"min_return_percentile={min_return_percentile}")


# Legacy alias — drop-in replacement for existing training scripts
def register_bc_loss(
    demo_path: str,
    bc_weight: float = 0.03,
    bc_decay_steps: int = 3000,
    bc_batch_size: int = 128,
    device: str = "cuda",
) -> None:
    """Legacy alias for ``register_dapg_loss``.

    Maintains backward compatibility with existing training scripts that
    call ``register_bc_loss()``. Now uses DAPG instead of separate-step NLL.
    """
    register_dapg_loss(
        demo_path=demo_path,
        bc_weight=bc_weight,
        bc_decay_steps=bc_decay_steps,
        bc_batch_size=bc_batch_size,
        device=device,
    )
