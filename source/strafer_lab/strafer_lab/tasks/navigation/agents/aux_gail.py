"""GAIL (Generative Adversarial Imitation Learning) auxiliary loss module.

Trains a discriminator to distinguish expert state-action pairs from policy
rollouts, then uses the discriminator output as an auxiliary loss signal.

Uses WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training:
- Expert should score high, policy low
- Gradient penalty (lambda=10) enforces Lipschitz constraint
- Discriminator is trained in ``on_update_start`` and frozen during the
  policy mini-batch loop

The discriminator operates on the policy's **encoded** observation (post visual
backbone, pre-RNN) concatenated with actions. This gives it scene-aware features
via the shared encoder without duplicating the backbone. The discriminator input
dimension is auto-detected from the policy on first use.

Usage:
    from strafer_lab.tasks.navigation.agents.strafer_ppo import (
        install_strafer_ppo, register_auxiliary,
    )
    from strafer_lab.tasks.navigation.agents.aux_gail import GAILAuxiliary

    install_strafer_ppo()
    register_auxiliary(GAILAuxiliary(
        demo_path="demos.h5",
        act_dim=3,
    ))
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .demo_buffer import ExpertDemoBuffer
from .strafer_ppo import AuxiliaryLoss


class GAILAuxiliary(AuxiliaryLoss):
    """GAIL auxiliary loss with WGAN-GP discriminator and shared visual backbone.

    The discriminator is a small MLP operating on concatenated
    (encoded_obs, action) pairs, where encoded_obs comes from the policy's
    visual encoder (frozen DeFM + trainable projection). It is trained to
    score expert pairs high and policy pairs low. The policy receives an
    auxiliary loss that encourages producing state-action pairs the
    discriminator classifies as expert-like.

    The discriminator is lazily constructed on first use — its input dimension
    is auto-detected from ``ppo.policy.encoded_obs_dim``.
    """

    def __init__(
        self,
        demo_path: str,
        act_dim: int = 3,
        reward_weight: float = 1.0,
        disc_lr: float = 3e-4,
        disc_hidden_dims: tuple[int, ...] = (256, 256),
        disc_updates_per_ppo: int = 1,
        disc_batch_size: int = 256,
        grad_penalty_weight: float = 10.0,
        min_return_percentile: float = 0.0,
        device: str = "cuda",
    ):
        """Initialize GAIL config and demo buffer.

        The discriminator MLP is NOT built here — it is lazily constructed on
        the first call to ``on_update_start`` when ``ppo.policy`` is available,
        so the input dimension can be auto-detected.

        Args:
            demo_path: Path to HDF5 demo file (obs must match training env dim).
            act_dim: Action dimension.
            reward_weight: Scale factor for GAIL loss contribution.
            disc_lr: Discriminator learning rate.
            disc_hidden_dims: Hidden layer sizes for discriminator MLP.
            disc_updates_per_ppo: Discriminator training steps per PPO update.
            disc_batch_size: Batch size for discriminator training.
            grad_penalty_weight: WGAN-GP gradient penalty coefficient.
            min_return_percentile: Drop demo episodes below this percentile.
            device: Torch device.
        """
        self.buffer = ExpertDemoBuffer(
            demo_path,
            device=device,
            min_return_percentile=min_return_percentile,
            action_noise_std=0.0,  # no noise for GAIL — discriminator handles it
        )
        self.act_dim = act_dim
        self.reward_weight = reward_weight
        self.disc_lr = disc_lr
        self.disc_hidden_dims = disc_hidden_dims
        self.disc_updates_per_ppo = disc_updates_per_ppo
        self.disc_batch_size = disc_batch_size
        self.grad_penalty_weight = grad_penalty_weight
        self.device = device

        # Lazy-initialized in _ensure_discriminator()
        self.discriminator: nn.Sequential | None = None
        self.disc_optimizer: optim.Adam | None = None

        # Cache for discriminator metrics from on_update_start
        self._disc_metrics: dict[str, float] = {}

        print(f"[GAILAuxiliary] reward_weight={reward_weight}, disc_lr={disc_lr}, "
              f"hidden={disc_hidden_dims}, updates_per_ppo={disc_updates_per_ppo}, "
              f"grad_penalty={grad_penalty_weight} (discriminator built on first use)")

    def _ensure_discriminator(self, ppo) -> None:
        """Lazily build the discriminator once the actor is available."""
        if self.discriminator is not None:
            return

        # StraferDepthRNNModel exposes encoded_obs_dim / encode_obs().
        # For plain MLPModel/RNNModel actors, fall back to raw obs dim
        # and identity encoding.
        if hasattr(ppo.actor, "encoded_obs_dim"):
            enc_dim = ppo.actor.encoded_obs_dim
        else:
            # Sum obs group dimensions as fallback
            enc_dim = sum(
                ppo.storage.observations[g].shape[-1]
                for g in ppo.actor.obs_groups
            )
        input_dim = enc_dim + self.act_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.disc_hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.discriminator = nn.Sequential(*layers).to(self.device)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.disc_lr)

        # Validate expert demo dimensions match the actor's obs space
        demo_obs_dim = self.buffer.obs.shape[1]
        policy_obs_dim = sum(
            ppo.storage.observations[g].shape[-1]
            for g in ppo.actor.obs_groups
        )
        if demo_obs_dim != policy_obs_dim:
            raise ValueError(
                f"GAIL demo obs_dim={demo_obs_dim} but policy expects "
                f"{policy_obs_dim}. Train with an env variant whose "
                f"obs_dim matches your demos (e.g. Depth for "
                f"{demo_obs_dim}-dim demos, NoCam for "
                f"{policy_obs_dim}-dim demos)."
            )

        print(f"[GAILAuxiliary] Discriminator built: input_dim={input_dim} "
              f"(encoded_obs={enc_dim} + act={self.act_dim})")

    @staticmethod
    def _encode(actor, obs_flat: torch.Tensor) -> torch.Tensor:
        """Encode observations through the actor's encoder if available."""
        if hasattr(actor, "encode_obs"):
            return actor.encode_obs(obs_flat)
        return obs_flat  # identity fallback for non-depth actors

    def _gradient_penalty(
        self,
        expert_input: torch.Tensor,
        policy_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty for Lipschitz constraint."""
        batch_size = min(expert_input.shape[0], policy_input.shape[0])
        expert_input = expert_input[:batch_size]
        policy_input = policy_input[:batch_size]

        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated = alpha * expert_input + (1 - alpha) * policy_input
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1.0) ** 2).mean()

    def on_update_start(self, ppo) -> None:
        """Train discriminator on expert vs rollout data (encoded space)."""
        self._ensure_discriminator(ppo)

        # Raw rollout observations and actions from PPO storage
        rollout_obs_raw = ppo.storage.observations["policy"]
        rollout_act = ppo.storage.actions

        # Flatten time and env dimensions
        T, N = rollout_obs_raw.shape[0], rollout_obs_raw.shape[1]
        rollout_obs_flat = rollout_obs_raw.reshape(T * N, -1)
        rollout_act_flat = rollout_act.reshape(T * N, -1)

        # Encode rollout obs through shared backbone (no grad — encoder
        # is trained by PPO, not the discriminator)
        with torch.no_grad():
            rollout_encoded = self._encode(ppo.actor, rollout_obs_flat)

        total_disc_loss = 0.0
        total_expert_score = 0.0
        total_policy_score = 0.0

        for _ in range(self.disc_updates_per_ppo):
            # Sample expert batch and encode through shared backbone
            expert_obs, expert_act = self.buffer.sample(self.disc_batch_size)
            with torch.no_grad():
                expert_encoded = self._encode(ppo.actor, expert_obs)
            expert_input = torch.cat([expert_encoded, expert_act], dim=-1)

            # Sample rollout batch (random indices from flattened storage)
            idx = torch.randint(0, T * N, (self.disc_batch_size,), device=self.device)
            policy_encoded = rollout_encoded[idx]
            policy_act = rollout_act_flat[idx]
            policy_input = torch.cat([policy_encoded, policy_act], dim=-1)

            # Discriminator scores
            expert_score = self.discriminator(expert_input)
            policy_score = self.discriminator(policy_input)

            # WGAN loss: expert should score high, policy low
            disc_loss = policy_score.mean() - expert_score.mean()

            # Gradient penalty
            gp = self._gradient_penalty(expert_input.detach(), policy_input.detach())
            disc_loss = disc_loss + self.grad_penalty_weight * gp

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            total_disc_loss += disc_loss.item()
            total_expert_score += expert_score.mean().item()
            total_policy_score += policy_score.mean().item()

        n = max(self.disc_updates_per_ppo, 1)
        self._disc_metrics = {
            "gail_disc_loss": total_disc_loss / n,
            "gail_disc_expert": total_expert_score / n,
            "gail_disc_policy": total_policy_score / n,
        }

    def compute(
        self,
        ppo,
        obs_batch,
        original_batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GAIL auxiliary loss: encourage expert-like state-actions.

        The discriminator is frozen (weights not updated here — trained in
        on_update_start). Gradients flow through the discriminator's
        computation graph into the policy via actions, so the policy learns
        to produce state-action pairs the discriminator classifies as
        expert-like.
        """
        self._ensure_discriminator(ppo)

        # Get current policy actions (with grad) and observations.
        # In rsl_rl 5.0.1, the actor's distribution params are stored after
        # the forward pass. The first param tuple element is the action mean.
        actions = ppo.actor.output_distribution_params[0]
        obs_flat = obs_batch["policy"]

        if obs_flat.dim() == 3:
            obs_flat = obs_flat[-1]
        if actions.dim() == 3:
            actions = actions[-1]

        actions = actions[:original_batch_size]
        obs_flat = obs_flat[:original_batch_size]

        # Encode through shared backbone (detach — encoder trained by PPO only)
        encoded = self._encode(ppo.actor, obs_flat).detach()

        # Ensure batch dims match (recurrent trajectory splitting can cause
        # obs and action_mean to have different trajectory counts)
        batch = min(encoded.shape[0], actions.shape[0])
        encoded = encoded[:batch]
        actions = actions[:batch]

        sa = torch.cat([encoded, actions], dim=-1)

        # Freeze discriminator weights but keep computation graph live so
        # gradients flow: gail_loss → d_score → sa → actions → policy
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        d_score = self.discriminator(sa)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # GAIL reward: -log(sigmoid(D(s,a)))
        # Higher when discriminator thinks the pair is expert-like.
        # We minimize this as a loss (hence the sign).
        gail_loss = -torch.log(torch.sigmoid(d_score) + 1e-8).mean()

        metrics = {
            "gail_reward": -gail_loss.item(),
        }
        metrics.update(self._disc_metrics)

        return self.reward_weight * gail_loss, metrics

    # --- CLI integration ---

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add GAIL-specific CLI arguments."""
        g = parser.add_argument_group("GAIL")
        g.add_argument("--gail_demos", type=str, default=None,
                        help="Path to HDF5 demo file for GAIL")
        g.add_argument("--gail_act_dim", type=int, default=3,
                        help="Action dimension (default: 3)")
        g.add_argument("--gail_reward_weight", type=float, default=1.0,
                        help="GAIL loss weight (default: 1.0)")
        g.add_argument("--gail_disc_lr", type=float, default=3e-4,
                        help="Discriminator learning rate (default: 3e-4)")
        g.add_argument("--gail_disc_updates", type=int, default=1,
                        help="Discriminator updates per PPO update (default: 1)")
        g.add_argument("--gail_disc_batch_size", type=int, default=256,
                        help="Discriminator batch size (default: 256)")
        g.add_argument("--gail_grad_penalty", type=float, default=10.0,
                        help="WGAN-GP gradient penalty weight (default: 10.0)")

    @classmethod
    def from_args(cls, args: argparse.Namespace, device: str = "cuda") -> GAILAuxiliary:
        """Create GAILAuxiliary from parsed CLI arguments."""
        return cls(
            demo_path=args.gail_demos,
            act_dim=args.gail_act_dim,
            reward_weight=args.gail_reward_weight,
            disc_lr=args.gail_disc_lr,
            disc_updates_per_ppo=args.gail_disc_updates,
            disc_batch_size=args.gail_disc_batch_size,
            grad_penalty_weight=args.gail_grad_penalty,
            device=device,
        )
