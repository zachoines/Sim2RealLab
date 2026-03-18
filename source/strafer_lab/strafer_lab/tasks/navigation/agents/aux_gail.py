"""GAIL (Generative Adversarial Imitation Learning) auxiliary loss module.

Trains a discriminator to distinguish expert state-action pairs from policy
rollouts, then uses the discriminator output as an auxiliary loss signal.

Uses WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training:
- Expert should score high, policy low
- Gradient penalty (λ=10) enforces Lipschitz constraint
- Discriminator is trained in ``on_update_start`` and frozen during the
  policy mini-batch loop

Usage:
    from strafer_lab.tasks.navigation.agents.strafer_ppo import (
        install_strafer_ppo, register_auxiliary,
    )
    from strafer_lab.tasks.navigation.agents.aux_gail import GAILAuxiliary

    install_strafer_ppo()
    register_auxiliary(GAILAuxiliary(
        demo_path="demos.h5",
        obs_dim=19,
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
    """GAIL auxiliary loss with WGAN-GP discriminator.

    The discriminator is a small MLP operating on concatenated (obs, action)
    pairs. It is trained to score expert pairs high and policy pairs low.
    The policy receives an auxiliary loss that encourages producing
    state-action pairs the discriminator classifies as expert-like.
    """

    def __init__(
        self,
        demo_path: str,
        obs_dim: int,
        act_dim: int,
        reward_weight: float = 1.0,
        disc_lr: float = 3e-4,
        disc_hidden_dims: tuple[int, ...] = (256, 256),
        disc_updates_per_ppo: int = 1,
        disc_batch_size: int = 256,
        grad_penalty_weight: float = 10.0,
        min_return_percentile: float = 0.0,
        device: str = "cuda",
    ):
        """Initialize GAIL discriminator and demo buffer.

        Args:
            demo_path: Path to HDF5 demo file.
            obs_dim: Observation dimension (policy obs, not depth).
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
        self.reward_weight = reward_weight
        self.disc_updates_per_ppo = disc_updates_per_ppo
        self.disc_batch_size = disc_batch_size
        self.grad_penalty_weight = grad_penalty_weight
        self.device = device

        # Build discriminator MLP
        input_dim = obs_dim + act_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in disc_hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.discriminator = nn.Sequential(*layers).to(device)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=disc_lr)

        # Cache for discriminator metrics from on_update_start
        self._disc_metrics: dict[str, float] = {}

        print(f"[GAILAuxiliary] reward_weight={reward_weight}, disc_lr={disc_lr}, "
              f"hidden={disc_hidden_dims}, updates_per_ppo={disc_updates_per_ppo}, "
              f"grad_penalty={grad_penalty_weight}")

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
        """Train discriminator on expert vs rollout data."""
        # Sample rollout transitions from PPO storage
        # observations is TensorDict[num_steps, num_envs, obs_dim]
        # actions is Tensor[num_steps, num_envs, act_dim]
        rollout_obs = ppo.storage.observations["policy"]
        rollout_act = ppo.storage.actions

        # Flatten time and env dimensions
        T, N = rollout_obs.shape[0], rollout_obs.shape[1]
        rollout_obs_flat = rollout_obs.reshape(T * N, -1)
        rollout_act_flat = rollout_act.reshape(T * N, -1)

        total_disc_loss = 0.0
        total_expert_score = 0.0
        total_policy_score = 0.0

        for _ in range(self.disc_updates_per_ppo):
            # Sample expert batch
            expert_obs, expert_act = self.buffer.sample(self.disc_batch_size)
            expert_input = torch.cat([expert_obs, expert_act], dim=-1)

            # Sample rollout batch (random indices from flattened storage)
            idx = torch.randint(0, T * N, (self.disc_batch_size,), device=self.device)
            policy_obs = rollout_obs_flat[idx]
            policy_act = rollout_act_flat[idx]
            policy_input = torch.cat([policy_obs, policy_act], dim=-1)

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

        The discriminator is frozen (trained in on_update_start). The GAIL
        signal acts like a shaped reward encouraging the policy to produce
        state-action pairs the discriminator classifies as expert-like.
        """
        # Get current policy actions and observations for this mini-batch
        actions = ppo.policy.action_mean[:original_batch_size]
        obs_flat = obs_batch["policy"][:original_batch_size]

        # For recurrent policies, obs may be (seq_len, batch, dim) — take last
        if obs_flat.dim() == 3:
            obs_flat = obs_flat[-1]

        with torch.no_grad():
            sa = torch.cat([obs_flat, actions], dim=-1)
            d_score = self.discriminator(sa)

        # GAIL reward: -log(1 - sigmoid(D(s,a)))
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
        g.add_argument("--gail_obs_dim", type=int, default=19,
                        help="Policy observation dimension (default: 19)")
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
            obs_dim=args.gail_obs_dim,
            act_dim=args.gail_act_dim,
            reward_weight=args.gail_reward_weight,
            disc_lr=args.gail_disc_lr,
            disc_updates_per_ppo=args.gail_disc_updates,
            disc_batch_size=args.gail_disc_batch_size,
            grad_penalty_weight=args.gail_grad_penalty,
            device=device,
        )
