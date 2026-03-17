"""Behavior cloning auxiliary loss for PPO training.

Loads expert demonstrations from HDF5 (collected via ``collect_demos.py``)
and adds a negative log-likelihood (NLL) loss that maximizes the probability
of expert actions under the policy's Gaussian distribution.

Why NLL instead of MSE on action means:
    RSL-RL's policy outputs a Gaussian N(μ(s), σ(s)).  MSE on the mean
    ignores σ entirely — the policy can have high variance (noisy exploration)
    and still score low MSE.  NLL jointly optimizes both μ AND σ:

        NLL = 0.5 * |a_expert - μ(s)|² / σ(s)² + log σ(s) + const

    This pushes μ toward expert actions AND shrinks σ for states where
    expert data is available, making the policy confident about imitation.
    PPO's entropy bonus (``entropy_coef``) prevents σ from collapsing to
    zero, maintaining exploration ability as the BC signal decays.

Integration: call ``register_bc_loss()`` before creating the PPO runner.
This monkey-patches ``PPO.update()`` to add the BC term after PPO's own
gradient step.

The BC weight decays linearly from ``bc_weight`` to 0 over ``bc_decay_steps``
training iterations, implementing a DAgger-like annealing where the policy
gradually transitions from imitation to pure RL.

Usage:
    from strafer_lab.tasks.navigation.agents.bc_loss import register_bc_loss
    register_bc_loss(
        demo_path="demos.h5",
        bc_weight=1.0,
        bc_decay_steps=2000,
        bc_batch_size=256,
    )
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn


class ExpertDemoBuffer:
    """Replay buffer of expert demonstrations loaded from HDF5."""

    def __init__(self, path: str, device: str = "cuda"):
        self.device = device
        obs_list = []
        act_list = []

        with h5py.File(path, "r") as f:
            num_eps = f.attrs["num_episodes"]
            print(f"[BC] Loading {num_eps} expert episodes from {path}")
            for i in range(num_eps):
                grp = f[f"episode_{i:04d}"]
                obs_list.append(np.array(grp["obs"]))
                act_list.append(np.array(grp["actions"]))

        self.obs = torch.from_numpy(np.concatenate(obs_list, axis=0)).float().to(device)
        self.actions = torch.from_numpy(np.concatenate(act_list, axis=0)).float().to(device)
        self.num_samples = self.obs.shape[0]
        print(f"[BC] Loaded {self.num_samples} expert transitions "
              f"(obs: {self.obs.shape}, actions: {self.actions.shape})")

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (obs, expert_actions)."""
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        return self.obs[indices], self.actions[indices]


# ---------------------------------------------------------------------------
# Monkey-patching PPO.update()
# ---------------------------------------------------------------------------

_BC_STATE: dict = {}


def register_bc_loss(
    demo_path: str,
    bc_weight: float = 1.0,
    bc_decay_steps: int = 2000,
    bc_batch_size: int = 256,
    device: str = "cuda",
) -> None:
    """Monkey-patch RSL-RL PPO to add behavior cloning NLL loss.

    Must be called BEFORE creating the OnPolicyRunner.

    Args:
        demo_path: Path to HDF5 file of expert demonstrations.
        bc_weight: Initial BC loss weight (decays linearly to 0).
        bc_decay_steps: Number of PPO update calls over which to decay weight.
        bc_batch_size: Number of expert transitions to sample per mini-batch.
        device: Torch device for the demo buffer.
    """
    from rsl_rl.algorithms.ppo import PPO

    # Load demos
    buffer = ExpertDemoBuffer(demo_path, device=device)

    # Store state
    _BC_STATE["buffer"] = buffer
    _BC_STATE["initial_weight"] = bc_weight
    _BC_STATE["decay_steps"] = bc_decay_steps
    _BC_STATE["batch_size"] = bc_batch_size
    _BC_STATE["update_count"] = 0

    # Save original update
    _original_update = PPO.update

    def _update_with_bc(self) -> dict[str, float]:
        """Wrapped PPO.update() that adds BC NLL loss."""
        buf = _BC_STATE["buffer"]
        step = _BC_STATE["update_count"]
        decay = _BC_STATE["decay_steps"]
        w0 = _BC_STATE["initial_weight"]

        # Linear decay: w = w0 * max(0, 1 - step/decay)
        bc_weight_now = w0 * max(0.0, 1.0 - step / decay) if decay > 0 else w0
        _BC_STATE["update_count"] = step + 1

        if bc_weight_now <= 0.0:
            # BC phase complete — run original PPO
            return _original_update(self)

        # Run original PPO update, then do a separate BC gradient step.
        loss_dict = _original_update(self)

        # Sample expert transitions
        bc_obs, bc_expert_actions = buf.sample(_BC_STATE["batch_size"])

        # Forward pass: builds the policy's N(μ, σ) distribution for expert obs
        # Both ActorCritic and StraferActorCritic expect a dict: obs["policy"]
        bc_obs_dict = {"policy": bc_obs}
        self.policy.act(bc_obs_dict)

        # NLL loss: -log π(a_expert | s)
        # For Gaussian: -log_prob = 0.5 * ((a - μ)/σ)² + log(σ) + 0.5*log(2π)
        # distribution.log_prob returns per-dimension log probs; sum over actions
        log_prob = self.policy.distribution.log_prob(bc_expert_actions)
        nll_loss = -log_prob.sum(dim=-1).mean()

        scaled_bc_loss = bc_weight_now * nll_loss

        self.optimizer.zero_grad()
        scaled_bc_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        loss_dict["bc_nll"] = nll_loss.item()
        loss_dict["bc_weight"] = bc_weight_now
        return loss_dict

    PPO.update = _update_with_bc
    print(f"[BC] Registered BC NLL loss: weight={bc_weight}, "
          f"decay_steps={bc_decay_steps}, batch_size={bc_batch_size}")
