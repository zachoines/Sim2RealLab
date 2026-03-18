"""DAPG (Demo Augmented Policy Gradient) as an auxiliary loss module.

Implements the DAPG approach from Rajeswaran et al. 2018 using the
``AuxiliaryLoss`` plugin interface from ``strafer_ppo.py``.

The BC term is:
    L_dapg = λ * mean(-log π(a_demo | s_demo))

where λ decays linearly from ``bc_weight`` to 0 over ``bc_decay_steps``
PPO update cycles.

Usage:
    from strafer_lab.tasks.navigation.agents.strafer_ppo import (
        install_strafer_ppo, register_auxiliary,
    )
    from strafer_lab.tasks.navigation.agents.aux_dapg import DAPGAuxiliary

    install_strafer_ppo()
    register_auxiliary(DAPGAuxiliary(
        demo_path="demos.h5",
        bc_weight=0.03,
        bc_decay_steps=3000,
        bc_batch_size=128,
    ))
"""

from __future__ import annotations

import argparse

import torch

from .demo_buffer import ExpertDemoBuffer
from .strafer_ppo import AuxiliaryLoss


class DAPGAuxiliary(AuxiliaryLoss):
    """Demo Augmented Policy Gradient auxiliary loss.

    Adds a negative log-likelihood term on expert demonstrations to the
    PPO loss, weighted by a linearly decaying coefficient.
    """

    def __init__(
        self,
        demo_path: str,
        bc_weight: float = 0.03,
        bc_decay_steps: int = 3000,
        bc_batch_size: int = 128,
        min_return_percentile: float = 0.0,
        action_noise_std: float = 0.05,
        device: str = "cuda",
    ):
        self.buffer = ExpertDemoBuffer(
            demo_path,
            device=device,
            min_return_percentile=min_return_percentile,
            action_noise_std=action_noise_std,
        )
        self.bc_weight = bc_weight
        self.decay_steps = bc_decay_steps
        self.batch_size = bc_batch_size
        self.update_count = 0
        self.bc_lambda = bc_weight

        print(f"[DAPGAuxiliary] weight={bc_weight}, decay_steps={bc_decay_steps}, "
              f"batch_size={bc_batch_size}, action_noise_std={action_noise_std}")

    def on_update_start(self, ppo) -> None:
        """Compute current lambda with linear decay."""
        step = self.update_count
        if self.decay_steps > 0:
            self.bc_lambda = self.bc_weight * max(0.0, 1.0 - step / self.decay_steps)
        else:
            self.bc_lambda = self.bc_weight
        self.update_count += 1

    def compute(
        self,
        ppo,
        obs_batch,
        original_batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DAPG NLL loss on expert demonstrations."""
        if self.bc_lambda <= 0.0:
            return torch.tensor(0.0, device=ppo.device), {
                "dapg_nll": 0.0,
                "dapg_weight": 0.0,
            }

        bc_obs, bc_actions = self.buffer.sample(self.batch_size)
        bc_obs_dict = {"policy": bc_obs}

        # Save/reset/restore RNN hidden state for independent demo samples
        if ppo.policy.is_recurrent:
            saved_h_a = ppo.policy.memory_a.hidden_state
            saved_h_c = ppo.policy.memory_c.hidden_state
            ppo.policy.memory_a.reset()
            ppo.policy.memory_c.reset()

        ppo.policy.act(bc_obs_dict)
        bc_log_prob = ppo.policy.get_actions_log_prob(bc_actions)

        if ppo.policy.is_recurrent:
            ppo.policy.memory_a.reset(hidden_state=saved_h_a)
            ppo.policy.memory_c.reset(hidden_state=saved_h_c)

        nll = -bc_log_prob.mean()
        return self.bc_lambda * nll, {
            "dapg_nll": nll.item(),
            "dapg_weight": self.bc_lambda,
        }

    # --- CLI integration ---

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add DAPG-specific CLI arguments."""
        g = parser.add_argument_group("DAPG")
        g.add_argument("--dapg_demos", type=str, default=None,
                        help="Path to HDF5 demo file for DAPG")
        g.add_argument("--dapg_weight", type=float, default=0.03,
                        help="Initial DAPG loss weight (default: 0.03)")
        g.add_argument("--dapg_decay", type=int, default=3000,
                        help="DAPG weight decay steps (default: 3000)")
        g.add_argument("--dapg_batch_size", type=int, default=128,
                        help="DAPG mini-batch size (default: 128)")
        g.add_argument("--dapg_min_return_pct", type=float, default=0.0,
                        help="Drop demo episodes below this return percentile")
        g.add_argument("--dapg_action_noise", type=float, default=0.05,
                        help="Std of Gaussian noise on demo actions (default: 0.05)")

    @classmethod
    def from_args(cls, args: argparse.Namespace, device: str = "cuda") -> DAPGAuxiliary:
        """Create DAPGAuxiliary from parsed CLI arguments."""
        return cls(
            demo_path=args.dapg_demos,
            bc_weight=args.dapg_weight,
            bc_decay_steps=args.dapg_decay,
            bc_batch_size=args.dapg_batch_size,
            min_return_percentile=args.dapg_min_return_pct,
            action_noise_std=args.dapg_action_noise,
            device=device,
        )
