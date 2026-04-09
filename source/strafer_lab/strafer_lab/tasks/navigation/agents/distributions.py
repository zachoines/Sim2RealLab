"""Affine-Beta distribution for rsl_rl 5.0.

Wraps the ``AffineBeta`` helper (Beta on ``[0, 1]`` mapped to ``[-1, 1]``)
into the ``rsl_rl.modules.distribution.Distribution`` interface so it can
be used as a drop-in ``distribution_cfg`` on any model config.

The MLP outputs ``2 * num_actions`` logits which are split into two halves,
softplus'd (plus a floor), and interpreted as Beta concentration parameters
``(α, β)``.  A symmetric initialization ensures a zero-mean policy at the
start of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from rsl_rl.modules.distribution import Distribution

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

# ---------------------------------------------------------------------------
# Helpers (shared with strafer_network — keep duplicated to avoid circular
# imports; these are trivial one-liners)
# ---------------------------------------------------------------------------

_LOG_TWO = torch.log(torch.tensor(2.0))


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive inputs."""
    return x + torch.log(-torch.expm1(-x))


def _symmetric_beta_concentration_for_std(std: float) -> float:
    """Return alpha=beta concentration giving the requested std on [-1, 1]."""
    std = float(max(min(std, 0.999), 1.0e-3))
    return 0.5 * ((1.0 / (std * std)) - 1.0)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


class AffineBetaDistribution(Distribution):
    """Beta distribution affinely mapped from ``[0, 1]`` to ``[-1, 1]``.

    Provides true bounded actions with well-defined entropy, avoiding the
    approximation errors of clipped Gaussian distributions.

    The MLP must output shape ``[..., 2, output_dim]`` (flat logits reshaped
    internally).  Each action dimension gets its own ``(α, β)`` pair.

    Constructor args (passed as ``**kwargs`` after ``class_name`` is popped
    from the config dict):

    Args:
        output_dim: Number of action dimensions.
        init_std: Desired initial standard deviation on ``[-1, 1]``.
        min_concentration: Floor for concentration parameters (prevents
            degenerate U-shaped distributions).
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.3,
        min_concentration: float = 0.2,
    ) -> None:
        super().__init__(output_dim)
        self.min_concentration = min_concentration

        # Pre-compute base concentration for weight init
        self._base_concentration = max(
            _symmetric_beta_concentration_for_std(init_std),
            self.min_concentration + 1.0e-4,
        )

        # Populated by update()
        self._concentration1: torch.Tensor | None = None
        self._concentration0: torch.Tensor | None = None
        self._beta: Beta | None = None

        # Disable args validation for speedup
        Beta.set_default_validate_args(False)

    # -- Distribution interface -----------------------------------------------

    @property
    def input_dim(self) -> list[int]:
        """MLP must output ``[2, output_dim]`` logits (alpha, beta halves)."""
        return [2, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        """Parse MLP logits into Beta concentration parameters.

        Args:
            mlp_output: Shape ``(..., 2, output_dim)``.
        """
        alpha_logits = mlp_output[..., 0, :]
        beta_logits = mlp_output[..., 1, :]
        self._concentration1 = F.softplus(alpha_logits) + self.min_concentration
        self._concentration0 = F.softplus(beta_logits) + self.min_concentration
        self._beta = Beta(self._concentration1, self._concentration0)

    def sample(self) -> torch.Tensor:
        """Reparameterized sample mapped to ``[-1, 1]``."""
        return self._beta.rsample().mul(2.0).sub(1.0)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Compute Beta mean on ``[-1, 1]`` from raw MLP logits."""
        alpha_logits = mlp_output[..., 0, :]
        beta_logits = mlp_output[..., 1, :]
        c1 = F.softplus(alpha_logits) + self.min_concentration
        c0 = F.softplus(beta_logits) + self.min_concentration
        return (c1 / (c1 + c0)).mul(2.0).sub(1.0)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module computing the Beta mean."""
        return _BetaDeterministicOutput(self.min_concentration)

    @property
    def mean(self) -> torch.Tensor:
        return self._beta.mean.mul(2.0).sub(1.0)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self._beta.variance).mul(2.0)

    @property
    def entropy(self) -> torch.Tensor:
        return (
            self._beta.entropy() + _LOG_TWO.to(self._concentration1.device, self._concentration1.dtype)
        ).sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._concentration1, self._concentration0)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(outputs.dtype).eps
        unit_value = outputs.add(1.0).mul(0.5)
        unit_value = torch.clamp(unit_value, min=eps, max=1.0 - eps)
        return (
            self._beta.log_prob(unit_value) - _LOG_TWO.to(outputs.device, outputs.dtype)
        ).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_dist = Beta(old_params[0], old_params[1])
        new_dist = Beta(new_params[0], new_params[1])
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the MLP's last layer for a symmetric zero-mean Beta prior."""
        # Find the last Linear layer (rsl_rl MLP ends with Linear, no trailing activation)
        last_linear = None
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            return

        bias_val = _inverse_softplus(
            torch.tensor(self._base_concentration - self.min_concentration, dtype=last_linear.bias.dtype)
        ).item()

        nn.init.zeros_(last_linear.weight)
        with torch.no_grad():
            last_linear.bias.fill_(bias_val)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


class _BetaDeterministicOutput(nn.Module):
    """Exportable module that computes the Beta mean on ``[-1, 1]``."""

    def __init__(self, min_concentration: float) -> None:
        super().__init__()
        self.min_concentration = min_concentration

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        c1 = F.softplus(mlp_output[..., 0, :]) + self.min_concentration
        c0 = F.softplus(mlp_output[..., 1, :]) + self.min_concentration
        return (c1 / (c1 + c0)).mul(2.0).sub(1.0)


# ---------------------------------------------------------------------------
# Isaac Lab config
# ---------------------------------------------------------------------------


@configclass
class BetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Configuration for the affine Beta distribution on ``[-1, 1]``."""

    class_name: str = "strafer_lab.tasks.navigation.agents.distributions:AffineBetaDistribution"
    """Fully-qualified class resolved by ``rsl_rl.utils.resolve_callable()``."""

    init_std: float = 0.3
    """Initial standard deviation on ``[-1, 1]``.  Controls the symmetric
    Beta concentration at the start of training."""

    min_concentration: float = 0.2
    """Floor for concentration parameters α, β.  Prevents degenerate
    U-shaped distributions that would collapse to boundary modes."""
