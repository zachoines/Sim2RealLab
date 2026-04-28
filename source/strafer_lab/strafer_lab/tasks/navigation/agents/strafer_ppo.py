"""Modular PPO update loop with auxiliary loss plugin architecture.

Replaces the fragile per-auxiliary monkey-patching pattern (where each
auxiliary copies the entire PPO.update() method) with a single patched
loop that calls registered auxiliary modules at a well-defined hook point.

The patched update is identical to upstream RSL-RL PPO.update() except for
one insertion point after surrogate_loss / value_loss / entropy are computed:

    aux_total = 0
    for aux in _AUXILIARIES:
        aux_loss, aux_metrics = aux.compute(ppo, obs_batch, batch_size)
        aux_total += aux_loss

    loss = surrogate + value_coef * value - entropy_coef * entropy + aux_total

Usage:
    from strafer_lab.tasks.navigation.agents.strafer_ppo import (
        install_strafer_ppo, register_auxiliary,
    )
    from strafer_lab.tasks.navigation.agents.aux_dapg import DAPGAuxiliary

    install_strafer_ppo()  # patches PPO.update() once
    register_auxiliary(DAPGAuxiliary(...))
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AuxiliaryLoss(ABC):
    """Interface for auxiliary losses injected into the PPO mini-batch loop."""

    @abstractmethod
    def compute(
        self,
        ppo,
        obs_batch,
        original_batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (scalar loss to add to total, dict of metrics to log).

        Args:
            ppo: The PPO algorithm instance (``rsl_rl.algorithms.ppo.PPO``).
            obs_batch: TensorDict of observations for the current mini-batch.
            original_batch_size: Batch size before any symmetric augmentation.
        """
        ...

    def on_update_start(self, ppo) -> None:
        """Called once per PPO.update(), before the mini-batch loop."""
        pass

    def on_update_end(self, ppo) -> None:
        """Called once per PPO.update(), after the mini-batch loop."""
        pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_AUXILIARIES: list[AuxiliaryLoss] = []


def register_auxiliary(aux: AuxiliaryLoss) -> None:
    """Register an auxiliary loss module to be called during PPO updates."""
    _AUXILIARIES.append(aux)
    print(f"[StraferPPO] Registered auxiliary: {type(aux).__name__}")


def get_auxiliaries() -> list[AuxiliaryLoss]:
    """Return the list of registered auxiliary modules."""
    return _AUXILIARIES


# ---------------------------------------------------------------------------
# Monkey-patching PPO.update()
# ---------------------------------------------------------------------------

_INSTALLED = False


def install_strafer_ppo() -> None:
    """Monkey-patch PPO.update() once to support auxiliary losses.

    Must be called BEFORE creating the OnPolicyRunner. The patched update
    loop is identical to upstream RSL-RL except for the auxiliary hook point.
    Safe to call multiple times (only patches once).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from rsl_rl.algorithms.ppo import PPO

    _original_update = PPO.update

    def _update_with_auxiliaries(self) -> dict[str, float]:
        """PPO.update() with auxiliary loss plugin support.

        Mirrors upstream rsl_rl 5.0.1 PPO.update() (actor/critic split,
        Batch namedtuple, distribution_params-based KL) with one insertion
        point for auxiliary losses after the standard PPO losses are computed.
        """
        auxiliaries = _AUXILIARIES

        # If no auxiliaries registered, use original (faster, no copy overhead)
        if not auxiliaries:
            return _original_update(self)

        # Notify auxiliaries that a new update cycle is starting
        for aux in auxiliaries:
            aux.on_update_start(self)

        # ---- Replicate PPO.update() with auxiliary injection ----
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # Beta distribution diagnostics (populated only for AffineBeta policies)
        beta_diag_accum: dict[str, float] = {}
        # Accumulate auxiliary metrics across mini-batches
        aux_metrics_accum: dict[str, float] = {}
        # RND / symmetry
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator (rsl_rl 5.0.1: yields Batch namedtuple)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Per-mini-batch advantage normalization
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            # Symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    obs=batch.observations, actions=batch.actions, env=self.symmetry["_env"],
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Forward pass: actor (stochastic) + critic
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive learning rate (KL divergence)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
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
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # ---- Auxiliary losses (DAPG, GAIL, etc.) ----
            aux_total = torch.tensor(0.0, device=self.device)
            for aux in auxiliaries:
                aux_loss, aux_metrics = aux.compute(self, batch.observations, original_batch_size)
                aux_total = aux_total + aux_loss
                for k, v in aux_metrics.items():
                    aux_metrics_accum[k] = aux_metrics_accum.get(k, 0.0) + v

            # Combined loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + aux_total
            )

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Accumulate metrics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

            # Beta distribution diagnostics
            dist = getattr(self.actor, "distribution", None)
            if dist is not None and hasattr(dist, "base_dist"):
                bd = dist.base_dist
                c1 = bd.concentration1[:original_batch_size]
                c0 = bd.concentration0[:original_batch_size]
                beta_diag_accum["beta/concentration1_mean"] = (
                    beta_diag_accum.get("beta/concentration1_mean", 0.0) + c1.mean().item()
                )
                beta_diag_accum["beta/concentration0_mean"] = (
                    beta_diag_accum.get("beta/concentration0_mean", 0.0) + c0.mean().item()
                )
                beta_diag_accum["beta/concentration_max"] = max(
                    beta_diag_accum.get("beta/concentration_max", 0.0),
                    c1.max().item(),
                    c0.max().item(),
                )
                # Use first distribution param (mu) as action_std proxy
                if len(distribution_params) >= 2:
                    beta_diag_accum["beta/action_std_mean"] = (
                        beta_diag_accum.get("beta/action_std_mean", 0.0)
                        + distribution_params[1].mean().item()
                    )

        # Average over all mini-batch updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Notify auxiliaries that the update cycle is complete
        for aux in auxiliaries:
            aux.on_update_end(self)

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        # Beta distribution diagnostics (averaged, except concentration_max)
        for k, v in beta_diag_accum.items():
            if k == "beta/concentration_max":
                loss_dict[k] = v
            else:
                loss_dict[k] = v / num_updates
        # Average auxiliary metrics
        for k, v in aux_metrics_accum.items():
            loss_dict[k] = v / num_updates
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    PPO.update = _update_with_auxiliaries
    print("[StraferPPO] Installed auxiliary loss support into PPO.update()")
