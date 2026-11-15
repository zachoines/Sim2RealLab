"""Pure (no-Kit) checks that the recurrent update encodes depth at valid steps only.

rsl_rl's recurrent mini-batch pads every trajectory to the rollout length.
``StraferDepthRNNModel`` runs its DeFM encoder on the valid steps alone and
leaves the padded steps at zero. These tests compare that against the model's
previous path, which encoded every padded row (``_padded_path_get_latent``
below, verbatim). They check that the two give the same valid-row embeddings,
actor and critic outputs, PPO losses, gradients and updated parameters through
rsl_rl's own ``PPO.update`` (the depth projection's gradients to within the
order of a float32 sum, bounded), and that padded values cannot reach the loss.
They also check that the rollout path and the CNN encoder are unchanged.

The DeFM backbone is a frozen, deterministic, DeFM-shaped stand-in (per-image
preprocess, conv, GroupNorm, SiLU, pooling, LayerNorm); the real one needs a
torch.hub download. CPU only.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage import rollout_storage as rollout_storage_module
from rsl_rl.utils import split_and_pad_trajectories

from strafer_lab.tasks.navigation.agents import depth_rnn_model
from strafer_lab.tasks.navigation.agents.depth_encoders import DeFMDepthEncoder, DepthEncoder
from strafer_lab.tasks.navigation.agents.depth_rnn_model import StraferDepthRNNModel
from strafer_lab.tasks.navigation.agents.distributions import BetaDistributionCfg

_DEPTH_DIM = depth_rnn_model._DEFAULT_DEPTH_OBS_DIM
_SCALAR_DIM = 19
_OBS_DIM = _SCALAR_DIM + _DEPTH_DIM
_NUM_ENVS = 4
_STEPS = 8
_NUM_MINI_BATCHES = 2
_NUM_EPOCHS = 2

# Episode ends per env (rollout step indices): 0, 1, 2 and 3 ends give 1 to 4
# trajectories per env, so the two mini-batches pad to different widths.
_DONE_STEPS = {0: (), 1: (3,), 2: (1, 5), 3: (0, 2, 6)}


class _StandInBackbone(nn.Module):
    """Frozen DeFM-shaped backbone: every op works on each image alone."""

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 5, stride=4, padding=2)
        self.norm = nn.GroupNorm(4, channels)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.silu(self.norm(self.conv(x)))
        return {"global_backbone": self.out_norm(x.mean(dim=(2, 3)))}


def _stand_in_preprocess(x, target_size, cnn_padding, device):
    """Per-image stand-in for DeFM's ``preprocess_depth_batch`` (log channels, min-max, resize)."""
    log_depth = torch.log1p(torch.clamp(x, min=0.0, max=100.0))
    flat = log_depth.flatten(1)
    lo = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    hi = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    rel = (log_depth - lo) / torch.where(hi > lo, hi - lo, torch.ones_like(hi))
    x3 = torch.cat([log_depth / 4.6, torch.clamp(log_depth / 2.3, 0.0, 1.0), rel], dim=1)
    return F.interpolate(x3, size=(32, 32), mode="bilinear", align_corners=False)


def _stand_in_defm_encoder(output_dim: int) -> DeFMDepthEncoder:
    encoder = DeFMDepthEncoder.__new__(DeFMDepthEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = _StandInBackbone().eval()
    for p in encoder.backbone.parameters():
        p.requires_grad = False
    encoder._preprocess = _stand_in_preprocess
    encoder.projection = nn.Sequential(nn.Linear(8, output_dim), nn.ELU())
    return encoder


def _obs(generator: torch.Generator) -> TensorDict:
    def group() -> torch.Tensor:
        scalar = torch.randn(_NUM_ENVS, _SCALAR_DIM, generator=generator)
        depth = torch.rand(_NUM_ENVS, _DEPTH_DIM, generator=generator)
        return torch.cat([scalar, depth], dim=-1)

    return TensorDict({"policy": group(), "critic": group()}, batch_size=[_NUM_ENVS])


def _model(obs: TensorDict, obs_set: str, encoder_type: str = "defm") -> StraferDepthRNNModel:
    is_actor = obs_set == "actor"
    return StraferDepthRNNModel(
        obs,
        {"actor": ["policy"], "critic": ["critic"]},
        obs_set,
        3 if is_actor else 1,
        hidden_dims=[32, 16],
        obs_normalization=True,
        distribution_cfg=BetaDistributionCfg().to_dict() if is_actor else None,
        rnn_type="gru",
        rnn_hidden_dim=16,
        depth_encoder_type=encoder_type,
        depth_embedding_dim=8,
    )


@pytest.fixture
def stand_in_defm(monkeypatch):
    real = depth_rnn_model.create_depth_encoder

    def create(encoder_type, output_dim, defm_model_name):
        if encoder_type == "defm":
            return _stand_in_defm_encoder(output_dim)
        return real(encoder_type, output_dim, defm_model_name)

    monkeypatch.setattr(depth_rnn_model, "create_depth_encoder", create)


def _padded_path_get_latent(self, obs, masks=None, hidden_state=None):
    """The model's latent before this change: every padded row goes through the encoder."""
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    raw = torch.cat(obs_list, dim=-1)
    if self._has_depth and self.depth_encoder is not None:
        encoded = self._encode_depth(raw)
    else:
        encoded = raw
    encoded = self.obs_normalizer(encoded)
    encoded = self.rnn(encoded, masks, hidden_state).squeeze(0)
    return encoded


def _use_padded_path(*models: StraferDepthRNNModel) -> None:
    for model in models:
        model.get_latent = types.MethodType(_padded_path_get_latent, model)


def _rollout(seed: int = 0) -> PPO:
    """A short PPO rollout through rsl_rl's own act/process_env_step, returns computed."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    obs = _obs(generator)
    actor, critic = _model(obs, "actor"), _model(obs, "critic")
    storage = RolloutStorage("rl", _NUM_ENVS, _STEPS, obs, [3], "cpu")
    alg = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=_NUM_EPOCHS,
        num_mini_batches=_NUM_MINI_BATCHES,
        lam=0.97,
        entropy_coef=0.005,
        learning_rate=3.0e-4,
        max_grad_norm=0.5,
        schedule="fixed",
    )
    alg.train_mode()
    with torch.inference_mode():
        for step in range(_STEPS):
            alg.act(obs)
            obs = _obs(generator)
            rewards = torch.randn(_NUM_ENVS, generator=generator)
            dones = torch.tensor([int(step in _DONE_STEPS[e]) for e in range(_NUM_ENVS)])
            alg.process_env_step(obs, rewards, dones, {})
        alg.compute_returns(obs)
    return alg


def _encoder_calls(model: StraferDepthRNNModel) -> list[torch.Tensor]:
    """Record every output of the model's depth encoder."""
    calls: list[torch.Tensor] = []
    model.depth_encoder.register_forward_hook(lambda _m, _i, out: calls.append(out.detach().clone()))
    return calls


def _record_gradients(monkeypatch, *algs: PPO) -> list[dict[str, torch.Tensor]]:
    """Record the gradients PPO.update hands to clip_grad_norm_ (before clipping), by parameter name."""
    names = {
        id(p): f"{tag}.{name}"
        for alg in algs
        for tag, model in (("actor", alg.actor), ("critic", alg.critic))
        for name, p in model.named_parameters()
    }
    recorded: list[dict[str, torch.Tensor]] = []
    clip = nn.utils.clip_grad_norm_

    def recording_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        recorded.append({names[id(p)]: p.grad.clone() for p in parameters if p.grad is not None})
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", recording_clip)
    return recorded


def _projection_rows(model: StraferDepthRNNModel) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Record the depth projection's input rows and the gradient reaching its output rows, once per pass."""
    linear = model.depth_encoder.projection[0]
    inputs: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    linear.register_forward_hook(lambda _m, args, _out: inputs.append(args[0].detach().clone()))
    linear.register_full_backward_hook(lambda _m, _gin, gout: grads.append(gout[0].detach().clone()))
    return inputs, grads


def _first_mini_batch(alg: PPO) -> RolloutStorage.Batch:
    return next(alg.storage.recurrent_mini_batch_generator(_NUM_MINI_BATCHES, 1))


def _trainable(alg: PPO) -> list[tuple[str, torch.Tensor]]:
    return [
        (f"{tag}.{name}", p)
        for tag, model in (("actor", alg.actor), ("critic", alg.critic))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def test_the_rollout_pads_the_mini_batches_unevenly(stand_in_defm) -> None:
    """Guard on the fixture: each mini-batch carries padding, and the two pad to different widths."""
    alg = _rollout()
    widths = []
    for batch in alg.storage.recurrent_mini_batch_generator(_NUM_MINI_BATCHES, 1):
        assert batch.masks.shape[0] == _STEPS
        assert int(batch.masks.sum()) == _STEPS * _NUM_ENVS // _NUM_MINI_BATCHES
        assert not bool(batch.masks.all())
        widths.append(batch.masks.shape[1])
    assert widths == [3, 7]


def test_the_encoder_sees_only_the_valid_steps_and_embeds_them_as_before(stand_in_defm) -> None:
    """Every valid row's embedding equals the padded path's; the encoder's batch is T x envs rows."""
    alg = _rollout()
    batch = _first_mini_batch(alg)
    present = copy.deepcopy(alg.actor)
    _use_padded_path(present)
    new_calls, present_calls = _encoder_calls(alg.actor), _encoder_calls(present)

    with torch.no_grad():
        new_latent = alg.actor.get_latent(batch.observations, batch.masks, batch.hidden_states[0])
        present_latent = present.get_latent(batch.observations, batch.masks, batch.hidden_states[0])

    steps, trajectories = batch.masks.shape
    assert [c.shape[0] for c in new_calls] == [int(batch.masks.sum())]
    assert [c.shape[0] for c in present_calls] == [steps * trajectories]
    present_valid = present_calls[0].view(steps, trajectories, -1)[batch.masks]
    assert torch.equal(new_calls[0], present_valid)
    assert torch.equal(new_latent, present_latent)


def test_actor_and_critic_outputs_on_valid_steps_are_unchanged(stand_in_defm) -> None:
    """The policy's distribution parameters, log-probabilities, entropy and the values are identical."""
    alg = _rollout()
    batch = _first_mini_batch(alg)
    present_actor, present_critic = copy.deepcopy(alg.actor), copy.deepcopy(alg.critic)
    _use_padded_path(present_actor, present_critic)

    def outputs(actor, critic):
        with torch.no_grad():
            torch.manual_seed(1)
            actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            values = critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            return (
                *actor.output_distribution_params,
                actor.get_output_log_prob(batch.actions),
                actor.output_entropy,
                values,
            )

    new, present = outputs(alg.actor, alg.critic), outputs(present_actor, present_critic)
    assert new[-1].shape == (_STEPS, _NUM_ENVS // _NUM_MINI_BATCHES, 1)
    for a, b in zip(new, present, strict=True):
        assert torch.equal(a, b)


def test_ppo_update_losses_gradients_and_parameters_are_unchanged(stand_in_defm, monkeypatch) -> None:
    """rsl_rl's own PPO.update over every mini-batch and epoch: same losses, gradients and new weights.

    Every gradient is identical except the depth projection's weight and bias, which are sums over the
    encoder's rows. The padded path sums the same per-row terms plus exact zeros for the padded rows, so
    only the order of a float32 sum differs. The test checks that the terms are identical and the zeros
    exact, and bounds the difference by the float32 summation error, (N - 1) eps sum|term| over N terms
    (N eps sum|term| for the weight's products).
    """
    alg = _rollout()
    present = copy.deepcopy(alg)
    _use_padded_path(present.actor, present.critic)
    valid_rows = [
        b.masks.reshape(-1) for b in alg.storage.recurrent_mini_batch_generator(_NUM_MINI_BATCHES, _NUM_EPOCHS)
    ]
    rows = {
        (run, tag): _projection_rows(getattr(a, tag))
        for run, a in (("new", alg), ("present", present))
        for tag in ("actor", "critic")
    }
    recorded = _record_gradients(monkeypatch, alg, present)
    projection_before = alg.actor.depth_encoder.projection[0].weight.detach().clone()

    torch.manual_seed(2)
    new_losses = alg.update()
    new_grads = list(recorded)
    recorded.clear()
    torch.manual_seed(2)
    present_losses = present.update()
    present_grads = list(recorded)

    assert new_losses == present_losses
    assert len(new_grads) == len(present_grads) == 2 * len(valid_rows)
    assert not torch.equal(alg.actor.depth_encoder.projection[0].weight, projection_before)
    eps = torch.finfo(torch.float32).eps
    for step, (g_new, g_present) in enumerate(zip(new_grads, present_grads, strict=True)):
        assert g_new.keys() == g_present.keys()
        tag, batch = next(iter(g_new)).split(".")[0], step // 2
        valid = valid_rows[batch]
        x_new, x_present = rows[("new", tag)][0][batch], rows[("present", tag)][0][batch]
        d_new, d_present = rows[("new", tag)][1][batch], rows[("present", tag)][1][batch]
        assert torch.equal(x_present[valid], x_new)
        assert torch.equal(d_present[valid], d_new)
        assert bool((d_present[~valid] == 0).all())
        n = d_present.shape[0]
        bound = {
            f"{tag}.depth_encoder.projection.0.bias": (n - 1) * eps * d_present.abs().double().sum(0),
            f"{tag}.depth_encoder.projection.0.weight": n * eps * d_present.abs().double().T @ x_present.abs().double(),
        }
        for name, g in g_new.items():
            if name in bound:
                assert bool(((g - g_present[name]).abs().double() <= bound[name]).all()), name
            else:
                assert torch.equal(g, g_present[name]), name
    for (name, p_new), (_, p_present) in zip(_trainable(alg), _trainable(present), strict=True):
        assert torch.equal(p_new, p_present), name


@pytest.mark.parametrize("padded_path", [True, False], ids=["padded-path", "valid-steps"])
def test_values_at_padded_steps_cannot_reach_the_loss(stand_in_defm, monkeypatch, padded_path) -> None:
    """Large random values in place of the zero padding leave the update exactly as it was.

    rsl_rl pads after each trajectory's last valid step (``split_and_pad_trajectories``), the GRU's
    output at a step depends only on earlier steps, and ``RNN.forward`` keeps only the valid steps
    (``unpad_trajectories``) before the heads and the loss.
    """
    alg = _rollout()
    if padded_path:
        _use_padded_path(alg.actor, alg.critic)
    filled = copy.deepcopy(alg)
    generator = torch.Generator().manual_seed(3)
    pad = split_and_pad_trajectories
    filled_rows: list[int] = []

    def pad_with_noise(tensor, dones):
        padded, masks = pad(tensor, dones)
        for key in padded.keys():
            value = padded[key]
            value[~masks] = 100.0 * torch.randn(value[~masks].shape, generator=generator)
            filled_rows.append(int((~masks).sum()))
        return padded, masks

    torch.manual_seed(4)
    zero_losses = alg.update()
    monkeypatch.setattr(rollout_storage_module, "split_and_pad_trajectories", pad_with_noise)
    torch.manual_seed(4)
    noise_losses = filled.update()

    assert filled_rows and all(n > 0 for n in filled_rows)
    assert noise_losses == zero_losses
    for (name, p_zero), (_, p_noise) in zip(_trainable(alg), _trainable(filled), strict=True):
        assert torch.equal(p_zero, p_noise), name


def test_the_rollout_path_without_masks_is_unchanged(stand_in_defm) -> None:
    """act/compute_returns call the model without masks: same outputs and hidden states, every row encoded."""
    generator = torch.Generator().manual_seed(5)
    obs = _obs(generator)
    model = _model(obs, "actor")
    present = copy.deepcopy(model)
    _use_padded_path(present)
    calls = _encoder_calls(model)

    with torch.no_grad():
        for step in range(3):
            step_obs = _obs(generator)
            torch.manual_seed(step)
            new = model(step_obs, stochastic_output=True)
            torch.manual_seed(step)
            old = present(step_obs, stochastic_output=True)
            assert torch.equal(new, old)
            assert torch.equal(model.get_hidden_state(), present.get_hidden_state())
    assert [c.shape[0] for c in calls] == [_NUM_ENVS] * 3


def test_the_cnn_encoder_keeps_the_padded_batch() -> None:
    """The CNN's BatchNorm normalizes over its batch, so in the update it still sees every padded row."""
    generator = torch.Generator().manual_seed(6)
    steps = [_obs(generator) for _ in range(_STEPS)]
    sequence = torch.stack([o["policy"] for o in steps])
    dones = torch.zeros(_STEPS, _NUM_ENVS, 1)
    dones[2, 1] = dones[5, 3] = 1
    padded, masks = split_and_pad_trajectories(sequence, dones)
    model = _model(steps[0], "actor", encoder_type="cnn")
    assert isinstance(model.depth_encoder, DepthEncoder)
    model.train()
    present = copy.deepcopy(model)
    _use_padded_path(present)
    calls = _encoder_calls(model)
    hidden = torch.zeros(1, masks.shape[1], 16)

    new = model.get_latent(TensorDict({"policy": padded}, batch_size=padded.shape[:2]), masks, hidden)
    old = present.get_latent(TensorDict({"policy": padded}, batch_size=padded.shape[:2]), masks, hidden)

    assert [c.shape[0] for c in calls] == [masks.numel()]
    assert torch.equal(new, old)
    for b_new, b_old in zip(model.depth_encoder.buffers(), present.depth_encoder.buffers(), strict=True):
        assert torch.equal(b_new, b_old)
