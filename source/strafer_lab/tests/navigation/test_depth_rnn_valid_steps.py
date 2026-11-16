"""Pure (no-Kit) checks of the recurrent update's depth encoding at valid steps only.

rsl_rl's recurrent mini-batch pads every trajectory to the rollout length.
``StraferDepthRNNModel`` runs its DeFM encoder on the valid steps alone and
leaves the padded steps at zero. These tests compare it with a padded-batch
reference that encodes every padded row (``_padded_batch_get_latent``), on
padded rollouts built by rsl_rl's own ``act`` and ``process_env_step``:

- the valid rows' embeddings, the latent, and the actor's and critic's
  outputs match the reference in every mini-batch;
- one ``PPO.update`` step from identical parameters gives the reference's
  losses and gradients, except the depth projection's gradients, which differ
  at most by the order of one float32 sum. From the first optimizer step on,
  the gradient-norm clip and Adam can carry that difference into any parameter
  of the same model by an ulp, so the parameters after the step are held to a
  stated bound;
- values at padded steps cannot reach the loss;
- without masks, and with the CNN encoder however it is built, every row is
  encoded as in the reference.

The DeFM backbone is a frozen, deterministic, DeFM-shaped stand-in (per-image
preprocess, conv, GroupNorm, SiLU, pooling, LayerNorm); the real one needs a
torch.hub download. The exact equalities are the stand-in's on the CPU: the real
backbone's arithmetic on a row moves by rounding with the batch size, and its
train-mode stochastic depth is beyond what these tests can see.
"""

from __future__ import annotations

import copy
import itertools
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

from strafer_lab.tasks.navigation.agents import depth_encoders, depth_rnn_model
from strafer_lab.tasks.navigation.agents.depth_encoders import DeFMDepthEncoder, DepthEncoder
from strafer_lab.tasks.navigation.agents.depth_rnn_model import StraferDepthRNNModel
from strafer_lab.tasks.navigation.agents.distributions import BetaDistributionCfg

_DEPTH_DIM = depth_rnn_model._DEFAULT_DEPTH_OBS_DIM
_SCALAR_DIM = 19
_NUM_ENVS = 4
_STEPS = 8
_NUM_MINI_BATCHES = 2
_NUM_EPOCHS = 2
_LEARNING_RATE = 3.0e-4
_ROLLOUT_SEED = 0

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


def _padded_batch_get_latent(self, obs, masks=None, hidden_state=None):
    """The padded-batch reference: every row of the mini-batch, padding included, goes through the encoder."""
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    raw = torch.cat(obs_list, dim=-1)
    if self._has_depth and self.depth_encoder is not None:
        encoded = self._encode_depth(raw)
    else:
        encoded = raw
    encoded = self.obs_normalizer(encoded)
    encoded = self.rnn(encoded, masks, hidden_state).squeeze(0)
    return encoded


def _use_padded_batch_reference(*models: StraferDepthRNNModel) -> None:
    for model in models:
        model.get_latent = types.MethodType(_padded_batch_get_latent, model)


def _rollout(seed: int = _ROLLOUT_SEED) -> PPO:
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
        learning_rate=_LEARNING_RATE,
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


def _mini_batches(alg: PPO) -> list[RolloutStorage.Batch]:
    return list(alg.storage.recurrent_mini_batch_generator(_NUM_MINI_BATCHES, 1))


def _only_mini_batch(alg: PPO, index: int) -> None:
    """Make PPO.update take a single optimizer step, on mini-batch ``index`` of the first epoch."""
    generator = alg.storage.recurrent_mini_batch_generator
    alg.storage.recurrent_mini_batch_generator = lambda *args: itertools.islice(generator(*args), index, index + 1)


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


def _trainable(alg: PPO) -> list[tuple[str, torch.Tensor]]:
    return [
        (f"{tag}.{name}", p)
        for tag, model in (("actor", alg.actor), ("critic", alg.critic))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def test_the_rollout_pads_the_mini_batches_unevenly(stand_in_defm) -> None:
    """Guard on the fixture: each mini-batch carries padding, and the two pad to different widths."""
    widths = []
    for batch in _mini_batches(_rollout()):
        assert batch.masks.shape[0] == _STEPS
        assert int(batch.masks.sum()) == _STEPS * _NUM_ENVS // _NUM_MINI_BATCHES
        assert not bool(batch.masks.all())
        widths.append(batch.masks.shape[1])
    assert widths == [3, 7]


def test_padded_steps_enter_the_normalizer_as_exact_zeros(stand_in_defm) -> None:
    """The encoded mini-batch holds each valid row's encoding in place and exact zeros at every padded step."""
    alg = _rollout()
    for batch in _mini_batches(alg):
        raw = torch.cat([batch.observations[g] for g in alg.actor.obs_groups], dim=-1)
        with torch.no_grad():
            encoded = alg.actor._encode_valid_steps(raw, batch.masks)
            valid = alg.actor._encode_depth(raw[batch.masks])
        assert encoded.shape == (*batch.masks.shape, valid.shape[-1])
        assert bool((encoded[~batch.masks] == 0).all())
        assert torch.equal(encoded[batch.masks], valid)


def test_the_encoder_sees_only_the_valid_steps_and_matches_the_padded_batch_reference(stand_in_defm) -> None:
    """In every mini-batch the encoder takes T x envs rows, and each valid row's embedding and the latent match the
    padded-batch reference."""
    alg = _rollout()
    reference = copy.deepcopy(alg.actor)
    _use_padded_batch_reference(reference)
    calls, reference_calls = _encoder_calls(alg.actor), _encoder_calls(reference)

    for i, batch in enumerate(_mini_batches(alg)):
        with torch.no_grad():
            latent = alg.actor.get_latent(batch.observations, batch.masks, batch.hidden_states[0])
            reference_latent = reference.get_latent(batch.observations, batch.masks, batch.hidden_states[0])
        steps, trajectories = batch.masks.shape
        assert calls[i].shape[0] == int(batch.masks.sum()) == _STEPS * _NUM_ENVS // _NUM_MINI_BATCHES
        assert reference_calls[i].shape[0] == steps * trajectories
        assert torch.equal(calls[i], reference_calls[i].view(steps, trajectories, -1)[batch.masks])
        assert torch.equal(latent, reference_latent)
    assert len(calls) == len(reference_calls) == _NUM_MINI_BATCHES


def test_actor_and_critic_outputs_on_valid_steps_match_the_padded_batch_reference(stand_in_defm) -> None:
    """In every mini-batch the policy's distribution parameters, log-probabilities and entropy, and the values, match
    the padded-batch reference."""
    alg = _rollout()
    reference_actor, reference_critic = copy.deepcopy(alg.actor), copy.deepcopy(alg.critic)
    _use_padded_batch_reference(reference_actor, reference_critic)

    def outputs(actor, critic, batch):
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

    for batch in _mini_batches(alg):
        got = outputs(alg.actor, alg.critic, batch)
        want = outputs(reference_actor, reference_critic, batch)
        assert got[-1].shape == (_STEPS, _NUM_ENVS // _NUM_MINI_BATCHES, 1)
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("index", range(_NUM_MINI_BATCHES), ids=["narrow-mini-batch", "wide-mini-batch"])
def test_one_ppo_step_matches_the_padded_batch_reference(stand_in_defm, monkeypatch, index) -> None:
    """One rsl_rl ``PPO.update`` step from identical parameters, on one mini-batch.

    The losses and every gradient match the padded-batch reference, except the depth projection's weight and bias
    gradients. Those are sums over the encoder's rows, and the reference sums the same per-row terms plus exact
    zeros for its padded rows, so they can differ only by the order of a float32 sum. The test checks that the
    terms are identical and the zeros exact, and bounds the difference by the float32 summation error:
    (N - 1) eps sum|term| over N terms, and N eps sum|term| for the weight's products.

    The gradient-norm clip scales every gradient of a model by its total norm, and an Adam step moves a parameter
    by at most about the learning rate, so the parameters after the step are bounded rather than equal: within
    4 eps (|p| + lr) of the reference's.

    The comparison starts after one ordinary update: the Beta head's output layer starts at zero weight, so the
    actor's first step reaches nothing below it.
    """
    alg = _rollout()
    reference = copy.deepcopy(alg)
    for a in (alg, reference):
        torch.manual_seed(1)
        a.update()
    for (name, p), (_, p_reference) in zip(_trainable(alg), _trainable(reference), strict=True):
        assert torch.equal(p, p_reference), name
    _use_padded_batch_reference(reference.actor, reference.critic)
    valid = _mini_batches(alg)[index].masks.reshape(-1)
    for a in (alg, reference):
        _only_mini_batch(a, index)
    rows = {
        (run, tag): _projection_rows(getattr(a, tag))
        for run, a in (("valid-steps", alg), ("reference", reference))
        for tag in ("actor", "critic")
    }
    recorded = _record_gradients(monkeypatch, alg, reference)

    torch.manual_seed(2)
    losses = alg.update()
    grads = list(recorded)
    recorded.clear()
    torch.manual_seed(2)
    reference_losses = reference.update()
    reference_grads = list(recorded)

    assert losses == reference_losses
    assert len(grads) == len(reference_grads) == 2
    for g, tag in zip(grads, ("actor", "critic"), strict=True):
        assert g[f"{tag}.depth_encoder.projection.0.weight"].abs().sum() > 0, f"no gradient reached the {tag}'s encoder"
    eps = torch.finfo(torch.float32).eps
    for g, g_reference in zip(grads, reference_grads, strict=True):
        assert g.keys() == g_reference.keys()
        tag = next(iter(g)).split(".")[0]
        (x,), (d,) = rows[("valid-steps", tag)]
        (x_reference,), (d_reference,) = rows[("reference", tag)]
        assert torch.equal(x_reference[valid], x)
        assert torch.equal(d_reference[valid], d)
        assert bool((d_reference[~valid] == 0).all())
        n = d_reference.shape[0]
        abs_d, abs_x = d_reference.abs().double(), x_reference.abs().double()
        bound = {
            f"{tag}.depth_encoder.projection.0.bias": (n - 1) * eps * abs_d.sum(0),
            f"{tag}.depth_encoder.projection.0.weight": n * eps * abs_d.T @ abs_x,
        }
        for name, grad in g.items():
            if name in bound:
                assert bool(((grad - g_reference[name]).abs().double() <= bound[name]).all()), name
            else:
                assert torch.equal(grad, g_reference[name]), name
    for (name, p), (_, p_reference) in zip(_trainable(alg), _trainable(reference), strict=True):
        tolerance = 4 * eps * (p_reference.abs() + _LEARNING_RATE)
        assert bool(((p - p_reference).abs() <= tolerance).all()), name


@pytest.mark.parametrize("path", ["padded-batch-reference", "valid-steps"])
def test_values_at_padded_steps_cannot_reach_the_loss(stand_in_defm, monkeypatch, path) -> None:
    """Large random values at the padded steps leave a whole update exactly as with zeros there.

    rsl_rl pads after each trajectory's last valid step (``split_and_pad_trajectories``), the GRU's output at a step
    depends only on earlier steps, and ``RNN.forward`` keeps only the valid steps (``unpad_trajectories``) before the
    heads and the loss. The reference takes the values in its padded observations, through the encoder; the
    valid-steps path takes them in its encoded batch, after the scatter, where they reach the normalizer and the GRU.
    """
    alg = _rollout()
    if path == "padded-batch-reference":
        _use_padded_batch_reference(alg.actor, alg.critic)
    filled = copy.deepcopy(alg)
    generator = torch.Generator().manual_seed(3)
    filled_rows: list[int] = []

    def noise_at(values: torch.Tensor, masks: torch.Tensor) -> None:
        values[~masks] = 100.0 * torch.randn(values[~masks].shape, generator=generator)
        filled_rows.append(int((~masks).sum()))

    if path == "padded-batch-reference":
        pad = split_and_pad_trajectories

        def pad_with_noise(tensor, dones):
            padded, masks = pad(tensor, dones)
            for key in padded.keys():
                noise_at(padded[key], masks)
            return padded, masks

        monkeypatch.setattr(rollout_storage_module, "split_and_pad_trajectories", pad_with_noise)
    else:
        for model in (filled.actor, filled.critic):
            encode = model._encode_valid_steps

            def encode_with_noise(raw, masks, encode=encode):
                encoded = encode(raw, masks).clone()
                noise_at(encoded, masks)
                return encoded

            model._encode_valid_steps = encode_with_noise

    torch.manual_seed(4)
    noise_losses = filled.update()
    monkeypatch.undo()
    torch.manual_seed(4)
    zero_losses = alg.update()

    assert filled_rows and all(n > 0 for n in filled_rows)
    assert noise_losses == zero_losses
    for (name, p_zero), (_, p_noise) in zip(_trainable(alg), _trainable(filled), strict=True):
        assert torch.equal(p_zero, p_noise), name


def test_without_masks_every_row_is_encoded_as_in_the_padded_batch_reference(stand_in_defm) -> None:
    """act/compute_returns call the model without masks: every row is encoded, and the outputs and hidden states
    match the padded-batch reference."""
    generator = torch.Generator().manual_seed(5)
    obs = _obs(generator)
    model = _model(obs, "actor")
    reference = copy.deepcopy(model)
    _use_padded_batch_reference(reference)
    calls = _encoder_calls(model)

    with torch.no_grad():
        for step in range(3):
            step_obs = _obs(generator)
            torch.manual_seed(step)
            got = model(step_obs, stochastic_output=True)
            torch.manual_seed(step)
            want = reference(step_obs, stochastic_output=True)
            assert torch.equal(got, want)
            assert torch.equal(model.get_hidden_state(), reference.get_hidden_state())
    assert [c.shape[0] for c in calls] == [_NUM_ENVS] * 3


@pytest.mark.parametrize("requested", ["cnn", "defm-fallback"])
def test_the_cnn_encoder_keeps_the_padded_batch(monkeypatch, requested) -> None:
    """The CNN's BatchNorm normalizes over its batch, so in the update it sees every padded row, whether it was
    asked for or is DeFM's fallback when the DeFM backbone cannot load."""

    def no_backbone(model_name):
        raise RuntimeError("no DeFM backbone in this test")

    monkeypatch.setattr(depth_encoders, "_load_defm_backbone", no_backbone)
    generator = torch.Generator().manual_seed(6)
    steps = [_obs(generator) for _ in range(_STEPS)]
    sequence = torch.stack([o["policy"] for o in steps])
    dones = torch.zeros(_STEPS, _NUM_ENVS, 1)
    dones[2, 1] = dones[5, 3] = 1
    padded, masks = split_and_pad_trajectories(sequence, dones)
    model = _model(steps[0], "actor", encoder_type="cnn" if requested == "cnn" else "defm")
    assert isinstance(model.depth_encoder, DepthEncoder)
    model.train()
    reference = copy.deepcopy(model)
    _use_padded_batch_reference(reference)
    calls = _encoder_calls(model)
    hidden = torch.zeros(1, masks.shape[1], 16)

    got = model.get_latent(TensorDict({"policy": padded}, batch_size=padded.shape[:2]), masks, hidden)
    want = reference.get_latent(TensorDict({"policy": padded}, batch_size=padded.shape[:2]), masks, hidden)

    assert [c.shape[0] for c in calls] == [masks.numel()]
    assert torch.equal(got, want)
    for b_got, b_want in zip(model.depth_encoder.buffers(), reference.depth_encoder.buffers(), strict=True):
        assert torch.equal(b_got, b_want)
