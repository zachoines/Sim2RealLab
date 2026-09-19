"""The per-environment subpixel disparity noise band.

``disparity_noise_px`` sets one stereo noise amplitude for the whole batch, so
every training frame carries per-pixel texture of the same size. The deploy
path's 8x8 block median produces frames with next to none, and a policy that
only ever saw one amplitude can read its presence as a feature of simulation.
``disparity_noise_px_range`` draws the amplitude per environment at reset, the
way the observation latency band already does, so the distribution reaches down
to the featureless field the deploy path actually delivers.

The band is a reparameterisation, not a second model: these tests pin that
leaving it unset changes nothing at all — not the output bits, not the RNG
state — and that a band pinned to a single value reproduces the fixed path for
that value exactly.

Pure torch on CPU — no Isaac Sim, no Kit boot. The same equivalences on CUDA are
measured in the record rather than gated here, so the suite needs no device.
"""

from __future__ import annotations

import pytest
import torch

from strafer_lab.tasks.navigation.mdp.noise_models import (
    DepthNoiseModel,
    DepthNoiseModelCfg,
)
from strafer_lab.tasks.navigation.sim_real_cfg import (
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    get_depth_noise,
)

DEVICE = "cpu"
HEIGHT, WIDTH = 4, 16
PIXELS = HEIGHT * WIDTH
NUM_ENVS = 8
TIERS = {"realistic": REAL_ROBOT_CONTRACT, "robust": ROBUST_TRAINING_CONTRACT}


def _cfg(contract, **overrides) -> DepthNoiseModelCfg:
    """A tier's depth noise over a toy frame, with the temporal terms stilled.

    Drops, holds and latency all replay a previous frame, which would mask the
    per-frame amplitude this suite is about.
    """
    cfg = get_depth_noise(contract)
    cfg.height, cfg.width = HEIGHT, WIDTH
    cfg.frame_drop_prob = 0.0
    cfg.failure_probability = 0.0
    cfg.hold_fraction_range = (0.0, 0.0)
    cfg.latency_steps = 0
    cfg.latency_steps_range = None
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _frames(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(NUM_ENVS, PIXELS, generator=generator) * 5.5 + 0.3


def _emit(cfg: DepthNoiseModelCfg, data: torch.Tensor, steps: int = 6,
          seed: int = 11) -> torch.Tensor:
    model = DepthNoiseModel(cfg, NUM_ENVS, DEVICE)
    torch.manual_seed(seed)
    return torch.stack([model(data.clone()) for _ in range(steps)])


class TestTheNeutralPath:
    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_no_band_is_bit_identical_to_the_fixed_amplitude(self, tier):
        data = _frames()
        assert torch.equal(
            _emit(_cfg(TIERS[tier]), data),
            _emit(_cfg(TIERS[tier], disparity_noise_px_range=None), data),
        )

    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_no_band_draws_no_randomness_of_its_own(self, tier):
        """Construction and reset must leave the generator where they found it,
        or every downstream draw in the episode shifts."""
        cfg = _cfg(TIERS[tier], disparity_noise_px_range=None)
        torch.manual_seed(29)
        before = torch.get_rng_state().clone()
        model = DepthNoiseModel(cfg, NUM_ENVS, DEVICE)
        model.reset(None)
        model.reset(torch.arange(3))
        assert torch.equal(before, torch.get_rng_state())

    def test_the_shipped_tiers_declare_no_band(self):
        """The band is wired to nothing until a training run turns it on."""
        for contract in TIERS.values():
            assert contract.sensors.depth_camera.disparity_noise_px_range is None
            assert get_depth_noise(contract).disparity_noise_px_range is None


class TestTheBandIsAReparameterisation:
    @pytest.mark.parametrize("sigma", [0.0, 0.08, 0.16])
    def test_a_band_pinned_to_one_value_reproduces_that_fixed_amplitude(self, sigma):
        data = _frames()
        fixed = _emit(_cfg(REAL_ROBOT_CONTRACT, disparity_noise_px=sigma), data)
        # The fixed field is left at a value the band cannot produce, so a
        # model reading the wrong one of the two cannot pass.
        pinned = _emit(_cfg(REAL_ROBOT_CONTRACT, disparity_noise_px=0.5,
                            disparity_noise_px_range=(sigma, sigma)), data)
        assert torch.equal(fixed, pinned)

    def test_a_reversed_band_is_read_as_the_interval_it_names(self):
        data = _frames()
        forward = _emit(_cfg(REAL_ROBOT_CONTRACT,
                             disparity_noise_px_range=(0.0, 0.16)), data)
        reversed_ = _emit(_cfg(REAL_ROBOT_CONTRACT,
                               disparity_noise_px_range=(0.16, 0.0)), data)
        assert torch.equal(forward, reversed_)

    def test_a_negative_end_cannot_produce_a_negative_amplitude(self):
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(-0.05, 0.0)),
            NUM_ENVS, DEVICE)
        assert model._disparity_range == (0.0, 0.0)
        assert torch.all(model._env_coeff == 0.0)


class TestThePerEnvDraw:
    def test_each_env_draws_its_own_amplitude_inside_the_band(self):
        torch.manual_seed(31)
        low, high = 0.0, 0.16
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(low, high)),
            512, DEVICE)
        sigma = model._env_coeff * model._focal_baseline
        assert sigma.min() >= low and sigma.max() <= high
        assert sigma.unique().numel() > 400  # a draw per env, not one shared
        # Uniform on the band: the mean sits at its midpoint.
        assert float(sigma.mean()) == pytest.approx(0.5 * (low + high), abs=0.01)

    def test_a_partial_reset_redraws_only_the_envs_it_names(self):
        torch.manual_seed(37)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(0.0, 0.16)),
            NUM_ENVS, DEVICE)
        before = model._env_coeff.clone()

        reset_ids = torch.tensor([1, 4, 6])
        model.reset(reset_ids)
        kept = [index for index in range(NUM_ENVS) if index not in reset_ids.tolist()]
        assert torch.equal(before[kept], model._env_coeff[kept])
        assert not torch.equal(before[reset_ids], model._env_coeff[reset_ids])

    def test_a_full_reset_redraws_every_env(self):
        torch.manual_seed(41)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(0.0, 0.16)),
            NUM_ENVS, DEVICE)
        before = model._env_coeff.clone()
        model.reset(None)
        assert not torch.equal(before, model._env_coeff)

    def test_an_env_drawn_near_zero_carries_less_texture_than_one_drawn_high(self):
        """The point of the band: the batch spans amplitudes, so a policy sees
        featureless depth and textured depth in the same update."""
        torch.manual_seed(43)
        data = torch.full((2, PIXELS), 4.0)
        cfg = _cfg(REAL_ROBOT_CONTRACT, hole_probability=0.0,
                   disparity_noise_px_range=(0.0, 0.16))
        model = DepthNoiseModel(cfg, 2, DEVICE)
        with torch.no_grad():
            model._env_coeff[0] = 0.0
            model._env_coeff[1] = 0.16 / model._focal_baseline
        emitted = model(data.clone())
        assert torch.equal(emitted[0], data[0])
        assert float(emitted[1].std()) > 0.0


class TestTheContractPlumbing:
    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_the_contract_field_reaches_the_noise_model(self, tier):
        contract = TIERS[tier]
        band = (0.0, contract.sensors.depth_camera.disparity_noise_px)
        contract.sensors.depth_camera.disparity_noise_px_range = band
        try:
            assert get_depth_noise(contract).disparity_noise_px_range == band
        finally:
            contract.sensors.depth_camera.disparity_noise_px_range = None
