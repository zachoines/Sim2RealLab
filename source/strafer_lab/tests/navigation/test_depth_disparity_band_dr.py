"""The per-environment subpixel disparity noise band.

``disparity_noise_px`` sets one stereo noise amplitude for the whole batch, so
every training frame carries per-pixel texture of the same size. The deploy
path's 8x8 block median produces frames with next to none, and a policy that
only ever saw one amplitude can read its presence as a feature of simulation.
``disparity_noise_px_range`` draws the amplitude per environment at reset, the
way the observation latency band already does, so the distribution reaches down
to the near-featureless field the deploy path actually delivers.

The draw is log-uniform. σ_d is a scale parameter spanning two decades of
interest, and a uniform draw over such a band puts almost all of its mass in the
loud decade; both ends must therefore be positive.

The band is a reparameterisation, not a second model: these tests pin that
leaving it unset changes nothing at all — not the output bits, not the RNG
state — and that a band pinned to a single value reproduces the fixed path for
that value, temporal terms live or stilled.

Pure torch on CPU — no Isaac Sim, no Kit boot. The same equivalences on CUDA are
measured in the record rather than gated here, so the suite needs no device.
"""

from __future__ import annotations

import hashlib

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
BAND = (0.002, 0.16)
TIERS = {"realistic": REAL_ROBOT_CONTRACT, "robust": ROBUST_TRAINING_CONTRACT}

# The bits each tier's depth noise produces with its band cleared -- the path
# the tier ran before it carried one. Frozen literals, because a tier that
# ships a band has no live fixed arm to compare against. Re-freeze by name when
# a depth-noise parameter on that tier changes deliberately.
_FIXED_PATH_FINGERPRINTS = {
    "realistic": "3d0d224c8e2fb5411a8472bd549cd05eeaa15af17c64761d986a894e56fdd3a7",
    "robust": "9401dc6518c944fe56f4f810cd5161b742094b379859f8594116e4016127ead1",
}


def _cfg(contract, **overrides) -> DepthNoiseModelCfg:
    """A tier's depth noise over a toy frame, with the temporal terms stilled.

    Drops, holds and latency all replay a previous frame, which would mask the
    per-frame amplitude most of this suite is about. ``_cfg_live`` keeps them.
    """
    cfg = _cfg_live(contract, **overrides)
    cfg.frame_drop_prob = 0.0
    cfg.failure_probability = 0.0
    cfg.hold_fraction_range = (0.0, 0.0)
    cfg.latency_steps = 0
    cfg.latency_steps_range = None
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _cfg_live(contract, **overrides) -> DepthNoiseModelCfg:
    """The tier as shipped, over a toy frame."""
    cfg = get_depth_noise(contract)
    cfg.height, cfg.width = HEIGHT, WIDTH
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _cfg_fixed(contract, **overrides) -> DepthNoiseModelCfg:
    """The tier's fixed-amplitude path: as shipped, with any band cleared.

    A tier that ships a band is on the drawn path by default, so the arm a
    band is compared against has to name the clearing rather than rely on the
    tier not having one.
    """
    return _cfg_live(contract, disparity_noise_px_range=None, **overrides)


def _frames(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(NUM_ENVS, PIXELS, generator=generator) * 5.5 + 0.3


def _emit(cfg: DepthNoiseModelCfg, data: torch.Tensor, steps: int = 6,
          build_seed: int = 5, run_seed: int = 11) -> torch.Tensor:
    """Both phases seeded, so two arms are compared from the same state.

    Construction already consumed randomness before this field existed — the
    hold process draws its per-env parameters there — so an arm built after
    another one starts from a different point unless the build is seeded too.
    """
    torch.manual_seed(build_seed)
    model = DepthNoiseModel(cfg, NUM_ENVS, DEVICE)
    torch.manual_seed(run_seed)
    return torch.stack([model(data.clone()) for _ in range(steps)])


class TestTheNeutralPath:
    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_a_cleared_band_reproduces_the_fixed_paths_own_bits(self, tier):
        """Clearing the band returns the tier to the amplitude it had before one.

        Pinned against a stored fingerprint rather than against another arm of
        the same cfg: a tier that ships a band has no live fixed arm to compare
        with, and comparing a cfg with itself would pass whatever the band did.
        The fingerprint moves when the tier's other depth-noise parameters move,
        and is re-frozen by name when they do.
        """
        digest = hashlib.sha256(
            _emit(_cfg_fixed(TIERS[tier]), data=_frames()).numpy().tobytes()
        ).hexdigest()
        assert digest == _FIXED_PATH_FINGERPRINTS[tier], (
            f"the {tier} tier's cleared-band output moved; if a depth-noise "
            f"parameter changed deliberately, re-freeze this fingerprint"
        )

    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_a_live_band_is_an_extra_draw_and_a_cleared_one_is_not(self, tier):
        """The band is the only randomness the field adds.

        Stated against a live band rather than against a second copy of the
        cleared cfg, which would pass whatever the band did. Clearing it has to
        leave the generator where the pre-band code left it, or every later
        draw in the episode lands somewhere else.
        """
        def rng_after(cfg):
            torch.manual_seed(29)
            model = DepthNoiseModel(cfg, NUM_ENVS, DEVICE)
            model.reset(None)
            model.reset(torch.arange(3))
            return torch.get_rng_state().clone()

        cleared = rng_after(_cfg_fixed(TIERS[tier]))
        banded = rng_after(_cfg_live(TIERS[tier], disparity_noise_px_range=BAND))
        assert not torch.equal(cleared, banded), (
            "a live band consumed no randomness, so clearing it cannot be what "
            "returns the generator to where the pre-band code left it"
        )

    def test_each_shipped_tier_declares_the_band_it_trains_on(self):
        """Robust draws over the band; realistic stays on one amplitude.

        The tiers are named individually rather than looped over, because the
        thing worth catching is one tier acquiring the other's law.
        """
        realistic = REAL_ROBOT_CONTRACT.sensors.depth_camera
        assert realistic.disparity_noise_px_range is None
        assert get_depth_noise(REAL_ROBOT_CONTRACT).disparity_noise_px_range is None

        robust = ROBUST_TRAINING_CONTRACT.sensors.depth_camera
        assert robust.disparity_noise_px_range == BAND
        assert get_depth_noise(ROBUST_TRAINING_CONTRACT).disparity_noise_px_range == BAND


class TestTheBandIsAReparameterisation:
    @pytest.mark.parametrize("sigma", [0.002, 0.08, 0.16])
    def test_a_band_pinned_to_one_value_reproduces_that_fixed_amplitude(self, sigma):
        data = _frames()
        fixed = _emit(_cfg(REAL_ROBOT_CONTRACT, disparity_noise_px=sigma), data)
        # The fixed field is left at a value the band cannot produce, so a
        # model reading the wrong one of the two cannot pass.
        pinned = _emit(_cfg(REAL_ROBOT_CONTRACT, disparity_noise_px=0.5,
                            disparity_noise_px_range=(sigma, sigma)), data)
        assert torch.equal(fixed, pinned)

    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_a_pinned_band_matches_the_fixed_path_with_the_temporal_terms_live(self, tier):
        """The band draws after every other per-env parameter, so switching it
        on does not move the hold process's or the delay buffer's draws. Pinned
        to the tier's own value it is invisible from the same seed."""
        data = _frames()
        contract = TIERS[tier]
        sigma = contract.sensors.depth_camera.disparity_noise_px
        fixed = _emit(_cfg_fixed(contract), data)
        pinned = _emit(_cfg_live(contract, disparity_noise_px=0.5,
                                 disparity_noise_px_range=(sigma, sigma)), data)
        assert torch.equal(fixed, pinned)

    def test_a_reversed_band_is_read_as_the_interval_it_names(self):
        data = _frames()
        forward = _emit(_cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=BAND), data)
        reversed_ = _emit(_cfg(REAL_ROBOT_CONTRACT,
                               disparity_noise_px_range=BAND[::-1]), data)
        assert torch.equal(forward, reversed_)

    @pytest.mark.parametrize("band", [(0.0, 0.16), (-0.05, 0.16), (0.0, 0.0)])
    def test_a_non_positive_end_is_refused(self, band):
        """A log-uniform draw has no zero end. Silently clamping one would turn
        the band into a different law at its low end."""
        with pytest.raises(ValueError, match="positive"):
            DepthNoiseModel(
                _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=band),
                NUM_ENVS, DEVICE)


class TestTheDrawLaw:
    def test_the_draw_is_log_uniform_not_uniform(self):
        """Over [0.001, 0.1] the two laws disagree on the median by 5x: the
        log-uniform median is the geometric mean 0.01, the uniform one 0.0505."""
        torch.manual_seed(53)
        low, high = 0.001, 0.1
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(low, high)),
            4096, DEVICE)
        sigma = model._env_coeff * model._focal_baseline
        median = float(sigma.median())
        assert median == pytest.approx((low * high) ** 0.5, rel=0.05)
        assert median != pytest.approx(0.5 * (low + high), rel=0.5)

    def test_each_decade_of_the_band_gets_equal_weight(self):
        """The property the law is chosen for: a band spanning two decades puts
        half its environments in each, where a uniform draw puts 9% in the
        quiet one."""
        torch.manual_seed(59)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=(0.001, 0.1)),
            4096, DEVICE)
        sigma = (model._env_coeff * model._focal_baseline).ravel()
        lower_decade = float((sigma < 0.01).to(torch.float64).mean())
        assert lower_decade == pytest.approx(0.5, abs=0.03)


class TestThePerEnvDraw:
    def test_each_env_draws_its_own_amplitude_inside_the_band(self):
        torch.manual_seed(31)
        low, high = BAND
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=BAND), 512, DEVICE)
        sigma = model._env_coeff * model._focal_baseline
        assert sigma.min() >= low and sigma.max() <= high
        assert sigma.unique().numel() > 400  # a draw per env, not one shared

    def test_a_partial_reset_redraws_only_the_envs_it_names(self):
        torch.manual_seed(37)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=BAND), NUM_ENVS, DEVICE)
        before = model._env_coeff.clone()

        reset_ids = torch.tensor([1, 4, 6])
        model.reset(reset_ids)
        kept = [index for index in range(NUM_ENVS) if index not in reset_ids.tolist()]
        assert torch.equal(before[kept], model._env_coeff[kept])
        assert not torch.equal(before[reset_ids], model._env_coeff[reset_ids])

    def test_a_full_reset_redraws_every_env(self):
        torch.manual_seed(41)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, disparity_noise_px_range=BAND), NUM_ENVS, DEVICE)
        before = model._env_coeff.clone()
        model.reset(None)
        assert not torch.equal(before, model._env_coeff)

    def test_an_env_drawn_quiet_carries_less_texture_than_one_drawn_loud(self):
        """The point of the band: the batch spans amplitudes, so a policy sees
        near-featureless depth and textured depth in the same update."""
        torch.manual_seed(43)
        data = torch.full((2, PIXELS), 4.0)
        model = DepthNoiseModel(
            _cfg(REAL_ROBOT_CONTRACT, hole_probability=0.0,
                 disparity_noise_px_range=BAND), 2, DEVICE)
        with torch.no_grad():
            model._env_coeff[0] = BAND[0] / model._focal_baseline
            model._env_coeff[1] = BAND[1] / model._focal_baseline
        emitted = model(data.clone())
        assert float(emitted[0].std()) < float(emitted[1].std()) / 10.0


class TestTheContractPlumbing:
    @pytest.mark.parametrize("tier", list(TIERS), ids=list(TIERS))
    def test_the_contract_field_reaches_the_noise_model(self, tier):
        contract = TIERS[tier]
        band = (0.002, contract.sensors.depth_camera.disparity_noise_px)
        # The contracts are module-level singletons, so the restore has to put
        # back what was there rather than the pre-band default: a tier that
        # ships a band would otherwise lose it for the rest of the process.
        shipped = contract.sensors.depth_camera.disparity_noise_px_range
        contract.sensors.depth_camera.disparity_noise_px_range = band
        try:
            assert get_depth_noise(contract).disparity_noise_px_range == band
        finally:
            contract.sensors.depth_camera.disparity_noise_px_range = shipped
