"""Tests for the temporal-texture randomization the training contracts carry.

Two mechanisms and one shared law. The law is the mixture-geometric hold
process; the mechanisms are the depth stream's bursty frame holds (a noise
model) and the chassis command hold (an action term). A third, smaller change
sits alongside them: the depth observation latency is drawn per environment at
reset instead of being one number the whole batch shares.

The suite pins what the mechanisms are only trustworthy if they get right —
that the realized hold fraction and run-length distribution match the profile
asked for, that a held frame really is the previous emission, that the process
is inert at its neutral parameters, and that the contract tiers keep the
realistic-vs-robust convention with no knob left declared and unread.

Pure numpy plus torch on CPU — no Isaac Sim, no Kit boot.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import eval_cadence_emulation as ece
from strafer_lab.tasks.navigation.mdp.hold_process import (
    HoldProcess,
    entry_probability,
    max_hold_fraction,
    mixture_mean_run,
)
from strafer_lab.tasks.navigation.mdp.noise_models import DelayBuffer, DepthNoiseModel
from strafer_lab.tasks.navigation.sim_real_cfg import (
    IDEAL_SIM_CONTRACT,
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    TimingCfg,
    get_action_config_params,
    get_depth_noise,
)


DEVICE = "cpu"


def _hold_runs(mask: np.ndarray) -> np.ndarray:
    """Closed run lengths of True in a (steps, envs) mask.

    A run still open at the last step is dropped rather than counted short —
    including it biases the mean run length downward.
    """
    runs: list[int] = []
    for column in mask.T:
        length = 0
        for flag in column:
            if flag:
                length += 1
            elif length:
                runs.append(length)
                length = 0
    return np.asarray(runs, dtype=np.float64)


def _roll(process: HoldProcess, steps: int) -> np.ndarray:
    return np.stack([process.step().cpu().numpy().copy() for _ in range(steps)])


# ---------------------------------------------------------------------------
# The run-length law
# ---------------------------------------------------------------------------


class TestHoldRunLaw:
    """The training-side law must stay the law the evaluation harness samples.

    The two implementations are deliberately separate — one is per-env torch
    state inside the env, the other numpy inside a rollout script — so the only
    thing keeping them from drifting is that both derive their parameters from
    the same arithmetic. These assertions are that check.
    """

    @pytest.mark.parametrize(
        "mean_run, burst_weight, burst_run",
        [(1.2, 0.0, 1.0), (1.5, 0.25, 6.0), (2.0, 0.5, 4.0)],
    )
    def test_mixture_mean_matches_the_harness(self, mean_run, burst_weight, burst_run):
        profile = ece.TemporalProfile(
            name="x",
            hold_fraction=0.2,
            mean_hold_run=mean_run,
            burst_weight=burst_weight,
            mean_burst_run=burst_run,
        )
        assert mixture_mean_run(mean_run, burst_weight, burst_run) == pytest.approx(
            profile.expected_hold_run()
        )

    def test_reachability_ceiling_matches_the_harness(self):
        profile = ece.TemporalProfile(
            name="x", hold_fraction=0.6, mean_hold_run=1.5,
            burst_weight=0.25, mean_burst_run=6.0,
        )
        effective = profile.expected_hold_run()
        ceiling = effective / (effective + 1.0)
        assert max_hold_fraction(effective) == pytest.approx(ceiling)
        # The harness raises above its ceiling; the law agrees on where it is.
        with pytest.raises(ValueError, match="unreachable"):
            ece.TemporalProfile(name="x", hold_fraction=0.9, mean_hold_run=1.0)

    def test_entry_probability_reproduces_the_harness_live_run(self):
        """1 / entry probability is the mean live gap between hold runs."""
        profile = ece.TemporalProfile(
            name="x", hold_fraction=0.611, mean_hold_run=1.5,
            burst_weight=0.25, mean_burst_run=6.0,
        )
        enter = entry_probability(profile.hold_fraction, profile.expected_hold_run())
        assert 1.0 / enter == pytest.approx(profile.expected_inferring_run(), rel=1e-9)

    def test_entry_probability_is_zero_at_a_zero_fraction(self):
        assert entry_probability(0.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# The hold process
# ---------------------------------------------------------------------------


class TestHoldProcess:
    def test_realized_statistics_match_a_pinned_profile(self):
        """A degenerate band isolates the process from the per-env draw, so the
        realized fraction and run length answer for the law alone."""
        torch.manual_seed(0)
        process = HoldProcess(
            256, DEVICE, hold_fraction_range=(0.4, 0.4), hold_run_range=(2.0, 2.0)
        )
        mask = _roll(process, 2000)
        runs = _hold_runs(mask)

        assert mask.mean() == pytest.approx(0.4, abs=0.01)
        assert runs.mean() == pytest.approx(2.0, abs=0.05)

    def test_burst_component_lengthens_runs_and_grows_a_tail(self):
        torch.manual_seed(1)
        base = HoldProcess(
            256, DEVICE, hold_fraction_range=(0.5, 0.5), hold_run_range=(1.5, 1.5)
        )
        bursty = HoldProcess(
            256, DEVICE,
            hold_fraction_range=(0.5, 0.5),
            hold_run_range=(1.5, 1.5),
            burst_weight=0.25,
            burst_run=6.0,
        )
        base_runs = _hold_runs(_roll(base, 2000))
        bursty_runs = _hold_runs(_roll(bursty, 2000))

        assert base_runs.mean() == pytest.approx(1.5, abs=0.05)
        assert bursty_runs.mean() == pytest.approx(
            mixture_mean_run(1.5, 0.25, 6.0), abs=0.1
        )
        # The fraction is held fixed across the pair: the burst buys run length,
        # not more holding.
        assert bursty_runs.max() > 3 * base_runs.max() / 2
        assert np.percentile(bursty_runs, 99) > np.percentile(base_runs, 99)

    def test_a_zero_band_leaves_the_process_inert(self):
        process = HoldProcess(
            32, DEVICE, hold_fraction_range=(0.0, 0.0), hold_run_range=(1.0, 2.0)
        )
        assert process.enabled is False
        assert not _roll(process, 200).any()

    def test_parameters_are_redrawn_per_env_at_reset(self):
        """The band is domain randomization, not one shared setting: envs must
        differ from each other and an env must differ across episodes."""
        torch.manual_seed(2)
        process = HoldProcess(
            64, DEVICE, hold_fraction_range=(0.0, 0.6), hold_run_range=(1.0, 2.0)
        )
        first = process._enter_p.clone()
        assert first.unique().numel() > 32

        process.reset(torch.arange(64))
        assert not torch.allclose(first, process._enter_p)

        # A partial reset touches only the named envs.
        before = process._enter_p.clone()
        process.reset(torch.tensor([0, 1, 2]))
        assert not torch.allclose(before[:3], process._enter_p[:3])
        assert torch.allclose(before[3:], process._enter_p[3:])

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_shipped_depth_band_never_needs_the_reachability_clamp(self, contract):
        """A hold fraction above what its run length can reach would be silently
        lowered. The shipped bands are chosen so that never happens; if a future
        widening breaks it, this fails rather than quietly delivering less."""
        torch.manual_seed(3)
        cfg = contract.sensors.depth_camera
        process = HoldProcess(
            512, DEVICE,
            hold_fraction_range=cfg.hold_fraction_range,
            hold_run_range=cfg.hold_run_range,
            burst_weight=cfg.hold_burst_weight,
            burst_run=cfg.hold_burst_run_steps,
        )
        for _ in range(20):
            process.reset(None)
        assert process.clamped_draws == 0

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_shipped_action_band_never_needs_the_reachability_clamp(self, contract):
        torch.manual_seed(4)
        timing = contract.timing
        process = HoldProcess(
            512, DEVICE,
            hold_fraction_range=timing.action_hold_fraction_range,
            hold_run_range=timing.action_hold_run_range,
        )
        for _ in range(20):
            process.reset(None)
        assert process.clamped_draws == 0

    def test_an_unreachable_request_is_clamped_and_counted(self):
        torch.manual_seed(5)
        # A mean run of 1 tick cannot hold more than half the steps.
        process = HoldProcess(
            64, DEVICE, hold_fraction_range=(0.9, 0.9), hold_run_range=(1.0, 1.0)
        )
        assert process.clamped_draws == 64
        assert _roll(process, 1000).mean() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# The depth stream
# ---------------------------------------------------------------------------


def _depth_cfg(**overrides):
    cfg = get_depth_noise(ROBUST_TRAINING_CONTRACT)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _emit(model: DepthNoiseModel, steps: int, num_envs: int, pixels: int = 64):
    """Roll the model on a distinct frame per step; return the repeat mask.

    Every step feeds fresh content, so an emitted frame equal to the previous
    emission can only have come from a drop or a hold.
    """
    repeats = []
    previous = None
    for step in range(steps):
        data = torch.full((num_envs, pixels), 1.0 + 0.001 * step)
        out = model(data)
        if previous is not None:
            repeats.append(torch.all(out == previous, dim=1).numpy().copy())
        previous = out.clone()
    return np.stack(repeats)


class TestDepthStreamTexture:
    def test_realized_texture_matches_the_configured_profile(self):
        """The observable the policy sees — a frame identical to the last one —
        has to carry the fraction and the run law the contract asked for."""
        torch.manual_seed(6)
        cfg = _depth_cfg(
            frame_drop_prob=0.0,
            latency_steps=0,
            latency_steps_range=None,
            hole_probability=0.0,
            hold_fraction_range=(0.45, 0.45),
            hold_run_range=(2.0, 2.0),
            hold_burst_weight=0.0,
            hold_burst_run_steps=1.0,
        )
        model = DepthNoiseModel(cfg, num_envs=128, device=DEVICE)
        mask = _emit(model, 1500, 128)

        assert mask.mean() == pytest.approx(0.45, abs=0.02)
        assert _hold_runs(mask).mean() == pytest.approx(2.0, abs=0.1)

    def test_the_shipped_robust_band_lands_where_the_contract_says(self):
        """Holds and the memoryless drop compose as a union, so the emitted
        repeat share is the hold fraction plus what the drop adds on top of it."""
        torch.manual_seed(7)
        cfg = _depth_cfg(latency_steps=0, latency_steps_range=None, hole_probability=0.0)
        band = cfg.hold_fraction_range
        model = DepthNoiseModel(cfg, num_envs=512, device=DEVICE)
        mask = _emit(model, 600, 512)

        mean_hold = 0.5 * (band[0] + band[1])
        expected = mean_hold + (1.0 - mean_hold) * cfg.frame_drop_prob
        assert mask.mean() == pytest.approx(expected, abs=0.03)

    def test_a_held_frame_is_the_previous_emission_byte_for_byte(self):
        torch.manual_seed(8)
        cfg = _depth_cfg(
            frame_drop_prob=0.0,
            latency_steps=0,
            latency_steps_range=None,
            hold_fraction_range=(0.5, 0.5),
            hold_run_range=(2.0, 2.0),
            hold_burst_weight=0.0,
            # Camera failure runs after the hold and blanks the row, which is
            # correct — a failed camera outranks a stale one — but it is not
            # what this assertion is about.
            failure_probability=0.0,
        )
        model = DepthNoiseModel(cfg, num_envs=16, device=DEVICE)

        previous = None
        checked = 0
        for step in range(400):
            out = model(torch.full((16, 32), 1.0 + 0.001 * step))
            held = model._hold._holding
            if previous is not None and bool(held.any()):
                assert torch.equal(out[held], previous[held])
                checked += int(held.sum())
            previous = out.clone()
        assert checked > 100

    def test_the_neutral_band_recovers_the_drop_only_stream(self):
        """With the hold inert the depth model must behave exactly as it did
        before the mechanism existed: repeats at the drop rate and nothing else."""
        torch.manual_seed(9)
        cfg = _depth_cfg(
            frame_drop_prob=0.05,
            latency_steps=0,
            latency_steps_range=None,
            hole_probability=0.0,
            hold_fraction_range=(0.0, 0.0),
            hold_burst_weight=0.0,
        )
        model = DepthNoiseModel(cfg, num_envs=512, device=DEVICE)
        assert model._hold.enabled is False

        mask = _emit(model, 400, 512)
        assert mask.mean() == pytest.approx(0.05, abs=0.01)
        # Memoryless drops give runs of mean 1/(1-p); a hold would lengthen them.
        assert _hold_runs(mask).mean() == pytest.approx(1.0 / 0.95, abs=0.05)

    def test_the_stream_is_reproducible_from_a_seed(self):
        cfg = _depth_cfg(latency_steps=0, latency_steps_range=None)
        masks = []
        for _ in range(2):
            torch.manual_seed(11)
            masks.append(_emit(DepthNoiseModel(cfg, 32, DEVICE), 200, 32))
        assert np.array_equal(masks[0], masks[1])

    def test_a_reset_redraws_the_per_env_texture(self):
        torch.manual_seed(12)
        cfg = _depth_cfg()
        model = DepthNoiseModel(cfg, num_envs=64, device=DEVICE)
        before = model._hold._enter_p.clone()
        model.reset(None)
        assert not torch.allclose(before, model._hold._enter_p)


# ---------------------------------------------------------------------------
# Per-env observation latency
# ---------------------------------------------------------------------------


class TestPerEnvDepthLatency:
    def test_each_env_reads_its_own_latency(self):
        torch.manual_seed(13)
        buffer = DelayBuffer(64, 1, 2, DEVICE, delay_steps_range=(0, 3))
        delays = buffer._delays.clone()

        emitted = []
        for step in range(40):
            emitted.append(buffer(torch.full((64, 1), float(step))).squeeze(-1).clone())
        emitted = torch.stack(emitted)

        # Past the zero-filled warm-up, env e reads the frame it wrote
        # ``delays[e]`` steps ago.
        for env in range(64):
            lag = int(delays[env])
            step = 30
            assert emitted[step, env].item() == pytest.approx(float(step - lag))

    def test_the_sampled_band_covers_its_range_and_redraws_on_reset(self):
        torch.manual_seed(14)
        buffer = DelayBuffer(512, 1, 2, DEVICE, delay_steps_range=(1, 3))
        assert set(buffer._delays.unique().tolist()) == {1, 2, 3}

        before = buffer._delays.clone()
        buffer.reset(None)
        assert not torch.equal(before, buffer._delays)

        # A partial reset redraws only the named envs.
        before = buffer._delays.clone()
        buffer.reset(torch.arange(400, 512))
        assert torch.equal(before[:400], buffer._delays[:400])

    def test_the_fixed_path_is_unchanged_without_a_range(self):
        buffer = DelayBuffer(4, 1, 2, DEVICE)
        emitted = [buffer(torch.full((4, 1), float(s))).squeeze(-1) for s in range(10)]
        # Two zero-filled warm-up steps, then a clean two-step shift.
        assert [float(e[0]) for e in emitted] == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_a_zero_latency_with_no_range_stays_a_passthrough(self):
        buffer = DelayBuffer(4, 1, 0, DEVICE)
        data = torch.rand(4, 1)
        assert torch.equal(buffer(data), data)


# ---------------------------------------------------------------------------
# The contract tiers
# ---------------------------------------------------------------------------


class TestContractTiers:
    def test_no_declared_timing_knob_is_left_without_a_consumer(self):
        """The timing contract's field set is pinned so a knob cannot be added
        and left unread — the failure mode this contract has already had, where
        a declared jitter percentage described a randomization the env never
        performed. Adding a field means choosing its consumer in the same
        change, then updating this list."""
        assert set(vars(TimingCfg())) == {
            "control_frequency_hz",
            "imu_latency_steps",
            "encoder_latency_steps",
            "depth_latency_steps",
            "depth_latency_steps_range",
            "rgb_latency_steps",
            "action_latency_steps",
            "action_latency_steps_range",
            "action_hold_fraction_range",
            "action_hold_run_range",
        }

    def test_both_tiers_range_and_robust_is_strictly_wider(self):
        real = REAL_ROBOT_CONTRACT.sensors.depth_camera
        robust = ROBUST_TRAINING_CONTRACT.sensors.depth_camera
        for cfg in (real, robust):
            assert cfg.hold_fraction_range[1] > cfg.hold_fraction_range[0]
            assert cfg.hold_run_range[1] > cfg.hold_run_range[0]
        assert robust.hold_fraction_range[1] > real.hold_fraction_range[1]
        assert robust.hold_run_range[1] > real.hold_run_range[1]
        assert robust.hold_burst_weight > real.hold_burst_weight

    def test_the_action_hold_stays_a_residual_under_the_depth_hold(self):
        """The command hold models what is left after the deployed node's
        inference cadence, so it must stay far below the depth hold on both
        tiers — a command band approaching the depth band would be modelling
        the cadence twice."""
        for contract in (REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT):
            action = contract.timing.action_hold_fraction_range
            depth = contract.sensors.depth_camera.hold_fraction_range
            assert action[0] == 0.0
            assert 0.0 < action[1] < 0.5 * depth[1]
        real = REAL_ROBOT_CONTRACT.timing.action_hold_fraction_range
        robust = ROBUST_TRAINING_CONTRACT.timing.action_hold_fraction_range
        assert robust[1] > real[1]

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_the_depth_latency_band_is_mean_preserving(self, contract):
        """The band randomizes the age of the depth observation without moving
        its centre — where the centre belongs is a bench measurement, not this
        knob."""
        timing = contract.timing
        lo, hi = timing.depth_latency_steps_range
        assert lo >= 0
        assert 0.5 * (lo + hi) == pytest.approx(timing.depth_latency_steps)

    def test_the_ideal_contract_carries_no_temporal_texture(self):
        depth = get_depth_noise(IDEAL_SIM_CONTRACT)
        assert depth is None  # noise off entirely on the ideal tier
        timing = IDEAL_SIM_CONTRACT.timing
        assert timing.action_hold_fraction_range == (0.0, 0.0)
        assert timing.depth_latency_steps_range is None

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_the_contract_reaches_the_depth_noise_model(self, contract):
        cfg = contract.sensors.depth_camera
        noise = get_depth_noise(contract)
        assert noise.hold_fraction_range == cfg.hold_fraction_range
        assert noise.hold_run_range == cfg.hold_run_range
        assert noise.hold_burst_weight == cfg.hold_burst_weight
        assert noise.hold_burst_run_steps == cfg.hold_burst_run_steps
        assert noise.latency_steps_range == contract.timing.depth_latency_steps_range

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_the_contract_reaches_the_action_term(self, contract):
        params = get_action_config_params(contract)
        assert params["hold_fraction_range"] == contract.timing.action_hold_fraction_range
        assert params["hold_run_range"] == contract.timing.action_hold_run_range
