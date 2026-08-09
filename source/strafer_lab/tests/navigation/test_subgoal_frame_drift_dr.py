"""Tests for the referent-frame drift the training contracts carry.

One mechanism, two error classes, and one law shared with the evaluation
harness. The law is an SE(2) Ornstein-Uhlenbeck process on the frame the
goal-shaped observations are read through; the classes are the smooth wander a
SLAM estimate accumulates and the discontinuous snap a loop closure produces.

The suite pins what the mechanism is only trustworthy if it gets right — that
the realized magnitude and correlation match the profile asked for, that all
four referent dims come from one perturbed vector rather than three independent
corruptions, that the privileged critic still reads truth, that the process is
inert (and draws nothing) at its neutral parameters, and that the contract
tiers keep the realistic-vs-robust convention with the jump band shipped off.

The harness carries its own numpy implementation on purpose — one is per-env
torch inside the env, the other numpy inside a rollout script — so the only
thing keeping them from diverging is that both derive from the same arithmetic.
The first class below is that check.

Pure numpy plus torch on CPU — no Isaac Sim, no Kit boot.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

import eval_cadence_emulation as ece
from strafer_lab.tasks.navigation.mdp import observations as obs
from strafer_lab.tasks.navigation.mdp.subgoal_drift import (
    SubgoalDriftProcess,
    axis_sigma,
    drift_referent,
    jump_probability,
    ou_coefficients,
)
from strafer_lab.tasks.navigation.sim_real_cfg import (
    IDEAL_SIM_CONTRACT,
    MEASURED_DRIFT_HEADING_DEG,
    MEASURED_DRIFT_POSITION_RMS_M,
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    LocalizationDriftCfg,
    get_subgoal_drift_params,
)


DEVICE = "cpu"
STEP_DT = 1.0 / 30.0


def _process(**overrides) -> SubgoalDriftProcess:
    kwargs = dict(
        step_dt=STEP_DT,
        position_rms_m=MEASURED_DRIFT_POSITION_RMS_M,
        heading_sigma_deg=MEASURED_DRIFT_HEADING_DEG,
        gain_range=(1.0, 1.0),
        tau_s=2.0,
    )
    num_envs = overrides.pop("num_envs", 256)
    kwargs.update(overrides)
    return SubgoalDriftProcess(num_envs, DEVICE, **kwargs)


def _roll(process: SubgoalDriftProcess, steps: int) -> np.ndarray:
    """(steps, envs, 3) of the offsets the process emitted."""
    return np.stack([process.step().clone().numpy() for _ in range(steps)])


# ---------------------------------------------------------------------------
# The shared law
# ---------------------------------------------------------------------------


class TestDriftLawMatchesTheHarness:
    @pytest.mark.parametrize("tau_s", [0.5, 2.0, 10.0])
    @pytest.mark.parametrize("step_dt", [1.0 / 30.0, 1.0 / 12.0])
    def test_ou_coefficients_match_the_harness(self, tau_s, step_dt):
        harness = ece.SubgoalDrift(
            num_envs=1,
            position_rms_m=MEASURED_DRIFT_POSITION_RMS_M,
            heading_sigma_rad=math.radians(MEASURED_DRIFT_HEADING_DEG),
            tau_s=tau_s,
            step_dt=step_dt,
            rng=np.random.default_rng(0),
        )
        decay, innovation = ou_coefficients(tau_s, step_dt)
        assert decay == pytest.approx(harness._decay, rel=1e-12)
        assert innovation == pytest.approx(harness._innovation, rel=1e-12)

    def test_the_axis_split_matches_the_harness(self):
        """The class is quoted as an RMS displacement of the 2-D offset, so a
        per-axis sigma that skipped the 1/sqrt(2) would train against a class
        41% larger than the one measured."""
        harness = ece.SubgoalDrift(
            num_envs=1,
            position_rms_m=MEASURED_DRIFT_POSITION_RMS_M,
            heading_sigma_rad=math.radians(MEASURED_DRIFT_HEADING_DEG),
            tau_s=2.0,
            step_dt=STEP_DT,
            rng=np.random.default_rng(0),
        )
        process = _process(num_envs=4)
        assert axis_sigma(MEASURED_DRIFT_POSITION_RMS_M) == pytest.approx(
            harness.position_sigma_axis_m
        )
        assert process._unit_sigma.numpy() == pytest.approx(harness._sigma)

    def test_the_transform_reproduces_the_harness_quartet(self):
        """Same referent, same offsets, same four dims.

        The harness rewrites the scaled observation quartet; the env perturbs
        the raw body-frame vector the quartet is derived from. Equal outputs
        are what makes an arm run against the harness comparable to a policy
        trained against the env."""
        rng = np.random.default_rng(7)
        rel = rng.normal(0.0, 2.0, size=(64, 2))
        offsets = np.concatenate(
            [rng.normal(0.0, 0.2, size=(64, 2)), rng.normal(0.0, 0.3, size=(64, 1))],
            axis=1,
        )
        dist_scale, heading_scale = 1.0 / 10.0, 1.0 / math.pi

        quartet = np.stack(
            [
                rel[:, 0] * dist_scale,
                rel[:, 1] * dist_scale,
                np.hypot(rel[:, 0], rel[:, 1]) * dist_scale,
                np.arctan2(rel[:, 1], rel[:, 0]) * heading_scale,
            ],
            axis=1,
        )
        expected = ece.drifted_quartet(
            quartet, offsets, dist_scale=dist_scale, heading_scale=heading_scale
        )

        drifted = drift_referent(
            torch.tensor(rel, dtype=torch.float64),
            torch.tensor(offsets, dtype=torch.float64),
        ).numpy()
        assert drifted[:, 0] * dist_scale == pytest.approx(expected[:, 0])
        assert drifted[:, 1] * dist_scale == pytest.approx(expected[:, 1])
        assert np.hypot(drifted[:, 0], drifted[:, 1]) * dist_scale == pytest.approx(
            expected[:, 2]
        )
        assert np.arctan2(drifted[:, 1], drifted[:, 0]) * heading_scale == pytest.approx(
            expected[:, 3]
        )

    def test_a_zero_offset_is_the_identity(self):
        rel = torch.randn(32, 2, dtype=torch.float64)
        assert torch.equal(drift_referent(rel, torch.zeros(32, 3, dtype=torch.float64)), rel)

    def test_rotation_preserves_the_distance(self):
        """Heading drift moves the bearing and must not move the range: the two
        are different errors and a policy that saw them coupled would be
        learning geometry no frame error produces."""
        rel = torch.randn(32, 2, dtype=torch.float64)
        offsets = torch.zeros(32, 3, dtype=torch.float64)
        offsets[:, 2] = torch.rand(32, dtype=torch.float64) * 2.0 - 1.0
        drifted = drift_referent(rel, offsets)
        assert drifted.norm(dim=-1).numpy() == pytest.approx(rel.norm(dim=-1).numpy())

    def test_jump_probability_is_the_poisson_arrival_law(self):
        assert jump_probability(0.0, STEP_DT) == 0.0
        assert jump_probability(1.0, STEP_DT) == pytest.approx(1.0 - math.exp(-STEP_DT))
        # At a small rate*dt the per-step probability is the rate times the step.
        assert jump_probability(0.05, STEP_DT) == pytest.approx(0.05 * STEP_DT, rel=1e-3)


# ---------------------------------------------------------------------------
# The wander
# ---------------------------------------------------------------------------


class TestDriftProcess:
    def test_realized_magnitude_matches_a_pinned_gain(self):
        """A degenerate gain band isolates the process from the per-env draw, so
        the realized sigma answers for the law alone. Burn-in is dropped: the
        state starts at the anchor and reaches its stationary spread over a few
        time constants."""
        torch.manual_seed(0)
        process = _process(num_envs=512)
        rolled = _roll(process, 1200)[300:]

        position_rms = math.sqrt(np.mean(rolled[..., 0] ** 2 + rolled[..., 1] ** 2))
        heading_sigma = math.degrees(math.sqrt(np.mean(rolled[..., 2] ** 2)))
        assert position_rms == pytest.approx(MEASURED_DRIFT_POSITION_RMS_M, rel=0.05)
        assert heading_sigma == pytest.approx(MEASURED_DRIFT_HEADING_DEG, rel=0.05)

    def test_the_gain_scales_both_magnitudes_together(self):
        torch.manual_seed(1)
        half = _roll(_process(num_envs=512, gain_range=(0.5, 0.5)), 1200)[300:]
        full = _roll(_process(num_envs=512, gain_range=(1.0, 1.0)), 1200)[300:]
        for axis in (0, 1, 2):
            ratio = np.std(full[..., axis]) / np.std(half[..., axis])
            assert ratio == pytest.approx(2.0, rel=0.08)

    def test_the_error_is_integrated_not_resampled(self):
        """The point of the mechanism. A localization offset persists over
        seconds; independent per-tick noise is something a recurrent policy
        averages away in a handful of ticks, which is the opposite of the
        deployed failure."""
        torch.manual_seed(2)
        rolled = _roll(_process(num_envs=256), 3000)[500:]
        decay, _ = ou_coefficients(2.0, STEP_DT)

        for axis in (0, 1, 2):
            series = rolled[..., axis]
            lag1 = np.mean(series[:-1] * series[1:]) / np.mean(series**2)
            assert lag1 == pytest.approx(decay, abs=0.02)
        # The correlation time is long against a control step: a whole second
        # later the offset is still most of what it was.
        series = rolled[..., 0]
        lag30 = np.mean(series[:-30] * series[30:]) / np.mean(series**2)
        assert lag30 == pytest.approx(decay**30, abs=0.05)

    def test_a_reset_re_anchors_and_redraws_the_magnitude(self):
        """A new episode is a new path off a fresh anchor, and the band is
        domain randomization rather than one shared setting."""
        torch.manual_seed(3)
        process = _process(num_envs=64, gain_range=(0.0, 1.25))
        _roll(process, 200)
        assert float(process.offsets.abs().max()) > 0.0

        sigma_before = process._sigma.clone()
        process.reset(None)
        assert torch.equal(process.offsets, torch.zeros(64, 3))
        assert not torch.allclose(sigma_before, process._sigma)
        assert process._sigma[:, 0].unique().numel() > 32

    def test_a_partial_reset_touches_only_the_named_envs(self):
        torch.manual_seed(4)
        process = _process(num_envs=64, gain_range=(0.0, 1.25))
        _roll(process, 100)
        state_before = process.offsets.clone()
        sigma_before = process._sigma.clone()

        process.reset(torch.arange(8))
        assert torch.equal(process.offsets[:8], torch.zeros(8, 3))
        assert torch.equal(process.offsets[8:], state_before[8:])
        assert torch.equal(sigma_before[8:], process._sigma[8:])

    def test_a_zero_gain_band_leaves_the_process_inert(self):
        process = _process(num_envs=32, gain_range=(0.0, 0.0))
        assert process.enabled is False
        assert not _roll(process, 200).any()

    def test_a_disabled_process_leaves_the_rng_stream_untouched(self):
        """Inertness is not just "never drifts" — it is "consumes nothing".

        The contract-golden attribution claims the ideal tier is unchanged by
        this mechanism, and that claim is false the moment a disabled process
        draws a single random number. Asserted directly rather than inferred
        from behaviour, matching the temporal-texture pin."""
        torch.manual_seed(5)
        process = _process(num_envs=128, gain_range=(0.0, 0.0))
        assert process.enabled is False

        before = torch.get_rng_state()
        process.reset(None)
        process.reset(torch.arange(64))
        for _ in range(100):
            assert not process.step().any()
        assert torch.equal(torch.get_rng_state(), before)

    def test_the_stream_is_reproducible_from_a_seed(self):
        rolled = []
        for _ in range(2):
            torch.manual_seed(6)
            rolled.append(_roll(_process(num_envs=32), 200))
        assert np.array_equal(rolled[0], rolled[1])

    def test_the_process_advances_once_per_control_step(self):
        """Three observation terms read the same offsets, and the observation
        group can be computed more than once in a step, so the advance is keyed
        on the step index rather than on the read."""
        torch.manual_seed(7)
        process = _process(num_envs=8)
        first = process.advance_to(0).clone()
        assert torch.equal(process.advance_to(0), first)
        assert torch.equal(process.advance_to(0), first)

        second = process.advance_to(1).clone()
        assert not torch.equal(second, first)


# ---------------------------------------------------------------------------
# The loop-closure jump
# ---------------------------------------------------------------------------


def _jumping(**overrides) -> SubgoalDriftProcess:
    kwargs = dict(
        num_envs=256,
        gain_range=(0.0, 0.0),
        jump_rate_hz=1.0,
        jump_position_range_m=(0.4, 0.4),
        jump_heading_range_deg=(10.0, 10.0),
    )
    kwargs.update(overrides)
    return _process(**kwargs)


class TestLoopClosureJumps:
    def test_the_shipped_band_leaves_the_jump_dormant(self):
        """The mechanism carries no measured distribution, so both of its knobs
        default to zero rather than to a guessed shape."""
        for contract in (REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT):
            drift = contract.localization
            assert drift.jump_rate_hz == 0.0
            assert drift.jump_position_range_m == (0.0, 0.0)
            assert drift.jump_heading_range_deg == (0.0, 0.0)

        torch.manual_seed(8)
        process = _process(num_envs=128)
        _roll(process, 500)
        assert process.jumps == 0

    def test_a_dormant_jump_leaves_the_rng_stream_untouched(self):
        """Same claim as the disabled wander, for the half that ships off: a
        zero band must not cost a draw, or the shipped tiers are not the streams
        the attribution says they are."""
        torch.manual_seed(9)
        process = _process(num_envs=64, gain_range=(0.0, 0.0), jump_rate_hz=5.0)
        assert process.enabled is False
        before = torch.get_rng_state()
        for _ in range(100):
            process.step()
        assert torch.equal(torch.get_rng_state(), before)

    def test_jumps_arrive_at_the_configured_rate(self):
        torch.manual_seed(10)
        process = _jumping(num_envs=512)
        steps = 600
        _roll(process, steps)
        expected = 512 * steps * jump_probability(1.0, STEP_DT)
        assert process.jumps == pytest.approx(expected, rel=0.1)

    def test_a_jump_is_discontinuous_where_the_wander_is_not(self):
        """The class this term exists for. A closure landing moves the frame in
        one step by an amount the correlated walk reaches only over seconds, so
        the two are separable in the step-to-step difference."""
        torch.manual_seed(11)
        jumps = _roll(_jumping(num_envs=256), 400)
        wander = _roll(_process(num_envs=256), 400)

        def biggest_step(rolled):
            delta = np.diff(rolled[..., :2], axis=0)
            return np.max(np.hypot(delta[..., 0], delta[..., 1]))

        assert biggest_step(jumps) == pytest.approx(0.4, rel=0.05)
        assert biggest_step(wander) < 0.4 * 0.5

    def test_a_jump_relaxes_on_the_same_time_constant(self):
        """One process, not two: the snap lands in the state the wander relaxes,
        so a closure error decays instead of persisting forever."""
        torch.manual_seed(12)
        # A rate this far above the tick makes the first step a certain jump,
        # of exactly the degenerate band's magnitude.
        process = _jumping(num_envs=1, jump_rate_hz=1.0e6)
        process.step()
        landed = float(torch.linalg.vector_norm(process.offsets[0, :2]))
        assert landed == pytest.approx(0.4, rel=1e-3)

        process._jump_enabled = False
        decay, _ = ou_coefficients(2.0, STEP_DT)
        for _ in range(30):
            process.step()
        after = float(torch.linalg.vector_norm(process.offsets[0, :2]))
        assert after == pytest.approx(landed * decay**30, rel=1e-3)


# ---------------------------------------------------------------------------
# The observation terms
# ---------------------------------------------------------------------------


def _stub_env(rel_xy: torch.Tensor, *, yaw: torch.Tensor | None = None):
    """Minimal env exposing what the goal-shaped observation terms read.

    The robot sits at the origin facing ``yaw`` and the referent sits at
    ``rel_xy`` in the robot's own frame, so the true body-frame offset the
    terms should produce is exactly ``rel_xy``.
    """
    num_envs = rel_xy.shape[0]
    if yaw is None:
        yaw = torch.zeros(num_envs)
    half = 0.5 * yaw
    quat = torch.zeros(num_envs, 4)
    quat[:, 2] = torch.sin(half)  # z
    quat[:, 3] = torch.cos(half)  # w

    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    goal = torch.zeros(num_envs, 3)
    goal[:, 0] = cos_y * rel_xy[:, 0] - sin_y * rel_xy[:, 1]
    goal[:, 1] = sin_y * rel_xy[:, 0] + cos_y * rel_xy[:, 1]

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=wp.from_torch(torch.zeros(num_envs, 3)),
            root_quat_w=wp.from_torch(quat),
        )
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device=DEVICE,
        step_dt=STEP_DT,
        common_step_counter=0,
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: goal),
    )


class TestDriftedObservationTerms:
    def test_the_terms_agree_with_the_true_referent_without_drift(self):
        rel = torch.tensor([[2.0, 0.5], [-1.0, 3.0], [0.25, -0.75]])
        env = _stub_env(rel, yaw=torch.tensor([0.0, 1.1, -2.4]))

        relative = obs.goal_position_relative(env, "goal_command")
        assert relative.numpy() == pytest.approx(rel.numpy(), abs=1e-5)
        assert obs.goal_distance(env, "goal_command").squeeze(-1).numpy() == pytest.approx(
            rel.norm(dim=-1).numpy(), abs=1e-5
        )
        assert obs.goal_heading_to_goal(env, "goal_command").squeeze(-1).numpy() == (
            pytest.approx(torch.atan2(rel[:, 1], rel[:, 0]).numpy(), abs=1e-5)
        )

    def test_all_four_dims_come_from_one_perturbed_vector(self):
        """The invariant the mechanism exists to keep. The four referent dims
        are near-redundant, so perturbing them independently would hand the
        policy a relative offset, a range and a bearing that no geometry can
        produce together."""
        torch.manual_seed(13)
        rel = torch.tensor([[2.0, 0.5], [-1.0, 3.0], [0.25, -0.75], [4.0, -1.5]])
        env = _stub_env(rel, yaw=torch.tensor([0.0, 1.1, -2.4, 0.6]))
        env._subgoal_drift = _process(num_envs=4)

        relative = obs.goal_position_relative(env, "goal_command")
        distance = obs.goal_distance(env, "goal_command").squeeze(-1)
        bearing = obs.goal_heading_to_goal(env, "goal_command").squeeze(-1)

        assert not torch.allclose(relative, rel, atol=1e-3)
        assert distance.numpy() == pytest.approx(relative.norm(dim=-1).numpy(), abs=1e-5)
        assert bearing.numpy() == pytest.approx(
            torch.atan2(relative[:, 1], relative[:, 0]).numpy(), abs=1e-5
        )

    def test_the_privileged_read_is_the_true_referent(self):
        """``perceived=False`` is what keeps the critic clean: the drift is
        applied inside the term, so the observation manager's corruption switch
        cannot strip it the way it strips a noise model."""
        torch.manual_seed(14)
        rel = torch.tensor([[2.0, 0.5], [-1.0, 3.0]])
        env = _stub_env(rel)
        env._subgoal_drift = _process(num_envs=2)

        assert obs.goal_position_relative(
            env, "goal_command", perceived=False
        ).numpy() == pytest.approx(rel.numpy(), abs=1e-5)
        assert obs.goal_distance(
            env, "goal_command", perceived=False
        ).squeeze(-1).numpy() == pytest.approx(rel.norm(dim=-1).numpy(), abs=1e-5)
        assert obs.goal_heading_to_goal(
            env, "goal_command", perceived=False
        ).squeeze(-1).numpy() == pytest.approx(
            torch.atan2(rel[:, 1], rel[:, 0]).numpy(), abs=1e-5
        )

    def test_the_three_terms_share_one_advance_per_step(self):
        """A per-read advance would decorrelate the three terms from each other
        and would run the process at three times its configured rate."""
        torch.manual_seed(15)
        rel = torch.tensor([[2.0, 0.5], [-1.0, 3.0]])
        env = _stub_env(rel)
        env._subgoal_drift = _process(num_envs=2)

        obs.goal_position_relative(env, "goal_command")
        obs.goal_distance(env, "goal_command")
        obs.goal_heading_to_goal(env, "goal_command")
        first = env._subgoal_drift.offsets.clone()

        env.common_step_counter = 1
        obs.goal_position_relative(env, "goal_command")
        obs.goal_distance(env, "goal_command")
        assert not torch.equal(env._subgoal_drift.offsets, first)

    def test_the_arrival_heading_reads_the_same_believed_yaw(self):
        """The referent quartet's rotation is the robot believing its heading is
        dtheta off, and the arrival-heading error is read against that same
        belief. The term is unused by every shipped variant, so this pins the
        law before something wires it in rather than after."""
        torch.manual_seed(16)
        rel = torch.tensor([[2.0, 0.5], [-1.0, 3.0]])
        env = _stub_env(rel, yaw=torch.tensor([0.3, -1.2]))
        env._subgoal_drift = _process(num_envs=2)

        truth = obs.goal_heading_relative(env, "goal_command", perceived=False)
        drifted = obs.goal_heading_relative(env, "goal_command")
        dtheta = env._subgoal_drift.offsets[:, 2]

        expected = torch.atan2(
            torch.sin(truth.squeeze(-1) + dtheta), torch.cos(truth.squeeze(-1) + dtheta)
        )
        assert drifted.squeeze(-1).numpy() == pytest.approx(expected.numpy(), abs=1e-6)
        assert not torch.allclose(drifted, truth)

    def test_the_drift_capable_terms_are_the_referent_derived_ones(self):
        """Pinned as a set, so a new observation function derived from the
        referent frame has to choose whether it drifts — and so the contract
        gate's signature-based discovery keeps matching what this module
        actually offers."""
        capable = {
            name
            for name, member in vars(obs).items()
            if not name.startswith("_")
            and inspect.isfunction(member)
            and member.__module__ == obs.__name__
            and "perceived" in inspect.signature(member).parameters
        }
        assert capable == {
            "goal_position_relative",
            "goal_distance",
            "goal_heading_to_goal",
            "goal_heading_relative",
        }

    def test_an_env_without_the_process_reads_truth(self):
        """The mechanism is opt-in per tier, and an env that never installed it
        must take the untouched path — including the observation manager's
        dimension probe, which runs before any reset event."""
        rel = torch.tensor([[2.0, 0.5]])
        env = _stub_env(rel)
        assert not hasattr(env, "_subgoal_drift")
        assert obs.goal_distance(env, "goal_command").squeeze(-1).numpy() == (
            pytest.approx(rel.norm(dim=-1).numpy(), abs=1e-5)
        )


# ---------------------------------------------------------------------------
# The contract tiers
# ---------------------------------------------------------------------------


class TestContractTiers:
    def test_no_declared_drift_knob_is_left_without_a_consumer(self):
        """The field set is pinned so a knob cannot be added and left unread —
        the failure mode this contract has already had. Adding a field means
        choosing its consumer in the same change, then updating this list."""
        assert set(vars(LocalizationDriftCfg())) == {
            "enable_drift",
            "position_rms_m",
            "heading_sigma_deg",
            "gain_range",
            "tau_s",
            "jump_rate_hz",
            "jump_position_range_m",
            "jump_heading_range_deg",
        }

    def test_both_tiers_range_and_robust_is_strictly_wider(self):
        real = REAL_ROBOT_CONTRACT.localization
        robust = ROBUST_TRAINING_CONTRACT.localization
        for drift in (real, robust):
            assert drift.enable_drift
            assert drift.gain_range[0] == 0.0
            assert drift.gain_range[1] > 0.0
        assert robust.gain_range[1] > real.gain_range[1]

    def test_the_bands_are_the_multiples_of_the_measured_class(self):
        """Both tiers are anchored to one recorded pair rather than to a shape
        someone liked: realistic reaches half of it, robust past it, the way the
        temporal bands reach past the measured arrival profiles."""
        assert REAL_ROBOT_CONTRACT.localization.gain_range == (0.0, 0.5)
        assert ROBUST_TRAINING_CONTRACT.localization.gain_range == (0.0, 1.25)
        for contract in (REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT):
            drift = contract.localization
            assert drift.position_rms_m == MEASURED_DRIFT_POSITION_RMS_M
            assert drift.heading_sigma_deg == MEASURED_DRIFT_HEADING_DEG
            assert drift.tau_s == 2.0

    def test_the_ideal_contract_carries_no_drift(self):
        assert IDEAL_SIM_CONTRACT.localization.enable_drift is False
        assert get_subgoal_drift_params(IDEAL_SIM_CONTRACT) is None

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_the_contract_reaches_the_event_term(self, contract):
        drift = contract.localization
        params = get_subgoal_drift_params(contract)
        assert params["position_rms_m"] == drift.position_rms_m
        assert params["heading_sigma_deg"] == drift.heading_sigma_deg
        assert params["gain_range"] == drift.gain_range
        assert params["tau_s"] == drift.tau_s
        assert params["jump_rate_hz"] == drift.jump_rate_hz
        assert params["jump_position_range_m"] == drift.jump_position_range_m
        assert params["jump_heading_range_deg"] == drift.jump_heading_range_deg
        # Every parameter the process takes beyond the ones the env supplies.
        assert set(params) == {
            "position_rms_m",
            "heading_sigma_deg",
            "gain_range",
            "tau_s",
            "jump_rate_hz",
            "jump_position_range_m",
            "jump_heading_range_deg",
        }

    @pytest.mark.parametrize(
        "contract", [REAL_ROBOT_CONTRACT, ROBUST_TRAINING_CONTRACT], ids=["real", "robust"]
    )
    def test_the_shipped_band_spans_the_sensitivity_arm(self, contract):
        """The arm that convicted this axis ran at 1x and cost 24-28% of
        completion. The robust band has to reach it; the realistic band is
        deliberately below it, because realistic models the link as measured
        rather than the worst it can be."""
        gain_hi = contract.localization.gain_range[1]
        assert gain_hi >= 1.0 if contract is ROBUST_TRAINING_CONTRACT else gain_hi < 1.0
