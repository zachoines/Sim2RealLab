"""Unit tests for ``train_strafer_navigation.attach_lr_schedule``.

Every assertion about the applied rate reads
``optimizer.param_groups[...]["lr"]`` at the moment the update runs, never
``alg.learning_rate``. The attribute is the value RSL-RL logs, and a schedule
that sets it while leaving the optimizer alone produces a TensorBoard curve
that decays over a run whose Adam never changed rate. Asserting the attribute
is therefore the same mistake the schedule itself made, so it is only ever
checked for *agreement* with the optimizer.

``_RecordingRunner.learn`` mirrors RSL-RL's loop shape, including the
assignment of ``current_learning_iteration`` *after* ``alg.update()`` returns:
a schedule reading that counter from inside the update sees a stale index.
"""

from __future__ import annotations

import math

import pytest
import torch

import train_strafer_navigation as train_script

LR_INIT = 3.0e-4
LR_MIN = 1.0e-5


def _expected_cosine(progress: float) -> float:
    return LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def _expected_linear(progress: float) -> float:
    return LR_INIT + (LR_MIN - LR_INIT) * progress


class _RecordingAlg:
    """Stand-in for ``rsl_rl.algorithms.PPO`` holding a real Adam optimizer."""

    def __init__(
        self,
        lr: float,
        num_param_groups: int = 1,
        apply_step: bool = False,
        schedule: str = "fixed",
    ) -> None:
        self.learning_rate = lr
        self.schedule = schedule
        self.params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(num_param_groups)]
        self.optimizer = torch.optim.Adam([{"params": [p]} for p in self.params], lr=lr)
        self.apply_step = apply_step
        self.optimizer_lrs: list[float] = []
        self.attribute_lrs: list[float] = []
        self.step_sizes: list[float] = []

    def update(self) -> dict:
        # Record what Adam actually holds when the update runs.
        self.optimizer_lrs.append(self.optimizer.param_groups[0]["lr"])
        self.attribute_lrs.append(self.learning_rate)
        if self.apply_step:
            before = self.params[0].detach().clone()
            self.params[0].grad = torch.ones_like(self.params[0])
            self.optimizer.step()
            self.step_sizes.append(float((self.params[0].detach() - before).abs().item()))
        return {"loss": 0.0}


class _RecordingRunner:
    """Reproduces the parts of ``OnPolicyRunner`` the schedule depends on."""

    def __init__(
        self,
        lr: float = LR_INIT,
        start_iteration: int = 0,
        num_param_groups: int = 1,
        apply_step: bool = False,
        schedule: str = "fixed",
    ) -> None:
        self.alg = _RecordingAlg(lr, num_param_groups, apply_step, schedule)
        self.current_learning_iteration = start_iteration

    def load(self, checkpoint_lr: float, iteration: int) -> None:
        """Stand in for ``runner.load()``: restore an optimizer rate and an iteration.

        rsl_rl restores the optimizer state dict, and torch's
        ``load_state_dict`` carries the checkpoint's ``lr`` into
        ``param_groups`` wholesale.
        """
        for param_group in self.alg.optimizer.param_groups:
            param_group["lr"] = checkpoint_lr
        self.current_learning_iteration = iteration

    def learn(self, num_iterations: int) -> None:
        start = self.current_learning_iteration
        for iteration in range(start, start + num_iterations):
            self.alg.update()
            # rsl_rl advances the counter only after the update returns.
            self.current_learning_iteration = iteration


def test_cosine_schedule_reaches_the_optimizer():
    """A cosine schedule must move Adam's rate, not just the logged attribute."""
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=10)
    runner.learn(10)

    observed = runner.alg.optimizer_lrs
    assert len(observed) == 10
    assert len(set(observed)) > 1, "optimizer rate never changed over the run"
    assert observed[0] == pytest.approx(LR_INIT)
    assert all(later < earlier for earlier, later in zip(observed, observed[1:]))
    assert observed[-1] < 0.1 * LR_INIT


def test_linear_schedule_reaches_the_optimizer():
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "linear", LR_INIT, LR_MIN, num_iterations=10)
    runner.learn(10)

    observed = runner.alg.optimizer_lrs
    assert len(set(observed)) > 1, "optimizer rate never changed over the run"
    assert observed == pytest.approx([_expected_linear(i / 10) for i in range(10)])


def test_every_update_runs_at_its_own_scheduled_rate():
    """No off-by-one: update *i* runs at the rate the schedule defines for *i*.

    The counter the runner exposes is one behind inside ``update()``, so a
    schedule that reads it applies iteration *i-1*'s rate to iteration *i*.
    """
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=8)
    runner.learn(8)

    assert runner.alg.optimizer_lrs == pytest.approx([_expected_cosine(i / 8) for i in range(8)])


def test_resumed_run_does_not_start_clamped_at_lr_min():
    """Resuming anneals across the iterations the run will reach, not ``max_iterations``.

    Progress is measured against ``start + num_iterations``. Measuring against
    ``num_iterations`` alone puts a resume from iteration 9999 past the end of a
    6000-iteration curve, pinning every update at ``lr_min``.
    """
    start, num_iterations = 9999, 6000
    total = start + num_iterations
    runner = _RecordingRunner(start_iteration=start)
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=num_iterations)
    runner.learn(num_iterations)

    observed = runner.alg.optimizer_lrs
    assert len(observed) == num_iterations
    assert observed[0] == pytest.approx(_expected_cosine(start / total))
    assert observed[0] > 5 * LR_MIN, "resumed run started clamped at lr_min"
    assert observed[0] < LR_INIT, "resumed run restarted the curve from lr_init"
    assert observed[-1] == pytest.approx(_expected_cosine((total - 1) / total))
    assert all(later < earlier for earlier, later in zip(observed, observed[1:]))


def test_resume_point_is_read_after_the_checkpoint_loads():
    """The schedule is attached before ``runner.load()`` moves the counter."""
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "linear", LR_INIT, LR_MIN, num_iterations=100)
    # Stand in for runner.load(): the checkpoint's iteration lands after attach.
    runner.current_learning_iteration = 400
    runner.learn(100)

    assert runner.alg.optimizer_lrs[0] == pytest.approx(_expected_linear(400 / 500))


def test_schedule_writes_every_param_group():
    runner = _RecordingRunner(num_param_groups=3)
    train_script.attach_lr_schedule(runner, "linear", LR_INIT, LR_MIN, num_iterations=4)
    runner.learn(4)

    rates = [group["lr"] for group in runner.alg.optimizer.param_groups]
    assert rates == pytest.approx([_expected_linear(3 / 4)] * 3)


def test_logged_attribute_matches_the_applied_rate():
    """The attribute RSL-RL logs must equal the rate Adam ran at."""
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=6)
    runner.learn(6)

    assert runner.alg.attribute_lrs == pytest.approx(runner.alg.optimizer_lrs)


def test_decay_shrinks_the_parameter_step():
    """The rate reaches the weights, not only the ``param_groups`` dict."""
    runner = _RecordingRunner(apply_step=True)
    train_script.attach_lr_schedule(runner, "linear", LR_INIT, LR_MIN, num_iterations=12)
    runner.learn(12)

    steps = runner.alg.step_sizes
    assert len(steps) == 12
    assert all(later < earlier for earlier, later in zip(steps, steps[1:]))
    assert steps[-1] < 0.2 * steps[0]


def test_original_update_is_still_called_and_its_result_returned():
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=3)

    assert runner.alg.update() == {"loss": 0.0}
    assert len(runner.alg.optimizer_lrs) == 1


def test_unknown_schedule_is_rejected_at_attach_time():
    runner = _RecordingRunner()
    with pytest.raises(ValueError, match="unknown LR schedule"):
        train_script.attach_lr_schedule(runner, "exponential", LR_INIT, LR_MIN, num_iterations=10)


def test_adaptive_algorithm_schedule_is_rejected_at_attach_time():
    """The KL controller rewrites ``param_groups`` from inside ``update()``.

    It would overwrite the decay after the wrapper has written it, reproducing
    the logged-versus-applied divergence this schedule exists to remove.
    """
    runner = _RecordingRunner(schedule="adaptive")
    with pytest.raises(ValueError, match="requires algorithm schedule 'fixed'"):
        train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=10)


def test_resume_without_the_flag_does_not_inherit_a_decayed_rate():
    """A checkpoint from a scheduled leg carries ``lr_min`` in its optimizer state.

    Resumed without ``--lr_schedule``, nothing rewrites it under
    ``schedule="fixed"``, so the leg would run ~30x slow against a config that
    says otherwise, and with no banner marking the difference.
    """
    runner = _RecordingRunner()
    runner.load(checkpoint_lr=LR_MIN, iteration=9999)
    # What load() leaves behind, and what the leg would run at unreset.
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(LR_MIN)

    replaced = train_script.reset_lr_after_resume(runner, LR_INIT, schedule=None)

    assert replaced == pytest.approx(LR_MIN), "the replaced rate is reported for the console record"
    runner.learn(3)
    assert runner.alg.optimizer_lrs == pytest.approx([LR_INIT] * 3)


def test_no_reset_is_reported_when_the_checkpoint_matches_the_config():
    runner = _RecordingRunner()
    runner.load(checkpoint_lr=LR_INIT, iteration=500)

    assert train_script.reset_lr_after_resume(runner, LR_INIT, schedule=None) is None
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(LR_INIT)


def test_an_adaptive_algorithm_keeps_ownership_of_a_resumed_rate():
    """The KL controller adapts from the restored rate; the reset must not clobber it."""
    runner = _RecordingRunner(schedule="adaptive")
    runner.load(checkpoint_lr=5.0e-5, iteration=500)

    assert train_script.reset_lr_after_resume(runner, LR_INIT, schedule=None) is None
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-5)


def test_the_schedule_keeps_ownership_of_a_resumed_rate():
    """With the flag there is nothing to reset — the schedule writes before its first update."""
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=6000)
    runner.load(checkpoint_lr=LR_MIN, iteration=9999)

    assert train_script.reset_lr_after_resume(runner, LR_INIT, schedule="cosine") is None
    runner.learn(1)
    assert runner.alg.optimizer_lrs[0] == pytest.approx(_expected_cosine(9999 / 15999))


def test_set_learning_rate_writes_every_param_group_and_the_attribute():
    runner = _RecordingRunner(num_param_groups=3)
    runner.load(checkpoint_lr=LR_MIN, iteration=500)

    train_script.set_learning_rate(runner, LR_INIT)

    assert [g["lr"] for g in runner.alg.optimizer.param_groups] == pytest.approx([LR_INIT] * 3)
    assert runner.alg.learning_rate == pytest.approx(LR_INIT)


def test_scheduled_resume_overrides_the_checkpoint_rate_on_the_first_update():
    """With the flag, the schedule owns the rate from the first update on."""
    start, num_iterations = 9999, 6000
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "cosine", LR_INIT, LR_MIN, num_iterations=num_iterations)
    runner.load(checkpoint_lr=LR_MIN, iteration=start)
    runner.learn(num_iterations)

    expected = _expected_cosine(start / (start + num_iterations))
    assert runner.alg.optimizer_lrs[0] == pytest.approx(expected)
    assert runner.alg.optimizer_lrs[0] > 5 * LR_MIN


def test_lr_at_progress_spans_the_curve():
    assert train_script.lr_at_progress("cosine", LR_INIT, LR_MIN, 0.0) == pytest.approx(LR_INIT)
    assert train_script.lr_at_progress("cosine", LR_INIT, LR_MIN, 1.0) == pytest.approx(LR_MIN)
    assert train_script.lr_at_progress("linear", LR_INIT, LR_MIN, 0.5) == pytest.approx(
        0.5 * (LR_INIT + LR_MIN)
    )
    with pytest.raises(ValueError, match="unknown LR schedule"):
        train_script.lr_at_progress("exponential", LR_INIT, LR_MIN, 0.0)


def test_the_final_iteration_lands_one_step_short_of_lr_min():
    """The loop is exclusive of ``start + num_iterations``, so progress never reaches 1.

    The last iteration executed is ``start + num_iterations - 1``, leaving the
    final rate exactly one schedule step above ``lr_min`` — negligible at a real
    iteration count, but it is what the code does and the docstring says so.
    """
    runner = _RecordingRunner()
    train_script.attach_lr_schedule(runner, "linear", LR_INIT, LR_MIN, num_iterations=100)
    runner.learn(100)

    final = runner.alg.optimizer_lrs[-1]
    assert final == pytest.approx(_expected_linear(99 / 100))
    assert final > LR_MIN
    assert final - LR_MIN == pytest.approx((LR_INIT - LR_MIN) / 100)
