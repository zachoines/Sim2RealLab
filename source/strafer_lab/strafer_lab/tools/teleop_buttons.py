"""Episode-end button-to-outcome translator for the harness teleop driver.

Pure-Python — extracted from the in-process teleop loop so the mapping
table from the harness brief's "Episode-end button mapping" section is
unit-testable from ``.venv_harness`` without booting Isaac Sim.

The translator takes a :class:`GamepadFrame`-shaped dict (buttons +
D-pad) and returns a :class:`EpisodeEndDecision` carrying the
``(outcome, outcome_category, hard_negative_category, keep)`` tuple the
writer needs on ``end_episode``.

Mapping table (mirror of
``docs/tasks/active/harness/harness-architecture.md`` §Episode-end
button mapping)::

  Y (north)             succeeded            on_course           null      keep
  B (east)              failed               on_course           null      keep
  X + D-pad up/down     wrong_instance       wrong_instance      wrong_instance  keep
  X + D-pad left/right  wrong_room           wrong_room          wrong_room      keep
  SELECT (share)        trajectory_violation trajectory_violation trajectory_violation keep
  Back (xbox view)      —                    —                   —         discard

The D-pad direction picks between the two ``X`` sub-modes:

- D-pad **up or down** → ``wrong_instance`` (same room, wrong specific
  object; the operator is pointing along the local axis at a sibling)
- D-pad **left or right** → ``wrong_room`` (operator is gesturing
  laterally toward a different room)

If ``X`` is held but no D-pad direction is registered, the decision is
``pending`` and the translator returns ``None`` — the loop should keep
polling rather than commit a half-specified hard negative. This is
intentional: the harness brief says the operator commits to the
specific failure mode at capture time, so the chord must be fully
specified before the episode is closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_VALID_OUTCOMES = {
    "succeeded", "failed",
    "wrong_instance", "wrong_room",
    "trajectory_violation", "discarded",
}


@dataclass(frozen=True)
class EpisodeEndDecision:
    """Decision returned by :func:`button_state_to_episode_outcome`.

    ``discard=True`` means ``end_episode(discard=True)`` — the episode
    index does not advance and nothing is persisted. Other outcomes
    flow into the LeRobot writer's ``end_episode`` kwargs verbatim.
    """

    outcome: str
    outcome_category: str
    hard_negative_category: str | None
    discard: bool

    def __post_init__(self) -> None:
        # Treat as soft assertions so the loop can fail loudly if it
        # ever constructs a decision outside the schema.
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"outcome {self.outcome!r} not in {sorted(_VALID_OUTCOMES)}",
            )


def button_state_to_episode_outcome(
    buttons: Mapping[str, bool],
    *,
    dpad_x: int = 0,
    dpad_y: int = 0,
) -> EpisodeEndDecision | None:
    """Translate one tick of button state into an episode-end decision.

    Returns ``None`` when the current chord is not yet committal — the
    operator has pressed ``X`` but not yet picked a D-pad direction, for
    example. The caller's polling loop should treat ``None`` as "keep
    capturing" and re-poll on the next tick.

    Button precedence (highest first):

    1. ``back`` → discard (always wins; lets the operator scrub a bad
       chord by tapping Back before releasing the other buttons).
    2. ``select`` → trajectory_violation.
    3. ``y`` → succeeded.
    4. ``b`` → failed.
    5. ``x`` + D-pad → wrong_instance / wrong_room (committal only when
       D-pad is non-zero).

    ``A`` / ``Start`` are reserved by the driver loop for other purposes
    (keep-on-reset, save-and-quit) and are intentionally ignored here.
    """
    if buttons.get("back"):
        return EpisodeEndDecision(
            outcome="discarded",
            outcome_category="discarded",
            hard_negative_category=None,
            discard=True,
        )
    if buttons.get("select"):
        return EpisodeEndDecision(
            outcome="trajectory_violation",
            outcome_category="trajectory_violation",
            hard_negative_category="trajectory_violation",
            discard=False,
        )
    if buttons.get("y"):
        return EpisodeEndDecision(
            outcome="succeeded",
            outcome_category="on_course",
            hard_negative_category=None,
            discard=False,
        )
    if buttons.get("b"):
        return EpisodeEndDecision(
            outcome="failed",
            outcome_category="on_course",
            hard_negative_category=None,
            discard=False,
        )
    if buttons.get("x"):
        # Committal only when the D-pad has been pushed in one of the
        # four cardinals. Otherwise, withhold the decision.
        if dpad_y != 0:
            return EpisodeEndDecision(
                outcome="wrong_instance",
                outcome_category="wrong_instance",
                hard_negative_category="wrong_instance",
                discard=False,
            )
        if dpad_x != 0:
            return EpisodeEndDecision(
                outcome="wrong_room",
                outcome_category="wrong_room",
                hard_negative_category="wrong_room",
                discard=False,
            )
        return None  # X held, waiting for D-pad direction.
    return None


def describe_button_layout() -> str:
    """Human-readable layout description for the console banner.

    The harness teleop driver prints this once on startup so the
    operator does not have to keep the brief open while teleoping.
    """
    return (
        "Episode-end buttons:\n"
        "  Y (triangle)       → succeed (kept)\n"
        "  B (circle)         → fail (kept)\n"
        "  X + D-pad ↑/↓      → wrong_instance hard negative (kept)\n"
        "  X + D-pad ←/→      → wrong_room hard negative (kept)\n"
        "  SELECT (share)     → trajectory_violation (kept)\n"
        "  Back / View        → discard (not persisted)\n"
        "Other:\n"
        "  Left stick         → world-frame velocity (vx, vy)\n"
        "  Right stick X      → angular velocity (omega_z)\n"
        "  Start (held 1s)    → save + quit cleanly"
    )
