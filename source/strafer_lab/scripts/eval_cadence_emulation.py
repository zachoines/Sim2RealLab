#!/usr/bin/env python3
"""Evaluate a checkpoint in closed-loop sim under an emulated inference cadence.

The deployed inference node runs one inference per *fresh* depth frame. A tick
whose depth frame has not advanced skips inference entirely, publishes nothing
(so the chassis keeps executing the last command), and leaves the recurrent
hidden state untouched. Separately, a publisher that stamps faster than it
renders satisfies that freshness gate with duplicate pixels, so an inference can
run on depth content identical to the previous one.

This harness reproduces both effects inside the sim rollout, which ticks at a
fixed rate with every frame fresh. Each env draws its own tick schedule:

    fresh  -- the policy runs on the live observation
    stale  -- the policy runs on live scalars and a cached depth block
    held   -- the policy does not run; the previous action is re-issued and the
              env's recurrent hidden state is left where it was

The held fraction sets the effective inference rate; the stale fraction sets the
duplicate-content rate among the inferences that do run. They are independent
axes and are requested, realized, and reported separately.

Held ticks are emulated by running the batched forward for every env and then
restoring the held envs' hidden-state columns and overwriting their action rows.
That is equivalent to never having called the policy for those envs -- the
recurrent state is independent across the batch dimension -- and it keeps the
batch width constant, so no env's convolution numerics shift with its schedule.

Examples:
    # Baseline plus two degraded profiles for one checkpoint, one session:
    $ISAACLAB -p source/strafer_lab/scripts/eval_cadence_emulation.py \\
        --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \\
        --checkpoint logs/rsl_rl/strafer_navigation/run_20260727_171735/model_998.pt \\
        --profile clean,band,degraded --num_envs 16 --episodes 100 --headless

    # Replay a temporal profile measured from an inference-node observation dump:
    $ISAACLAB -p source/strafer_lab/scripts/eval_cadence_emulation.py \\
        --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \\
        --checkpoint logs/rsl_rl/strafer_navigation/run_20260727_171735/model_998.pt \\
        --profile measured --profile-dump captures/inference_obs.jsonl --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Sequence

import numpy as np

TICK_FRESH = 0
TICK_STALE = 1
TICK_HELD = 2

# Ranked so an episode that trips several terms on one step gets the most
# informative label; arrival is the success signal and outranks the rest.
TERMINATION_PRIORITY = (
    "path_complete",
    "off_path_divergence",
    "sustained_collision",
    "robot_flipped",
    "time_out",
)


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------


def field_slices(variant) -> dict[str, slice]:
    """Map each ``PolicyVariant`` field key to its slice of the flat obs vector.

    The sim's policy observation group concatenates its terms in declaration
    order, which matches the variant's field order term-for-term.
    """
    slices: dict[str, slice] = {}
    offset = 0
    for obs_field in variant.fields:
        slices[obs_field.key] = slice(offset, offset + obs_field.dims)
        offset += obs_field.dims
    if offset != variant.obs_dim:
        raise ValueError(
            f"field walk produced {offset} dims but {variant.name}.obs_dim is "
            f"{variant.obs_dim}"
        )
    return slices


def bearing_key(variant) -> str:
    """Key of the signed heading-to-referent field for ``variant``."""
    for obs_field in variant.fields:
        if obs_field.key.endswith(("heading_to_goal", "heading_to_subgoal")):
            return obs_field.key
    raise ValueError(f"{variant.name} has no signed heading field")


def relative_key(variant) -> str:
    """Key of the two-component body-frame referent offset for ``variant``."""
    for obs_field in variant.fields:
        if obs_field.key in ("goal_relative", "subgoal_relative"):
            return obs_field.key
    raise ValueError(f"{variant.name} has no relative-position field")


# ---------------------------------------------------------------------------
# Direction offset
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap radians into (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def signed_direction_offset(
    action_xy: np.ndarray,
    bearing_rad: np.ndarray,
    *,
    min_command: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed angle from the referent bearing to the commanded direction.

    ``action_xy`` is the raw normalized ``(vx, vy)`` policy output. Both
    components carry the same velocity scale and the downstream L1 clamp scales
    them jointly, so the commanded direction is ``atan2(vy, vx)`` with no
    denormalization. Positive offset means the command points counter-clockwise
    (left) of the referent.

    Returns ``(offset, valid)``; ``valid`` is False where the commanded speed is
    at or below ``min_command``, since the direction is undefined there.
    """
    action_xy = np.asarray(action_xy, dtype=np.float64)
    bearing_rad = np.asarray(bearing_rad, dtype=np.float64)
    speed = np.hypot(action_xy[..., 0], action_xy[..., 1])
    commanded = np.arctan2(action_xy[..., 1], action_xy[..., 0])
    return wrap_to_pi(commanded - bearing_rad), speed > min_command


def summarize_offsets(offsets: Sequence[float]) -> dict:
    """Median, spread, and left-share of a signed-offset sample."""
    arr = np.asarray(offsets, dtype=np.float64)
    if arr.size == 0:
        return {
            "samples": 0,
            "median_deg": None,
            "mean_deg": None,
            "p10_deg": None,
            "p90_deg": None,
            "median_abs_deg": None,
            "fraction_left": None,
        }
    deg = np.degrees(arr)
    return {
        "samples": int(arr.size),
        "median_deg": float(np.median(deg)),
        "mean_deg": float(np.mean(deg)),
        "p10_deg": float(np.percentile(deg, 10.0)),
        "p90_deg": float(np.percentile(deg, 90.0)),
        "median_abs_deg": float(np.median(np.abs(deg))),
        "fraction_left": float(np.mean(arr > 0.0)),
    }


# ---------------------------------------------------------------------------
# Temporal profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalProfile:
    """A parametric tick-schedule specification.

    Attributes:
        name: Label carried into the results.
        hold_fraction: Long-run share of ticks on which no inference runs.
        mean_hold_run: Mean length in ticks of a base hold run.
        burst_weight: Share of hold runs drawn from the burst component.
        mean_burst_run: Mean length in ticks of a burst hold run.
        stale_fraction: Share of the *inferring* ticks fed a duplicate depth
            block.
        mean_stale_run: Mean length in inferring ticks of a duplicate run.
    """

    name: str
    hold_fraction: float = 0.0
    mean_hold_run: float = 1.0
    burst_weight: float = 0.0
    mean_burst_run: float = 1.0
    stale_fraction: float = 0.0
    mean_stale_run: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hold_fraction < 1.0:
            raise ValueError("hold_fraction must be in [0, 1)")
        if not 0.0 <= self.stale_fraction < 1.0:
            raise ValueError("stale_fraction must be in [0, 1)")
        if not 0.0 <= self.burst_weight <= 1.0:
            raise ValueError("burst_weight must be in [0, 1]")
        for name in ("mean_hold_run", "mean_burst_run", "mean_stale_run"):
            if getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be at least 1 tick")
        if self.hold_fraction > 0.0:
            ceiling = self.expected_hold_run() / (self.expected_hold_run() + 1.0)
            if self.hold_fraction > ceiling + 1e-9:
                raise ValueError(
                    f"hold_fraction {self.hold_fraction:.3f} is unreachable with a "
                    f"mean hold run of {self.expected_hold_run():.2f} ticks (max "
                    f"{ceiling:.3f}); lengthen the hold runs"
                )
        if self.stale_fraction > 0.0:
            ceiling = self.mean_stale_run / (self.mean_stale_run + 1.0)
            if self.stale_fraction > ceiling + 1e-9:
                raise ValueError(
                    f"stale_fraction {self.stale_fraction:.3f} is unreachable with a "
                    f"mean stale run of {self.mean_stale_run:.2f} (max {ceiling:.3f})"
                )

    def expected_hold_run(self) -> float:
        base = (1.0 - self.burst_weight) * self.mean_hold_run
        return base + self.burst_weight * self.mean_burst_run

    def expected_inferring_run(self) -> float:
        """Mean length of a run of consecutive inferring ticks."""
        if self.hold_fraction <= 0.0:
            return math.inf
        share = (1.0 - self.hold_fraction) / self.hold_fraction
        return self.expected_hold_run() * share

    def expected_stale_gap(self) -> float:
        """Mean length of a run of consecutive live-depth inferring ticks."""
        if self.stale_fraction <= 0.0:
            return math.inf
        share = (1.0 - self.stale_fraction) / self.stale_fraction
        return self.mean_stale_run * share

    def expected_inference_hz(self, tick_hz: float) -> float:
        return tick_hz * (1.0 - self.hold_fraction)


# ``band`` is the inference rate the rig sustained across historical measurement
# sessions. ``degraded`` is the mid-arm arrival rate and duplicate share of the
# slowest session recorded to date; its two fractions are independent -- roughly
# three ticks in five carry no new frame, and of the inferences that do run,
# roughly two in five see pixels identical to the previous inference.
PRESET_PROFILES: dict[str, TemporalProfile] = {
    "clean": TemporalProfile(name="clean"),
    "band": TemporalProfile(
        name="band",
        hold_fraction=0.233,
        mean_hold_run=1.2,
    ),
    "degraded": TemporalProfile(
        name="degraded",
        hold_fraction=0.611,
        mean_hold_run=1.5,
        burst_weight=0.25,
        mean_burst_run=6.0,
        stale_fraction=0.383,
        mean_stale_run=1.0,
    ),
}


@dataclass
class EmpiricalProfile:
    """Tick schedule replayed from measured inter-inference statistics.

    ``interval_ticks`` is the empirical distribution of the gap in ticks between
    consecutive inferences: 1 means back-to-back inferences, k means k-1 held
    ticks in between. ``stale_run_lengths`` is the empirical distribution of
    duplicate-content run lengths, counted as the number of repeat inferences
    following the frame that introduced the content.
    """

    name: str
    interval_ticks: np.ndarray
    stale_run_lengths: np.ndarray
    clean_run_lengths: np.ndarray
    source: str = ""
    records: int = 0

    def hold_fraction(self) -> float:
        if self.interval_ticks.size == 0:
            return 0.0
        return float(1.0 - 1.0 / np.mean(self.interval_ticks))

    def stale_fraction(self) -> float:
        total = self.stale_run_lengths.sum() + self.clean_run_lengths.sum()
        if total == 0:
            return 0.0
        return float(self.stale_run_lengths.sum() / total)

    def expected_inference_hz(self, tick_hz: float) -> float:
        return tick_hz * (1.0 - self.hold_fraction())


def _run_lengths(flags: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Split a boolean sequence into its runs of True and runs of False."""
    true_runs: list[int] = []
    false_runs: list[int] = []
    current: bool | None = None
    length = 0
    for raw in flags:
        flag = bool(raw)
        if flag == current:
            length += 1
            continue
        if current is True:
            true_runs.append(length)
        elif current is False:
            false_runs.append(length)
        current = flag
        length = 1
    if current is True:
        true_runs.append(length)
    elif current is False:
        false_runs.append(length)
    return (
        np.asarray(true_runs, dtype=np.int64),
        np.asarray(false_runs, dtype=np.int64),
    )


def load_dump_profile(
    path: str,
    step_dt: float,
    *,
    depth_start: int,
    name: str = "measured",
    max_interval_ticks: int = 200,
) -> EmpiricalProfile:
    """Extract a temporal profile from an inference-node observation dump.

    The dump carries one JSON object per inference with a ``t_sim`` stamp and the
    full flat observation vector, so consecutive ``t_sim`` deltas are the
    inter-inference intervals and bit-identical depth blocks on consecutive lines
    are duplicate-content events.

    Args:
        path: Dump file, one JSON object per line.
        step_dt: Control period the intervals are quantized against.
        depth_start: Index at which the depth block begins in the obs vector.
        name: Label carried into the results.
        max_interval_ticks: Intervals longer than this are dropped as gaps
            between capture segments rather than as inference stalls.
    """
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")

    stamps: list[float] = []
    duplicate_flags: list[bool] = []
    previous_key: bytes | None = None
    records = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if "t_sim" not in record or "obs" not in record:
                raise ValueError(
                    f"{path}:{line_no} lacks the 't_sim'/'obs' pair the dump "
                    f"schema requires"
                )
            obs = record["obs"]
            if len(obs) <= depth_start:
                raise ValueError(
                    f"{path}:{line_no} has {len(obs)} obs dims, too short for a "
                    f"depth block starting at {depth_start}"
                )
            key = np.asarray(obs[depth_start:], dtype=np.float32).tobytes()
            stamps.append(float(record["t_sim"]))
            duplicate_flags.append(previous_key is not None and key == previous_key)
            previous_key = key
            records += 1

    if records < 2:
        raise ValueError(f"{path} holds {records} records; at least 2 are needed")

    deltas = np.diff(np.asarray(stamps, dtype=np.float64))
    if np.any(deltas <= 0.0):
        raise ValueError(
            f"{path} has non-monotonic t_sim; a dump must cover one launch only"
        )
    intervals = np.maximum(1, np.rint(deltas / step_dt).astype(np.int64))
    intervals = intervals[intervals <= max_interval_ticks]
    if intervals.size == 0:
        raise ValueError(f"{path} has no inter-inference interval under the cap")

    # The first record has no predecessor, so its flag carries no information.
    stale_runs, clean_runs = _run_lengths(duplicate_flags[1:])

    return EmpiricalProfile(
        name=name,
        interval_ticks=intervals,
        stale_run_lengths=stale_runs,
        clean_run_lengths=clean_runs,
        source=os.path.abspath(path),
        records=records,
    )


# ---------------------------------------------------------------------------
# Schedule sampling
# ---------------------------------------------------------------------------


def _geometric_draw(rng: np.random.Generator, mean: float, size: int) -> np.ndarray:
    """Draw from a geometric distribution on {1, 2, ...} with the given mean."""
    prob = min(1.0, 1.0 / max(mean, 1.0))
    return rng.geometric(prob, size=size).astype(np.int64)


def _empirical_draw(
    rng: np.random.Generator, samples: np.ndarray, size: int, fallback: int
) -> np.ndarray:
    if samples.size == 0:
        return np.full(size, fallback, dtype=np.int64)
    return rng.choice(samples, size=size, replace=True).astype(np.int64)


_HUGE = np.iinfo(np.int64).max


class ScheduleSampler:
    """Per-env fresh / stale / held tick schedule.

    Each env alternates runs of inferring ticks with runs of held ticks, and
    within the inferring ticks alternates runs of live-depth ticks with runs of
    duplicate-depth ticks. Envs are independent: a hold on one implies nothing
    about any other.

    ``warmup_ticks`` ticks after every episode boundary are forced fresh. A new
    episode is a new mission, and the node cannot infer before that mission's
    first frame arrives; the forced ticks also cover the depth delay buffer's
    zero-filled warm-up, which would otherwise be what a cache captured.
    """

    def __init__(
        self,
        profile: TemporalProfile | EmpiricalProfile,
        num_envs: int,
        rng: np.random.Generator,
        *,
        warmup_ticks: int = 0,
    ) -> None:
        self.profile = profile
        self.num_envs = num_envs
        self.warmup_ticks = max(0, warmup_ticks)
        self._rng = rng

        if isinstance(profile, EmpiricalProfile):
            self._holds_enabled = profile.interval_ticks.size > 0
            self._stale_enabled = profile.stale_run_lengths.size > 0
            self._draw_hold: Callable[[int], np.ndarray] = lambda n: np.maximum(
                0, _empirical_draw(self._rng, profile.interval_ticks, n, 1) - 1
            )
            self._draw_inferring: Callable[[int], np.ndarray] = lambda n: np.ones(
                n, dtype=np.int64
            )
            self._draw_stale: Callable[[int], np.ndarray] = lambda n: _empirical_draw(
                self._rng, profile.stale_run_lengths, n, 1
            )
            self._draw_clean: Callable[[int], np.ndarray] = lambda n: _empirical_draw(
                self._rng, profile.clean_run_lengths, n, 1
            )
        else:
            self._holds_enabled = profile.hold_fraction > 0.0
            self._stale_enabled = profile.stale_fraction > 0.0
            inferring_mean = profile.expected_inferring_run()
            clean_mean = profile.expected_stale_gap()
            self._draw_hold = self._draw_hold_mixture
            self._draw_inferring = lambda n: _geometric_draw(self._rng, inferring_mean, n)
            self._draw_stale = lambda n: _geometric_draw(
                self._rng, profile.mean_stale_run, n
            )
            self._draw_clean = lambda n: _geometric_draw(self._rng, clean_mean, n)

        self._hold_remaining = np.zeros(num_envs, dtype=np.int64)
        self._inferring_remaining = np.zeros(num_envs, dtype=np.int64)
        self._stale_remaining = np.zeros(num_envs, dtype=np.int64)
        self._clean_remaining = np.zeros(num_envs, dtype=np.int64)
        self._warmup_left = np.zeros(num_envs, dtype=np.int64)

        self._observed_kind = np.full(num_envs, -1, dtype=np.int8)
        self._observed_len = np.zeros(num_envs, dtype=np.int64)
        self._tick_counts = np.zeros(3, dtype=np.int64)
        self._hold_runs: list[int] = []
        self._stale_runs: list[int] = []

        self.reset(np.ones(num_envs, dtype=bool))

    def _draw_hold_mixture(self, n: int) -> np.ndarray:
        profile = self.profile
        base = _geometric_draw(self._rng, profile.mean_hold_run, n)
        if profile.burst_weight <= 0.0:
            return base
        burst = _geometric_draw(self._rng, profile.mean_burst_run, n)
        return np.where(self._rng.random(n) < profile.burst_weight, burst, base)

    def reset(self, env_mask: np.ndarray) -> None:
        """Restart the schedule for the masked envs at an episode boundary."""
        idx = np.flatnonzero(np.asarray(env_mask, dtype=bool))
        if idx.size == 0:
            return
        self._flush_runs(idx)
        self._hold_remaining[idx] = 0
        self._stale_remaining[idx] = 0
        self._warmup_left[idx] = self.warmup_ticks
        if self._holds_enabled:
            self._inferring_remaining[idx] = np.maximum(1, self._draw_inferring(idx.size))
        else:
            self._inferring_remaining[idx] = _HUGE
        if self._stale_enabled:
            self._clean_remaining[idx] = np.maximum(1, self._draw_clean(idx.size))
        else:
            self._clean_remaining[idx] = _HUGE

    def _flush_runs(self, idx: np.ndarray) -> None:
        """Close out the in-progress observed runs for the given envs."""
        for env_index in idx.tolist():
            kind = int(self._observed_kind[env_index])
            length = int(self._observed_len[env_index])
            if length <= 0:
                continue
            if kind == TICK_HELD:
                self._hold_runs.append(length)
            elif kind == TICK_STALE:
                self._stale_runs.append(length)
        self._observed_kind[idx] = -1
        self._observed_len[idx] = 0

    def _record(self, kinds: np.ndarray) -> None:
        for kind in (TICK_FRESH, TICK_STALE, TICK_HELD):
            self._tick_counts[kind] += int(np.count_nonzero(kinds == kind))
        changed = np.flatnonzero(kinds != self._observed_kind)
        if changed.size:
            self._flush_runs(changed)
        self._observed_kind[:] = kinds
        self._observed_len += 1

    def next(self) -> np.ndarray:
        """Advance one tick and return the per-env tick kind."""
        kinds = np.full(self.num_envs, TICK_FRESH, dtype=np.int8)

        warming = self._warmup_left > 0
        self._warmup_left[warming] -= 1
        active = np.flatnonzero(~warming)

        if active.size:
            hold_now = self._hold_remaining[active] > 0
            holding = active[hold_now]
            inferring = active[~hold_now]
            kinds[holding] = TICK_HELD
            self._hold_remaining[holding] -= 1

            if inferring.size and self._stale_enabled:
                stale_now = self._stale_remaining[inferring] > 0
                stale_idx = inferring[stale_now]
                clean_idx = inferring[~stale_now]
                kinds[stale_idx] = TICK_STALE
                self._stale_remaining[stale_idx] -= 1
                if clean_idx.size:
                    self._clean_remaining[clean_idx] -= 1
                    spent = clean_idx[self._clean_remaining[clean_idx] <= 0]
                    if spent.size:
                        self._stale_remaining[spent] = np.maximum(
                            1, self._draw_stale(spent.size)
                        )
                        self._clean_remaining[spent] = np.maximum(
                            1, self._draw_clean(spent.size)
                        )

            if inferring.size and self._holds_enabled:
                self._inferring_remaining[inferring] -= 1
                spent = inferring[self._inferring_remaining[inferring] <= 0]
                if spent.size:
                    self._hold_remaining[spent] = np.maximum(0, self._draw_hold(spent.size))
                    self._inferring_remaining[spent] = np.maximum(
                        1, self._draw_inferring(spent.size)
                    )

        self._record(kinds)
        return kinds

    def realized(self, tick_hz: float) -> dict:
        """Realized schedule statistics, for auditing against the request."""
        self._flush_runs(np.arange(self.num_envs))
        fresh, stale, held = (int(count) for count in self._tick_counts)
        ticks = fresh + stale + held
        inferring = fresh + stale
        hold_runs = np.asarray(self._hold_runs, dtype=np.float64)
        stale_runs = np.asarray(self._stale_runs, dtype=np.float64)
        return {
            "ticks": ticks,
            "fresh_ticks": fresh,
            "stale_ticks": stale,
            "held_ticks": held,
            "hold_fraction": (held / ticks) if ticks else 0.0,
            "stale_fraction": (stale / inferring) if inferring else 0.0,
            "inference_hz": (tick_hz * inferring / ticks) if ticks else 0.0,
            "mean_hold_run": float(hold_runs.mean()) if hold_runs.size else 0.0,
            "max_hold_run": int(hold_runs.max()) if hold_runs.size else 0,
            "hold_runs": int(hold_runs.size),
            "mean_stale_run": float(stale_runs.mean()) if stale_runs.size else 0.0,
            "max_stale_run": int(stale_runs.max()) if stale_runs.size else 0,
            "stale_runs": int(stale_runs.size),
        }


def requested_profile_summary(
    profile: TemporalProfile | EmpiricalProfile, tick_hz: float
) -> dict:
    """The profile as requested, shaped like the realized statistics."""
    if isinstance(profile, EmpiricalProfile):
        gaps = np.maximum(0, profile.interval_ticks - 1)
        gaps = gaps[gaps > 0]
        return {
            "kind": "measured",
            "name": profile.name,
            "source": profile.source,
            "records": profile.records,
            "hold_fraction": profile.hold_fraction(),
            "stale_fraction": profile.stale_fraction(),
            "inference_hz": profile.expected_inference_hz(tick_hz),
            "mean_hold_run": float(gaps.mean()) if gaps.size else 0.0,
            "mean_stale_run": (
                float(np.mean(profile.stale_run_lengths))
                if profile.stale_run_lengths.size
                else 0.0
            ),
        }
    return {
        "kind": "parametric",
        "name": profile.name,
        "hold_fraction": profile.hold_fraction,
        "stale_fraction": profile.stale_fraction,
        "inference_hz": profile.expected_inference_hz(tick_hz),
        "mean_hold_run": profile.expected_hold_run(),
        "mean_inferring_run": profile.expected_inferring_run(),
        "mean_stale_run": profile.mean_stale_run,
        "burst_weight": profile.burst_weight,
        "mean_burst_run": profile.mean_burst_run,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_arm(episodes: Sequence[dict]) -> dict:
    """Completion, progress, and cause breakdown over an arm's episodes."""
    total = len(episodes)
    if total == 0:
        return {"episodes": 0}
    causes: dict[str, int] = {}
    for episode in episodes:
        causes[episode["cause"]] = causes.get(episode["cause"], 0) + 1
    progress = np.asarray([ep["progress_fraction"] for ep in episodes], dtype=np.float64)
    arc = np.asarray([ep["along_track_m"] for ep in episodes], dtype=np.float64)
    steps = np.asarray([ep["steps"] for ep in episodes], dtype=np.float64)
    complete = causes.get("path_complete", 0)
    return {
        "episodes": total,
        "completions": complete,
        "completion_rate": complete / total,
        "cause_counts": causes,
        "cause_fractions": {key: value / total for key, value in causes.items()},
        "progress_fraction_mean": float(progress.mean()),
        "progress_fraction_median": float(np.median(progress)),
        "along_track_m_mean": float(arc.mean()),
        "along_track_m_median": float(np.median(arc)),
        "near_arrival_rate": float(np.mean(progress >= 0.95)),
        "steps_mean": float(steps.mean()),
    }


def _cell(value, spec: str, width: int) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return format(value, spec)


def format_table(arms: Sequence[dict]) -> str:
    """One row per arm: cadence, completion, progress, direction offset."""
    header = (
        f"{'arm':<12} {'Hz':>6} {'hold':>6} {'dup':>6} {'eps':>5} {'compl':>7} "
        f"{'near':>6} {'prog':>6} {'offset':>8} {'left':>6}"
    )
    rows = [header, "-" * len(header)]
    for arm in arms:
        summary = arm["summary"]
        if summary.get("episodes", 0) == 0:
            rows.append(f"{arm['arm']:<12} {'no episodes scored':>44}")
            continue
        realized = arm["realized_profile"]
        offset = arm["direction_offset"]
        rows.append(
            f"{arm['arm']:<12} "
            f"{realized['inference_hz']:>6.2f} "
            f"{realized['hold_fraction']:>6.3f} "
            f"{realized['stale_fraction']:>6.3f} "
            f"{summary['episodes']:>5d} "
            f"{summary['completion_rate']:>7.3f} "
            f"{summary['near_arrival_rate']:>6.3f} "
            f"{summary['progress_fraction_mean']:>6.3f} "
            f"{_cell(offset['median_deg'], '>+8.2f', 8)} "
            f"{_cell(offset['fraction_left'], '>6.3f', 6)}"
        )
    return "\n".join(rows)


def format_cause_table(arms: Sequence[dict]) -> str:
    """Per-arm episode-end cause breakdown."""
    header = f"{'arm':<12} " + " ".join(f"{name[:15]:>16}" for name in TERMINATION_PRIORITY)
    rows = [header, "-" * len(header)]
    for arm in arms:
        fractions = arm["summary"].get("cause_fractions", {})
        cells = " ".join(
            f"{fractions.get(name, 0.0):>16.3f}" for name in TERMINATION_PRIORITY
        )
        rows.append(f"{arm['arm']:<12} {cells}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OVERRIDE_FIELDS = (
    "hold_fraction",
    "mean_hold_run",
    "burst_weight",
    "mean_burst_run",
    "stale_fraction",
    "mean_stale_run",
)


def profile_overrides(args: argparse.Namespace) -> dict:
    """Collect the profile knobs the caller set explicitly."""
    overrides = {}
    for name in _OVERRIDE_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return overrides


def resolve_profiles(
    names: Sequence[str],
    *,
    overrides: dict | None = None,
    dump_loader: Callable[[str], EmpiricalProfile] | None = None,
) -> list[TemporalProfile | EmpiricalProfile]:
    """Turn profile names into profile objects, applying any overrides.

    ``measured`` resolves through ``dump_loader``; every other name must be a
    preset. Overrides apply to parametric profiles only.
    """
    overrides = overrides or {}
    resolved: list[TemporalProfile | EmpiricalProfile] = []
    for name in names:
        if name == "measured":
            if dump_loader is None:
                raise ValueError("the 'measured' profile needs an observation dump")
            resolved.append(dump_loader(name))
            continue
        if name not in PRESET_PROFILES:
            known = sorted(PRESET_PROFILES) + ["measured"]
            raise ValueError(f"unknown profile {name!r}; choose from {known}")
        profile = PRESET_PROFILES[name]
        if overrides:
            profile = replace(profile, **overrides)
        resolved.append(profile)
    return resolved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint under an emulated inference cadence"
    )
    parser.add_argument("--env", type=str, required=True, help="Registered task ID")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a model_*.pt produced by train_strafer_navigation.py",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="clean",
        help="Comma-separated profiles, run in order within one session: "
        f"{', '.join(sorted(PRESET_PROFILES))}, measured",
    )
    parser.add_argument(
        "--profile-dump",
        type=str,
        default=None,
        help="Observation-dump JSONL the 'measured' profile is replayed from",
    )
    parser.add_argument("--hold-fraction", type=float, default=None,
                        help="Override the profile's held-tick share")
    parser.add_argument("--mean-hold-run", type=float, default=None,
                        help="Override the mean hold-run length in ticks")
    parser.add_argument("--burst-weight", type=float, default=None,
                        help="Override the share of hold runs drawn from the burst arm")
    parser.add_argument("--mean-burst-run", type=float, default=None,
                        help="Override the mean burst hold-run length in ticks")
    parser.add_argument("--stale-fraction", type=float, default=None,
                        help="Override the duplicate-depth share of inferring ticks")
    parser.add_argument("--mean-stale-run", type=float, default=None,
                        help="Override the mean duplicate-depth run length")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override scene.num_envs (default: the cfg's play count)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes to score per arm before moving on (default: 100)")
    parser.add_argument("--max-steps", type=int, default=6000,
                        help="Per-arm step ceiling if the episode budget is not met")
    parser.add_argument("--episode-length-s", type=float, default=None,
                        help="Override the episode cap in seconds (default: the cfg's)")
    parser.add_argument("--warmup-ticks", type=int, default=2,
                        help="Ticks forced fresh after each episode boundary, covering "
                             "the depth delay buffer's zero-filled warm-up")
    parser.add_argument("--min-command", type=float, default=0.05,
                        help="Normalized commanded speed below which the direction "
                             "offset is undefined and the tick leaves that metric")
    parser.add_argument("--disable-obs-corruption", action="store_true",
                        help="Turn the policy observation group's noise models off, "
                             "isolating the emulated cadence from the trained noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str,
                        default="logs/rsl_rl/strafer_navigation/cadence_emulation",
                        help="Directory for the results JSONL")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _run_arm(
    *,
    env,
    policy,
    torch,
    profile,
    sampler: ScheduleSampler,
    depth_span: slice,
    bearing_index: int,
    relative_span: slice,
    heading_scale: float,
    episode_budget: int,
    step_budget: int,
    min_command: float,
    tick_hz: float,
) -> dict:
    """Roll one profile out to its episode budget and score it."""
    num_envs = env.num_envs
    device = env.unwrapped.device

    command_term = env.unwrapped.command_manager.get_term("goal_command")
    termination_manager = env.unwrapped.termination_manager
    active_terms = set(termination_manager.active_terms)
    cursor = command_term.path_cursor
    rnn = getattr(policy, "rnn", None)

    episodes: list[dict] = []
    offsets: list[float] = []
    offsets_relative: list[float] = []
    dropped_offsets = 0

    ep_steps = np.zeros(num_envs, dtype=np.int64)
    ep_kinds = np.zeros((num_envs, 3), dtype=np.int64)
    ep_progress = np.zeros(num_envs, dtype=np.float64)
    ep_arc = np.zeros(num_envs, dtype=np.float64)

    obs = env.get_observations()
    action_dim = env.unwrapped.action_manager.total_action_dim
    prev_actions = torch.zeros(num_envs, action_dim, device=device)
    depth_cache = obs["policy"][:, depth_span].clone()

    steps = 0
    while len(episodes) < episode_budget and steps < step_budget:
        steps += 1
        kinds = sampler.next()
        held_mask = kinds == TICK_HELD
        stale_mask = kinds == TICK_STALE
        fresh_mask = kinds == TICK_FRESH

        held_idx = torch.as_tensor(np.flatnonzero(held_mask), device=device,
                                   dtype=torch.long)
        stale_idx = torch.as_tensor(np.flatnonzero(stale_mask), device=device,
                                    dtype=torch.long)
        fresh_idx = torch.as_tensor(np.flatnonzero(fresh_mask), device=device,
                                    dtype=torch.long)
        scored = np.flatnonzero(~held_mask)

        with torch.inference_mode():
            live_flat = obs["policy"]
            policy_flat = live_flat
            if stale_idx.numel():
                policy_flat = live_flat.clone()
                policy_flat[stale_idx, depth_span] = depth_cache[stale_idx]
            policy_obs = dict(obs)
            policy_obs["policy"] = policy_flat

            hidden_before = None
            if rnn is not None and rnn.hidden_state is not None:
                hidden_before = rnn.hidden_state.clone()

            actions = policy(policy_obs)

            if held_idx.numel():
                if hidden_before is not None:
                    rnn.hidden_state[:, held_idx, :] = hidden_before[:, held_idx, :]
                actions[held_idx] = prev_actions[held_idx]

            if fresh_idx.numel():
                depth_cache[fresh_idx] = live_flat[fresh_idx][:, depth_span]

            # Scored off the observation the policy consumed: the step swaps in
            # a reset episode's values for any env that terminates on it.
            if scored.size:
                scored_idx = torch.as_tensor(scored, device=device, dtype=torch.long)
                bearing = (
                    live_flat[scored_idx, bearing_index].detach().float().cpu().numpy()
                    / heading_scale
                )
                relative = (
                    live_flat[scored_idx][:, relative_span].detach().float().cpu().numpy()
                )
                action_xy = actions[scored_idx][:, 0:2].detach().float().cpu().numpy()
                offset, valid = signed_direction_offset(
                    action_xy, bearing, min_command=min_command
                )
                offsets.extend(offset[valid].tolist())
                dropped_offsets += int(np.count_nonzero(~valid))
                rel_offset, rel_valid = signed_direction_offset(
                    action_xy,
                    np.arctan2(relative[:, 1], relative[:, 0]),
                    min_command=min_command,
                )
                offsets_relative.extend(rel_offset[rel_valid].tolist())

            prev_actions = actions.clone()

            # The command term's public arc-length readout is a per-tick delta
            # that is dropped once at every path install, so the monotone cursor
            # is the only exact progress signal; both are rewound by the reset
            # inside step(), hence the pre-step snapshot.
            pre_arc = cursor._cursor.detach().float().cpu().numpy().copy()
            pre_total = cursor.total_arc.detach().float().cpu().numpy().copy()

            obs, _, dones, _ = env.step(actions)

            done_np = dones.detach().cpu().numpy().astype(bool)
            fired = {
                name: termination_manager.get_term(name).detach().cpu().numpy().copy()
                for name in TERMINATION_PRIORITY
                if name in active_terms
            }
            policy.reset(dones)

            ep_steps += 1
            ep_kinds[np.arange(num_envs), kinds.astype(np.int64)] += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                progress = np.where(pre_total > 0.0, pre_arc / pre_total, 0.0)
            ep_progress = np.maximum(ep_progress, progress)
            ep_arc = np.maximum(ep_arc, pre_arc)

            if done_np.any():
                for env_index in np.flatnonzero(done_np):
                    cause = "unattributed"
                    for name in TERMINATION_PRIORITY:
                        if name in fired and bool(fired[name][env_index]):
                            cause = name
                            break
                    episodes.append(
                        {
                            "env": int(env_index),
                            "steps": int(ep_steps[env_index]),
                            "cause": cause,
                            "progress_fraction": float(ep_progress[env_index]),
                            "along_track_m": float(ep_arc[env_index]),
                            "fresh_ticks": int(ep_kinds[env_index, TICK_FRESH]),
                            "stale_ticks": int(ep_kinds[env_index, TICK_STALE]),
                            "held_ticks": int(ep_kinds[env_index, TICK_HELD]),
                        }
                    )
                ep_steps[done_np] = 0
                ep_kinds[done_np] = 0
                ep_progress[done_np] = 0.0
                ep_arc[done_np] = 0.0
                sampler.reset(done_np)
                done_idx = torch.as_tensor(np.flatnonzero(done_np), device=device,
                                           dtype=torch.long)
                prev_actions[done_idx] = 0.0
                depth_cache[done_idx] = obs["policy"][done_idx][:, depth_span]

    return {
        "profile": profile.name,
        "steps": steps,
        "episodes": episodes,
        "summary": summarize_arm(episodes),
        "requested_profile": requested_profile_summary(profile, tick_hz),
        "realized_profile": sampler.realized(tick_hz),
        "direction_offset": summarize_offsets(offsets),
        "direction_offset_from_relative": summarize_offsets(offsets_relative),
        "direction_offset_dropped_ticks": dropped_offsets,
        "budget_exhausted": len(episodes) < episode_budget,
    }


def main() -> None:
    args = _parse_args()

    profile_names = [name.strip() for name in args.profile.split(",") if name.strip()]
    if not profile_names:
        raise SystemExit("--profile resolved to an empty list")
    if "measured" in profile_names and args.profile_dump is None:
        raise SystemExit("--profile measured requires --profile-dump")
    if "NoCam" in args.env:
        raise SystemExit(
            "cadence emulation models the depth freshness gate; a camera-free "
            "variant has no gate to emulate"
        )
    args.enable_cameras = True

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import importlib.metadata as _metadata

    from strafer_shared.policy_interface import PolicyVariant

    import strafer_lab  # noqa: F401  (registers envs)

    env_cfg = parse_env_cfg(args.env, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    if args.disable_obs_corruption:
        env_cfg.observations.policy.enable_corruption = False

    agent_cfg = load_cfg_from_registry(args.env, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    env = gym.make(args.env, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if getattr(policy, "training", False):
        raise SystemExit(
            "the inference policy is in train mode; batch rows would couple and "
            "the per-env schedule would be meaningless"
        )

    variant = PolicyVariant.DEPTH_SUBGOAL if "Subgoal" in args.env else PolicyVariant.DEPTH
    slices = field_slices(variant)
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    if obs_dim != variant.obs_dim:
        raise SystemExit(
            f"the env's policy observation is {obs_dim} dims but {variant.name} "
            f"declares {variant.obs_dim}; the depth block cannot be located"
        )
    depth_span = slices["depth_image"]
    bearing_field = bearing_key(variant)
    relative_field = relative_key(variant)
    heading_scale = next(f.scale for f in variant.fields if f.key == bearing_field)

    step_dt = env.unwrapped.step_dt
    tick_hz = 1.0 / step_dt

    def _load_measured(name: str) -> EmpiricalProfile:
        return load_dump_profile(
            args.profile_dump, step_dt, depth_start=depth_span.start, name=name
        )

    profiles = resolve_profiles(
        profile_names,
        overrides=profile_overrides(args),
        dump_loader=_load_measured if args.profile_dump else None,
    )

    corruption = "off" if args.disable_obs_corruption else "on"
    print(
        f"[INFO] {env.num_envs} envs at {tick_hz:.2f} Hz, {args.episodes} episodes per "
        f"arm, observation corruption {corruption}, bearing from {bearing_field}"
    )

    arms: list[dict] = []
    try:
        for arm_index, profile in enumerate(profiles):
            print(f"[INFO] arm {arm_index + 1}/{len(profiles)}: {profile.name}")
            if arm_index > 0:
                env.reset()
                policy.reset()
            sampler = ScheduleSampler(
                profile,
                env.num_envs,
                np.random.default_rng(args.seed + 1000 * arm_index),
                warmup_ticks=args.warmup_ticks,
            )
            result = _run_arm(
                env=env,
                policy=policy,
                torch=torch,
                profile=profile,
                sampler=sampler,
                depth_span=depth_span,
                bearing_index=slices[bearing_field].start,
                relative_span=slices[relative_field],
                heading_scale=heading_scale,
                episode_budget=args.episodes,
                step_budget=args.max_steps,
                min_command=args.min_command,
                tick_hz=tick_hz,
            )
            result.update(
                arm=profile.name,
                checkpoint=os.path.abspath(args.checkpoint),
                env_id=args.env,
                num_envs=int(env.num_envs),
                seed=args.seed,
                obs_corruption=not args.disable_obs_corruption,
                bearing_field=bearing_field,
                warmup_ticks=args.warmup_ticks,
                tick_hz=tick_hz,
            )
            arms.append(result)
            if result["budget_exhausted"]:
                print(
                    f"[WARN] arm {profile.name} hit the {args.max_steps}-step ceiling "
                    f"with {len(result['episodes'])}/{args.episodes} episodes"
                )
    except KeyboardInterrupt:
        print("[WARN] interrupted; reporting the arms completed so far")

    if arms:
        print()
        print(format_table(arms))
        print()
        print(format_cause_table(arms))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"cadence_{timestamp}.jsonl")
        with open(out_path, "w", encoding="utf-8") as handle:
            for arm in arms:
                handle.write(json.dumps(arm) + "\n")
        print(f"\n[INFO] Results: {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
