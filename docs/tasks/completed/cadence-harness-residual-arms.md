# Extend the cadence harness to the residual attribution candidates

**Status:** Shipped 2026-08-08 in `12a1e95` (DGX). The harness landed earlier
in `ff5d93d` (PR #193); this change records the 2026-08-08 evaluation session
and the read-out that closes the brief. Subgoal-frame drift is the dominant
sim-testable contributor — 1× costs 24–28% of completion, monotone across the
sweep — while the recurrent horizon is not implicated at today's horizons.
Timer-driven stale reuse is adopted: at the degraded operating point,
inference-rate parity recovers the entire loss, 0.610 → 0.910 at matched depth
novelty. The composition does not collapse the policy (0.700 of baseline), so
the sim-unreachable axes stay open behind a rig re-validation.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/PENDING

**Type:** investigation (harness extension + one operator-launched run)
**Owner:** DGX
**Priority:** P1 — the enriched-lane advance failure is now unattributed on
every axis that has been tested, and the held retrain has no trigger until one
of these arms implicates a trainable one.
**Estimate:** M (four knobs plus their pure tests in-PR; one evaluation session
on the play env)
**Branch:** `task/cadence-harness-residual-arms`

## Story

As the **coordinator attributing the depth subgoal policy's advance failure**,
I want **the two deploy-only regimes the cadence harness does not yet reproduce
— an unbounded recurrent horizon and a drifting subgoal frame — swept alone, at
magnitude, and composed with the cadence band, alongside a discriminant for a
proposed change to the node's inference semantics**, so that **the attribution
either lands on an axis or is shown to live outside what closed-loop sim can
reach.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

[`cadence-emulation-eval`](cadence-emulation-eval.md) exonerated
the temporal axis as sufficient: at the 22–25 Hz band every recorded deploy
session actually ran in, emulated temporal texture costs ≤3% of completion, and
at the 12 Hz / 36%-duplicate profile it costs ~⅓ — a real loss, but far from the
rig's zero completions. That read-out left the enriched-lane `mission` ✗ result
unexplained by scene, by anchoring semantics, or by temporal texture, and named
four live candidates
([`cadence-emulation-eval.md`'s read-out](cadence-emulation-eval.md#read-out--scored-against-the-pre-registered-rule),
item 2): SLAM-frame anchoring noise, planner path-geometry distribution,
unbounded recurrent-state horizon, and residual observation-chain deltas.

**Two of the four are reachable from the existing harness and two are not.**
The recurrent horizon is one guarded call. The subgoal frame is a rewrite of
four observation dimensions the harness already slices. The planner
path-geometry candidate needs recorded Nav2 plans. The instrument exists and its
statistics already accept an arbitrary polyline
(`source/strafer_lab/strafer_lab/tools/path_statistics.py`, with a
shipped-generator reference distribution); only its convenience entry point
plans, so what is missing is a capture rather than a rewiring, and no such
capture has ever been run. The observation-chain candidate needs the rig's own
dump. Both wait on rig time; this brief takes the two that do not.

**The horizon claim needs correcting before it is tested.** Deploy does reset
the GRU — once per mission, on a new action-server goal
(`strafer_inference/inference_node.py:788-791`, the only `policy.reset()` call
site in the node, per
[recurrent-policy-contract §4](../context/recurrent-policy-contract.md#4-reset-trigger)).
What it never does is reset *within* a mission, so the horizon is
mission-length, not episode-length: `mission_timeout_s` is 60.0 s
(`strafer_inference/config/inference.yaml:58`), which at the nominal 30 Hz tick
is up to 1800 hidden-state advances against training's 600 (20.0 s × 30 Hz).
That is a 3× horizon extension, not an unbounded one — and in the failing 12 Hz
regime the freshness gate suppressed ~61% of ticks, so the *node* advanced state
only ~700 times. A watchdog trip freezes rather than clears the
state, which lengthens it further. Arm A tests horizon length by chaining
episodes without a reset; the quantity to read is the shape of the decay against
chain depth, not the presence of a cliff.

**Arm D is a different kind of arm.** It does not test a failure candidate — it
scores a proposed deploy change before the change is written. Today the node
infers once per fresh depth frame: a tick whose depth has not advanced returns
before observation assembly, so no inference, no publish, and no hidden-state
advance (`inference_node.py:954-956`). The proposal is timer-driven inference at
30 Hz that reuses the last depth frame when none arrived, which restores GRU
rate, proprioception and subgoal freshness, and command rate to training parity
and leaves depth staleness as the only residual axis. The harness already
expresses exactly that as `hold_fraction = 0` with a non-zero `stale_fraction`.
Scoring it here means the node PR is written against a measured number rather
than an argument.

That proposal retires a prior ruling rather than contradicting it silently:
[`inference-cadence-shortfall`](inference-cadence-shortfall.md)
shipped with "the freshness gate is NOT loosened (one inference per fresh depth
frame is preserved exactly, and is now structural)". That choice traded rate
parity for depth freshness, and it was made before any measurement showed which
half was expensive. Arm D is that measurement. If it reads low, the ruling
stands on new evidence rather than on precedent.

## What each arm changes in the harness

Every change is additive inside `source/strafer_lab/scripts/eval_cadence_emulation.py`
and its pure test file. `_run_arm` is already keyword-only and already drivable
against mocks without Kit, so each knob gets tests that do not need a GPU.

### Arm A — recurrent-state horizon

One guarded call: `policy.reset(dones)` (`eval_cadence_emulation.py:1062`)
becomes conditional on a new keyword-only parameter fed by `--no-hidden-reset`.
Environments still reset normally — the auto-reset happens inside `env.step`
before it returns, and nothing in the harness triggers it. The per-arm global
clear in `reset_between_arms` stays unconditional so arms cannot inherit each
other's state.

Two boundary details are pre-registered rather than left to the implementer.
`--warmup-ticks` keeps forcing fresh ticks after each boundary, and the choice
is a knowing one: the harness gives the forced ticks two justifications, and
this arm negates one of them. A new episode is no longer a new mission once the
hidden state carries, so that half lapses; the other half — covering the depth
delay buffer's zero-filled warm-up, which is exactly 2 steps on this tier — does
not, and holding the setting fixed is what keeps the arm comparable to the
baseline. It is retained as a comparability control, not because both of its
reasons survive.

The `prev_actions[done_idx] = 0.0` zeroing stays. At any `--warmup-ticks ≥ 1`
the first post-boundary tick is fresh, so the latch is overwritten before
anything can read it and the zeroing is inert.

The arm needs one new per-episode field: the env-local chain index (how many
episodes that env has completed within the arm), so completion can be read
against chain depth rather than pooled.

### Arm B — subgoal-frame noise

The subgoal quartet is `slice(10, 14)` of the policy observation: relative
subgoal position in the **robot body frame** (dims 10:12, scaled 1/10), scalar
distance (dim 12, scaled 1/10), and signed bearing `atan2(Δy, Δx) − yaw` wrapped
to (−π, π] (dim 13, scaled 1/π). The four are near-redundant — all four are
determined by one body-frame 2-vector on a flat floor — so the perturbation is
applied to that vector and the other three dims recomputed from it. Perturbing
the four independently would produce an observation no geometry can generate.
With a body-frame offset `t` and a yaw error `dθ`, the rewrite is
`rel' = R(dθ)(rel + t)`, then distance and bearing recomputed from `rel'`.

Two things about that rewrite have to be pre-registered rather than discovered.

**It is a planar approximation of a non-planar referent.** The env does not
compute the four dims from one 2-vector: the relative position is a full
quaternion-inverse rotation of the 3-D displacement, the distance is a
world-planar norm, and the bearing subtracts a quaternion-derived yaw. Under any
non-zero roll or pitch those three disagree at second order, so recomputing
distance and bearing from `rel'` replaces the full-quaternion referent with a
planar one and moves the observation even at zero drift. The arm therefore
**skips the rewrite entirely when the drift is exactly zero**, which is what
makes a zero-drift no-op byte-identical and keeps the baseline arm clean; the
planar substitution is a property of the perturbed arms and is recorded, not
hidden.

**The startup layout check does not cover the term being perturbed.**
`verify_term_layout` checks the term count, per-index widths, that the bearing
term's name ends in `heading_to_goal`/`heading_to_subgoal`, and that the
trailing term is the depth image. It never name-checks the relative term — and
`body_velocity_xy` is width-identical, so an equal-width reordering would pass
the check while Arm B silently perturbed body velocity. Extend the check to
name the relative and distance terms before shipping the arm.

The perturbation is a per-env Ornstein-Uhlenbeck process, not i.i.d. jitter:
localization error is integrated, not resampled. The repo already carries the
same reasoning twice — [`goal-noise-training`](../active/trained-policy/goal-noise-training.md)'s drift
component ("integrated, not re-sampled — that matches SLAM drift (cumulative),
not white noise") and the planner's shipped `perturb_waypoints`, which samples
at correlation-length control points because independent per-waypoint noise
"would crinkle the polyline".

**The magnitude class is cited; the time constant is assumed.** The 1× class is
the map→odom movement recorded in the 2026-08-02 session's discarded arm —
0.166 m and 6.7°. That figure reaches this repo only as a derived clause in the
read-out; the measurement protocol, its duration, and any drift rate live in the
Jetson-side findings file and are not recoverable here, so no drift rate can be
cited. The time constant is therefore a stated assumption, **τ = 2.0 s**,
justified only by RTAB-Map's 1–10 Hz `map→odom` refresh; the sweep is on
magnitude, and τ is held fixed so the sweep has one axis. The measurement that
would pin τ is Phase 1 item 6 of
[`domain-randomization-audit`](../active/trained-policy/domain-randomization-audit.md), which is also
where the training-side mechanism for this axis is already designed.

Parameterization at 1×: the 2-D offset is sampled with stationary per-axis
σ = 0.166/√2 = 0.117 m so its RMS displacement is the recorded 0.166 m, and the
yaw offset with stationary σ = 6.7°. `--subgoal-drift-gain` scales both together
at 0.5× / 1× / 2×. **The two knobs are near-equal in effect, so they do not
separate**: only the offset component perpendicular to the bearing produces
bearing error, and its σ is the per-axis 0.117 m, which at the 1.0 m nominal
subgoal lookahead is 6.7° — the same size as the yaw σ. Their combined 1σ
bearing error is ~9.5° at the nominal lookahead and ~11.6° at the ROBUST band's
0.7 m floor. Sweeping them together is therefore deliberate; a position-only and
a heading-only arm would be measuring nearly the same thing, and separating them
is a follow-up.

Three implementation hazards, all load-bearing:

- **Aliasing.** `policy_flat` aliases the env's own observation tensor unless a
  row is stale (`eval_cadence_emulation.py:990`). An in-place quartet write on
  the aliased path would corrupt the env's buffer and poison the metric reads,
  which happen after the forward pass. It cannot reach the depth cache — that
  slice is disjoint — but the perturbation path must still clone
  unconditionally.
- **Referent.** Both existing direction-offset accumulators — the one from the
  bearing dim and the one from the relative dims — read the *unperturbed*
  observation, so both measure command-versus-truth. Arm B adds a perceived
  counterpart of each, from the perturbed quartet. A policy tracking its noisy
  subgoal perfectly is a different finding from one that has stopped tracking,
  and only truth and perceived together separate them.
- **Boundaries.** The perturbation state resets at episode boundaries in every
  arm, including the composed one, so that only the hidden state carries across
  a chain. A new episode is a new path, and re-anchoring at a mission boundary
  is what deploy does too.

Training's own subgoal perturbation is worth stating so the magnitude is read
correctly: `randomize_goal_noise` is `None` for every subgoal variant regardless
of tier, the quartet carries no noise model at any tier, and the only
perturbation the subgoal lane sees is `waypoint_noise_std_m = 0.025 m` applied
at path reset, not per tick. The 1× class is 6.6× that on RMS displacement and
4.7× per axis — and on an axis training never randomized at all.

### Arm C — composition

No new mechanism: `band` profile with A and B both on. It needs one thing
though — **the arm label must fold A and B in**. The harness renames an arm only
when a *profile* field is overridden, so a `band` arm carrying A and B would
still be labelled `band`, which breaks the invariant that an arm's label
identifies what was done to it. Extend the arm naming so the recorded label
carries every active knob.

### Arm D — inference semantics

Expressible with existing knobs plus one launch-ergonomics change, but three
things about the dispatched cell definition have to be corrected first.

**The validation ceiling.** The harness enforces
`stale_fraction ≤ mean_stale_run / (mean_stale_run + 1)`. At `degraded`'s
`mean_stale_run = 1.0` the ceiling is 0.5, so a bare `--stale-fraction 0.61`
raises. The ceiling is really the statement that the mean *fresh* gap between
stale runs cannot fall below one tick, so each cell sets `mean_stale_run` above
the minimum its own fraction admits — 1.57 for 0.61, 3.17 for 0.76. The chosen
values leave mean fresh gaps of 1.28, 1.26 and 3.9 ticks: the two cells carrying
the decision sit only ~0.27 ticks above degenerate, which is acceptable because
the fraction is what the rule reads, but it does mean their duplicate-run law is
close to periodic and must not be described as bursty.

**"The degraded profile's burst lengths" cannot be transplanted.**
`burst_weight` and `mean_burst_run` feed the *hold* mixture only; the stale axis
is a single geometric. At `hold_fraction = 0` the burst knobs are inert by
construction. The cells below therefore use a single-geometric duplicate-run law
and set its mean explicitly — recorded as a deviation from the dispatched
wording, not a silent substitution. Adding a stale burst mixture is possible
(~25 lines plus validation plus tests) and is **out of scope** here: the
load-bearing parameter for the decision rule is the fraction, not the run-length
law.

**The novel-depth arithmetic does not match the dispatched cell.** `degraded` is
`hold 0.611` × `stale 0.383`, so inference runs at 11.7 Hz and *distinct* depth
content reaches the policy at 30 × 0.389 × 0.617 = **7.2 Hz**. A `hold 0 /
stale 0.61` cell infers at 30 Hz and sees distinct content at 30 × 0.39 =
**11.7 Hz** — which equals `degraded`'s *inference* rate, not its novel-content
rate. So that cell is not novelty-matched to `degraded`: it models a world where
every arriving frame is new. The faithful emulation of the proposal *at the
degraded arrival regime* — where the publisher's own duplication is still
present — is `hold 0 / stale 0.76`, which restores 7.2 Hz of novel content while
inferring at 30 Hz. Both run; the decision rule reads the novelty-matched one.

**The 0.61 cell is a second reading, not merely an upper bound.** A world where
every arriving frame carries new content is not hypothetical — it is what the
deploy lane looks like once the sim's render-duplication defect is fixed, and
what a physical D555 looks like already, since the sensor does not emit
duplicate content. So that cell is the projected post-fix operating point and
the closest sim analogue of real hardware, and the decision rule reads it as
such rather than discarding it.

The band-equivalent cell has no such problem: `band` carries no duplicate axis,
so setting the stale fraction to `band`'s own hold fraction — 0.233, not a
rounded 0.22 — makes it novelty-matched by construction at 23.0 Hz.

**Match on realized, not requested.** `--warmup-ticks` dilutes the duplicate
axis by ~2 points, so a requested 0.76 realizes ≈0.74 and delivers ≈7.8 Hz of
distinct content, while `degraded` itself realized hold 0.600 / duplicate 0.363
and so delivered 7.64 Hz rather than the requested 7.20. Those two land within
~2% of each other, which is the match the decision rule needs — but it is a
match of realized against realized and the session must verify it rather than
assume it. If the realized distinct-content rate differs from `degraded`'s
realized 7.64 Hz by more than 5%, the cell is re-requested and re-run before it
is scored.

**Launch ergonomics — deliberately not improved.** Overrides apply to every
parametric profile named in one `--profile` list, so each cell that differs in a
profile field costs its own Kit launch, and the hidden-reset and drift knobs are
not profile fields either. The session is therefore **ten launches**: nine for
v2, one for v1. Per-arm override syntax in `resolve_profiles` (for example
`degraded:stale_fraction=0.76`) would collapse the three duplicate-axis cells
into one and bring it to eight, and extending the same syntax to the new knobs
would take it to two. Neither is built here: an ergonomics change to a
just-verified instrument, made immediately before the measurement that
instrument exists to make, is not worth two Kit launches. That syntax is the
shape to build if the harness sees continued service afterwards.

## Acceptance criteria

- [ ] Pure tests cover: the guarded hidden-state reset in both states; the
      drift process's stationary statistics and its time constant against the
      step period; the quartet rewrite's self-consistency (distance and bearing
      agree with the perturbed relative vector); the zero-drift skip, which must
      be byte-identical; the extended term-layout check rejecting an
      equal-width reordering; the unconditional clone on the perturbation path;
      the arm-label folding; and every duplicate-axis cell's parameters against
      the validation ceiling.
- [ ] The label invariant holds: an arm labelled `clean` is the untouched
      baseline, and every arm carrying a knob names it in its recorded label.
- [ ] The session's launch count is recorded in the results rather than left
      implicit. Ten is the count as filed, and the harness's override ergonomics
      are deliberately unchanged for this run.
- [ ] Provenance stays in this brief. The harness carries parameter values and
      the invariants they enforce — never the session date, the rig arm the
      magnitude came from, arm or cell labels, or references to another brief's
      phases. That applies to `--help` strings and test names as much as to
      comments.
- [ ] Harness dry-run with realized-schedule statistics printed and inside
      tolerance of the request for every Arm D cell, and with the perturbation's
      realized RMS displacement and yaw σ printed per arm.
- [ ] The baseline STOP gate runs **first**: a `clean` arm with every new knob
      off must reproduce the recorded 0.900 within binomial sampling noise
      (0.84–0.96 at 100 episodes). Outside that band, stop and report — the
      extension has changed the baseline and no arm below is readable.
- [ ] Per-arm results table (completion, near-arrival, progress, both direction
      offsets, realized profile, realized perturbation) plus the raw JSONL in
      the PR description, and the per-episode chain index recorded for Arm A.
- [ ] Failure counts broken out by cause. `off_path_divergence` versus
      `sustained_collision` separates a tracking failure from a control failure,
      and `near_arrival` separates a dwell-gate failure from never having
      arrived.
- [ ] Requested versus realized profile recorded for every arm. `--warmup-ticks`
      dilutes the hold axis by ~1 point and the duplicate axis by ~2, so the
      duplicate-axis cells are scored on their realized distinct-content rate.
- [ ] No changes to envs, cfgs, noise models, the inference node, or the play
      script — the harness is additive.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — the pure
      `strafer_lab` suite stays green and the play script is untouched.

## Arms

Env `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, seed 42,
16 envs, 100 episodes per arm, 20 s episode cap, observation corruption on,
`--warmup-ticks 2` — the recorded run's settings, held fixed so its `clean`
arms remain comparable. `--num_envs 16` has to be passed explicitly; the play
config's own default is 8, and taking it would silently break the comparison.

| arm | policy | cell |
|---|---|---|
| B2 | depth v2 `run_20260727_171735/model_998.pt` | `clean`, all knobs off — the STOP gate, mandatory, runs first |
| T2A | depth v2 | `clean` + no hidden reset |
| T2B₁ / T2B₂ / T2B₃ | depth v2 | `clean` + subgoal drift at 0.5× / 1× / 2× |
| T2C | depth v2 | `band` + no hidden reset + subgoal drift 1× |
| T2D₁ | depth v2 | `hold 0 / stale 0.61`, `mean_stale_run 2.0` — rate parity at full novelty (11.7 Hz distinct): the post-render-fix operating point, and the closest analogue of real hardware |
| T2D₂ | depth v2 | `hold 0 / stale 0.76`, `mean_stale_run 4.0` — novelty-matched to `degraded` (7.2 Hz distinct) |
| T2D₃ | depth v2 | `hold 0 / stale 0.233`, `mean_stale_run 1.2` — band-equivalent, novelty-matched by construction (23.0 Hz distinct) |
| T1C | depth v1 `run_20260708_005923/model_500.pt` | the T2C cell — calibration control, arm C only |

The recorded 2026-08-04 `clean` results (v2 0.900, v1 0.840) are the reference
denominators. They may be re-used as denominators only while B2 passes its STOP
gate; if it lands outside the band, every ratio is recomputed against the
in-session baseline and the discrepancy is the headline finding.

## Decision rule (pre-registered)

Committed before the session runs, so the outcome cannot be read backwards.
Ratios are against the v2 `clean` baseline of 0.900.

The rows below are as committed; the trailing **scored** column was added
afterwards from the 2026-08-08 measurements and changes no other text.

| prediction | reading if it fails | scored |
|---|---|---|
| v2 under A alone holds ≥ 0.70 of baseline | state longevity alone is a first-order defect, and the trainable axis is episode length | **held** — 0.890, 0.989 of baseline |
| v2 under B at 1× loses ≥ 15% of baseline | the subgoal-following gain is lower than assumed and SLAM-frame noise is not the residual | **held** — 0.650, a 27.8% loss |
| v2 under B is monotone across 0.5× / 1× / 2× | the perturbation is not acting through the channel it was built for — verify the quartet rewrite before reading any level | **held** — 0.810 / 0.650 / 0.340, strictly monotone; no re-verification owed |
| v2 at T2D₁ scores ≥ 0.80 of baseline | rate parity recovers less than predicted even on fully novel content; the consequence is whatever the Arm D table assigns to the level measured, and this row records only that the prediction missed | **held** — 0.860, 0.956 of baseline |
| v2 at T2D₂ scores ≥ 0.70 of baseline | reuse at the *current* duplicated-content operating point is worse than predicted; same reading, and the T2D₁ level then carries the adoption question | **held** — 0.910, 1.011 of baseline |

| outcome | consequence | scored |
|---|---|---|
| C below 0.30 of baseline | compositional sufficiency is established: individually survivable axes compose to the rig's failure, and the remedy list is deploy-side — arrival rate and SLAM quality — not a retrain | does not fire (C = 0.700) |
| C at or above 0.60 of baseline | the residual lives in the axes sim cannot reach (node observation chain, planner path distribution); the next discriminant is rig-side, after the render stall is cleared | **fires** — C = 0.630 raw, 0.700 of baseline. Qualified below: the two deploy changes ship and the rig is re-validated first; only a still-failing rig read opens those discriminants |
| C between 0.30 and 0.60 | composition contributes but is not sufficient: rank A and B by their solo losses, carry the larger into the rig-side discriminant, and no retrain trigger fires | does not fire |
| A decays with chain depth **and** C is below 0.30 | the trainable axis is episode length; that is the one path back to a retrain, and it re-opens the held decision | does not fire — A is flat with chain depth and C is 0.700 |
| B at 1× loses ≥ 15% | the training-side mechanism already designed in the DR audit's Phase 2 block is licensed for measurement, not yet for a training run | **fires** — 27.8% loss. Superseded upward: the drift model joins the v2.1 retrain rather than stopping at measurement |

**Arm D, scored on T2D₂** — novelty-matched to the stream deploy sees today —
with T2D₁ read as the post-render-fix and real-hardware operating point, and
T2D₃ as the band-regime reading:

| T2D₂ against baseline | consequence | scored |
|---|---|---|
| ≥ 0.85 | adopt timer-driven stale reuse. The node change is small and bounded: the freshness gate's skip path becomes bounded reuse, `tick_on_depth` goes false, and the reuse budget stays strictly shorter than `depth_timeout_s`. The depth-age watchdog still caps it, and the planner refusal and starvation guards are untouched | **fires** — T2D₂ = 1.011 of baseline, well clear of the line. Adoption is ordered; the node change is filed separately and is not part of this brief |
| 0.75 – 0.85 | a judgment call taken with the operator, informed by the T2D₁ – T2D₂ spread: a wide spread means the gain came from depth novelty rather than rate parity, and the deploy change would not deliver it | does not fire |
| < 0.75, **but T2D₁ ≥ 0.85** | adoption stays live, coupled to the sim render-duplication fix. The semantics pay off at the operating point that fix produces, and on real hardware, but not at today's duplicated-content one. This comes back as a judgment call rather than a rejection, and the deploy change waits on the render fix landing | does not fire — and the coupling it anticipated is moot: T2D₁ (0.956) and T2D₂ (1.011) bracket the line from above, so adoption does not wait on the render fix |
| < 0.75 with T2D₁ also short | the current gate stands, and the freshness-gate ruling is re-affirmed on measurement rather than precedent | does not fire |

Predictions committed now: **≥ 0.80 at T2D₁** and **≥ 0.70 at T2D₂**, on the
grounds that training's own render duplication already exposed the policy to
stale content at full rate, and that fresh proprioception and a fresh subgoal on
every tick remove the worse half of the staleness skew. T2D₂ is pitched lower
because it carries the publisher's duplication on top of the node's own reuse.
Both sit below the 0.85 adoption line on purpose — the predictions are a
calibration record, not the rule.

One interaction to read, not to gate on: adopting Arm D's semantics lengthens
the recurrent horizon in deploy from ~700 advances per mission to the full 1800,
which is the regime Arm A measures. If A shows decay with chain depth, the
Arm D adoption note must carry that cost.

## Results — 2026-08-08 session

The harness landed in `ff5d93d` (PR #193). This section records the ten-launch
session it was built for.

**Tree.** Run from detached `dfd8a25`, the first parent of the #195 merge —
the last commit carrying the harness extensions and predating the temporal-DR
change, which adds a native stream-hold band to the ROBUST contract and would
otherwise layer under the emulated schedules and break comparability with the
recorded `clean` baseline. Verified before launching: `de514a8` (#193) is an
ancestor and no commit of #195 is; `hold_fraction_range` has zero occurrences
in the checkout; and no commit has touched `source/strafer_lab/strafer_lab/`
since the 2026-08-04 baseline run, so the env, cfgs and noise models are
identical to the tree that produced the 0.900.

**Settings**, held fixed across all ten launches:
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, seed 42, 16 envs
(passed explicitly), 100 episodes per arm, 20 s cap, observation corruption on,
`--warmup-ticks 2`. Launches were serialised — the runner refuses to start
while any compute process holds the GPU. No arm hit its step ceiling; every arm
completed its full 100 episodes.

**Launch count: ten**, as filed. The override ergonomics were left unchanged.

**Completion and failure cause.** Ratios are against the recorded v2 `clean`
0.900 and, for T1C, the recorded v1 `clean` 0.840. The `/B2` column divides by
the same-session baseline instead. `robot_flipped` and `time_out` are 0 in
every arm, so the three causes shown are the complete breakdown.

| arm | recorded label | compl | ratio | /B2 | near | prog | offset | left | complete / off-path / collision |
|---|---|---|---|---|---|---|---|---|---|
| B2 | `clean` | 0.860 | 0.956 | 1.000 | 0.490 | 0.900 | +4.70° | 0.627 | 86 / 2 / 12 |
| T2A | `clean+nohreset` | 0.890 | 0.989 | 1.035 | 0.580 | 0.920 | +4.46° | 0.610 | 89 / 0 / 11 |
| T2B₁ | `clean+drift0.5x` | 0.810 | 0.900 | 0.942 | 0.590 | 0.872 | +4.36° | 0.589 | 81 / 3 / 16 |
| T2B₂ | `clean+drift1x` | 0.650 | 0.722 | 0.756 | 0.440 | 0.810 | +4.49° | 0.570 | 65 / 22 / 13 |
| T2B₃ | `clean+drift2x` | 0.340 | 0.378 | 0.395 | 0.310 | 0.670 | +1.57° | 0.519 | 34 / 37 / 29 |
| T2C | `band+nohreset+drift1x` | 0.630 | 0.700 | 0.733 | 0.540 | 0.836 | +4.71° | 0.579 | 63 / 19 / 18 |
| T2D₁ | `clean+stale0.61+stale_run2` | 0.860 | 0.956 | 1.000 | 0.580 | 0.903 | +4.72° | 0.623 | 86 / 2 / 12 |
| T2D₂ | `clean+stale0.76+stale_run4` | 0.910 | 1.011 | 1.058 | 0.700 | 0.921 | +4.34° | 0.625 | 91 / 2 / 7 |
| T2D₃ | `clean+stale0.233+stale_run1.2` | 0.850 | 0.944 | 0.988 | 0.650 | 0.916 | +4.42° | 0.624 | 85 / 4 / 11 |
| T1C (v1) | `band+nohreset+drift1x` | 0.480 | 0.571 | — | 0.520 | 0.787 | +2.80° | 0.536 | 48 / 35 / 17 |

The label invariant holds: the only arm labelled `clean` is the untouched
baseline, and every arm carrying a knob names it — including the composed arms,
which fold all three.

**Requested versus realized.** Distinct-content rate is the rate at which new
depth pixels reach the policy, `tick_hz × (1 − hold) × (1 − duplicate)`.

| arm | hold | dup | infer Hz | distinct Hz | dup run mean / max | drift RMS (of request) | drift σ (of request) | steps |
|---|---|---|---|---|---|---|---|---|
| B2 | 0.0000 | 0.0000 | 30.000 | 30.000 | — | — | — | 853 |
| T2A | 0.0000 | 0.0000 | 30.000 | 30.000 | — | — | — | 920 |
| T2B₁ | 0.0000 | 0.0000 | 30.000 | 30.000 | — | 0.0724 m (0.873) | 2.88° (0.860) | 820 |
| T2B₂ | 0.0000 | 0.0000 | 30.000 | 30.000 | — | 0.1456 m (0.877) | 6.19° (0.925) | 861 |
| T2B₃ | 0.0000 | 0.0000 | 30.000 | 30.000 | — | 0.2811 m (0.847) | 11.62° (0.867) | 808 |
| T2C | 0.2300 | 0.0000 | 23.101 | 23.101 | — | 0.1497 m (0.902) | 6.14° (0.916) | 961 |
| T2D₁ | 0.0000 | 0.5939 | 30.000 | 12.183 | 1.97 / 12 | — | — | 892 |
| T2D₂ | 0.0000 | 0.7395 | 30.000 | 7.814 | 3.86 / 30 | — | — | 872 |
| T2D₃ | 0.0000 | 0.2263 | 30.000 | 23.211 | 1.19 / 5 | — | — | 878 |
| T1C | 0.2285 | 0.0000 | 23.146 | 23.146 | — | 0.1506 m (0.902) | 5.96° (0.889) | 946 |

Requests were hold 0.233 and duplicate 0.610 / 0.760 / 0.233; the realized
shortfall of ~1 point on the hold axis and ~2 on the duplicate axis is
`--warmup-ticks` forcing fresh ticks after each boundary, as anticipated. Drift
realizes at 0.85–0.90 of request against the ~0.89 the re-anchoring expression
predicts at the observed mean episode length, so the shortfall is the modelled
effect rather than a defect.

**T2D₂ is novelty-matched on realized figures**, which is what the brief
requires before the cell may be scored: its 7.814 Hz of distinct content sits
2.15% from `degraded`'s realized 7.650 Hz, inside the 5% gate. No re-request
was owed.

**Command direction against truth and against the perturbed referent.** Only
the drift arms carry a perceived counterpart.

| arm | truth median | truth median abs | perceived median | perceived median abs |
|---|---|---|---|---|
| B2 | +4.70° | 14.78° | — | — |
| T2A | +4.46° | 15.22° | — | — |
| T2B₁ | +4.36° | 16.66° | +4.78° | 15.60° |
| T2B₂ | +4.49° | 18.97° | +3.89° | 15.87° |
| T2B₃ | +1.57° | 23.74° | +3.40° | 18.35° |
| T2C | +4.71° | 18.66° | +4.62° | 15.87° |
| T2D₁ | +4.72° | 15.33° | — | — |
| T2D₂ | +4.34° | 13.71° | — | — |
| T2D₃ | +4.42° | 13.84° | — | — |
| T1C (v1) | +2.80° | 23.73° | +3.55° | 21.97° |

**Completion against chain depth**, for the arms that carry the state across
episodes. Bin sizes fall from 16 at low depth to single digits at the tail.

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| T2A | 0.875 | 1.000 | 0.938 | 0.750 | 0.875 | 0.857 | 1.000 | — | — |
| T2C | 0.812 | 0.750 | 0.400 | 0.600 | 0.600 | 0.700 | 0.625 | 0.500 | 0.000 |
| T1C (v1) | 0.625 | 0.562 | 0.562 | 0.375 | 0.429 | 0.333 | 0.286 | 0.667 | — |

### Gates and deviations

- **The baseline STOP gate passed.** B2 is 0.860, inside the pre-registered
  0.84–0.96 band, so the recorded 0.900 and 0.840 denominators stay valid and
  every arm below is readable. It landed 0.040 under the reference and the run
  is not bit-reproducible against 2026-08-04 despite an identical env, seed and
  settings — GPU nondeterminism, which is the sampling noise the band exists to
  absorb. The `/B2` column is carried alongside so no reading depends on which
  denominator is chosen; the two agree on every verdict.
- **A GPU-free pre-flight preceded the session.** The pure suite ran 123/123,
  and an offline replay of the sampler and the drift process against the
  recorded episode-boundary sequence confirmed each duplicate-axis cell clears
  the validation ceiling and projected T2D₂ at 7.86 Hz distinct content — 2.7%
  from the match target — before any Kit launch. Realized 7.814.
- **The horizon Arm A reaches is shorter than deploy's, and differs by arm.**
  Held ticks do not advance the recurrent state, so the horizon is the count of
  inferring ticks per env: 920 for T2A, 740 for T2C, 730 for T1C. T2C and T1C
  land almost exactly on the ~700 advances derived for the failing 12 Hz deploy
  regime; T2A exceeds it but reaches roughly half the 1800 of a full 60 s
  mission at 30 Hz, because the 100-episode budget caps the chain. Arm A
  therefore characterises the horizon deploy occupies today, not the horizon
  the adopted inference semantics will create.
- **No repository file was modified by the session itself.** No changes to
  envs, cfgs, noise models, the inference node, or the play script.

### Where the evidence lives

`logs/` is gitignored, so the session's JSONL is not in version control and the
tables above are its durable record. On the machine that produced it the ten
files are in `logs/rsl_rl/strafer_navigation/cadence_emulation/`, one per
launch, in arm order: `cadence_20260808_195228` (B2), `_195508` (T2A),
`_195730` / `_195957` / `_200218` (T2B₁₋₃), `_200459` (T2C), `_200731` /
`_201000` / `_201229` (T2D₁₋₃), `_201506` (T1C). Each carries the per-episode
records — with `episode_index` as the env-local chain index — alongside
`requested_profile`, `realized_profile`, `subgoal_drift`, and both
direction-offset summaries.

## Read-out — scored against the pre-registered rule

- **(a) Arm A — passes; the recurrent horizon is not implicated at today's
  horizons.** T2A is 0.890 (0.989 of reference, 1.035 of the same-session
  baseline) and the chain-depth curve is flat. Carried forward: the session
  reached ~920 inferring ticks, and the inference semantics adopted under (d)
  push deploy's effective horizon toward ~1800, which nothing here has tested.
  The v2.1 acceptance grid gains a long-horizon chained arm.
- **(b) Arm B — passes; subgoal-frame drift is the dominant sim-testable
  contributor.** 1× drift costs 24–28% (0.650; 0.722 of reference, 0.756 of the
  same-session baseline), the sweep is monotone across 0.5× / 1× / 2×
  (0.810 / 0.650 / 0.340), and the truth-versus-perceived split shows the
  policy faithfully tracking a displaced frame: perceived offset holds near the
  baseline's 14.78° while truth widens with magnitude. The failure is the
  frame, not the tracking. **The drift model joins the v2.1 retrain**, which
  is more than the pre-registered row licensed — it read "measurement, not yet
  a training run" — and the upgrade rests on the monotone sweep plus the
  discriminator agreeing on mechanism.
- **(c) Arm C — the ≥ 0.60 branch fires, with the chase ordered rather than
  opened.** T2C is 0.630 (0.700 of reference), so the composition of band
  cadence, unbounded horizon and 1× drift does not collapse v2; T2C ≈ T2B₂
  means drift carries essentially the whole compositional cost. The gap between
  0.63 in sim and ~0.0 on the rig is real and lives in the axes sim cannot
  reach. The branch's literal consequence — go to the rig-side discriminants —
  is qualified: the two deploy changes ship first and the rig is re-validated
  with them live, and only a still-failing read then licenses the
  path-geometry and observation-chain discriminants. T1C at 0.571 of its own
  baseline confirms the composition is generically harsh rather than
  v2-specific.
- **(d) Arm D — adopt, decisively.** T2D₂, the novelty-matched cell, reads
  0.910 — 1.011 of reference — far above the 0.85 adoption line, and T2D₁ at
  0.956 beat its ≥ 0.80 prediction; both committed predictions held. The
  sharpest single figure is the regime contrast: holding distinct content fixed
  and moving only inference rate, the degraded operating point goes 0.610 →
  0.910 while the band point is neutral (0.870 → 0.850). The cost of
  degradation was the frozen-state and held-command half, almost never the
  stale-content half. That training itself ran amid render duplication remains
  the explanation for why duplicate content is nearly free.

**Consequences.** The cadence-setpoint question dissolves — 30 Hz stands and no
20 Hz branch is needed. The command-hold bands stay as shipped. The
host-capacity concern is largely defused for the policy loop, since at today's
collapsed 5–8 Hz arrival stale reuse puts the node in T2D₂'s regime, pending
the depth-age bound; capacity work proceeds on its own merits for SLAM, whose
quality feeds the very drift axis (b) convicts.

**What this brief licenses, filed separately.** Two changes follow and neither
belongs to this brief: timer-driven stale-reuse inference semantics in the
node, config-gated with the current behaviour as a named fallback; and an SE(2)
drift randomization on the goal-shaped observation terms in `strafer_lab`,
calibrated to the measured class, sharing `SubgoalDrift` as the reference
definition, and carrying a default-off discontinuous-jump component for loop
closures. A rig-side logger for loop-closure jump statistics rides along with
the next session that has SLAM up, and pins the τ = 2.0 s assumption this brief
had to state rather than cite. Pre-registered now for the v2.1 acceptance grid:
**drift-1× ratio ≥ 0.85 of its own clean**, up from today's 0.722, alongside
the long-horizon chained arm from (a).

## Investigation pointers

- The hidden-state reset Arm A guards: `eval_cadence_emulation.py:1062`
  (`policy.reset(dones)`), with the per-arm global clear in `reset_between_arms`
  which must stay unconditional.
- Held-tick semantics Arm A must not disturb: held rows are realized by running
  the batched forward for every env and then restoring the held rows' hidden
  columns, so batch width never changes with a schedule.
- The quartet layout and its startup verification: `field_slices` and
  `verify_term_layout` in the same file. The observation functions carry no
  `subgoal_` prefix — the subgoal env reuses `goal_position_relative`,
  `goal_distance` and `goal_heading_to_goal` from
  `strafer_lab/tasks/navigation/mdp/observations.py` against a `goal_command`
  term that is a `SubgoalCommand`, so grepping for the subgoal name finds
  nothing.
- The bearing is the bearing *to the subgoal*, not the path tangent. The
  tangent exists on the command term and is never observed, so Arm B perturbs
  everything the policy can see about the subgoal frame.
- **Launching it.** `isaaclab.sh -p` resolves whatever `python` is on `PATH`
  and falls back to base conda without erroring, so the cheatsheet's
  `$ISAACLAB -p` line only works from a shell that already has the Isaac
  environment active. From a fresh one it dies on `ModuleNotFoundError: torch`
  before Kit boots.
- **Realized drift reads below requested, and should.** The process re-anchors
  at every episode boundary and needs roughly three time constants to reach its
  stationary magnitude, so an arm's applied RMS is `sqrt(1 - (tau/2T)(1 -
  exp(-2T/tau)))` of the request for mean episode length `T`. At the 20 s cap
  that is ~97%; on a short dry run it is visibly less. Read the shortfall
  against that expression before treating it as a defect.
- Scale constants: `GOAL_DIST_SCALE = 1/10`, `HEADING_SCALE = 1/π`,
  `GOAL_DIST_MAX = 10.0`, `SUBGOAL_LOOKAHEAD_M = 1.0` randomized to (0.7, 1.3)
  on the ROBUST tier — the lookahead band is why the position and heading knobs
  couple.
- **Scoring stays on ground truth under Arm B.** The perturbation touches only
  the policy observation. The dwell gate, `off_path_divergence` and the arc
  cursor are all computed inside the env from the true robot pose and the true
  path, so a drift arm degrades what the policy sees without moving what it is
  measured against.
- Completion is dwell-gated: 10 consecutive steps inside a 0.30 m radius under
  0.1 m/s. `near_arrival_rate` is the discriminator between a dwell-gate failure
  and never having arrived.
- `off_path_divergence` fires at 0.3 m cross-track, which is the same order as
  the 1× drift class. A drift-induced off-path termination is expected and is
  not by itself evidence that the policy stopped tracking — the perceived
  direction offset is what separates those two.
- Progress is latched pre-step from the monotone path cursor, so a completed
  episode reports `progress_fraction` just under 1.0.
- The deploy-side gate Arm D models, and the guards a node change must not
  route around: `inference_node.py:954-956` (skip before assembly),
  `_note_depth_content` (whose meaning changes under reuse — every reused tick
  becomes a counted repeat), `watchdog.py`'s `stale_sources` with
  `depth_timeout_s` at 0.5 s, and the refusal and starvation guards in
  `subgoal_generator_node.py`, whose whole purpose is breaking the
  zero-twist self-lock.

## Out of scope

- **Any inference-node change.** Arm D scores a proposal; the node PR belongs to
  the deploy lane and is gated on this read-out. Nothing in this brief touches
  `source/strafer_ros/`.
- **The planner path-geometry candidate.** Its statistics already accept an
  arbitrary polyline, so what it needs is a capture of Nav2 plans — rig time
  this brief cannot buy — not new instrument code. The shipped-generator
  reference distribution is the comparison waiting for it.
- **The observation-chain candidate.** Same rig dependency as the archived
  measured-profile cell.
- **A stale burst mixture on the duplicate axis**, and separating Arm B's
  position and heading knobs. Both are follow-ups that only matter if the
  fraction-level arms read positive.
- **Wiring the dead randomization knobs, or any widening of a randomization
  band.** Measurement first. The DR audit's Phase 2 block is the designed home
  if Arm B confirms.
- **The retrain**, which stays held. It shrinks to nothing unless A or C
  implicates a trainable axis.
- **Re-running the temporal profiles.** `clean`, `band`, and `degraded` are
  recorded; this brief re-runs `clean` only, as its STOP gate.
