# Extend the cadence harness to the residual attribution candidates

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

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

[`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md) exonerated
the temporal axis as sufficient: at the 22–25 Hz band every recorded deploy
session actually ran in, emulated temporal texture costs ≤3% of completion, and
at the 12 Hz / 36%-duplicate profile it costs ~⅓ — a real loss, but far from the
rig's zero completions. That read-out left the enriched-lane `mission` ✗ result
unexplained by scene, by anchoring semantics, or by temporal texture, and named
four live candidates
([`cadence-emulation-eval.md`'s read-out](../../completed/cadence-emulation-eval.md#read-out--scored-against-the-pre-registered-rule),
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
[recurrent-policy-contract §4](../../context/recurrent-policy-contract.md#4-reset-trigger)).
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
[`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md)
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
same reasoning twice — [`goal-noise-training`](goal-noise-training.md)'s drift
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
[`domain-randomization-audit`](domain-randomization-audit.md), which is also
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
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
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

| prediction | reading if it fails |
|---|---|
| v2 under A alone holds ≥ 0.70 of baseline | state longevity alone is a first-order defect, and the trainable axis is episode length |
| v2 under B at 1× loses ≥ 15% of baseline | the subgoal-following gain is lower than assumed and SLAM-frame noise is not the residual |
| v2 under B is monotone across 0.5× / 1× / 2× | the perturbation is not acting through the channel it was built for — verify the quartet rewrite before reading any level |
| v2 at T2D₁ scores ≥ 0.80 of baseline | rate parity recovers less than predicted even on fully novel content; the consequence is whatever the Arm D table assigns to the level measured, and this row records only that the prediction missed |
| v2 at T2D₂ scores ≥ 0.70 of baseline | reuse at the *current* duplicated-content operating point is worse than predicted; same reading, and the T2D₁ level then carries the adoption question |

| outcome | consequence |
|---|---|
| C below 0.30 of baseline | compositional sufficiency is established: individually survivable axes compose to the rig's failure, and the remedy list is deploy-side — arrival rate and SLAM quality — not a retrain |
| C at or above 0.60 of baseline | the residual lives in the axes sim cannot reach (node observation chain, planner path distribution); the next discriminant is rig-side, after the render stall is cleared |
| C between 0.30 and 0.60 | composition contributes but is not sufficient: rank A and B by their solo losses, carry the larger into the rig-side discriminant, and no retrain trigger fires |
| A decays with chain depth **and** C is below 0.30 | the trainable axis is episode length; that is the one path back to a retrain, and it re-opens the held decision |
| B at 1× loses ≥ 15% | the training-side mechanism already designed in the DR audit's Phase 2 block is licensed for measurement, not yet for a training run |

**Arm D, scored on T2D₂** — novelty-matched to the stream deploy sees today —
with T2D₁ read as the post-render-fix and real-hardware operating point, and
T2D₃ as the band-regime reading:

| T2D₂ against baseline | consequence |
|---|---|
| ≥ 0.85 | adopt timer-driven stale reuse. The node change is small and bounded: the freshness gate's skip path becomes bounded reuse, `tick_on_depth` goes false, and the reuse budget stays strictly shorter than `depth_timeout_s`. The depth-age watchdog still caps it, and the planner refusal and starvation guards are untouched |
| 0.75 – 0.85 | a judgment call taken with the operator, informed by the T2D₁ – T2D₂ spread: a wide spread means the gain came from depth novelty rather than rate parity, and the deploy change would not deliver it |
| < 0.75, **but T2D₁ ≥ 0.85** | adoption stays live, coupled to the sim render-duplication fix. The semantics pay off at the operating point that fix produces, and on real hardware, but not at today's duplicated-content one. This comes back as a judgment call rather than a rejection, and the deploy change waits on the render fix landing |
| < 0.75 with T2D₁ also short | the current gate stands, and the freshness-gate ruling is re-affirmed on measurement rather than precedent |

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
