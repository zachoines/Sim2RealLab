# Train across the referent frame's drift

**Type:** implementation (contract field + one mechanism + tests)
**Owner:** DGX (`strafer_lab` lane — env config)
**Priority:** P1 — the residual-attribution arms convicted this axis as the
dominant sim-testable contributor to the deployed failure, and the goal-shaped
observations carry no noise model at all. The mechanism is cheap; what it is
waiting for is the retrain, which is scheduled elsewhere.
**Estimate:** S–M (one design-and-implementation session, no training run)
**Branch:** `task/subgoal-frame-drift-dr`

## Story

As a **DGX operator preparing the next depth checkpoint**, I want **the
training contracts to randomize the SLAM frame the goal-shaped observations are
read through**, so that **the policy learns to track a referent that wanders
the way the deployed one does, instead of one that is exact to the millimetre.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`cadence-harness-residual-arms`](../../completed/cadence-harness-residual-arms.md) —
  the arm that convicted the axis and the source of the magnitude class.
- [`temporal-texture-training-dr`](../../completed/temporal-texture-training-dr.md) —
  the shape this follows: one law, tier bands, an attributed golden re-freeze.
- [`domain-randomization-audit`](domain-randomization-audit.md) — the designed
  home for this axis, and the doc whose TF row this work rewrites.

## Context

Arm B of the residual-attribution run swept a correlated SE(2) perturbation of
the referent frame across 0.5× / 1× / 2× of the class recorded on the rig. It
cost 24–28% of completion at 1× (0.650 against a 0.900 baseline) and moved
monotonically with magnitude — 0.810 / 0.650 / 0.340. The truth-versus-perceived
discriminator in the same arm showed the policy tracking the displaced frame
faithfully, so the failure is the frame rather than the tracking. Arm C then
showed that the composition of band cadence, recurrent horizon and 1× drift is
essentially the drift alone: T2C 0.630 against T2B₂ 0.650.

Training modelled none of it. `goal_position`, `goal_distance` and
`goal_heading_to_goal` are the only observation terms in the policy tensor with
no noise model on any tier — the referent they carry is recomputed exactly from
the command term at every tick. Deploy reads the same quantities through
`(map→odom) ⊗ (odom→base_link)`, and RTAB-Map moves `map→odom`.

### Why an integrated process rather than per-tick noise

Localization error accumulates. An offset that is 0.1 m to the left this tick is
still roughly 0.1 m to the left a tick later and only wanders over seconds.
Independent per-tick noise models something a recurrent policy averages away in
a handful of ticks, which is the opposite of the deployed failure. The same
reasoning is already in the repo twice — `goal-noise-training`'s drift component
and the planner's `perturb_waypoints`, which samples at correlation-length
control points because independent per-waypoint noise "would crinkle the
polyline".

### What is measured and what is assumed

The **magnitude class is cited**: 0.166 m RMS displacement and 6.7° heading, the
`map→odom` movement recorded in the 2026-08-02 session. The **time constant is
an assumption** — τ = 2.0 s, justified only by RTAB-Map's 1–10 Hz `map→odom`
refresh. The measurement that would pin it is Phase 1 item 6 of the DR audit,
which this work extends to also record it. The **closure-jump distribution is
neither**: the mechanism ships with its band at zero.

The two magnitudes scale together under one per-env gain rather than being drawn
independently, because they do not separate in effect. Only the offset component
perpendicular to the bearing produces bearing error, and its per-axis σ of
0.117 m at the 1.0 m nominal subgoal lookahead is 6.7° — the same size as the
heading σ. The arm swept them together for that reason, and the band inherits it.

## What lands

### The law

`mdp/subgoal_drift.py` — a per-env SE(2) process. Each of the three components
is an Ornstein-Uhlenbeck path discretised exactly, so the stationary standard
deviation is the configured σ at any step size:

    decay      = exp(-dt / tau)
    innovation = sqrt(1 - decay^2)
    x         <- decay * x + innovation * sigma * n

State is zeroed at reset because a new episode is a new path off a fresh anchor,
and drift accumulates from the anchor. The class is quoted as an RMS
displacement of the 2-D offset, so each position axis carries `1/sqrt(2)` of it.

This is the same law the evaluation harness's `SubgoalDrift` samples, and the
two implementations are deliberately separate — per-env torch state inside the
env, numpy inside a rollout script. What keeps them from diverging is that both
derive from the same arithmetic, and the suite asserts the agreement against the
harness directly: the coefficients, the axis split, and the four-dim transform
element-for-element. **The harness itself is not touched** — it is the
acceptance instrument for the retrain this feeds.

### The loop-closure jump

A second error class lives on this axis and no random walk produces it: when
RTAB-Map accepts a closure, `map→odom` moves in one step. It ships as a Poisson
component of the same process — a rate and a magnitude band — so a snap lands in
the state the wander relaxes and decays on the same τ rather than persisting.
**Its band ships at zero on every tier.** The mechanism is in the tree so the
contract goldens move once, in this re-freeze, and turning it on later is a
config change; the distribution waits on the rig ride-along. A counter reports
jumps taken, so a non-zero band is visible rather than inferred.

### Where the perturbation is applied

Inside the three goal-shaped observation terms, on the body-frame referent
vector they all derive from. Translate, then rotate; the relative offset, the
range and the signed bearing are then read off one vector.

This is deliberately **not** a per-term noise model, and the reason is that the
law cannot be expressed as one. The perturbation is a single SE(2) transform
whose effect on each dim depends on the others — the drifted range needs the
bearing, the drifted bearing needs the range — so three independent noise models
cannot reproduce it, and Isaac Lab's `@configclass` deep-copies each term's cfg
independently, so they cannot share one process either (verified before the
design was chosen). Perturbing the four dims independently instead would hand
the policy a relative offset, a range and a bearing that no geometry can produce
together.

The cost of applying it in the term rather than the noise hook is that the
observation manager's `enable_corruption` switch no longer strips it for the
privileged group. So the critic's goal-shaped terms carry `perceived=False`
explicitly and read the true referent, and a contract test pins that on every
composed variant. The policy terms are untouched, which is what keeps the
layout golden still.

The command the rewards and terminations read is never perturbed. That
separation is what the arm's truth-versus-perceived discriminator rests on, and
it is what makes this domain randomization rather than a task change.

### Where it applies

Every randomized RL variant — the drift is a realism-tier property, like the
sensor corruptions. The ideal tier carries no term at all rather than an inert
one, so an ideal env stays structurally identical to a tree without this
mechanism.

The bridge and capture variants opt out through
`RealismCfg(localization_drift=False)`. Their policy-facing observation is a
record, not a policy input: the bridge hands its scene to a deploy stack that
reads its own drifting TF, so a sim drift on top would double-count it and would
corrupt the sim-versus-deploy observation parity the gym-side dump exists to
measure.

The event term is declared **last** in every tier that carries it, so the draw
it adds cannot shift the random stream the room generation and the robot reset
consume ahead of it.

### Tier bands

| knob | REAL | ROBUST | source |
|---|---|---|---|
| gain on both magnitudes | `(0.0, 0.5)` | `(0.0, 1.25)` | multiples of the measured class |
| position RMS at gain 1 | 0.166 m | 0.166 m | recorded `map→odom` movement |
| heading σ at gain 1 | 6.7° | 6.7° | same |
| τ | 2.0 s | 2.0 s | assumption — RTAB-Map's 1–10 Hz refresh |
| jump rate | 0.0 Hz | 0.0 Hz | not measured; ships off |
| jump magnitude | `(0.0, 0.0)` | `(0.0, 0.0)` | not measured; ships off |

Robust reaches past the measured class the way the temporal bands reach past the
measured arrival profiles: the sensitivity arm cost 24–28% at 1×, so the band
has to span it rather than sit on its edge. Realistic stays below it on purpose
— realistic models the link as measured, not the worst it can be.

## The contract goldens

Every one of the 22 composed-variant hashes and the depth-obs golden moved,
because the drift is an event term and the critic's goal-shaped terms gained the
truth parameter, and the snapshot walks both. The cause was isolated before
anything was re-frozen: **dropping only the two new key names
(`randomize_subgoal_drift`, `perceived`) from the serializer reproduces all 23
stored goldens byte-for-byte**, so the movement is their presence alone — no
term reordered, no scale changed, no existing parameter touched.

The serializer gained one line to make that possible: `drop` now subtracts
term-parameter names as well as cfg attributes. A term parameter is as much a
named field of the contract as an attribute is, and the attribution recipe has
to be able to subtract either.

**The policy-observation layout golden did not move at all**, which is the
stronger statement than the attribution: the drift never reaches the layout a
deployed checkpoint is hashed against.

## Acceptance criteria

- [ ] The drift law agrees with the evaluation harness's own `SubgoalDrift` —
      the OU coefficients across τ and step size, the RMS-to-axis split, and the
      four-dim transform element-for-element — asserted against the harness
      rather than restated.
- [ ] Realized in-env magnitude verified against the configured profile at a
      pinned gain, and the gain verified to scale both magnitudes together.
- [ ] The process is integrated, not resampled: the lag-1 and lag-30
      autocorrelations match the configured τ.
- [ ] All four referent dims come from one perturbed vector — distance equals
      the drifted relative offset's norm and bearing equals its `atan2`, in the
      terms themselves and not just in the helper.
- [ ] The privileged critic reads the true referent, on every composed variant.
- [ ] Per-env magnitudes differ across envs and are re-drawn at reset; a partial
      reset touches only the named envs.
- [ ] Inert at neutral parameters, and inert means *consumes nothing*: a zero
      gain band and a zero jump band each leave the torch RNG stream
      bit-identical.
- [ ] The jump is discontinuous where the wander is not, arrives at the
      configured Poisson rate, and relaxes on the same τ; the shipped band is
      zero on every tier.
- [ ] Tier convention: both tiers range, robust strictly wider, robust spans the
      measured class and realistic sits below it; the ideal tier carries no term.
- [ ] `LocalizationDriftCfg`'s field set is pinned, and no field is left without
      a consumer.
- [ ] The bridge and capture variants carry no drift term.
- [ ] Contract-hash movement attributed to the two new keys before re-freezing,
      and the policy layout golden verified unmoved.
- [ ] The evaluation harness, the play script, the training loop and the
      inference node are untouched.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Pre-registered acceptance for the retrain this feeds

Committed here, before the run, so the outcome cannot be read backwards. Scored
on the same evaluation-harness grid, and added to the grid the temporal work
already registered.

| quantity | current | target |
|---|---|---|
| drift-1× completion, as a ratio of own clean baseline | 0.722 | ≥ 0.85 |

The standing rows from the temporal work — degraded ≥ 0.85, band ≥ 0.95, clean
≥ 0.88 absolute — are unchanged, and the grid gains a long-horizon chained arm
at ~1800 advances now that the node's adopted tick semantics triples deploy's
effective horizon.

**One hazard for whoever scores it.** Once this merges, the harness's drift arms
compose *on top of* env-native drift, exactly as the temporal work's holds now
do. "Clean" at scoring time means env DR as trained, with no harness emulation
layered on it; an arm run against a v2.1 checkpoint measures the sum of the two.

**The retrain is not in this PR.** Its scope is now final — the temporal DR from
[`temporal-texture-training-dr`](../../completed/temporal-texture-training-dr.md)
plus this axis — and it runs when this merges.

## Out of scope

- **The retrain itself**, and any evaluation of it. See above.
- **Any change to the evaluation harness.** It is the acceptance instrument for
  the run this feeds; this work shares its law by asserting agreement rather
  than by rewiring it.
- **The jump band's values.** Measurement-first: the mechanism ships, the
  distribution waits on a rig session that logs `map→odom` against RTAB-Map's
  closure events. The same log pins τ.
- **TF staleness as an *age*.** An offset that is wrong and a reference that is
  old are different quantities, and only the second couples to the robot's own
  motion. That half of the DR audit's row stays open, with its designed
  mechanism unchanged.
- **The deploy-side admission rules.** Gross frame jumps are handled by the
  cross-track re-anchor already shipped on the node; this is the training-side
  half.
- **Perturbing the command.** The command drives rewards and terminations, so
  drifting it would move the truth the arm's discriminator is measured against.
- **Separating the position and heading knobs.** They are near-equal in effect
  at the nominal lookahead; splitting them is a follow-up with its own arm.

## Investigation pointers

- The law and its per-env process:
  `source/strafer_lab/strafer_lab/tasks/navigation/mdp/subgoal_drift.py`.
- Where the offsets are applied: `_drift_offsets` and `_referent_body_xy` in
  `mdp/observations.py`, consulted by all three goal-shaped terms.
- Why the advance is keyed on the step counter rather than performed per read:
  the observation group is computed more than once on a recorder step, and three
  terms read the same offsets — a per-read advance would run the process at
  three times its configured rate and decorrelate the terms from each other.
- Where the process is created and re-anchored: `randomize_subgoal_drift` in
  `mdp/events.py`, which reads the env's own control period so τ stays a
  quantity in seconds.
- The opt-out for the capture lane: `RealismCfg.localization_drift`, applied in
  `_ComposedStraferNavEnvCfg.__post_init__` next to the subgoal lane's existing
  `randomize_goal_noise` strip.
- The harness's own implementation, for the agreement assertions:
  `SubgoalDrift` and `drifted_quartet` in `scripts/eval_cadence_emulation.py`.
