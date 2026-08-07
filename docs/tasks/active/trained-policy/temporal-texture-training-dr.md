# Train across the deploy stream's temporal texture

**Type:** implementation (contract fields + two mechanisms + tests)
**Owner:** DGX (`strafer_lab` lane — env config)
**Priority:** P1 — the training distribution contains almost none of the
temporal texture deploy produces, and the measurement that quantified the cost
has already run. The mechanism is cheap; what it is waiting for is a retrain,
which is scheduled elsewhere.
**Estimate:** S–M (one design-and-implementation session, no training run)
**Branch:** `task/temporal-texture-training-dr`

## Story

As a **DGX operator preparing the next depth checkpoint**, I want **the
training contracts to randomize the depth stream's hold structure, the depth
observation's age, and the chassis command rate across the band the deploy rig
can actually produce**, so that **the deploy contract becomes a band the policy
is flat across instead of a point it was fitted to.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`domain-randomization-audit`](domain-randomization-audit.md) — the designed
  home for these fields, and the doc whose temporal rows this work rewrites.
- [`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md) — the
  measurement the tier ranges are set from.

## Context

The cadence-emulation eval measured the policy's sensitivity to deploy temporal
texture in closed-loop sim. At the 22–25 Hz band every recorded deploy session
actually ran in, emulated texture costs ≤3% of completion. At the degraded
profile — 12 Hz effective arrival, ~36% duplicate content — it costs about a
third, and it costs it to **both** the v1 and v2 checkpoints at near-identical
rates (0.667 and 0.678 of their own baselines). That symmetry is what makes the
axis a *training* gap rather than a v2 defect: the profile is generically harsh,
and neither policy has ever seen it.

Training modelled almost none of it. The shipped checkpoint trained against a
fixed 2-step depth latency, a 1% i.i.d. frame drop whose runs average ~1.01
frames, a structurally fixed 30 Hz tick, and a constant per-modality skew.
Three contract fields — `control_frequency_jitter_pct`, `obs_latency_steps`,
`obs_latency_steps_range` — were declared with zero consumers, so the config
surface implied a randomization the env never performed.

The gap is run structure, not rate. A per-step Bernoulli drop pins its own mean
run length at `1 / (1 - p)`, so at the rates a nominally healthy stream implies,
essentially every repeat is one tick long. Deploy stalls in runs. Reproducing
that needs the fraction and the run length to be independent knobs, which is
what a mixture-geometric hold process gives.

### Why holds and not latency

The two are different axes and the distinction is load-bearing. **Latency
shifts a modality in time; it never repeats one.** A fixed-per-episode latency
is a constant offset a recurrent policy simply recalibrates away — which is
also why the depth latency is now drawn per env rather than shared by the
batch. The new fields model holds and duplicates; the latency fields keep
modelling latency.

### What is approximated rather than reproduced

Deploy's full held-tick semantics — the policy is not called at all, so the
recurrent state does not advance — cannot be reproduced inside the training
rollout without invasive surgery, and that surgery is out of scope. The
training-side model is two mechanisms that together approximate it:

- **Obs-staleness** carries duplicate content and the staleness skew, where
  depth is stale while proprioception and the subgoal advance. The eval showed
  those are the axes that carry the cost.
- **Action-hold** carries "no new command this tick", so the chassis
  re-executes its last command.

The documented residual is the **GRU stepping rate**, which stays at 30 Hz in
training. This is likely temporary: a pending deploy-semantics decision may
move the inference node to timer-driven 30 Hz ticks with bounded stale-depth
reuse, which restores recurrent-rate parity and leaves depth staleness as the
only temporal axis — i.e. the obs-staleness mechanism becomes the whole story.
The obs-staleness mechanism is primary regardless of that outcome.

That same pending decision is why the **action-hold ranges are deliberately
small**. Under today's node semantics a held depth tick and a held command are
the *same event*; under timer-driven inference they decouple entirely and held
commands shrink to timer-miss residuals. The mechanism is built because it is
small and the machinery was adjacent, its REAL range sits near zero, ROBUST
carries a modest band, and the ranges are config the retrain resets to whatever
semantics is in force by then.

## What lands

### The shared law

`mdp/hold_process.py` — a two-state per-env process whose hold-run lengths are
drawn from a mixture of two geometrics. A geometric run of mean `m` is exactly a
per-step exit probability `1/m`, and the mixture component is chosen once when a
run begins, so a run's law never changes mid-run. Stationary relations, given a
mean hold run `m` and target fraction `f`:

    exit  = 1 / m                     mean hold run = m
    enter = f / (m * (1 - f))         mean live run = 1 / enter
    f     = enter / (enter + exit)    reachable iff f <= m / (m + 1)

The reachability ceiling is the statement that the mean live gap between holds
cannot fall below one tick. Requests above it are clamped and counted; the
shipped bands are chosen so it never fires, pinned by a test.

This is the same law the evaluation harness samples, and the two implementations
are deliberately separate — per-env torch state inside the env, numpy inside a
rollout script. What keeps them from drifting is that both derive their
parameters from the same arithmetic, and the suite asserts the agreement against
the harness's own `TemporalProfile` directly. The harness itself is **not
touched**: it is the acceptance instrument for the retrain this work feeds, and
it is being extended concurrently on its own branch.

### Mechanism 1 — depth stream holds (noise-model layer)

`DepthNoiseModel` gains the hold process. A held step re-emits the previous
*noisy* frame, the existing `_prev_frame` semantics. The memoryless
`frame_drop_probability` is unchanged and layered underneath: the two model
different causes of "this frame did not advance" — the sensor's own dropout and
a stalled transport or consumption stream — and compose as a union, so a repeat
is either. Keeping them separate is also what makes the change inert at neutral
parameters.

### Mechanism 2 — command holds (action-term layer)

`MecanumWheelAction` gains the same process, applied after the delay buffer and
before slew limiting, so a held command is re-executed through the actuator
model exactly as a re-published one would be.

### Per-env depth latency

`DelayBuffer` gains an optional `delay_steps_range`, drawn per env at every
reset, reading through a per-env ring index. The fixed path is untouched when no
range is given. This lands `depth_latency_steps_range`, which the DR audit
designed and left without a consumer.

### Tier ranges

Both tiers range and robust is strictly wider, per the convention. Sources are
the eval's realized profiles.

| knob | REAL | ROBUST | source |
|---|---|---|---|
| depth hold fraction | `(0.0, 0.35)` | `(0.0, 0.60)` | realized 0.224 at the band profile; 0.600 at degraded |
| depth hold run | `(1.0, 1.6)` | `(1.0, 2.0)` | requested mean 1.2 / 1.5 |
| depth burst weight / run | off | `0.25` / `6.0` steps | the degraded profile's burst mixture |
| depth latency | 1 step, band `(0, 2)` | 2 steps, band `(1, 3)` | mean-preserving around the shipped fixed value |
| command hold fraction | `(0.0, 0.05)` | `(0.0, 0.25)` | residual after inference cadence; not measured |
| command hold run | `(1.0, 1.2)` | `(1.0, 1.5)` | same |

The command-hold rows are the one pair not derived from a measurement, and the
brief says so rather than dressing them as one. They are sized to be visibly
subordinate to the depth band — a test asserts the fraction stays under half of
it — because sizing them like the cadence would model the same stall twice.

### Deletions

`control_frequency_jitter_pct`, `obs_latency_steps` and `obs_latency_steps_range`
are removed. Randomizing the tick period needs rollout surgery rather than a
config field, so the field is deleted rather than left implying a randomization
that never ran. `TimingCfg`'s field set is pinned by a test, so a future field
cannot be added without choosing its consumer in the same change — the failure
mode this contract has already had.

## The contract goldens

Every one of the 22 composed-variant hashes and the depth-obs golden moved,
because the new fields sit on the depth noise model and the action term and the
snapshot walks both. The cause was isolated before anything was re-frozen:
**dropping only the five new field names from the serializer reproduces all 23
stored goldens byte-for-byte**, so the movement is the fields' presence alone —
no term reordered, no scale changed, no existing parameter touched. The other
184 env assertions stayed green throughout.

Re-frozen with sign-off, and the gate gained a piece it was missing. A
**policy-observation layout golden** now hashes the policy group with every
noise model dropped: terms, order, params and scales only. There are exactly two
— one per observation profile — and they are tier-invariant by construction,
because realism selects the corruption and never the layout. That separation is
what makes a randomization change readable: re-ranging a sensor's noise moves
every contract golden, correctly, and must not move this one, because a deployed
checkpoint is fed the same quantities in the same order either way.

## Acceptance criteria

- [x] The hold law's arithmetic agrees with the evaluation harness's own
      `TemporalProfile` on the mixture mean, the reachability ceiling, and the
      mean live run, asserted against the harness rather than restated.
- [x] Realized in-env texture verified against the configured profile: hold
      fraction and mean run length from the emitted frames, at a pinned profile
      and at the shipped ROBUST band, plus the union arithmetic with the
      memoryless drop.
- [x] A held frame is the previous emission byte-for-byte; a held command is
      the previous command byte-for-byte.
- [x] Inert at neutral parameters: a zero band recovers the drop-only stream at
      the drop rate with runs of mean `1/(1-p)`.
- [x] The reachability clamp never fires at either shipped band, and does fire
      and count when a request is genuinely unreachable.
- [x] Per-env parameters differ across envs and are re-drawn at reset; per-env
      latency reads its own lag and redraws on reset.
- [x] Tier convention: both tiers range, robust strictly wider; the depth
      latency band is mean-preserving; the ideal tier carries no temporal
      texture; the command-hold band stays subordinate to the depth band.
- [x] `TimingCfg`'s field set pinned, and no field is left without a consumer.
- [x] Frozen-vanilla law: the ProcRoom generator's vanilla stream is untouched
      — this change reaches no generator code — and the guard suite confirms it.
- [x] Contract-hash movement attributed to the new keys before re-freezing, and
      a layout golden added that a randomization change cannot move.
- [x] The evaluation harness, the play script, and the inference node are
      untouched.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the workflows the touched code supports.

## Pre-registered acceptance for the retrain this feeds

Committed here, before the run, so the outcome cannot be read backwards. Scored
on the **same** evaluation-harness grid that produced the numbers the ranges
were set from — `clean` / `band` / `degraded` × the new checkpoint and the
current one — and scored before any rig time.

| quantity | current | target |
|---|---|---|
| `degraded` completion, as a ratio of own `clean` baseline | 0.678 | ≥ 0.85 |
| `band` completion, as a ratio of own `clean` baseline | 0.967 | ≥ 0.95 |
| `clean` completion, absolute | 0.900 | ≥ 0.88 |

The clean row is the one that makes the other two mean anything: robustness
bought by giving up baseline performance is not the outcome this randomization
is for. A degraded ratio that clears 0.85 while clean falls below 0.88 reads as
a failed change, not a partial success.

**The retrain is not in this PR.** It is scheduled for when the
residual-attribution arms report, so every ratified axis lands in one training
run rather than several. The hold and its lifting condition live in
[`procroom-depth-enrichment`](procroom-depth-enrichment.md); this brief's
contribution to that run is an axis and the three numbers above.

## Out of scope

- **The retrain itself**, and any evaluation of it. See above.
- **Rollout or RL-loop surgery.** Deploy's full held-tick semantics — no
  inference, frozen recurrent state — is not reproducible at the noise-model or
  action-term layer, and reaching it would mean rewriting the training loop. The
  two mechanisms here are the idiomatic approximation, and the residual is
  documented rather than hidden.
- **Any change to the evaluation harness.** It is the acceptance instrument, it
  is being extended concurrently for the residual-attribution arms, and this
  work shares its law by asserting agreement rather than by rewiring it. Its
  `band` and `degraded` arms becoming standing regression checks is a later,
  separate decision.
- **Moving the depth latency centre.** The band is mean-preserving on purpose;
  where the centre belongs is the DR audit's Phase 1 bench measurement.
- **A cadence setpoint change.** The setpoint is a single shared constant and
  gains no plumbing here.
- **TF staleness on the subgoal frame.** Still a designed field with no
  mechanism; it stays the DR audit's.
- **RGB stream holds.** The depth-policy variants do not observe RGB, so the
  mechanism would train nothing.

## Investigation pointers

- The law and its per-env process:
  `source/strafer_lab/strafer_lab/tasks/navigation/mdp/hold_process.py`.
- Where a held depth frame is emitted: `DepthNoiseModel.__call__`, after the
  memoryless drop and before the camera-failure path — a failed camera outranks
  a stale one, which is why the byte-identity test disables failures.
- Where a held command is emitted: `MecanumWheelAction.process_actions`, between
  the delay buffer and the slew limiter.
- The per-env latency read: `DelayBuffer.__call__`, which gathers on a per-env
  ring index only when a range was configured.
- The action tests' isolation helper (`test_sim/actions/conftest.py`) now
  silences the hold by default. Without that, a delay or slew measurement is
  quietly taken through a repeated command.
- The layout golden's serializer drops the `noise` attribute only, and hashes
  `observations.policy` rather than the whole obs cfg — the container's class
  name differs per tier and is not part of the tensor.
