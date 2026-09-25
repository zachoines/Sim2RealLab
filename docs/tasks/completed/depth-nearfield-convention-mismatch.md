# Reconcile the near-field depth convention training writes with the one deployment writes

**Status:** Shipped 2026-09-13 in `fe510df` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/219

**Type:** bug (train/deploy parity, depth observation)
**Owner:** DGX (`strafer_lab` depth pipeline; the training-side decision is the
whole of it)
**Priority:** P0 — it is the attributed cause of the 2026-08-17 mission gate
reaching 0 of 6, and it silently corrupts roughly a fifth of every depth frame
the policy trains on. No depth artifact trained under the current convention can
be trusted on hardware, so it gates the depth-subgoal lane's behavioural
acceptance and every retrain that would otherwise be queued behind it.
**Estimate:** S for the code (one `torch.where`, or one post-processing hook), L
for the consequence — whichever direction is taken, the depth family needs a
retrain, and the fix direction has to be chosen before that retrain is spent.
**Branch:** `task/depth-nearfield-convention-mismatch`

## Story

As the **depth policy crossing from sim to the robot**, I need **the near-field
depth I train on and the near-field depth the deploy pipeline assembles to mean
the same thing**, so that **a pixel the camera cannot resolve does not read
"6 metres of open floor" in training and "an obstacle 0.2 m ahead" on hardware.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [context/env-composition-contract.md](../context/env-composition-contract.md)

## Context — the defect

Two sources write opposite values into the same pixels.

**Training.** The observation term fills the near field first: every pixel
closer than `nearfield_clip` = 0.4 m becomes `nearfield_fill` = 0.2 m, then the
image is clamped to [0, 6.0]
([`observations.py:613-678`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L613)).
The realism noise model then runs on that output and slams everything below
`min_range` to `max_range`
([`noise_models.py:561-562`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L561)),
with holes written the same way
([`:555-557`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L555))
and `min_range` = 0.2 m / `max_range` = 6.0 m as defaults no config overrides
([`:644-645`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L644),
[`sim_real_cfg.py:258-262`](../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L258)).

**Deployment.** `downsample_depth` rescues non-finite values to 6.0, takes the
8×8 block median, applies the identical nearfield rule — `DEPTH_MIN` 0.4 →
`DEPTH_NEARFIELD_FILL` 0.2 — and clamps
([`obs_pipeline.py:44-85`](../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py#L44),
[`constants.py:99-134`](../../../source/strafer_shared/strafer_shared/constants.py#L99)).
There is no `too_close` stage. The node therefore agrees exactly with the
**clean** training term — both write the same float32 `0.03333334` — and
disagrees with what the training policy actually saw.

**The threshold sits on the fill value.** By the time the noise model runs no
pixel is genuinely below 0.2 m; the term has already written exactly 0.2 m
everywhere it applies. The slam rule fires on that constant, dithered by the
stereo term, whose σ at 0.2 m is 5.0e-5 m. A symmetric dither about a threshold
placed on the population's own value sends half of it across — and it is a
property of the threshold's position, not of the noise magnitude, so both
realism tiers behave the same. Measured on 30 paired ticks: **49.90%** of the
nearfield class crosses, which is **18.89% of the whole frame** reading 6.0 m in
training where deployment reads 0.2 m, concentrated in rows 22–44 (the floor
band) at 36.9% against 0.02% in rows 0–21.

`min_range` is a `DepthCameraNoiseCfg` field whose stated job is "closer =
invalid". It is not doing that job: nothing reaching it is closer, and what it
inverts is the near-obstacle signal the observation term deliberately created.

## Measured, 2026-08-22 attribution

v2 (`model_998`) learned those statistics as load-bearing; v1 (`model_500`) did
not. Full evidence chain, with the analysis scripts and their outputs:
[`measurements/goal-a-attribution-2026-08-22`](../../measurements/goal-a-attribution-2026-08-22/README.md).

The offline replay reproduces the robot's own commands from its own captured
observations to mean absolute differences of 3.6e-4 / 2.2e-4 / 4.7e-4 across
the three command components, so the failure belongs to the artifact and not to
the on-device TensorRT path, and it is present in the first inference of the
mission — 173° off the goal direction over the first 8 s sim, before the subgoal
referent has moved once. A same-pose Kit probe rebuilds the capture's room from
seed determinism and places the robot 2.9 cm from the recorded pose; from there
v2's first command is 9.4° off its referent instead of 82° off. Substituting one
field at a time at that tick, every non-depth field can be replaced with its
clean-sim value and the command moves by at most 0.570°, while depth moves it by
95.8° and does so in both directions. Swept across all 30 records, the
noise-bearing rows command toward the referent **29/30**; the single-field arm —
the node's own observation with only its depth dims replaced by clean sim — is
**0/30**; and the whole-clean-row arm is **1/30**, its records 2–29 converging
into a 3°-wide band around −79° to −82°. The patch replays and both published
sweeps re-run bit-exactly (worst deviation 0.000e+00) from the shipped
artifacts.

The record also carries what the evidence does not support: the exoneration of
the node's observation assembly is causal and single-tick, not a whole-mission
claim; a specificity probe that would have isolated the slammed pixel class from
"near-field depth content" in general did **not** separate, so the narrower
mechanism claim is open (below).

## The decision — direction B, refined

**Decided: the noise stage emulates what the deploy pipeline OUTPUTS for a
failed pixel, not what the sensor emits.** `too_close` pixels take the near-fill
value, which is `min_range` and is the same number the observation term has
already written at every pixel it filled; holes take the median of their valid
3×3 neighbours, falling back to the near fill where the neighbourhood is invalid
too. Both are selectable — `too_close_fill`, `hole_fill` — with the new values as
defaults, so the pre-fix statistics stay reproducible for the A/B. `min_range`
stays 0.2.

The reasoning, against the three directions below. (B) was taken because the
collision disappears *by construction* rather than by retuning: writing the fill
value at a threshold placed on the fill value makes the comparison idempotent, so
no pixel is left below `min_range` and nothing can be reclassified. The refinement
— holes to a neighbourhood median rather than to a constant — is the third
sub-question below answered in the affirmative: the node's rescue is a reduction,
not a constant, and matching the reduction is closer to parity than matching
either literal. (A) remains the stronger answer to future drift and is explicitly
not taken; it also covers the 8×8 block median, which training still does not
perform, so that half of the divergence is open. (C) was rejected because it
trains the policy to ignore a signal the near field is supposed to carry.

A latent defect surfaced during implementation and is fixed with it:
`DepthNoiseModelCfg.height`/`width` were 60×80 against a 45×80 frame. Nothing
read them, so the error was inert; a neighbourhood median has to unflatten, so it
could not stay inert. They now come from `strafer_shared.constants`, and a shape
mismatch raises.

**What the fix does not do.** Measured after the fact: feeding the reconciled
depth to v2 does not make it emit the rig class — it still commands toward its
referent. The stereo Gaussian alone, 0.029 m with no class structure, already
carries the command 52° off the clean answer. So the artifact's sensitivity is to
the presence of realism noise, not to the convention, which is open item 1 below
measured directly. The divergence this brief names is removed; the retrain is
what has to produce an artifact that does not key on noise signature.

Evidence: [`measurements/depth-convention-fix-2026-09-13`](../../measurements/depth-convention-fix-2026-09-13/README.md).

## Fix directions — as put for the decision

All three make training and deployment agree; they disagree about which
convention is the right one and about where parity is enforced.

**(A) Route the training depth through deploy-equivalent post-processing.**
Pair the term with the deploy reduction the way `enrich_depth` equivalence is
already pinned, so the two paths are the same code path by construction rather
than by two literals that happen to match today. Strongest against future drift
and the only option that also covers the 8×8 block median, which training does
not perform at all (it renders one ray per policy pixel). Largest surface.

**(B) Flip the noise model's `too_close` and hole writes to the near-fill
convention.** Two `torch.where` values. Makes training agree with what both the
clean term and the node already write, and removes the threshold collision
outright. Smallest change; leaves the two conventions as two literals that must
stay in step, and asserts that "unresolvable" should read as "something very
close" everywhere — which is what the observation term's own docstring argues
and what the deploy pipeline already does.

**(C) Randomize the convention during training.** Buy robustness instead of
matching: sample per-episode (or per-frame) which value the near field takes, so
no artifact can key on either. Survives a deploy-side change without a retrain,
and is the only option that would have made v2 tolerate the mismatch. Costs
sample efficiency, and it trains the policy to ignore a signal the near field is
supposed to carry.

Sub-questions the choice has to answer either way:

- **`min_range` 0.2 m against the real D555's 0.4 m.** The cfg comment says the
  margin is deliberate. With the threshold at 0.2 it collides with the fill
  value; at 0.4 it would slam the entire nearfield class rather than half of it,
  which is worse, not better. The coherent readings are "remove the rule" or
  "give it a genuinely separate sentinel", not "retune the number".
- **Holes should probably not be a constant at all.** The node's rescue is
  non-finite → 6.0 followed by a block **median**, so an isolated invalid pixel
  is outvoted by its neighbours and never reaches the policy as a value. The
  noise model writes 6.0 per pixel with no such reduction. Matching the node's
  median-rescue is closer to parity than matching either constant.
- **Which pixel classes exist.** Deployment currently cannot distinguish
  *invalid* from *too close*; [`d555-depth-decode-validity`](../active/trained-policy/d555-depth-decode-validity.md)
  is the brief that gives it an explicit validity mask, and it asks for invalid →
  6.0 m while genuine sub-0.4 m returns keep the fill. Whatever this brief lands
  must agree with that split, or the two fixes will fight.

## Acceptance criteria

- [x] One convention, stated once, for what the near field means, and both the
      training path and `obs_pipeline.downsample_depth` produce it. The decision
      and its rationale are recorded in the brief before the code lands.
- [x] The threshold collision cannot recur — by the second route, not the
      first. The threshold is *still* the value the pipeline writes, deliberately:
      `test_the_threshold_sits_on_the_value_the_pipeline_writes` asserts the two
      are one number, and `test_nearfield_fill_survives_the_model` asserts the
      class survives whole, failing on the pre-fix tree at 0.5003. The coincidence
      is made harmless rather than avoided, which is why the pinning test asserts
      survival rather than a split.
- [x] A unit test drives the training noise model with an image the observation
      term has already filled and asserts the surviving near-field share matches
      the chosen convention. Verified failing on the pre-fix noise model at
      **0.5003** of the frame, against the predicted ~50%.
- [x] A parity test feeds one image through both paths and asserts they agree on
      the near-field class. `tests/contracts/test_depth_nearfield_parity.py`; it
      reaches the rclpy-free `obs_pipeline` by path so it runs on the training
      host rather than skipping there.
- [x] The affected share is measured on the training distribution, before and
      after, on the enriched robust env and its realistic sibling — see the
      record's fingerprint section. Reported as the far-clamp share of the frame
      rather than as a rollout return, which is the quantity the rule moves.
- [x] Two clauses that were pre-registered for the implementing change went
      unmet, and the record states both as measured deviations rather than as
      passes. A band of p95 ≤ 0.01 scaled for the node against noise-bearing
      depth is **unreachable by a correct change**: the node already differs from
      the *clean* sim referent by p95 0.02004 on reconstruction geometry that no
      convention touches, and the bisection that defined the band puts
      p95 > 0.01 on its *material* side, so the clause asks for the non-material
      side of a threshold clean sim already exceeds. The change takes that pair
      from 0.96667 to 0.02095, which is convergence onto the floor. And the
      deployed artifact does **not** return to the rig class on reconciled depth;
      open item 1 below is why.
- [x] Retrain scope: **every depth artifact is invalidated**, v1 and v2 both.
      Both trained under the retired convention — v1 at a smaller affected share
      because its pre-enrichment env put less surface inside 0.4 m, but under the
      same rule, so neither is a valid artifact for the reconciled observation.
      The nocam family is untouched, having no depth term. The magnitude under
      v1's own env is still unmeasured and unmeasurable from the artifacts; it
      needs a Kit run on the pre-enrichment tree, which the retired pin no longer
      makes convenient.
- [x] Source docstrings and comments asserting the retired convention are
      corrected in the same commit — `sim_real_cfg.py`'s "closer = invalid" and
      hole docstring, the noise model's class docstring and `min_range` comment,
      and the Kit hole suite's derivation block. The keyword sweep the conventions
      section prescribes comes back empty on every context module, package README,
      the top-level `Readme.md` and every non-exempt guide under `docs/`. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression: the composition contracts re-freeze with the movement
      attributed to the three changed keys and both layout goldens held, and the
      full `strafer_lab` gate passes. See the record's gate section.

## Training provenance — what armed it between v1 and v2

The mechanism did not change; the exposure did. `noise_models.py`,
`sim_real_cfg.py`, `observations.py` and `d555_cfg.py` are **byte-identical**
between the two artifacts' export trees (`eeacccc`, 2026-07-08 → `69014c6`,
2026-07-26). The collision is two-sided and both halves long predate the runs:
the `too_close` slam with its `min_range = 0.2` default entered in `52e1bd5` on
2026-01-12, and the `nearfield_fill = 0.2` the threshold lands on entered in
`c50a76a` on 2026-03-23 — that second commit is where the two numbers met.
Neither has been edited since. Neither candidate change separates the two runs:
#153 merged 2026-07-18, inside the window, but re-derives a variance test's
confidence interval and says in its own body that the noise models are not
touched, which the empty diff confirms; #143's 80×60 → 80×45 policy camera
(vertical FOV ~71° → 56.4°) has its last commit nineteen minutes before v1's run
begins, and v1's own stdout log is named `depth_subgoal_vfov8045`, so both
artifacts trained on the 80×45 camera.

What differs is the task ID, and through it two config fields.

| | v1 | v2 |
|---|---|---|
| run | `run_20260708_005923`, iterations 0 → 574 | `run_20260726_221955` 0 → 499, then `run_20260727_171735` 500 → 998 resuming leg 1 |
| task | `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0` | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` |
| deployed | `model_500.pt` | `model_998.pt` |

v2 is a fresh run under a task ID that did not exist at v1's tree, not a
continuation of v1. `enrich_depth` False → True raises near-camera surface
everywhere — walls 1.0 → 2.7 m, a per-episode ceiling at p = 0.7, shelf /
cabinet / tall-cylinder heights 0.8 / 0.6 / 0.7 → 2.0 / 2.1 / 1.8 m, two
mid-room columns at p = 0.5, difficulty un-pinned from 7 to U[4, 7], and spawn
inflation 2 → 1 so the robot starts closer to obstacles. `level` `real` →
`robust` triples `hole_probability`, doubles `disparity_noise_px`, enables
camera failure and takes depth latency 1 → 2 steps; it leaves `min_range` and
`max_range` alone, so it does not move the slam rate. Every one of those levers
raises the share of the frame the rule rewrites, which is the whole difference
between an artifact that tolerates the mismatch and one that depends on it.

**The direction is argued from the parameters; the magnitude is not measured.**
No measurement of the affected share under v1's environment exists on either
host and none can be made from the artifacts — it needs a Kit run on the
pre-enrichment tree. Neither run saved a config: `logs/rsl_rl/*/git/` is empty
for all three run directories, the checkpoints carry only `iter` and state
dicts, and the TensorBoard files carry no hyperparameter records. The per-run
stdout logs are the only surviving statement of what was run, and they are
machine-local. [`training-run-provenance-manifest`](../active/trained-policy/training-run-provenance-manifest.md)
is the brief that would have made this recoverable instead of reconstructed;
this is its second collection of evidence.

## Open items

1. **The slammed pixel class is not isolated.** The bisection separates
   corruption-bearing depth from clean depth cleanly, but a control that
   far-clamps an equal number of *randomly chosen* pixels moves the command
   nearly as far (48–52° against the 45° criterion), and a uniform rescale with
   no class structure crosses it. So the evidence supports "v2's command is
   acutely sensitive to near-field depth content"; it does not yet support "v2
   keys on the slammed pixels specifically". Isolating it needs a design holding
   mean depth and affected area fixed while moving only class membership. This
   does not disturb the fix — the convention divergence is real in the source
   and is the instance of the sensitivity that deployment actually presents —
   but it does bound how the mechanism should be described.
2. **The quoted command magnitudes do not reconcile.** Figures of `vy` +0.145
   and +0.224 at duty 1.00 have circulated for this failure. The capture's raw
   policy output never exceeds `vy` +0.0602 in the positive direction, and the
   node publishes the action as m/s capped at `NAV_LINEAR_VEL` 0.7841, so no
   constant in `strafer_shared.constants` maps one onto the other. The one-signed,
   zero-sign-change shape those figures describe does appear, in the last 20 s
   sim at `vy` mean −0.0568. Which log, window and unit produced the quoted
   magnitudes is a `strafer_inference` question and is parked here rather than
   answered.
3. **The one gate success is consistent but unproven.** `PILOT_uncontrolled_heading`
   reached 0.299 m of a 3.029 m goal. It is the only run in the gate record whose
   start heading differs — every scored mission and every repeat started at
   heading error −2.6° to −2.8°, the pilot at −115.5° — so it is also the only
   one whose body-fixed camera faced a materially different part of the room,
   which fits a content-dependent affected share. No depth was captured for it,
   and goal bearing alone does not separate the set (M2 failed at −107.2°). Not
   evidence for the change; a prediction it should retire.
4. **The last-half-metre parking behaviour is a separate thread.** Both
   artifacts fail to close the final ~0.5 m, v1 included, and that survives this
   fix. It is not in scope here and needs its own brief when someone picks it up.
5. **This change and [`d555-depth-decode-validity`](../active/trained-policy/d555-depth-decode-validity.md)
   now disagree about what "invalid" means, and it has to be settled there.**
   Training writes the *neighbourhood median* at an invalid pixel, so an isolated
   stereo failure reads as the surface around it. That brief asks for invalid →
   6.0 m while genuine sub-0.4 m returns keep the fill. Those are different
   answers to the same pixel class. The deployed node's actual output is what
   defines truth for the pair, not either brief's preference: today
   `downsample_depth` rescues non-finite values to 6.0 m and then lets the 8×8
   block median outvote them, so an isolated failure already reads as the
   surface and only a saturated block reads far. Whatever that brief lands has to
   name both cases, and it should decide the camera-failure row at the same time —
   a failed frame still writes `max_range` at every pixel on the training side,
   which is the last `max_range`-means-invalid write in the noise model, and the
   node's counterpart is a zero Twist from the watchdog rather than a cleared
   frame. Settle it there before either brief ships further.
6. **The same-pose probe's seed determinism does not survive a pinned-pair
   change, and that needs its own brief.** The probe anchors on whatever pose the
   environment spawns the robot at and treats the capture's map coordinates as an
   offset from it. Room generation at seed 42 moved across the Isaac Lab pair
   flip, so the robot landed 2 m away facing open space and the clean near-field
   share of its frame was 0.0000 where the original run's was 0.3786 — an empty
   denominator for the metric the probe exists to produce. The probe itself is
   fine; the assumption that a seed reproduces a room is not. A pose-stable sim
   probe needs a captured scene USD rather than seed determinism, and that is the
   brief to file.

## Optional validation, designed and deliberately not run

A diagnostic-only shim on the node's depth path that mirrors the training
convention — rewrite the nearfield-fill pixels to the far clamp before the
observation is assembled — predicts that v2 drives on the rig without any
retrain. On the captured frame it already swings the single-tick command from
82° off the referent to 38° toward it. It would be a strong end-to-end
confirmation and it costs one rig session.

It is **never** a production configuration. It deliberately feeds the policy a
false reading of the space directly in front of the robot, on hardware, and its
whole purpose is to reproduce a defect. If it is run it must be behind a flag
that is off by default, named as diagnostic in the code, and removed in the PR
that lands the real fix.

## Out of scope

- `observations.py` and `obs_pipeline.py`. The fix lands entirely in
  `noise_models.py`: the observation term and the deploy pipeline already agreed
  with each other, and the noise stage is the only thing that disagreed.
- The retrain itself, and the choice of which checkpoints it starts from.
- The 16UC1 decode and validity mask
  ([`d555-depth-decode-validity`](../active/trained-policy/d555-depth-decode-validity.md)) — adjacent and
  must agree with whatever lands here, but separately owned.
- The subgoal-generator clock and cadence-report defects from the same gate
  ([`subgoal-generator-sim-clock-freshness`](subgoal-generator-sim-clock-freshness.md),
  [`cadence-report-window-never-resets`](../active/reliability/cadence-report-window-never-resets.md)),
  both measured non-causal for this failure and both still worth fixing.
- The last-half-metre parking behaviour.
