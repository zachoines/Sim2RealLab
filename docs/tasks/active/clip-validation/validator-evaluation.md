# Wire and evaluate the CLIP transit-monitor + verify_arrival path

**Type:** task / investigation
**Owner:** Either (the wiring half is in `strafer_autonomy` /
`strafer_ros`; the measurement half is DGX-led against harness
output)
**Priority:** P1 (gates whether mid-mission validation graduates,
becomes a co-trained-validator follow-up
([`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)),
or gets retired). **Pickup gate: now AFTER
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)**
— the cascade ships on the shared trunk the bake-off chooses (the
frozen text-capable trunk per
[`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)),
not on whatever ONNX happens to sit in `~/.strafer/models/`. The
roadmap reorder (backbone chosen first, then v1) is recorded in the
spine's Roadmap section; it is affordable because the project is
greenfield on weights. Also blocked
on harness output existing from **any driver mode**
(teleop via
[`harness-architecture`](../harness/harness-architecture.md)'s
[Driver: teleop](../harness/harness-architecture.md#driver-teleop)
is the fastest-to-produce; bridge via `next-integration-round`
also works) and the chosen trunk's exported visual + text encoders
existing under `~/.strafer/models/` (a bake-off output — see
Prerequisites below).
**Estimate:** L (~multi-day; small-but-broad implementation +
calibration measurement + write-up)
**Branch:** task/clip-mid-mission-validator-evaluation

## Story

As an **operator deciding whether to keep, replace, or retire CLIP
mid-mission validation**, I want **the existing
`SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` +
`verify_arrival` path actually wired into the production executor
and measured against harness output**, so that **the
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
recommendation has the empirical numbers it pre-registered, and
the follow-up decision (escalate to a learned validator vs. wrap
the cheap path with a VLM arbiter) is anchored in measurement
rather than guessed at**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
- [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
  — the frozen-trunk spine. This brief's cascade is the validator
  consumer (case-1 image-vs-image + case-2 image-vs-text) on the
  shared trunk; the spine fixes the trunk family, the freeze
  invariant, the backbone dimension, and the roadmap order this
  brief sits in.

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md).
This brief is the §4 next step. The §1.3 wiring gap and the §2.6
measurement gap together are the work.

## Context

### Prerequisites

This brief assumes two artifacts exist on the DGX before the
measurement step runs. Neither is the brief's job to produce; the
brief work checks them at the start and stops with a clear error
if missing.

| Prerequisite | Why | How it gets there |
|---|---|---|
| Populated LeRobot v3 dataset at `data/sim_in_the_loop/<scene>/` for ≥ 3 scenes × ≥ 30 missions | Without harness episodes, no metrics can be computed | **Fastest path: teleop.** [`harness-architecture`](../harness/harness-architecture.md)'s [Driver: teleop](../harness/harness-architecture.md#driver-teleop) gets to ≥ 3 scenes × 30 success + 30 hard-negative episodes in ~one operator-evening. **Slower path: bridge.** [`next-integration-round`](../investigations/next-integration-round.md) produces bridge-driver output as part of its acceptance, blocked on MPPI / Nav2 stability. Either driver's output works — the eval script consumes via HF `LeRobotDataset` regardless of source. |
| The chosen shared trunk's exported visual + text encoders under `~/.strafer/models/` | The eval script and the image-vs-text tripwire both encode against the runtime trunk; the cascade must ship on the **bake-off's chosen trunk**, not on an arbitrary pre-existing ONNX | A [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md) output. The bake-off picks the frozen text-capable trunk (SigLIP-2-B lead per the spine) and exports its encoders; that export is this brief's hard upstream, **not** a step this brief owns. The two live 512-dim sentinels and the dim-pinned ChromaDB store are made backbone-dim-aware inside the backbone-switch PR (see the spine's Backbone-dim hygiene section), so the eval inherits a dim-consistent runtime. The brief work checks the encoders exist at the start and stops with a clear error if missing. |

The brief's wiring + protocol-refactor work doesn't need either
prerequisite — those steps land independently, and the
measurement step gates on the prerequisites being satisfied.

### CLIP is the SEMANTIC validator — not a geometric one

The single most important scoping decision in this brief: **the
CLIP cascade answers "does the current view match the right
*place/region* and the right *target description*?" — it does not
answer "is the robot off the geometric path?"** Geometric deviation
is owned elsewhere:

- **No-progress / stuck detection** under the reactive
  [`strafer_direct`](../trained-policy/strafer-direct-sim-validation.md)
  backend (RL, no explicit planned path) is a locomotion watchdog —
  pose-advance + yaw-rate over a window — not a perception job. See
  [`nav-stall-multilayer-watchdog`](../../parked/reliability/nav-stall-multilayer-watchdog.md).
- **Plan-freshness / off-path tracking** under the `hybrid` (Nav2)
  backend is the Nav2 controller's own concern: Nav2 owns the path
  and knows when the robot is off its own plan.

A geodesic-A\* lateral-deviation rule is therefore **dropped from
CLIP's scope.** The earlier draft's per-window `off_course`
definition (lateral distance from the geodesic-A\* path to the
mission target) is **retired** as a CLIP signal — it conflates
legitimate exploration and "via the dining room" detours with
genuine error, and it is a geometric question the backends already
answer better. What remains for CLIP is purely semantic, validated
**per leg against that leg's sub-goal**.

### Per-leg sub-goal validation + leg-type suppression

The validator no longer validates every frame against the
*mission's final target*. It validates against the **current leg's
sub-goal**, with a per-leg-type suppression flag the executor
supplies:

| Leg type | Validator state | Validates against |
|---|---|---|
| `scan` | **OFF** | nothing — no locomotion to validate |
| `explore_until_visible` | **DISARMED** | exploration is not deviation; the robot is *meant* to leave the line toward the target |
| `navigate` (staging hop) | **ARMED (semantic)** | the **hop's room**, not the final target's room |
| `navigate` (final target) | **FULLY ARMED** | case-1 (target room) + case-2 (target instance) |

This is the **same leg-type granularity** the multi-room re-test
brief makes authoritative; the full, canonical DEVIATION CONTRACT
(every primitive, both backends, the soft-ranking carve-out for
exploration) is **owned by**
[`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md).
This brief states only the v1 single-room subset it needs to wire
the suppression flag; it does **not** duplicate the full contract.

To do this, the cascade accepts a per-leg context tuple from the
executor at `activate()`:
`(leg_type, sub_goal_room, sub_goal_target_phrase)`. `leg_type`
selects the suppression state above; `sub_goal_room` /
`sub_goal_target_phrase` replace "the mission target" as the
case-1 / case-2 reference for that leg.

### Signal emphasis — image-vs-text is PRIMARY, image-vs-image is a WARM booster

The earlier draft headlined the image-vs-image place-recognition
signal (the `TransitMonitor`). **Invert this.** Per the spine's
deployment-context section (one robot, one home, warm map is the
steady state but cold-start is a real transient):

- **Image-vs-text (case-2 instance discrimination)** is the
  **PRIMARY** signal — it is **map-free and cold-start-safe**, the
  only thing that works on a first visit, and it directly answers
  the case-2 question. This is the `TextAlignmentMonitor`.
- **Image-vs-image place recognition** (the `TransitMonitor`) and
  **Step B implicit memory** are **WARM boosters** — they layer on
  once the map has populated and degrade to their no-retrieval
  baseline on a cold/empty map by construction (the `sparse_map`
  permanent-`on_track=True` mode is the documented degradation, not
  a bug). They sharpen case-1 when warm; they never replace the
  primary.

The acceptance metrics, the OR-fusion, and the McNemar test below
are unchanged in mechanics, but the **framing** is flipped:
image-vs-text carries the cold-start bar; image-vs-image is
reported as the warm-leg booster.

### Brier stratified by contrast-pool composition

The image-vs-text softmax probability is taken over a contrast pool
that is **sourced dynamically per leg** (VLM sibling labels, then
semantic-map priors, then the fixed fallback anchors). A pool of
scene-grounded siblings and a pool of generic fallback anchors
produce **non-comparable** probabilities — a single aggregate Brier
score mixes them. **Report Brier stratified by contrast-pool
composition** (`grounded-siblings` / `map-prior` / `fallback-anchor`,
matching the pool actually used at each leg's `activate()`), not as
one aggregate number.

### Letterbox-vs-center-crop reconciliation (flag)

The runtime preprocessing in
[`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
does **resize + center-crop**, but the harness `info.json` `cameras`
block carries `preprocessing_hint: "letterbox-to-square for ViT
inputs"` (see
[`harness-architecture`](../harness/harness-architecture.md)). The
eval re-encode and the runtime encode **must share one preprocessing
path**, or the offline AUC measures a different input distribution
than the robot sees. The `cameras` preprocessing-hint side is the
**orchestrator's** to set; this brief flags the mismatch and pins
both to one path (and the backbone-switch PR may move the runtime
default to letterbox as part of choosing the trunk). Resolve before
the metric pass — a preprocessing skew silently invalidates the
calibration.

### What's already in the repo

The semantic-map package is complete-but-orphaned:

- [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py),
  [`manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py),
  [`background_mapper.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/background_mapper.py),
  [`transit_monitor.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py)
  exist and have unit tests under
  [`source/strafer_autonomy/tests/`](../../../../source/strafer_autonomy/tests/).
- [`mission_runner.py`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  accepts `semantic_map=` and `background_mapper=` and uses them in
  `_verify_arrival`, `_activate_transit_monitor`,
  `_try_query_before_scan`, `_store_scan_observation`,
  `_log_arrival_failure`.
- [`build_command_server`](../../../../source/strafer_autonomy/strafer_autonomy/executor/command_server.py#L246)
  accepts both kwargs and forwards them.

But [`executor/main.py:146-151`](../../../../source/strafer_autonomy/strafer_autonomy/executor/main.py#L146-L151)
calls `build_command_server` **without** `semantic_map=` /
`background_mapper=`. In production every CLIP-touching path
short-circuits on `None`. The
[`strafer_autonomy/README.md` skill table](../../../../source/strafer_autonomy/README.md#L137)
overstates this — it says `verify_arrival` is "always appended"
and runs CLIP top-k ranking; the appending is true (the compilers
do emit it), the ranking part is dead code.

### What this brief produces

Three artifacts in one PR:

1. **Wiring** in `executor/main.py` so the production executor
   constructs `SemanticMapManager` and `BackgroundMapper` and
   passes them through. Gated on env vars so a misconfigured
   Jetson degrades gracefully instead of hard-failing.
2. **An image-vs-target-text parallel tripwire**
   (`semantic_map/text_alignment_monitor.py`, parallel to
   `transit_monitor.py`) that uses the existing CLIP text tower
   to encode **both the mission target phrase and a set of
   alternate phrases**, then runs CLIP zero-shot
   classification on each live frame from `BackgroundMapper`'s
   capture loop. The monitor emits both a **continuous
   softmax-probability score** for `p(target)` (calibrated, for
   ROC-AUC) and a **discrete top-1 classification** (for
   threshold-free time-to-decision). Fires `abort=True` when
   `argmax != target` for N consecutive captures **and** the
   softmax-probability of the target stays below a
   baseline-relative threshold. **No model fine-tune required**
   — the runtime artifact reuses the already-loaded CLIP text
   encoder.

   **Why multi-text contrast, not single-cosine.** Single-text
   cosine is dominated by visual content unrelated to the
   target (lighting, viewpoint, camera distance), and the
   "below baseline by 0.15" threshold the earlier draft
   proposed bakes those nuisance variables into the signal. CLIP
   was originally designed for *zero-shot classification* — encode
   N candidate texts, take argmax — and that is exactly the
   case-2 instance-discrimination question this tripwire is
   trying to answer ([Radford et al., 2021](https://arxiv.org/abs/2103.00020),
   §3.1 zero-shot transfer protocol). Multi-text contrast also
   produces a calibrated probability (the softmax) that the
   §4.1 Brier-calibration metric can score, which a bare cosine
   cannot.

   **Where the alternates come from.**

   | Source | When | What |
   |---|---|---|
   | **VLM grounding context** (primary, real-deployment safe) | At `scan_for_target` start, the planner submits the target + any sibling labels the operator's mission text mentions; the VLM grounder returns multiple bboxes when same-label siblings are visible. The monitor consumes the **other** detection labels as alternate phrasings. | "the south window" + alternates ["the north window", "the doorway", "the bookshelf"] from the same `scan_for_target` rotation. |
   | **Semantic-map prior observations** (secondary, deployment-safe; degrades on first visit) | When `SemanticMapManager` has prior observations of same-label objects with different `text_description` fields, those alternates feed the contrast pool. Falls back to (1) on cold-start. | Adds historical room-context phrasings without depending on the current `scan_for_target` having found siblings in-frame. |
   | **`scene_metadata.json`** (sim-eval only, **not deployment**) | The measurement script's eval pass draws alternates from `scene_metadata.json`'s `objects[]` with the same `target_label` but a different instance, or with a different label in the same room. | Eval-time ground truth for case-2 alternates; **never** consumed by the live executor on the real robot. The runtime monitor uses only sources (1) and (2). |

   **Cold-start behavior.** If no alternates can be sourced at
   `activate()` — fresh deployment, no `scan_for_target` siblings,
   empty semantic map — the monitor falls back to a fixed
   "negative anchor pool" of `STRAFER_TEXT_ALIGN_FALLBACK_ANCHORS`
   (default: a small list of common indoor categories the target
   is **not**, e.g., `["a blank wall", "an empty hallway", "a
   closed door"]`). The fallback gives the monitor a non-trivial
   contrast even on the first leg in a new house, at the cost of
   weaker discrimination than scene-grounded alternates.

   **Threshold calibration.** The fire condition is **dual-gated**:
   `argmax != target` for `N=3` consecutive captures AND
   `p(target) < baseline − STRAFER_TEXT_ALIGN_MARGIN`
   (default margin 0.10 on the softmax probability, not on the
   raw cosine). Baseline is the softmax `p(target)` on the
   scan-found view, the moment the target was last grounded.
   The dual gate prevents the tripwire from firing when both the
   target and a near-synonym alternate are highly probable
   (which would manifest as argmax flipping under viewpoint
   noise but the target probability staying high).
3. **A `Validator` protocol refactor.** `BackgroundMapper`
   currently holds a concrete `TransitMonitor`. Replace with a
   `Validator` protocol (lives in `semantic_map/protocols.py`)
   that both `TransitMonitor` and the new
   `TextAlignmentMonitor` implement. This is small (~30 lines)
   but lets the future
   [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
   brief plug in enhanced cascade variants without touching
   `BackgroundMapper` again.
4. **A measurement script** that consumes the harness's per-mission
   output (LeRobot v3 schema per
   [`harness-architecture`](../harness/harness-architecture.md)),
   replays the CLIP path (both signals)
   against it in offline mode (no live executor needed for the
   metric pass), and emits
   `data/transit_monitor_eval/<run_id>/report.json` with the
   pre-registered metrics from
   [`MISSION_VALIDATION_ARCHITECTURE.md` §4.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#41-falsifiable-success-criteria-for-the-next-brief).

### Pre-registered metrics — industry-standard binary-classifier statistics

Computed against a harness episode set covering ≥ 3 distinct
Infinigen scenes and ≥ 30 missions total. Metrics are
**disaggregated by failure case** per
[`MISSION_VALIDATION_ARCHITECTURE.md` §2.3](../../../MISSION_VALIDATION_ARCHITECTURE.md):
case 1 (wrong-room), case 2 (wrong-instance, same room). Case 3
(trajectory-shape) is excluded — it requires a planner-side
prerequisite.

The reporting framework follows standard practice for
binary-classifier evaluation in the perception-cascade
literature, *not* an arbitrary TPR-at-fixed-FPR threshold:

| Metric | Definition | Case scope |
|---|---|---|
| **ROC-AUC + 95% bootstrap CI** | Area under the ROC curve over the tripwire's confidence-output sweep, with bootstrap-resampled confidence intervals (1000 resamples). | Per case, per signal (image-vs-image, image-vs-text, OR-fused) |
| **PR-AUC + 95% bootstrap CI** | Area under the precision-recall curve. More informative than ROC-AUC when the positive class (failure) is rare, which it is here. | Per case, per signal |
| **Confusion matrix at production threshold** | Confusion matrix at whatever operating threshold the cascade ships with in production. Sets the operational TPR / FPR for the running system, but as a *report*, not a *bar*. | Per case, per signal |
| **Brier score** | Probability calibration of the tripwire's soft signal. The cascade emits a continuous score; we want it well-calibrated so the arbiter's threshold is principled. | Per case, per signal |
| **McNemar's test** | Compares the image-vs-image and image-vs-text tripwires on the same trial set. Tests whether they catch *overlapping* failures or *distinct* ones — informs whether OR-fusion is justified. | Per case |
| **Time-to-decision CDF** | From leg start to first `abort=True`. Reported as a CDF; median + p95 callouts. | Per case |
| **Cascade end-to-end** (after VLM `/describe` arbiter + LLM-as-judge against mission text) | All of the above, computed on the cascade's final decision (post-arbiter) rather than the tripwire alone. Reports lift from the arbiter pass over tripwire-only. | Per case |
| **Frame-to-frame CLIP cosine σ + top-1 flip rate per meter** | Diagnostic only; sanity-check of the cascade's signal stability. Not a primary acceptance metric. | Diagnostic |

The five-way mission label (`on_course` / `wrong_room` /
`wrong_instance` / `trajectory_violation` / `ambiguous`) requires
post-processing the harness output, not new sim runs. The
measurement script's `--root-cause-pass` mode appends a
`root_cause` field to each mission record. After the harness
brief ships, the natural target is `mission.json`'s record (one
per mission); for legacy harness output, `root_cause` is written
into the per-mission summary in `frames.jsonl`. Either way, the
labeling logic is:

1. **`on_course` vs. failed:** mission's `mission_state` +
   `reachable` flags.
2. **`wrong_room` vs. `wrong_instance` split (failed missions
   only):** check whether the final pose `(x, y)` lies inside the
   room polygon containing `target_position_3d` (room polygons
   come from `scene_metadata.json`'s `rooms[]`). Outside the room ⇒
   `wrong_room`; inside the room but > R m from
   `target_position_3d` AND scene metadata shows ≥ 2 objects with
   the same `target_label` ⇒ `wrong_instance`.
3. **VLM cross-check (failed missions only):** call
   `strafer_vlm`'s `/describe` on the final-pose frame and use
   an LLM-as-judge prompt to compare against the mission target
   phrase, **with the bias-mitigation protocol in the next
   subsection — single-shot Qwen-VL-3B-as-judge against
   Qwen-VL-3B-as-grounder is self-enhancement-biased**.
   Disagreement with step 2 (after the bias-mitigated protocol
   converges) ⇒ `ambiguous`; excluded from TPR / FPR.
4. **`trajectory_violation`:** **always empty** in this brief —
   the planner does not yet emit trajectory constraints, so no
   mission can be labelled here. Reserved field.

### Per-window labels alongside per-leg labels

The mission-level label above answers "did the mission end in the
right place?" The tripwire fires *mid-leg*, so the right TPR /
FPR pairing for the tripwire's actual decision is **per-window**,
not per-leg. A mission can be on-course for 80 % of its leg and
drift off at the last 20 %; another can be off-course from t = 0
because the planner emitted a wrong goal. Collapsing both into a
single mission-level boolean costs the eval its time-resolved
signal.

Precedent for per-subgoal / per-window robotic eval: StepEval
([ElMallah et al., 2025](https://arxiv.org/abs/2509.19524)) makes
the case that "single binary success rate obscures where a
policy succeeds or fails along a multi-step task" and uses a
VLM as judge of per-subgoal outcomes. The same critique applies
to mid-mission tripwire eval — `reachability` is the leg outcome,
not the per-tick truth.

The measurement script computes both:

| Granularity | Label source | What it measures |
|---|---|---|
| **Per-leg** | `root_cause` from the labeling pipeline above | "Did the tripwire fire at any point during a failed leg?" — the user-visible save-the-mission signal. |
| **Per-window** | A 5-capture rolling window is `on_target` if the **robot's view** at the end of the window is *semantically* consistent with the current leg's sub-goal — the room the leg is committed to (case-1) and, on a final-target leg, the target instance (case-2). Otherwise `off_target`. The window truth is sourced from the room-polygon / instance ground truth (`scene_metadata.json`, **eval-only**), per leg's sub-goal, **not** from a geodesic-A\* deviation. | "Was the robot's view actually wrong-room / wrong-instance at this tick?" — the tripwire's actual decision boundary. |

Per-window TPR / FPR / time-to-decision are reported alongside
per-leg metrics. Per-window is the *primary* signal for choosing
the tripwire's operating threshold; per-leg is the user-facing
"did we save the mission" signal.

**Geometric off-path is deliberately not a per-window truth here.**
The window label is a *semantic* room/instance match against the
leg's sub-goal — not "is the robot off the A\* line." Geometric
drift is the backends' job (the `strafer_direct` no-progress
watchdog and the `hybrid` Nav2 plan-freshness tracker, per the
CLIP-is-semantic scoping above), and scoring it here would label
legitimate "via the dining room" detours and exploration as failures.
On `scan` / `explore_until_visible` legs the window label is
**suppressed** (no per-window row emitted), matching the
leg-type suppression flag.

### Eval-set construction — scene holdout, cold-start protocol, label authority

The §4.1 success bar is meaningless if the eval set is drawn
from the same distribution the validator was trained on (Step A
of [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
trains on harness frames; if cotrained's *eval* is also drawn
from harness frames without a holdout split, the reported AUC
double-counts). Three explicit holdout protocols, anchored to
peer practice (RoboCasa task splits, Habitat-Lab eval methodology,
StepEval's policy-and-model-agnostic design):

1. **Scene-seed holdout.** The eval set's scene seeds are
   declared at the top of `report.json` (a `held_out_seeds`
   field). Any downstream training brief that consumes the
   harness corpus **must** exclude these seeds from training.
   The cotrained brief's Step A and Step B both reference this
   field at training time. Default split: **2 / 3 scenes for
   training, 1 / 3 scenes for eval**, by seed not by scene name
   (so the room layouts differ, not just the lighting). Run the
   v1 eval on the held-out seeds only; the report records
   per-seed disaggregation so cross-seed generalization is
   legible.

2. **Cold-start vs. warm-map disaggregation.** The
   `TransitMonitor`'s image-vs-image retrieval signal is
   structurally near-random on an empty map (per
   [`MISSION_VALIDATION_ARCHITECTURE.md` §2.4](../../../MISSION_VALIDATION_ARCHITECTURE.md#24-structural-the-transitmonitors-decision-rule-has-known-cold-start-sparse-map-and-dead-locale-failure-modes)
   — `sparse_map` returns `on_track=True` permanently until
   the map is populated). Computing a single ROC-AUC across
   both cold-start and warm-map legs measures the wrong
   distribution. The script's labeling pipeline emits a
   `map_state ∈ {cold, warming, warm}` field per leg based on
   ChromaDB node count at leg-start:
   `cold` if 0 nodes, `warming` if 1 – 4 nodes,
   `warm` if ≥ 5 nodes. This split applies to the **warm-booster**
   image-vs-image signal, which is structurally degraded on a cold
   map: per-case AUC for it is reported separately for `warm` and
   (`warming` + `cold`) legs, and the warm-leg case-1 AUC is the
   bar that signal is held to (cold-start is a fixed-degradation
   diagnostic, not gated). The **primary** image-vs-text signal is
   map-free and cold-start-safe by construction, so it is **not**
   split by `map_state` — its case-2 AUC is reported as a single
   number across all map states and carries the primary §4.1 bar.
   This is the inversion in practice: the cold-start-safe primary
   bar rides on image-vs-text; image-vs-image is gated only on the
   warm legs where it is meant to contribute.

3. **Label-authority audit on a held-out human-scored
   subsample.** The five-way `root_cause` labeling pipeline
   (room-polygon check + VLM `/describe` + LLM-as-judge) is
   itself a model output and inherits the biases in the next
   subsection. Before its labels are treated as ground truth
   for TPR / FPR, **at least 50 missions** (stratified across
   `on_course` / `wrong_room` / `wrong_instance` /
   `ambiguous`) are inspected by a human and the agreement rate
   is reported as a `label_authority_cohens_kappa` in the §4.4
   addendum. Below Cohen's κ ≈ 0.7 (substantial agreement, per
   the Landis-Koch scale), the cascade's reported AUC is
   flagged as label-noise-bounded and a refinement to the
   labeling pipeline is filed as a follow-up.

### Arbiter and LLM-as-judge bias mitigation

The cascade end-to-end pass (tripwire fires → `/describe` +
LLM-as-judge against mission text → final decision) is a binary
classifier whose head is the LLM judge. The 2025 LLM-as-judge
literature has documented at least four biases that the brief
must mitigate, not handwave:

| Bias | Documented magnitude | Mitigation this brief adopts |
|---|---|---|
| **Position bias** | ~40 % GPT-4 inconsistency on swapped (A, B) vs. (B, A) prompts ([Wang et al. 2024](https://arxiv.org/abs/2406.07791)) | Every judge call runs twice with `[mission_text, scene_description]` and `[scene_description, mission_text]` orderings. Accept the call as valid only if both orderings agree; otherwise the leg is `ambiguous` and excluded from TPR / FPR. |
| **Self-enhancement bias** | LLM judges score their own-family outputs higher; mitigated by cross-family ensemble ([Zheng et al. 2024](https://arxiv.org/abs/2306.05685), [LLM-Judge-bias survey 2025](https://arxiv.org/abs/2406.22891)) | The arbiter judge is **not the same model as the grounder**. The grounder is Qwen2.5-VL-3B at port 8100 (`strafer_vlm`); the judge is the planner-LLM at port 8200 (a different model family — see `strafer_autonomy.planner` config). If the judge can be switched to a *third* model family (e.g., the local Qwen3-4B already cached on the DGX per `vla-v2-architecture` § Investigation pointers, *but* not the deployed VLM), do so for the eval pass; document the choice. |
| **Verbosity / style bias** | ~15 % score inflation for verbose answers; style bias 0.76 – 0.92 across judge models (the dominant bias overall per [the 2025 survey](https://arxiv.org/abs/2406.23178)) | The judge prompt uses a fixed 1 – 4 calibration scale (1 = clearly off-target, 4 = clearly on-target) with single-token output (no free-form rationale; the rationale lives in a separate prompt that the eval ignores for binary classification). |
| **Tier-1A vs. small-judge gap** | [Tier-1A judge analysis 2025](https://arxiv.org/abs/2510.09738) places Qwen2.5-72B in "human-like" tier with Cohen's κ z-score 0.14 — but the deployed **3B** model is not Tier 1A. | If the deployed Qwen2.5-VL-3B is the only available judge, **report a Cohen's κ against the held-out human-scored subsample** (item 3 in the previous subsection) and flag the cascade-end-to-end AUC as judge-bound if κ < 0.7. The brief does **not** ship if the judge agreement is below this floor — the §4.4 addendum either reports a passing judge or files a follow-up to swap in a stronger judge. |

The cascade end-to-end ROC-AUC bar (≥ 0.90 lower 95 % CI) is
computed *after* these mitigations; numbers without the
mitigations are reported as a diagnostic but not against the
bar.

### Acceptance — single-bar pass / fail with the statistics framework

The brief ships when **all of the above metrics are computed,
reported with confidence intervals, and the cascade meets a
**published-baseline-comparable** ROC-AUC threshold per case.**

Per the perception-cascade literature, ROC-AUC ≥ 0.85 at the
**lower 95 % confidence bound** is the standard "this signal is
useful" bar; ROC-AUC ≥ 0.90 lower-bound is a strong signal.
This brief targets ≥ 0.85 lower-bound on case-1 image-vs-image
and case-2 image-vs-text, with the cascade end-to-end ≥ 0.90
lower-bound on both. Below ≥ 0.85 the cascade is documented as
not-yet-useful and the brief defers to follow-ups for
improvements rather than escalating to a different validator
class.

Decision branches collapse to two:

- **Cascade meets the bar.** Ships to production behind
  `STRAFER_SEMANTIC_MAP_ENABLED`. The §4.4 addendum records the
  ROC/PR-AUC numbers, McNemar's test outcome, calibration
  curve, and time-to-decision CDF.
- **Cascade fails the bar.** Recorded honestly in the addendum.
  Improvements are filed via
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
  (research-flavored follow-up; co-trained CLIP + speaker +
  retrieval-augmented inference). The end-to-end VLA research
  arm at
  [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md)
  also remains as an alternative path. **No "small learned
  validator" escalation** — that path was retired (see
  [`completed/learned-mid-mission-validator`](../../completed/learned-mid-mission-validator.md)
  for the rationale).

The PR's write-up appends a §4.4 to
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
with the full statistics report, the cascade decision, and any
follow-up brief filed.

## Acceptance criteria

- [ ] **Wiring.** `executor/main.py` constructs and passes a
      `SemanticMapManager` + `BackgroundMapper` to
      `build_command_server`, gated on `STRAFER_SEMANTIC_MAP_ENABLED`
      env var (default `1`; `0` preserves the current degraded
      behavior). On startup, log whether CLIP ONNX models loaded
      successfully — the existing graceful-degrade in
      `clip_encoder.py:_load_models` should be visible in operator
      logs, not silent.
- [ ] **`Validator` protocol + per-leg context.**
      `source/strafer_autonomy/strafer_autonomy/semantic_map/protocols.py`
      defines a `Validator` protocol with `activate`, `deactivate`,
      `check`, `is_active` matching the existing
      `TransitMonitor` shape. `activate()` accepts a per-leg
      context tuple `(leg_type, sub_goal_room,
      sub_goal_target_phrase)` from the executor: `leg_type`
      selects the suppression state (`scan` → OFF;
      `explore_until_visible` → DISARMED; staging-hop `navigate` →
      ARMED semantic against the hop's room; final `navigate` →
      FULLY ARMED case-1 + case-2), and `sub_goal_room` /
      `sub_goal_target_phrase` replace the mission's final target
      as the per-leg case-1 / case-2 reference. `TransitMonitor`
      declares conformance. `BackgroundMapper` accepts a
      `Validator` instead of a concrete `TransitMonitor`. Unit
      test under `tests/` confirms the substitution and that a
      `scan` / `explore_until_visible` leg never fires an abort.
- [ ] **Image-vs-target-text tripwire (multi-text contrast).**
      `semantic_map/text_alignment_monitor.py` implements the
      `Validator` protocol. At `activate()` it encodes the
      mission target phrase **and** an alternate phrasings pool
      sourced (in order) from: VLM grounding's sibling-detection
      labels from the latest `scan_for_target`, the
      `SemanticMapManager`'s prior same-label observations,
      and a fixed fallback anchor pool from
      `STRAFER_TEXT_ALIGN_FALLBACK_ANCHORS` if (1) and (2) are
      empty. On each `check()` the monitor emits both a
      softmax `p(target)` over the contrast pool (continuous,
      for ROC-AUC + Brier scoring) and a top-1 argmax
      (discrete, for tripwire decisions). Fires after `N=3`
      consecutive captures with `argmax != target` **and**
      `p(target) < baseline − STRAFER_TEXT_ALIGN_MARGIN` (default
      `0.10` on softmax probability, not raw cosine).
      `STRAFER_TEXT_ALIGN_FALLBACK_ANCHORS` defaults to a
      three-item list (`["a blank wall", "an empty hallway", "a
      closed door"]`) and is operator-overridable for
      site-specific tuning. The eval pass populates alternates
      from `scene_metadata.json`; the runtime monitor never
      reads `scene_metadata.json` and degrades to the fallback
      anchor pool on cold-start. Both monitors run in parallel;
      `BackgroundMapper` OR-fires the `divergence_flag` if
      either trips.
- [ ] **Per-window labels in addition to per-leg.** The
      measurement script emits a `per_window` table alongside
      the per-leg `root_cause` table, each window labelled
      `on_target` / `off_target` from the **semantic** sub-goal
      match above (room match for case-1, instance match on a
      final-target leg for case-2) — **not** from a geodesic-A\*
      deviation rule, which is dropped from CLIP's scope. Windows
      on `scan` / `explore_until_visible` legs are suppressed
      (no row emitted). The §4.4 addendum reports per-window
      ROC-AUC separately from per-leg ROC-AUC and names per-window
      as the *primary* signal for threshold selection.
- [ ] **Brier stratified by contrast-pool composition.** The
      image-vs-text Brier score is reported per contrast-pool
      kind (`grounded-siblings` / `map-prior` / `fallback-anchor`)
      — the pool actually sourced at each leg's `activate()` — not
      as one aggregate, because a dynamically-sourced pool mixes
      non-comparable softmax probabilities.
- [ ] **Letterbox-vs-center-crop reconciliation.** The eval
      re-encode and the runtime `clip_encoder` preprocessing share
      one path. `clip_encoder.py` does resize + center-crop; the
      harness `cameras` `preprocessing_hint` says letterbox. The
      brief pins both to a single path and records which (the
      `cameras` hint is the orchestrator's to set; this brief flags
      and resolves the mismatch before the metric pass, since a
      preprocessing skew silently invalidates the calibration).
- [ ] **Scene holdout protocol.** `report.json` declares a
      `held_out_seeds` field listing which Infinigen seeds the
      eval drew from. Default split is 2 / 3 seeds for the
      training pool (consumed by future cotrained briefs)
      and 1 / 3 for eval. Per-seed AUC disaggregation is
      reported. The held-out seeds list is committed into the
      §4.4 addendum so downstream briefs can reference it.
- [ ] **Cold-start vs. warm-map disaggregation.** Each leg in
      `report.json` carries a `map_state ∈ {cold, warming, warm}`
      field based on ChromaDB node count at leg-start (0,
      1 – 4, ≥ 5). The split gates the **warm-booster**
      image-vs-image signal: its case-1 AUC is reported separately
      for `warm` and (`warming` + `cold`) legs, and only the
      warm-leg case-1 AUC is held to a bar (cold-start is a
      diagnostic, not gated). The **primary** map-free image-vs-text
      signal reports a single case-2 AUC across all map states and
      carries the primary §4.1 bar — the cold-start-safe primary,
      warm booster framing from the inversion.
- [ ] **Label-authority audit.** ≥ 50 missions drawn from the
      held-out eval set, stratified across `on_course` /
      `wrong_room` / `wrong_instance` / `ambiguous`, are
      inspected by a human and the agreement against the
      automated `root_cause` labeling pipeline is reported as
      `label_authority_cohens_kappa` in the addendum. The
      cascade-end-to-end AUC is treated as label-noise-bounded
      below κ = 0.7; below this floor the brief files a
      follow-up to refine the labeling pipeline before
      committing to a ship decision.
- [ ] **Arbiter bias mitigation.** Every LLM-as-judge call in
      `--root-cause-pass` and in the cascade-end-to-end arbiter
      pass uses the position-swap protocol (call twice with
      swapped argument order; accept only on agreement). The
      arbiter judge is *not* `strafer_vlm`'s Qwen2.5-VL-3B
      (which is also the grounder; using it as judge invites
      self-enhancement bias); it is instead the planner-LLM at
      port 8200 or, if available on the DGX, a third-family
      model documented in the addendum. The judge prompt uses a
      fixed 1 – 4 scale with single-token output. Judge–human
      agreement on the label-authority subsample is reported as
      Cohen's κ; below 0.7 the brief defers the ship decision
      to a follow-up that swaps in a stronger judge.
- [ ] **Smoke test.** A single-mission run on the Infinigen
      `scene_fast_singleroom_000_seed0` scene produces a
      non-zero-row `verify_arrival` outcome (verified or
      unverified, but not `semantic_map_disabled`) AND at least
      one `BackgroundMapper` capture during the nav leg. Captured
      in the PR description as a one-line log excerpt.
- [ ] **Measurement script** at
      `Scripts/eval_transit_monitor.py` (DGX-side) that:
  - Takes `--episodes-root data/sim_in_the_loop/<scene>` and
    `--clip-model ~/.strafer/models/`.
  - Walks the harness output and replays CLIP encoding offline
    against each frame. Consumes LeRobot v3 datasets per
    [`harness-architecture`](../harness/harness-architecture.md)
    via HF `LeRobotDataset`. Populates
    an in-memory `SemanticMapManager`, runs both
    `TransitMonitor` (image-vs-image) and `TextAlignmentMonitor`
    (image-vs-text) in the same loop the production executor
    would.
  - Computes the five metrics above and writes
    `data/transit_monitor_eval/<run_id>/report.json` with
    per-scene + aggregate breakdowns.
  - Has a `--root-cause-pass` mode that walks
    `scene_metadata.json` room polygons + label inventory and
    calls `strafer_vlm` to label each mission as `on_course` /
    `wrong_room` / `wrong_instance` / `trajectory_violation`
    (always-empty) / `ambiguous`. Per-case TPR / FPR are computed
    from the resulting label, not from the binary `reachability`
    flag alone.
- [ ] **Run the script** against ≥ 3 Infinigen scenes and ≥ 30
      missions. Commit the run report under
      `data/transit_monitor_eval/<run_id>/report.json` (the
      directory is gitignored under `data/`, so the PR commits
      a copy under `docs/artifacts/transit_monitor_eval/` for the
      record).
- [ ] **Decision addendum.** Append a §4.4 to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the full statistics report:
  - ROC curves + 95% CIs per case + signal.
  - PR curves + 95% CIs per case + signal.
  - McNemar's test result comparing image-vs-image vs.
    image-vs-text on the same trial set.
  - Calibration diagram + Brier score per signal.
  - Time-to-decision CDF.
  - Cascade end-to-end (post-arbiter) numbers vs. tripwire-only.
  Links the run report under
  `docs/artifacts/transit_monitor_eval/`. Names whether the
  cascade meets the ≥ 0.85 ROC-AUC lower-bound bar; if so, the
  cascade ships. If not, files
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
  as a follow-up for improvements; no other escalation.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports
      (call out `make sim-bridge` and the
      `strafer-executor` smoke test in the PR description).

## Investigation pointers

- The unwired wiring point:
  [`source/strafer_autonomy/strafer_autonomy/executor/main.py:146-151`](../../../../source/strafer_autonomy/strafer_autonomy/executor/main.py#L146-L151).
- The default storage path (already used by `SemanticMapManager`):
  `~/.strafer/semantic_map/` — the executor's working dir is
  whatever `strafer-executor` is launched from on the Jetson.
  Decide and document.
- Existing offline replay precedent: the perception-writer +
  harness format is consumed offline by
  [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py)
  and
  [`prepare_vlm_finetune_data.py`](../../../../source/strafer_lab/scripts/prepare_vlm_finetune_data.py).
  Mirror that pattern.
- For the `--root-cause-pass`: re-use
  [`strafer_autonomy.clients.vlm_client.HttpGroundingClient`](../../../../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py)
  and the deployed Qwen2.5-VL-3B service.
- The eval script wants the *same* exported encoders as the
  production executor — the **bake-off's chosen trunk** under
  `~/.strafer/models/` (per
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  and the spine's Backbone-dim hygiene section) — and fails loudly
  if missing. Do **not** hard-code a `laion2b` ViT-B/32 path; the
  trunk family and embedding dim are the bake-off's output, not an
  assumption baked into this brief.

## Out of scope

- **Training a new validator class.** The
  small-frozen-head learned-validator option was retired (see
  [`completed/learned-mid-mission-validator`](../../completed/learned-mid-mission-validator.md)).
  CLIP cascade improvements live in
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md);
  end-to-end VLA exploration lives in
  [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md).
- **Multi-room navigation evaluation AND the full deviation
  contract.** This brief's *measurement* runs on single-room
  subsets of the (now multi-room-default) harness corpus. The
  case-1 / case-2 bars in §4.1 are calibrated against single-room
  data initially; this brief wires only the v1 single-room subset
  of the leg-type suppression flag. The **authoritative,
  full DEVIATION CONTRACT** — every plan primitive
  (`scan_for_target`, `explore_until_visible`, staging-hop
  `navigate_to_pose`, final-target `navigate_to_pose`,
  `verify_arrival`), both navigation backends, and the
  soft-LLM-ranking carve-out for exploration — is **owned by**
  [`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md)
  (parked, filed-on-trigger when
  [`autonomy-stack`](../multi-room/autonomy-stack.md) ships). That
  brief re-runs the per-case ROC-AUC metrics on multi-room data
  and recalibrates the §4.1 bars. This brief **references** the
  contract; it does **not** duplicate the full leg-type table here.
  Keeping the v1 measurement single-room is deliberate: it gives
  an achievable bar for the cheap CLIP path before multi-room
  raises the difficulty.

  **Coordination flag (exploration ≠ deviation).** This is why CLIP
  is scoped semantic-only and the geodesic-A\* rule is dropped: the
  multi-room runtime deliberately leaves the geometric line toward
  the final target.
  `explore_until_visible` from
  [`autonomy-stack`](../multi-room/autonomy-stack.md) walks
  ranked frontiers, and the frontier-ranking variants
  ([`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md),
  [`frontier-cognitive-fsm`](../../parked/multi-room/frontier-cognitive-fsm.md))
  visit plausible-but-wrong frontiers and re-verify candidate
  rooms by design; a staging-hop `navigate_to_pose` from
  [`planner-far-target-staging`](../multi-room/planner-far-target-staging.md)
  routes "via the dining room." Scoring any of those legs with a
  geodesic-to-final-target rule would label normal autonomy as
  off-course. The deviation definition is therefore
  **per-leg / sub-goal-aware** — validate against the *current*
  leg's sub-goal and DISARM during exploration — and the canonical
  form of that definition is the remeasure brief's contract above.
- **Real-robot validation.** Sim-side only. A future brief may
  layer real-robot data in once the runtime path is calibrated.
- **Choosing the backbone, and any CLIP fine-tune cycle.**
  Backbone selection is
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  — and per the spine's roadmap reorder it runs **before** this
  brief, not after it (the prior "v1 on existing ONNX → bake-off
  later" order is reversed; it is affordable because the project is
  greenfield on weights). This brief ships the cascade on the trunk
  the bake-off **already chose** (SigLIP-2-B lead per the spine),
  not on whatever ONNX happens to sit in `~/.strafer/models/`, and
  it does **not** hard-assume OpenCLIP ViT-B/32. Fine-tune /
  co-training cycles belong to
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md);
  the freeze-the-base, LoRA-only adaptation invariant is the spine's
  Training-discipline section.
- **Wrapping the abort with a VLM arbiter (§3.5).** Implemented
  inside this brief if the cascade-end-to-end metrics need
  arbiter post-processing to clear the AUC bar; otherwise filed
  as a small follow-up. Either way the implementation is small
  (one `/describe` call + LLM-as-judge call per tripwire fire).
- **Validating trajectory-shape constraints (case 3).** Requires
  the planner to decompose constraints like "hug the wall" into a
  checkable spec. Filed as a future brief
  (`planner-trajectory-constraint-decomposition.md`) only when a
  real mission requires it; never measured by this brief.
- **Distilling an MVP-as-teacher VLA (§3.6).** That's a separate
  brief filed once
  [`next-integration-round`](../investigations/next-integration-round.md) ships and
  the action-labeled corpus exists. Independent of this brief's
  decision branches.
