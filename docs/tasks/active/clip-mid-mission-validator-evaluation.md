# Wire and evaluate the CLIP transit-monitor + verify_arrival path

**Type:** task / investigation
**Owner:** Either (the wiring half is in `strafer_autonomy` /
`strafer_ros`; the measurement half is DGX-led against harness
output)
**Priority:** P1 (gates whether mid-mission validation graduates,
becomes a learned-validator follow-up, or gets retired). Blocked
on harness output existing from **any driver mode**
(teleop via [`harness-teleop-driver`](harness-teleop-driver.md)
is the fastest-to-produce; bridge via `next-integration-round`
also works) and a CLIP ONNX existing under `~/.strafer/models/`
(one-time prerequisite — see Prerequisites below).
**Estimate:** L (~multi-day; small-but-broad implementation +
calibration measurement + write-up)
**Branch:** task/clip-mid-mission-validator-evaluation

## Story

As an **operator deciding whether to keep, replace, or retire CLIP
mid-mission validation**, I want **the existing
`SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` +
`verify_arrival` path actually wired into the production executor
and measured against harness output**, so that **the
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
recommendation has the empirical numbers it pre-registered, and
the follow-up decision (escalate to a learned validator vs. wrap
the cheap path with a VLM arbiter) is anchored in measurement
rather than guessed at**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md).
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
| Populated `data/{teleop,sim_in_the_loop}/<scene>/episode_NNNN/` for ≥ 3 scenes × ≥ 30 missions | Without harness episodes, no metrics can be computed | **Fastest path: teleop.** [`harness-teleop-driver`](harness-teleop-driver.md) gets to ≥ 3 scenes × 30 success + 30 hard-negative episodes in ~one operator-evening. **Slower path: bridge.** [`next-integration-round`](next-integration-round.md) produces bridge-driver output as part of its acceptance, blocked on MPPI / Nav2 stability. Either driver's output works — the eval script is schema-flexible. |
| `~/.strafer/models/clip_visual.onnx` + `clip_text.onnx` | The eval script and the new image-vs-text tripwire both depend on the runtime ONNX | One-time `finetune_clip.py --no-export-onnx false` run, OR (faster) export the laion2b ViT-B/32 weights without fine-tuning so the eval can run against the unfinetuned baseline. The brief work owns this small step at the start; it is **not** a separate brief. |

The brief's wiring + protocol-refactor work doesn't need either
prerequisite — those steps land independently, and the
measurement step gates on the prerequisites being satisfied.

### What's already in the repo

The semantic-map package is complete-but-orphaned:

- [`semantic_map/clip_encoder.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py),
  [`manager.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py),
  [`background_mapper.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/background_mapper.py),
  [`transit_monitor.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py)
  exist and have unit tests under
  [`source/strafer_autonomy/tests/`](../../../source/strafer_autonomy/tests/).
- [`mission_runner.py`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  accepts `semantic_map=` and `background_mapper=` and uses them in
  `_verify_arrival`, `_activate_transit_monitor`,
  `_try_query_before_scan`, `_store_scan_observation`,
  `_log_arrival_failure`.
- [`build_command_server`](../../../source/strafer_autonomy/strafer_autonomy/executor/command_server.py#L246)
  accepts both kwargs and forwards them.

But [`executor/main.py:146-151`](../../../source/strafer_autonomy/strafer_autonomy/executor/main.py#L146-L151)
calls `build_command_server` **without** `semantic_map=` /
`background_mapper=`. In production every CLIP-touching path
short-circuits on `None`. The
[`strafer_autonomy/README.md` skill table](../../../source/strafer_autonomy/README.md#L137)
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
   to encode the mission target phrase at leg start, cosines
   against each live frame from `BackgroundMapper`'s capture
   loop, and fires `abort=True` when the cosine drops below a
   baseline-relative threshold for N consecutive captures. **No
   model fine-tune required** — the runtime artifact reuses the
   already-loaded CLIP text encoder. The threshold's baseline is
   the cosine score on the scan-found view (the moment the
   target was last grounded), with a relative margin (default
   `baseline − 0.15`).
3. **A `Validator` protocol refactor.** `BackgroundMapper`
   currently holds a concrete `TransitMonitor`. Replace with a
   `Validator` protocol (lives in `semantic_map/protocols.py`)
   that both `TransitMonitor` and the new
   `TextAlignmentMonitor` implement. This is small (~30 lines)
   but lets the future
   [`clip-cotrained-retrieval-augmented`](clip-cotrained-retrieval-augmented.md)
   brief plug in enhanced cascade variants without touching
   `BackgroundMapper` again.
4. **A measurement script** that consumes the harness's per-mission
   output (file layout per
   [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
   when that brief has shipped; falls back to today's
   `frames.jsonl` if not), replays the CLIP path (both signals)
   against it in offline mode (no live executor needed for the
   metric pass), and emits
   `data/transit_monitor_eval/<run_id>/report.json` with the
   pre-registered metrics from
   [`MISSION_VALIDATION_ARCHITECTURE.md` §4.1](../../MISSION_VALIDATION_ARCHITECTURE.md#41-falsifiable-success-criteria-for-the-next-brief).

### Pre-registered metrics — industry-standard binary-classifier statistics

Computed against a harness episode set covering ≥ 3 distinct
Infinigen scenes and ≥ 30 missions total. Metrics are
**disaggregated by failure case** per
[`MISSION_VALIDATION_ARCHITECTURE.md` §2.3](../../MISSION_VALIDATION_ARCHITECTURE.md):
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
   phrase. Disagreement with step 2 ⇒ `ambiguous`; excluded from
   TPR / FPR.
4. **`trajectory_violation`:** **always empty** in this brief —
   the planner does not yet emit trajectory constraints, so no
   mission can be labelled here. Reserved field.

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
  [`clip-cotrained-retrieval-augmented`](clip-cotrained-retrieval-augmented.md)
  (research-flavored follow-up; co-trained CLIP + speaker +
  retrieval-augmented inference). The end-to-end VLA research
  arm at
  [`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md)
  also remains as an alternative path. **No "small learned
  validator" escalation** — that path was retired (see
  [`completed/learned-mid-mission-validator`](../completed/learned-mid-mission-validator.md)
  for the rationale).

The PR's write-up appends a §4.4 to
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
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
- [ ] **`Validator` protocol.**
      `source/strafer_autonomy/strafer_autonomy/semantic_map/protocols.py`
      defines a `Validator` protocol with `activate`, `deactivate`,
      `check`, `is_active` matching the existing
      `TransitMonitor` shape. `TransitMonitor` declares
      conformance. `BackgroundMapper` accepts a `Validator`
      instead of a concrete `TransitMonitor`. Unit test under
      `tests/` confirms the substitution works.
- [ ] **Image-vs-target-text tripwire.**
      `semantic_map/text_alignment_monitor.py` implements the
      `Validator` protocol; encodes the mission target phrase
      via `CLIPEncoder.encode_text` at `activate()`; computes
      cosine on each `check()`; fires after `N=3` consecutive
      below-threshold captures. The threshold is calibrated
      from the first 3 captures of the leg (baseline) with a
      relative margin from `STRAFER_TEXT_ALIGN_MARGIN` (default
      `0.15`). Both monitors run in parallel; `BackgroundMapper`
      OR-fires the `divergence_flag` if either trips.
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
    against each frame. Schema-flexible: prefers
    `frames_skill.jsonl` + `frames_tick.jsonl` from
    [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
    when present; falls back to legacy `frames.jsonl`. Populates
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
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
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
  [`clip-cotrained-retrieval-augmented`](clip-cotrained-retrieval-augmented.md)
  as a follow-up for improvements; no other escalation.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports
      (call out `make sim-bridge` and the
      `strafer-executor` smoke test in the PR description).

## Investigation pointers

- The unwired wiring point:
  [`source/strafer_autonomy/strafer_autonomy/executor/main.py:146-151`](../../../source/strafer_autonomy/strafer_autonomy/executor/main.py#L146-L151).
- The default storage path (already used by `SemanticMapManager`):
  `~/.strafer/semantic_map/` — the executor's working dir is
  whatever `strafer-executor` is launched from on the Jetson.
  Decide and document.
- Existing offline replay precedent: the perception-writer +
  harness format is consumed offline by
  [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
  and
  [`prepare_vlm_finetune_data.py`](../../../source/strafer_lab/scripts/prepare_vlm_finetune_data.py).
  Mirror that pattern.
- For the `--root-cause-pass`: re-use
  [`strafer_autonomy.clients.vlm_client.HttpGroundingClient`](../../../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py)
  and the deployed Qwen2.5-VL-3B service.
- CLIP fine-tune target's `pretrained` default is
  `laion2b_s34b_b79k`; `--no-export-onnx` exists for offline
  runs. The eval script wants the *same* ONNX as the production
  executor — load from
  `~/.strafer/models/clip_visual.onnx` + `clip_text.onnx` and
  fail loudly if missing.

## Out of scope

- **Training a new validator class.** The
  small-frozen-head learned-validator option was retired (see
  [`completed/learned-mid-mission-validator`](../completed/learned-mid-mission-validator.md)).
  CLIP cascade improvements live in
  [`clip-cotrained-retrieval-augmented`](clip-cotrained-retrieval-augmented.md);
  end-to-end VLA exploration lives in
  [`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md).
- **Multi-room navigation evaluation.** This brief's
  *measurement* runs on single-room subsets of the (now
  multi-room-default) harness corpus. The case-1 / case-2 TPR /
  FPR bars in §4.1 are calibrated against single-room data
  initially. A multi-room re-test follow-up brief
  (`clip-multi-room-validator-remeasure.md`) is filed after
  [`multi-room-autonomy-stack`](multi-room-autonomy-stack.md)
  ships — it re-runs the same metrics on multi-room data and
  recalibrates the bars. Keeping the v1 measurement single-room
  is deliberate: it gives an achievable bar for the cheap CLIP
  path before multi-room raises the difficulty.
- **Real-robot validation.** Sim-side only. A future brief may
  layer real-robot data in once the runtime path is calibrated.
- **Replacing CLIP with a non-CLIP backbone (DINOv2, MobileCLIP,
  SigLIP) and any CLIP fine-tune cycle.** Belongs to
  [`clip-cotrained-retrieval-augmented`](clip-cotrained-retrieval-augmented.md),
  filed as a follow-up if the cascade fails the AUC bar or the
  team wants to push improvements regardless. This brief
  evaluates whatever ONNX is currently in
  `~/.strafer/models/` — fine-tuning is a downstream concern.
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
  [`next-integration-round`](next-integration-round.md) ships and
  the action-labeled corpus exists. Independent of this brief's
  decision branches.
