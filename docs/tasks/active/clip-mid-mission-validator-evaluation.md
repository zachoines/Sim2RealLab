# Wire and evaluate the CLIP transit-monitor + verify_arrival path

**Type:** task / investigation
**Owner:** Either (the wiring half is in `strafer_autonomy` /
`strafer_ros`; the measurement half is DGX-led against harness
output)
**Priority:** P1 (gates whether mid-mission validation graduates,
becomes a learned-validator follow-up, or gets retired)
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

Two artifacts in one PR:

1. **Wiring** in `executor/main.py` so the production executor
   constructs `SemanticMapManager` and `BackgroundMapper` and
   passes them through. Gated on env vars so a misconfigured
   Jetson degrades gracefully instead of hard-failing.
2. **A measurement script** that walks
   `data/sim_in_the_loop/<scene_name>/episode_NNNN/frames.jsonl`
   trees, replays the CLIP path against them in offline mode (no
   live executor needed for the metric pass), and emits
   `data/transit_monitor_eval/<run_id>/report.json` with the
   pre-registered metrics from
   [`MISSION_VALIDATION_ARCHITECTURE.md` §4.1](../../MISSION_VALIDATION_ARCHITECTURE.md#41-falsifiable-success-criteria-for-the-next-brief).

### Pre-registered metrics

Computed against a harness episode set covering ≥ 3 distinct
Infinigen scenes and ≥ 30 missions total:

| Metric | Definition |
|---|---|
| Frame-to-frame CLIP cosine σ | Std-dev of cosine similarity between consecutive same-leg embeddings, after L2-norm. Per-leg, then aggregated. |
| Top-1 flip rate per meter | Fraction of consecutive embedding pairs whose top-1 ANN match in the map flips, normalized by the robot's traversed metric distance between captures. |
| `TransitMonitor.check` true-positive rate | On legs labelled `reachable=False, root_cause=wrong_locale`, fraction where the monitor's `abort=True` fires before arrival. |
| False-positive rate | On legs labelled `reachable=True`, fraction where the monitor's `abort=True` fires. |
| Time-to-decision (median, p95) | From leg start to first `abort=True`. |

`root_cause=wrong_locale` is a new label; it requires
post-processing the harness output, not new sim runs. The
measurement script appends a `root_cause` field to each
`frames.jsonl` mission record by:

1. Comparing the final-pose `(x, y)` against the mission's
   `target_position_3d`.
2. Calling `strafer_vlm`'s `/describe` once per failed mission
   on the final-pose frame and matching keyword presence against
   `target_label`.
3. Disagreement between (1) and (2) marks the mission `ambiguous`
   and excludes it from TPR/FPR.

### Decision criteria

Per [`MISSION_VALIDATION_ARCHITECTURE.md` §4](../../MISSION_VALIDATION_ARCHITECTURE.md#section-4--recommendation):

- **Pass (escalate to §3.5 — wrap with VLM arbiter):** TPR ≥ 0.7 at
  FPR ≤ 0.1 on the full set; per-scene worst-case TPR ≥ 0.5.
  Follow-up brief: a small "wrap `TransitMonitor.abort` with one
  `/describe` call before actually canceling Nav2."
- **Fail (escalate to §3.2 — train a learned validator):** TPR <
  0.5 at any FPR ≤ 0.2. Follow-up brief:
  [`learned-mid-mission-validator.md`](learned-mid-mission-validator.md)
  takes over.
- **In-between:** apply the §3.1 cheap fixes (letterbox preprocess,
  same-region contrastive head, MobileCLIP / SigLIP backbone swap,
  TRT-EP FP16) and re-measure once. If still in-between, fail to
  §3.2.

The PR's write-up appends a one-section addendum to
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
recording which branch fired and linking the run report.

## Acceptance criteria

- [ ] **Wiring.** `executor/main.py` constructs and passes a
      `SemanticMapManager` + `BackgroundMapper` to
      `build_command_server`, gated on `STRAFER_SEMANTIC_MAP_ENABLED`
      env var (default `1`; `0` preserves the current degraded
      behavior). On startup, log whether CLIP ONNX models loaded
      successfully — the existing graceful-degrade in
      `clip_encoder.py:_load_models` should be visible in operator
      logs, not silent.
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
  - Walks `frames.jsonl`, replays CLIP encoding offline against
    each frame, populates an in-memory `SemanticMapManager`,
    invokes `TransitMonitor` in the same loop the production
    executor would.
  - Computes the five metrics above and writes
    `data/transit_monitor_eval/<run_id>/report.json` with
    per-scene + aggregate breakdowns.
  - Has a `--root-cause-pass` mode that calls `strafer_vlm` to
    label `reachable=False` missions as `wrong_locale` /
    `nav_timeout` / `ambiguous`.
- [ ] **Run the script** against ≥ 3 Infinigen scenes and ≥ 30
      missions. Commit the run report under
      `data/transit_monitor_eval/<run_id>/report.json` (the
      directory is gitignored under `data/`, so the PR commits
      a copy under `docs/artifacts/transit_monitor_eval/` for the
      record).
- [ ] **Decision addendum.** Append a §4.3 to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
      that names which decision branch fired (`pass`/`in-between`/`fail`),
      links the run report under `docs/artifacts/transit_monitor_eval/`,
      and either retires this option (fail → file the
      learned-validator brief) or files the §3.5 arbiter brief
      (pass).
- [ ] If the in-between branch fires, run the §3.1 cheap-fix pass:
      letterbox preprocess in `clip_encoder._preprocess_image`,
      add a same-region contrastive head training script under
      `source/strafer_lab/scripts/`, retrain, re-export, re-run
      the eval script, and append the second-pass numbers to the
      §4.3 addendum.
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

- **Training a new validator.** That's
  [`learned-mid-mission-validator.md`](learned-mid-mission-validator.md).
  This brief only re-exports the existing CLIP path with the §3.1
  cheap fixes if the in-between branch fires.
- **Multi-room navigation evaluation.** Per
  [`STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../STRAFER_AUTONOMY_NEXT.md),
  multi-room is deferred. Single-room scenes only.
- **Real-robot validation.** Sim-side only. A future brief may
  layer real-robot data in once the runtime path is calibrated.
- **Replacing CLIP with a non-CLIP backbone (DINOv2, MobileCLIP,
  SigLIP).** Reserve for the §3.1 cheap-fix pass *only* if
  in-between fires; if pass or fail fires, the backbone choice
  belongs to a separate brief.
- **Wrapping the abort with a VLM arbiter (§3.5).** That's the
  follow-up brief filed if `pass` fires.
