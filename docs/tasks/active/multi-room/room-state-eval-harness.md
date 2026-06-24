# Eval harness for runtime room-state quality

**Type:** new feature / investigation
**Owner:** DGX agent (sim-side eval; eval script lives in
`source/strafer_lab/scripts/` per the existing eval-script
pattern; consumes `SemanticMapManager`'s public API per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (measurement infrastructure; should ship
before any quality-improvement brief from the v2 room-state
series so each improvement has a quantitative bar to clear)
**Estimate:** M (~3–5 days; eval script + scene set + metric
implementations + baseline report against v1)
**Branch:** task/room-state-eval-harness

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
has merged (shipped — provides the v1 `SemanticMapManager` API
this scores) AND
[`harness-architecture`](../harness/harness-architecture.md)'s
[Tier 3](../harness/harness-architecture.md#tier-3--scripted-driver--queuecaptionercoverage-mission-sources-pr-d)
has shipped the `scripted × coverage` capture path (provides the
LeRobot v3 datasets this eval consumes).

**Adversarial / open-plan scene set scope (added 2026-06-23):** this harness's
adversarial scene set (scenes that stress region + room-type reasoning) is an
**eval-only mini-capture**, deliberately NOT folded into the harness *bulk*
capture run (decided off the 2026-06-23 harness data-requirements analysis).
True open-plan is hard to generate with Infinigen (it partitions walled rooms);
scope the adversarial set to Infinigen-producible cases — duplicate room types +
dense clutter (e.g. seed2 already has 2 bedrooms / 2 bathrooms / 2 closets, which
stress room-type-uniqueness). Capture this small set when this harness is built;
the harness bulk run stays on standard scenes.

## Story

As an **engineer measuring v2 room-state improvements
(feature+space region partitioning, open-vocab labeling, and
the learned-spatial-encoder escape valve) against the v1
baseline shipped in
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)**,
I want **a reproducible eval that scores `SemanticMapManager`'s
room-state output against Infinigen ground-truth on a curated
multi-room scene set**, so that **each v2 quality brief has a
quantitative bar to clear, v2 changes can be ranked head-to-head
rather than vibe-checked, and the operator can read off whether
a given improvement actually improved anything**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — where this brief sits in the v1 / v1.5 / v2 / v2.5 / v3 / escape-valve stack.
- [`harness-architecture`](../harness/harness-architecture.md) — produces the LeRobot v3 datasets this eval reads. The [Scripted × coverage](../harness/harness-architecture.md#scripted--coverage-new--for-room-state-eval-and-vpr-training) mission source is the canonical capture path for the eval scene set; bridge / teleop captures of the same scenes are also valid input as long as every room is visited.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — the v1 baseline. This eval scores its `known_rooms` /
  `current_room` / `connectivity` outputs.
- [`scene-connectivity-validation`](scene-connectivity-validation.md)
  — provides the sim-side ground-truth `connectivity[]` block
  this eval compares the runtime output against.
- [`validator-evaluation`](../clip-validation/validator-evaluation.md)
  — sibling measurement brief for CLIP mid-mission validation;
  shape and rigor of the metrics here match.

## Context

### Why measurement-first

The v2 room-state work
([`semantic-region-partition`](semantic-region-partition.md),
open-vocab labeling, loop closure) and the parked escape valves
(learned region head, learned VPR, CLIO-style dynamic
granularity) each claim a quality lift. Without a measurement
harness, every "this should help" claim is unfalsifiable, and
merge order becomes guesswork. This brief sets the contract:
every subsequent v2+ brief reports its delta against the metrics
defined here, in its PR description.

### Scope after the 2026-05-24 harness consolidation

Pre-consolidation, this brief owned its own trajectory capture
(`logs/eval/room_state/<scene>/trajectory.jsonl` + repeated
traversals + raw RGB storage). Post-consolidation, the
[`harness-architecture`](../harness/harness-architecture.md)
brief's
[Scripted × coverage](../harness/harness-architecture.md#scripted--coverage-new--for-room-state-eval-and-vpr-training)
mission source is the canonical capture path. This brief shrinks
to:

- The eval *script* (`eval_room_state.py`)
- The scene set curation (multi-bedroom + open-plan adversarials
  + the existing scene + 2 new seeds)
- The metric implementations (V-measure / label P/R /
  time-to-converge / connectivity P/R / cold-start None rate /
  precision@5 + MRR for `query_room_by_text`)
- The aggregate report tool
- The baseline report (v1 numbers as the anchor every subsequent
  v2 brief diffs against)

Capture is **not** this brief's responsibility; it consumes
LeRobot v3 datasets produced by the harness epic.

### Eval input — LeRobot v3 datasets

The eval reads any LeRobot v3 dataset that satisfies:

1. Covers every room in the scene at least once (or the eval
   skips uncovered rooms and reports partial coverage).
2. Carries the strafer per-episode `scene_id` column (in
   `meta/episodes/`'s chunked Parquet per LeRobot v3) resolving
   to `meta/scenes/<scene_id>/scene_metadata.json` (per
   [`scene-connectivity-validation`](scene-connectivity-validation.md)'s
   `connectivity[]` block + Infinigen room polygons + objects).
3. Captures per-frame pose in `observation.state.pose` and per-
   frame RGB in `observation.images.perception`. Depth is
   optional (not used for room-state metrics).

The canonical input is a `scripted × coverage` dataset (one
trajectory per scene, every room visited ≥ 2× per the harness
brief's coverage spec; repeated approaches from different
headings included). `bridge × *` and `teleop × *` datasets are
also valid input if they happen to cover all rooms — the eval
doesn't care which driver produced the data, only that the
coverage is sufficient.

### Scene set

A curated set of Infinigen multi-room scenes with ground-truth
`scene_metadata.json` carrying room polygons, `room_adjacency`,
and `connectivity[]` (the latter from
[`scene-connectivity-validation`](scene-connectivity-validation.md)).
Minimum:

- `scene_high_quality_dgx_000_seed0` (existing — the v1 smoke
  scene).
- 2 additional multi-room seeds generated via
  [`prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py)
  with `--seed 1 --seed 2`.
- 1 **multi-bedroom** adversarial scene (two rooms sharing the
  same `room_type`) — exercises same-label disambiguation; the
  hard-negative case for clustering and place recognition.
- 1 **open-plan** adversarial scene (kitchen + dining + living
  with no demarcating walls) — the case where wall geometry
  can't define regions, which v2's
  [`semantic-region-partition`](semantic-region-partition.md)
  must split by visual content. Without this scene the v2
  open-plan claim is untestable.

The scene set is checked into `Assets/generated/scenes/` per
the existing scene-asset layout. LeRobot v3 datasets captured
against each scene live under
`data/sim_in_the_loop/<scene_name>/` per the harness brief's
layout.

### Metrics

Each metric is computed per-scene and aggregated. Source datasets
are LeRobot v3; the eval script reads them via the HF `lerobot`
loader and replays each frame through a fresh
`SemanticMapManager.add_observation`.

| Metric | Definition | Why |
|---|---|---|
| Cluster purity (V-measure) | sklearn `homogeneity_completeness_v_measure` over per-node cluster assignment vs. Infinigen `room_idx` | Are clusters actually rooms? |
| Label precision / recall | per `RoomEntry.label` vs. Infinigen `room_type` | Are room labels correct? |
| Time-to-correct-classification | frame index into the traversal at which `current_room(pose).label` first matches ground-truth and stays correct | How fast does the inference converge? |
| Connectivity precision / recall | `manager.connectivity()` edges vs. sim-side `connectivity[]` | Does the runtime graph match reality? |
| Cold-start `None` rate | fraction of poses in unmapped territory where `current_room` returns `None` (correct) vs. a wrong label (incorrect) | Pessimistic-correctness check |
| Open-vocab precision@5 / MRR (`query_room_by_text`) | For each Infinigen `room_type` in the scene, query the manager with a hand-curated free-form description ("the room with the cooking surface and refrigerator" for kitchen, etc.); report precision@1 + precision@5 + MRR against ground-truth. **Closes the deferred bar on [`query-room-by-text-v1`](../../completed/query-room-by-text-v1.md) PR #54.** Target: precision@1 ≥ 0.7; precision@5 ≥ 0.9 on the v1 four-scene eval set. Below those bars, file a follow-up to fine-tune via [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md). | Open-vocab query quality |
| Calibration ECE | (only when v2 uncertainty work lands) Expected Calibration Error of `RoomEntry.confidence` | Are confidences honest? |

Mission-success delta — running the autonomy stack end-to-end —
is **out of scope here**; that's
[`validator-evaluation`](../clip-validation/validator-evaluation.md)'s
shape.

### Ground-truth `room_idx` per pose — point-in-polygon

The cluster-purity metric requires a ground-truth `room_idx` per
captured pose. Use the existing helper at
[`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148)
to map `(x, y)` to the containing room's index against
`scene_metadata.json`'s `rooms[].footprint_xy` polygons. Poses
that fall outside every polygon (corridor pinches, doorway
crossings) are labelled `room_idx = None` and excluded from
per-node cluster-purity counts — they remain in the
time-to-converge and connectivity metrics so the eval still
scores doorway behavior. The helper is sim-side only, lives in
`strafer_lab`, and is the canonical source for this lookup; do
not re-derive in `strafer_autonomy`.

### Geodesic-A* utility — share with `validator-evaluation`

The
[`validator-evaluation`](../clip-validation/validator-evaluation.md)
brief ships a geodesic-A* helper (per-window on-course /
off-course labeling against the global costmap). The
room-state eval's time-to-converge metric benefits from the
same geodesic distance — "the robot is approaching the
kitchen along a path that passes through the living room"
needs the geodesic path, not the Euclidean midpoint.
Whichever brief ships first lifts the helper to
`source/strafer_lab/strafer_lab/tools/geodesic.py`; the other
consumes it. **Do not** copy the A* implementation across
briefs.

### Tie-in to state-of-the-art

The metric set is standard topological-SLAM /
scene-understanding evaluation:

- **Cluster homogeneity / V-measure**: sklearn standard.
- **Time-to-converge**: HoVer-SG (arXiv:2310.08864) reports
  the same.
- **Connectivity precision/recall**: Kimera-Multi
  (arXiv:2106.14367) reports this against ground-truth scene
  graphs.
- **ECE**: Guo et al. 2017 — temperature scaling calibration
  measurement.

## Acceptance criteria

- [ ] **Eval script.**
      `source/strafer_lab/scripts/eval_room_state.py` loads a
      LeRobot v3 dataset directory + scene metadata, replays the
      trajectory through a fresh `SemanticMapManager`, then
      computes and emits the metrics defined above to a JSON
      file at `logs/eval/room_state/<scene>/<git_sha>.json`.
- [ ] **Scene set.** At least 3 multi-room scenes (existing
      `scene_high_quality_dgx_000_seed0` + 2 new seeds) plus
      **two** adversarial scenes — multi-bedroom (same-label
      disambiguation) AND open-plan (kitchen + dining + living,
      no walls). The open-plan scene is what makes v2's
      open-plan-split claim testable.
- [ ] **Per-scene LeRobot v3 datasets.** A `scripted × coverage`
      run is captured per scene per the harness brief's coverage
      spec (every room visited ≥ 2× including repeated approaches
      from different headings). Capture procedure is documented
      in the harness brief, not duplicated here.
- [ ] **Ground-truth `room_idx` lookup.** The cluster-purity
      metric pulls ground-truth per-pose `room_idx` from
      [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148).
      Poses outside every polygon are labelled `room_idx = None`
      and excluded from per-node purity counts.
- [ ] **Metrics implemented.** Cluster purity (V-measure), label
      precision/recall, time-to-correct-classification,
      connectivity precision/recall, cold-start None rate,
      open-vocab precision@5 + MRR. ECE plumbed but no-op until
      v2 uncertainty work lands.
- [ ] **Baseline numbers.** PR description carries the v1
      baseline numbers on all 5 scenes — these are the anchor
      every subsequent v2 brief reports its delta against.
- [ ] **Aggregate report.** A small `aggregate_eval.py` reads
      `<git_sha>.json` files and produces a Markdown table for
      inclusion in v2 PR descriptions.
- [ ] **Closes the `query-room-by-text-v1` deferred bar.** The
      precision@5 + MRR row on the v1 scene set is reported in
      the PR description; the
      [`query-room-by-text-v1`](../../completed/query-room-by-text-v1.md)
      brief's deferred acceptance criterion is marked closed.
- [ ] **README documentation.** Operator-facing section in
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      "Semantic-map room state" subsection points at the eval
      script + the harness capture procedure.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites continue to pass.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- v1 baseline implementation:
  [`source/strafer_autonomy/strafer_autonomy/semantic_map/`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/)
  — `manager.py`, `room_state.py`, `models.py`.
- Sim-side ground truth (once
  [`scene-connectivity-validation`](scene-connectivity-validation.md)
  ships): the enriched `connectivity[]` block in
  `scene_metadata.json`.
- Sibling harness shape:
  [`validator-evaluation`](../clip-validation/validator-evaluation.md)
  — same pattern (offline replay + scored metrics).
- LeRobot v3 loader: HF `lerobot` library's `LeRobotDataset`
  class; see the
  [`harness-architecture`](../harness/harness-architecture.md)
  brief's Output format section for the schema this consumes.
- Capture procedure: see the harness brief's
  [Scripted × coverage](../harness/harness-architecture.md#scripted--coverage-new--for-room-state-eval-and-vpr-training)
  section. Do not duplicate the procedure here.

## Out of scope

- **Trajectory capture mechanism.** Owned by
  [`harness-architecture`](../harness/harness-architecture.md);
  this brief consumes its output.
- **Real-robot eval.** Sim only — no D555 / Jetson runtime
  measurement. Real-robot quality is tracked separately via
  the `next-integration-round` brief's mission success rate.
- **End-to-end mission grading.** This harness measures the
  room-state *output*; whether the autonomy stack consuming
  that output produces a successful mission is
  [`validator-evaluation`](../clip-validation/validator-evaluation.md)'s
  shape.
- **Backbone-specific eval.** The harness runs against whatever
  backbone the manager has loaded; comparing backbones
  head-to-head belongs in
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md).
- **Long-horizon eval.** Single-session trajectories only;
  multi-day map-lifecycle measurement is the
  [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
  brief's responsibility.
- **Training a region head / VPR head against this corpus.**
  That's
  [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)'s
  scope; this brief produces the eval, not the training run.
