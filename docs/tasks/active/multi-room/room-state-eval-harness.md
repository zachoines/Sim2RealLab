# Measurement harness for runtime room-state quality

**Type:** new feature / investigation
**Owner:** DGX agent (sim-side eval; harness lives in
`source/strafer_lab/scripts/` per the existing eval-script
pattern; consumes
`SemanticMapManager`'s public API, which is owned by the DGX
edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (measurement infrastructure; should ship
before any quality-improvement brief from the v2 room-state
series so each improvement has a quantitative bar to clear)
**Estimate:** M (~3–5 days; scene set, metrics, runner,
baseline report against v1)
**Branch:** task/room-state-eval-harness

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
merges (this brief scores that brief's output). Not parked —
file as active and pick up directly when #41 lands.

## Story

As an **engineer measuring v2 room-state improvements
(feature+space region partitioning, open-vocab labeling, and
the learned-region-head / VPR escape valves) against the v1
baseline shipped in `observation-derived-room-state`**, I want
**a reproducible eval harness that scores
`SemanticMapManager`'s room-state output against Infinigen
ground-truth on a fixed multi-room scene set**, so that **each
v2 quality brief has a quantitative bar to clear, v2 changes
can be ranked head-to-head rather than vibe-checked, and the
operator can read off whether a given improvement actually
improved anything**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — where this brief sits in the v1 / v1.5 / v2 / v2.5 / v3 / escape-valve stack, and which planner-side consumers depend on its output.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — the v1 baseline. This harness scores its
  `known_rooms` / `current_room` / `connectivity` outputs.
- [`scene-connectivity-validation`](scene-connectivity-validation.md)
  — provides the sim-side ground-truth `connectivity[]` block
  the harness compares the runtime output against.
- [`validator-evaluation`](../clip-validation/validator-evaluation.md)
  — sibling measurement brief for CLIP mid-mission validation;
  shape and rigor of the metrics here should match.

## Context

### Why measurement-first

The v2 room-state work (feature+space region partitioning per
[`semantic-region-partition`](semantic-region-partition.md),
open-vocab labeling, loop closure) and the parked escape
valves (learned region head, learned VPR, CLIO-style dynamic
granularity) each claim a quality lift. Without a harness,
every "this should help" claim is unfalsifiable, and merge
order becomes guesswork. Filing this brief first sets the
contract: every subsequent v2+ brief reports its delta against
the metrics defined here, in its PR description.

### Eval set

A fixed set of Infinigen multi-room scenes with ground-truth
`scene_metadata.json` (room polygons, `room_adjacency`, and
the enriched `connectivity[]` block from
[`scene-connectivity-validation`](scene-connectivity-validation.md))
plus a teleop trajectory through each scene that visits every
room. Minimum:

- `scene_high_quality_dgx_000_seed0` (existing, the v1 smoke
  scene)
- 2 additional multi-room seeds generated via
  [`prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py)
  with `--seed 1 --seed 2`
- 1 adversarial scene: multi-bedroom (two rooms with the same
  `room_type`) to exercise the v1 same-label merge limitation
  documented in `observation-derived-room-state`'s
  Out-of-scope

Trajectories captured by the sim-bridge harness or replay
from `collect_demos.py`; stored under
`logs/eval/room_state/<scene>/trajectory.jsonl`.

### Metrics

Each metric is computed per-scene and aggregated:

| Metric | Definition | Why |
|---|---|---|
| Cluster purity (homogeneity / completeness / V-measure) | sklearn `homogeneity_completeness_v_measure` over per-node cluster assignment vs. Infinigen `room_idx` | Are clusters actually rooms? |
| Label precision / recall | per `RoomEntry.label` vs. Infinigen `room_type` | Are room labels correct? |
| Time-to-correct-classification | observations into the traversal at which `current_room(pose).label` first matches ground-truth and stays correct | How fast does the inference converge? |
| Connectivity precision / recall | `manager.connectivity()` edges vs. sim-side `connectivity[]` | Does the runtime graph match reality? |
| Cold-start `None` rate | fraction of poses in unmapped territory where `current_room` returns `None` (correct) vs. a wrong label (incorrect) | Pessimistic-correctness check |
| Calibration ECE | (only when v2's uncertainty work lands) Expected Calibration Error of `RoomEntry.confidence` | Are confidences honest? |

Mission-success delta — running the autonomy stack
end-to-end against the harness — is **out of scope here**;
that's `validator-evaluation`'s shape and belongs in a
separate mission-grading harness.

### Reproducibility

- Eval script at
  `source/strafer_lab/scripts/eval_room_state.py`. Loads a
  scene + trajectory, instantiates `SemanticMapManager`,
  replays the trajectory through `add_observation`, then
  computes the metrics.
- Output as JSON to `logs/eval/room_state/<scene>/<git_sha>.json`
  so PR descriptions can diff numbers across commits.
- A small `aggregate_eval.py` companion that reads
  `<git_sha>.json` files and produces a Markdown table.
- The harness should be `python` runnable; no Isaac Sim
  required for *replay* (Isaac Sim is needed only to
  *capture* the trajectories).

### Trajectory storage — raw RGB, not pre-encoded embeddings

Each trajectory frame is stored as **raw RGB** (a path to the
captured PNG/JPEG plus pose + timestamp), **not** as a
pre-computed CLIP embedding. Rationale: the
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
brief and the
[`semantic-region-partition`](semantic-region-partition.md)
v2 work both need to re-encode the same trajectories under a
different backbone (DINOv3, SigLIP-2, MobileCLIP-2) or a
re-clustered partition. Embedding-baked trajectories
would force a re-capture per backbone, defeating the
held-out-set contract. The eval script re-encodes through
whichever `CLIPEncoder` backbone is loaded at eval time
(reads `~/.strafer/models/`).

Storage layout:

```
logs/eval/room_state/<scene>/
  trajectory.jsonl       # one record per frame: pose, ts, rgb_path
  rgb/0000.png ...       # raw RGB captures
  scene_metadata.json    # symlink or copy of the sim-side GT
```

### Ground-truth `room_idx` per pose — point-in-polygon

The cluster-purity metric requires a ground-truth `room_idx`
per captured pose. Use the existing helper at
[`scene_labels.py:get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148)
to map `(x, y)` to the containing room's index against
`scene_metadata.json`'s `rooms[].footprint_xy` polygons.
Poses that fall outside every polygon (corridor pinches,
doorway crossings) are labelled `room_idx = None` and
excluded from per-node cluster-purity counts — they remain
in the time-to-converge and connectivity metrics so the
eval still scores doorway behavior. The helper is sim-side
only, lives in `strafer_lab`, and is the canonical source
for this lookup; do not re-derive in `strafer_autonomy`.

### Geodesic-A* utility — share with `validator-evaluation`

The [`validator-evaluation`](../clip-validation/validator-evaluation.md)
brief ships a geodesic-A* helper (per-window on-course /
off-course labeling against the global costmap). The
room-state eval's time-to-correct-classification metric
benefits from the same geodesic distance — "the robot is
approaching the kitchen along a path that passes through the
living room" needs the geodesic path, not the Euclidean
midpoint. Whichever brief ships first lifts the helper to
`source/strafer_lab/strafer_lab/tools/geodesic.py`; the
other consumes it. **Do not** copy the A* implementation
across briefs.

### Tie-in to state-of-the-art

The metric set is standard topological-SLAM /
scene-understanding evaluation:

- **Cluster homogeneity / V-measure**: sklearn standard.
- **Time-to-converge**: HoVer-SG (arXiv:2310.08864) reports
  the same.
- **Connectivity precision/recall**: Kimera-Multi
  (arXiv:2106.14367) reports this against ground-truth scene
  graphs.
- **ECE**: Guo et al. 2017 — temperature scaling
  calibration measurement.

## Acceptance criteria

- [ ] **Eval script.**
      `source/strafer_lab/scripts/eval_room_state.py`
      loads a scene's `scene_metadata.json` + a trajectory
      JSONL, replays the trajectory through a fresh
      `SemanticMapManager`, then computes and emits the
      metrics defined above to a JSON file.
- [ ] **Scene set.** At least 3 multi-room scenes (existing
      `scene_high_quality_dgx_000_seed0` + 2 new seeds) plus
      1 adversarial multi-bedroom scene staged under
      `logs/eval/room_state/`.
- [ ] **Trajectory capture.** Document the trajectory-capture
      procedure in the script's module docstring (teleop via
      `collect_demos.py` or harness replay via
      `run_sim_in_the_loop.py --mode harness`). Trajectories
      stored alongside scene metadata.
- [ ] **Raw-RGB storage.** Each trajectory frame stores the
      raw RGB capture (PNG/JPEG path + pose + ts), **not** a
      pre-encoded CLIP embedding. The eval script re-encodes
      at run time through whichever `CLIPEncoder` backbone is
      loaded — required for
      [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
      and
      [`semantic-region-partition`](semantic-region-partition.md)
      to re-run against the same trajectories without
      re-capture.
- [ ] **Ground-truth `room_idx` lookup.** The cluster-purity
      metric pulls ground-truth per-pose `room_idx` from
      [`scene_labels.py:get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148).
      Poses outside every polygon are labelled `room_idx =
      None` and excluded from per-node purity counts.
- [ ] **Metrics implemented.** Cluster purity (V-measure),
      label precision/recall, time-to-correct-classification,
      connectivity precision/recall, cold-start None rate.
      ECE plumbed but no-op until v2 uncertainty work lands.
- [ ] **Baseline numbers.** PR description carries the v1
      baseline numbers on all four scenes — these are the
      anchor every subsequent v2 brief reports its delta
      against.
- [ ] **Aggregate report.** A small `aggregate_eval.py` reads
      `<git_sha>.json` files and produces a Markdown table
      for inclusion in v2 PR descriptions.
- [ ] **README documentation.** Operator-facing section in
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)'s
      "Semantic-map room state" subsection pointing at the
      eval script + the trajectory-capture procedure.
- [ ] **No regression.** Existing `test_semantic_map.py` and
      `test_room_state.py` suites continue to pass.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit. See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
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
- Trajectory capture entry points:
  [`Scripts/`](../../../../Scripts/) and
  [`source/strafer_lab/scripts/collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py).

## Out of scope

- **Real-robot eval.** Sim only — no D555 / Jetson runtime
  measurement. Real-robot quality is tracked separately via
  the `next-integration-round` brief's mission success rate.
- **End-to-end mission grading.** This harness measures the
  room-state *output*; whether the autonomy stack consuming
  that output produces a successful mission is
  `validator-evaluation`'s shape.
- **Backbone-specific eval.** The harness should run against
  whatever backbone the manager has loaded; comparing
  backbones head-to-head belongs in the
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  brief.
- **Long-horizon eval.** Single-session trajectories only;
  multi-day map-lifecycle measurement is the
  [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
  brief's responsibility.
