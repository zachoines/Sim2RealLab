# Runtime ergonomics on `SemanticMapManager` room-state APIs

**Type:** refactor / robustness
**Owner:** DGX agent (`source/strafer_autonomy/strafer_autonomy/semantic_map/`
is in the DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md)).
**Priority:** P3 (manager-internal ergonomics; v1 acceptable,
each finding has a clear filed-on-trigger condition)
**Estimate:** S (~½ day for both findings combined; per-label
index + retry-on-disable hook + one regression test each)
**Branch:** task/room-state-runtime-ergonomics

**Pickup gate:** Filed-on-trigger. Pick up when **any** of:

- v1 deployment regularly exceeds ~1K nodes (Finding A's
  linear-scan cost becomes user-noticeable; the lifecycle
  brief
  [`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md)
  sizes the long-term layer at ~10K nodes where this matters
  enough to warrant the work).
- A real-robot deployment surfaces a CLIP late-bind failure
  (Finding B) where the encoder finishes loading after the
  first `add_observation` call.

Un-park via `git mv` per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As an **operator running long-duration deployments where the
semantic map grows past v1 home sizes**, I want
**`SemanticMapManager`'s room-state runtime to handle the two
ergonomic gaps surfaced by the
[`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
ship audit** (linear `room_anchor` scans at scale, and a
sticky `classifier.enabled = False` state when CLIP loads
late), so that **scaling to longer missions and recovering
from late-binding model loads stays automatic instead of
becoming a manual workaround**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — v1 ship; the audit on its PR surfaced these two
  ergonomic gaps.
- [`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md)
  — the long-term-layer brief that sizes the regime where
  Finding A's linear scan becomes a real cost.

## Context

### Finding A — `room_anchor` linear-scan

`SemanticMapManager.room_anchor(label)` walks every node in
the graph on every call and returns the most-recent pose
tagged with `metadata["room_label"] == label`. The
[`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
compiler emits a `navigate_to_pose(room_anchor(target_room))`
transit step on every cross-room mission, so this scan fires
once per planning pass.

v1 home sizes (10s–100s of nodes) make the scan trivial. The
[`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md)
brief sizes the long-term layer at ~10K nodes — at which
point the linear scan is ~10 ms per call. Still survivable
but starts to add up across multi-target missions and the
planning loop's other lookups.

The fix is the same shape as the v1 cluster cache: a
per-label index keyed on `metadata["room_label"]`, updated
incrementally at `add_observation` time and invalidated /
rebuilt on `prune` / `clear` / explicit cache reset. Lookup
becomes O(1) for the label, plus an O(K) walk over that
label's entries to pick the most-recent.

### Finding B — `RoomClassifier` enabled-state is sticky

`RoomClassifier._ensure_text_embeddings` caches the result of
the CLIP encoder's `enabled` check on first call:

```python
if self._enabled is not None:
    return self._enabled
if not getattr(self._encoder, "enabled", False):
    self._enabled = False
    return False
```

If `_classify_room` is called once before the CLIP encoder
finishes loading (e.g., an ONNX model file dropped in after
startup, or a late-binding model load triggered by the
BackgroundMapper), the classifier latches `self._enabled =
False` and stays disabled forever — even after the encoder
becomes ready.

Real-world impact is low: CLIP typically loads at startup or
never, not late. But the failure mode is silent (no room
labels stamped on subsequent nodes) and discoverable only by
reading the manager state from a debugger. A
retry-on-disable hook makes the failure mode recoverable
without operator intervention.

The fix is a small state machine: keep the disable status
"tentative" for the first N calls, retry the encoder probe,
and only latch to `False` once a configurable number of
consecutive probes fail (or after a wall-clock timeout).
Alternatively: treat `enabled = False` as tentative always,
and re-probe on every call when disabled — the probe cost
is one `hasattr` + bool check, well below the cost of the
matmul that the enabled path runs.

## Acceptance criteria

### Finding A — `room_anchor` index

- [ ] **Per-label index added.** `SemanticMapManager` carries
      a `dict[str, list[tuple[float, str]]]` mapping room
      label → list of `(timestamp, node_id)` for nodes
      tagged with that label, sorted by timestamp descending
      (or maintained as a heap / sorted list).
- [ ] **Updated incrementally.** `add_observation` appends
      to the index when `merged_metadata["room_label"]` is
      stamped. `prune` (if it exists) removes entries.
      `clear` resets the index. The smoothing brief's
      in-place label mutation (per
      [`room-state-temporal-smoothing`](../../active/multi-room/room-state-temporal-smoothing.md))
      also updates the index — coordinate with that brief if
      it ships first.
- [ ] **`room_anchor` rewritten.** The public method becomes
      a constant-time index lookup followed by a sorted-list
      pick of the most-recent entry. Behaviour is
      byte-identical to the v1 linear scan on the same map
      state (regression-tested via a recorded scan + index
      query producing the same pose).
- [ ] **Cache invalidation hook.** A public
      `invalidate_room_index()` method lets external
      label-mutators (the smoothing brief, future
      consumers) rebuild the index after bulk metadata
      changes.

### Finding B — `RoomClassifier` recoverable disable

- [ ] **Tentative-disable semantics.** The classifier no
      longer latches `enabled = False` permanently. On a
      disabled probe, the classifier either (a) re-probes
      on every subsequent call (simplest, recommended), or
      (b) re-probes after N consecutive disabled calls / a
      configurable timeout.
- [ ] **Test fixture.** A test simulates a CLIP encoder
      that returns `enabled = False` on the first probe and
      `enabled = True` on subsequent probes; the classifier
      transitions from "no label" to "label correctly
      assigned" without manager-instance recycling.
- [ ] **No regression** in the steady-state path: the
      enabled classifier's behaviour is unchanged (still
      caches the text embeddings on first successful
      encode, still single matmul per `classify` call).

### Shared

- [ ] **Runtime-legal inputs only.** No
      `scene_metadata.json` access — same sim-to-real rule
      as the v1 brief.
- [ ] **Doc surface updates.** If `room_anchor`'s docstring
      promised "O(N) linear scan" anywhere (it doesn't
      today, but check before shipping), the doc moves with
      the implementation.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- v1 manager:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `room_anchor` is the linear scan (~10 lines); the
  cluster-cache pattern (`_invalidate_cluster_cache`,
  `_cluster_cache_stale`) is the existing template for the
  per-label index.
- v1 room classifier:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `RoomClassifier._ensure_text_embeddings` is where the
  sticky-disable lives.
- Smoothing brief's in-place label mutation:
  [`room-state-temporal-smoothing`](../../active/multi-room/room-state-temporal-smoothing.md)
  — coordinate the index-update hook with that brief if it
  ships first.

## Out of scope

- **Cluster cache redesign.** The growth-fraction cache
  invalidation for `known_rooms` / `current_room` /
  `connectivity` is fine for v1 and is already addressed
  by the temporal-smoothing brief's "cluster cache
  invalidation already triggers re-smoothing" disposition.
  Don't fold cache-design work into this brief.
- **`current_room` lookup-radius parameterisation.** A
  separate concern that may or may not surface from v2's
  frontier-enrichment work — the v2 brief
  ([`llm-guided-frontier-gain`](llm-guided-frontier-gain.md))
  is the right home if it does.
- **`RoomEntry.confidence` calibration.** Owned by
  [`room-state-uncertainty-calibration`](../../active/multi-room/room-state-uncertainty-calibration.md);
  not a runtime ergonomics concern.
- **Multi-instance room disambiguation.** Same-label cluster
  merge is v1-by-design; disambiguation work is filed
  elsewhere when it surfaces real failures (e.g., a
  multi-bedroom adversarial mission).
- **Finding B obsolescence.** If
  [`learned-region-head`](learned-region-head.md) ships and
  retires the v1 `RoomClassifier` entirely, Finding B
  (sticky-disable retry) becomes moot — there's no
  `RoomClassifier` left to fix. Until then, the finding
  stands. Finding A (`room_anchor` index) survives the
  head ship; `room_anchor()` is still O(N) regardless of
  what populates `metadata["room_label"]`.
