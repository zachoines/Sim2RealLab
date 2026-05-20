# VLM-refined per-node room labels

**Type:** new feature
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (the headline accuracy lift in the v2
quality series; takes CLIP zero-shot from baseline to
production-grade by composing with VLM detections already
flowing through the system)
**Estimate:** M (~3–5 days; prototype-objects map + heuristic
combiner + A/B eval against the harness + tests)
**Branch:** task/room-label-vlm-refinement

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](observation-derived-room-state.md)
merges. Strongly recommended ordering:
[`room-state-eval-harness`](room-state-eval-harness.md)
ships first so the label-quality lift can be measured.

## Story

As a **room classifier whose CLIP zero-shot baseline mislabels
~15–20% of nodes on busy scenes**, I want **the labeler to
also consume the VLM-detected `DetectedObjectEntry` list
already populated on mission-driven nodes (`scan_for_target`,
`verify_arrival`, etc.) and use an object-to-room prototype
map to refine the label**, so that **nodes with strong
object evidence (sink + stove + refrigerator → kitchen)
override the noisier CLIP prior and label accuracy lifts
measurably without adding any new VLM service calls**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — v1 implementation. This brief extends its `RoomClassifier`.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  required to measure the label-precision lift.
- [`room-state-uncertainty-calibration`](room-state-uncertainty-calibration.md)
  — sibling brief; the calibrated probabilities make the
  CLIP-vs-VLM override decision principled rather than
  threshold-tuned.

## Context

### The free lunch

VLM detections already flow through the manager via
`add_observation(detected_objects=[...])` populated by mission
skills:

- `scan_for_target` calls `HttpGroundingClient.locate_semantic_target()`
- `verify_arrival` calls `HttpGroundingClient.describe_scene()`
  / `detect_objects()`
- These produce `DetectedObjectEntry` lists with labels +
  3D positions

The `BackgroundMapper` itself doesn't call VLM (movement-gated
encoder-only capture) — so nodes split into two cohorts:

- **VLM-rich nodes**: mission-driven captures with one or
  more `detected_objects[*].label` set.
- **CLIP-only nodes**: BackgroundMapper captures with no
  detections.

v1 ignores the object evidence entirely for room labeling.
This brief uses it where available, with zero new VLM calls.

### Hybrid architecture

```
add_observation(image, detected_objects):
    base_label, base_conf = clip_classifier.classify(image)

    if detected_objects:
        vlm_label, vlm_conf = refine_label_from_objects(
            detected_objects, prototype_map,
        )
        if vlm_conf > base_conf:
            final_label, final_conf = vlm_label, vlm_conf
        else:
            final_label, final_conf = base_label, base_conf
    else:
        final_label, final_conf = base_label, base_conf

    metadata["room_label"] = final_label
    metadata["room_conf"] = final_conf
```

The override condition is "VLM is more confident" — clean
once
[`room-state-uncertainty-calibration`](room-state-uncertainty-calibration.md)
ships and both numbers are real probabilities. Until that
ships, gate behind a calibrated threshold or a simple
"VLM-derived dominates when prototype-match ≥ N objects."

### The prototype map

A configurable label → prototype-objects mapping lives in
[`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py):

```python
DEFAULT_ROOM_PROTOTYPES = {
    "kitchen": ["sink", "stove", "refrigerator", "oven",
                "microwave", "dishwasher", "cabinet"],
    "bedroom": ["bed", "nightstand", "dresser", "wardrobe",
                "lamp", "pillow"],
    "living room": ["couch", "sofa", "tv", "television",
                    "coffee table", "armchair", "rug"],
    "bathroom": ["toilet", "bathtub", "shower", "sink",
                 "mirror", "towel"],
    "office": ["desk", "monitor", "keyboard", "chair",
               "bookshelf"],
    "hallway": [],   # weak prototype; CLIP wins here
    "garage": ["car", "tool", "shelving", "bicycle"],
}
```

Note: hallways legitimately have no signature objects;
prototype map handles this by returning zero confidence,
falling back to CLIP.

### Refinement scoring

```python
def refine_label_from_objects(
    detected_objects, prototype_map,
) -> tuple[str, float]:
    """Score each candidate room by observed-prototype overlap."""
    seen_labels = {o.label.lower() for o in detected_objects}
    best_label = None
    best_score = 0.0
    for room, prototypes in prototype_map.items():
        if not prototypes:
            continue
        hits = sum(1 for p in prototypes if p in seen_labels)
        score = hits / len(prototypes)
        if score > best_score:
            best_score = score
            best_label = room
    return (best_label, best_score)
```

Score is the fraction of the room's prototype set observed in
this node. Tunable via the dict — operators can add
home-specific object→room hints by editing the constant.

### Why heuristic, not learned

v1.5 intentionally ships a hand-set prototype map for three
reasons:

1. The prototype map is small (~50 entries) and trivially
   readable — operators can audit and edit.
2. Learning room prototypes from data requires a labeled
   object→room corpus; we don't have one and Infinigen's
   `room_idx` per-object would be the only source. The
   parked [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
   brief's eval harness extension is the natural fit if we
   later want to learn the prototypes.
3. The heuristic is a strong baseline. Adding learned weights
   on top is a v2.5 follow-up if the heuristic plateaus.

### Tie-in to state-of-the-art

- **OK-Robot** (arXiv:2401.12202) — uses VLM detections to
  build object-grounded scene representations for navigation.
  Same shape: detections inform room understanding.
- **ConceptGraphs** (arXiv:2309.16650) — open-vocab 3D scene
  graph where objects are first-class. The prototype map here
  is a simplified, hand-set analogue of their learned object→
  region associations.
- **HOV-SG / NLMap** — both fuse object detection with scene
  embeddings for fine-grained spatial understanding. v1.5
  takes the cheaper "object dominates when confident" shape.

## Acceptance criteria

- [ ] **Prototype map.** A `DEFAULT_ROOM_PROTOTYPES` constant
      ships in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
      with the seven default room labels populated. Operator-
      facing comment in the module pointing at the dict for
      home-specific tuning.
- [ ] **Refinement function.**
      `refine_label_from_objects(detected_objects, prototype_map) -> tuple[str | None, float]`
      lives in `room_state.py`, pure on the inputs.
- [ ] **Wired into `add_observation`.** When
      `detected_objects` is non-empty, the refinement scores
      are computed and compared against the CLIP base label;
      the higher-confidence label wins. The non-winning
      label + score are stored on the node metadata
      (`metadata["room_label_clip"]`,
      `metadata["room_label_vlm"]`) for inspectability.
- [ ] **Quality lift measured.** PR description carries
      label precision + cluster purity on the four
      [`room-state-eval-harness`](room-state-eval-harness.md)
      scenes, comparing v1 baseline / v1+smoothing /
      v1+smoothing+VLM-refinement. The VLM refinement should
      lift label precision ≥ 0.10 absolute on at least two of
      the four scenes (relaxed on scenes with sparse
      detections).
- [ ] **Configurable prototype map.** `RoomClassifier`
      constructor accepts a `prototype_map: dict[str, list[str]] | None`
      argument; `None` uses the default.
- [ ] **No new VLM calls.** Confirm by inspection that the
      refinement path doesn't call any HTTP / RPC; it consumes
      the `detected_objects` already on the observation.
- [ ] **Unit tests.** Refinement function tested against
      synthetic detection lists with expected scoring;
      manager-level test verifying VLM override happens when
      detected_objects has prototype matches; manager-level
      test verifying CLIP retains label when detected_objects
      is empty.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- Existing `DetectedObjectEntry`:
  [`semantic_map/models.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py).
- VLM detection consumers:
  [`HttpGroundingClient.detect_objects`](../../../../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py)
  — what produces the labels we're consuming.
- Existing per-node stamping:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `add_observation` → `_classify_room`.
- Prototype-map literature: OK-Robot (arXiv:2401.12202),
  ConceptGraphs (arXiv:2309.16650).

## Out of scope

- **Open-vocab labeling API.** Retiring the discrete label set
  entirely in favor of an embedding-space query lives in the
  parked
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  extension. This brief preserves the v1 label-set API
  shape.
- **Learned prototypes.** v1.5 is hand-set. Learned weights
  fit against the eval harness is a v2.5 follow-up.
- **VLM-call-on-demand.** The brief explicitly does NOT add
  new VLM service calls; it reuses detections already flowing
  through the manager. A more aggressive design that calls
  the VLM from the BackgroundMapper is filed under
  [`semantic-graph-object-centric-hierarchical`](semantic-graph-object-centric-hierarchical.md)
  as part of the broader scene-graph upgrade.
- **Object-position-based room inference.** This brief uses
  object *labels* only. The 3D positions in
  `DetectedObjectEntry.position_mean` could refine
  intra-room sub-region labeling but that's the
  [`semantic-graph-object-centric-hierarchical`](semantic-graph-object-centric-hierarchical.md)
  brief's territory.
