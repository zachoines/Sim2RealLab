# Define the map-conditioning contract for the v2 VLA

**Type:** investigation / docs / new feature
**Owner:** DGX agent (training + eval lives DGX-side; the executor-
side adapter to surface the conditioning input is a small Jetson-
lane edit, same cross-lane shape as
[`vla-v2-architecture`](vla-v2-architecture.md))
**Priority:** P3 — filed-on-trigger. Becomes pickable when
**both**
[`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
ships (the hierarchy this brief might condition on exists) and
[`vla-v2-architecture`](vla-v2-architecture.md) has a working
training run on the harness corpus (we have a model to condition).
File this brief if v2's first ablations show the bare
`(frame, text)` conditioning leaves measurable mission-success
headroom on cross-room missions; do not pick up preemptively.
**Estimate:** L (~1–2 wk; conditioning-shape design + dataset
plumbing + retrofit of one v2 ablation row + write-up)
**Branch:** task/vla-v2-map-conditioning

## Story

As an **operator running the v2 VLA on cross-room missions where
the model has access to a persistent
[`SemanticMapManager`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
+ hierarchical scene graph at inference**, I want **a concrete
decision on how the v2 VLA consumes that map state (text
serialization in the prompt, learned cross-attention over a memory
bank, or no consumption at all — the hierarchy stays sim-eval
only)**, so that **the v2 VLA's training corpus, inference
adapter, and the
[`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
brief's "map-side substrate" framing are anchored by a written
contract instead of an unstated assumption, and the cross-room
mission-success number the v2 ablation reports is interpretable
against the room-state quality the v1 graph layer delivers**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`vla-v2-architecture`](vla-v2-architecture.md) — the v2 VLA
  itself. This brief is its map-conditioning contract.
- [`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
  — the hierarchical substrate this brief decides whether and how
  to feed into the VLA. That brief calls itself the "map-side
  substrate for the parked vla-v2-architecture bet"; this brief
  is the missing other half.
- [`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)
  — sibling research arm that already proposes a memory-bank +
  cross-attention layer for the cascade validator. Same primitive,
  different consumer.
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — the v1 flat-graph + room API the contract must remain valid
  against if the operator picks the "no consumption" branch and
  the v2 VLA reads only `current_room` + `known_rooms` as a
  text prefix.

## Context

### The gap this brief closes

The active brief at
[`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
positions itself as "the map-side substrate for the parked
[`vla-v2-architecture`](vla-v2-architecture.md) bet" — but
`vla-v2-architecture.md` does **not** mention the semantic map,
hierarchical graph, room state, or any conditioning shape that
consumes them. The two briefs share a stated dependency that is
unwritten on the v2 side. Without a contract:

- The v2 VLA training script doesn't know which conditioning
  tensor to expect alongside `(frame, text, action)`.
- The
  [`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)
  cross-attention layer and the hierarchical-graph nodes both
  exist but neither knows whether they are siblings (one
  conditioning shape) or alternatives (two conditioning shapes).
- The
  [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  metrics (cluster purity, label precision, connectivity) are
  scoring the *map*; the v2 mission-success metric is scoring
  the *VLA*. Without a contract there is no chain of causation
  linking them, so "improving the map" cannot be argued to
  improve VLA performance.

### Three viable conditioning shapes — pick one

| Option | Shape | Training cost | Inference cost | When it fits |
|---|---|---|---|---|
| **A. Text serialization** | The v1 `RoomEntry` and (optionally) the hierarchical layer are flattened into a short string ("Robot is in living_room; known_rooms include kitchen (sink, stove), bedroom (bed); kitchen reachable from living_room.") and prepended to the mission text. | None — the existing planner-LLM-pattern already serializes scene context into prompts; v2 reuses the same shape. | A few hundred tokens per request. | Quickest to test; matches how [GraphPilot (arXiv:2511.11266)](https://arxiv.org/pdf/2511.11266) conditions a driving VLA on a scene graph. |
| **B. Cross-attention over a memory bank** | Past observations' CLIP embeddings (or
[`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)'s memory-augmented embeddings) are retrieved per-step and consumed via a cross-attention layer in the VLA. The hierarchical layer is *not* a runtime input — it's a training-time annotation that shapes which memories the retrieval pool returns. | New cross-attention layer; new dataset plumbing for `(frame, text, retrieved, action)` quads. | ~5–10 ms cross-attention + ChromaDB top-K. | Most flexible long-term; aligns with the [HOV-SG (arXiv:2403.17846)](https://arxiv.org/pdf/2403.17846) and OpenScene / VLMaps / CLIP-Fields lineage where the consumer reads an *implicit* feature field. Composes with the cascade validator's memory bank. |
| **C. No consumption** | The v2 VLA reads only `(frame, text, action)` as in
[OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246). The semantic map is sim-eval / planner-side reasoning only. The hierarchical brief's "map-side substrate" framing is downgraded to "interpretability + planner-side queries"; the VLA path is untouched. | None. | None. | Best baseline; ships v2 with the smallest moving parts. If A and B both fail to lift cross-room mission success above this baseline, the hierarchy is not load-bearing for the VLA path. |

The brief's job is to pick **one** by running a measurement, not
to default to A or B because they sound smarter.

### Why B is *not* obviously the right answer

The
[`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)
brief already proposes B for the cascade validator and reports
the literature evidence (Re-CLIP, MemoryBank-CLIP, Atlas, RA-DIT)
that **training-time retrieval beats inference-only retrieval**.
But:

- The cascade validator's task is **binary classification** (is
  the leg off-course?). Memory-bank retrieval helps because the
  validator wants to compare the live frame against past frames
  from the same place.
- The v2 VLA's task is **action prediction**. Retrieved memories
  may help when the action depends on "which room am I in" or
  "where did I last see the target"; they may *not* help (or
  hurt) when the action is "drive forward 1 m" with no
  cross-room semantics. The literature on this for VLAs is
  thin — OpenVLA / π0 / GR00T don't ship with a memory bank,
  and the published action-prediction-with-retrieval results
  ([3D-VLA, CVPR 2024](https://arxiv.org/pdf/2403.17846); LERF;
  the OpenSceneGraph line) are all single-paper claims at
  research-preview maturity, not production patterns.

The measurement bar below pins this: B ships **only** if the
cross-room mission-success lift over A or C is at least one
CI-width. Otherwise the brief documents that the v1 flat-graph
+ text-serialization is sufficient and the hierarchical
brief's "substrate" claim is downgraded.

### Sim-to-real considerations specific to this brief

The chosen conditioning shape must survive sim-to-real. Three
failure modes specific to map conditioning:

- **Text-serialization (A)** is robust to sim-to-real: the
  string itself is sim-distribution-free. Risk: the VLA learns
  to ignore the prefix.
- **Memory-bank (B)** is fragile: the v2 VLA trained against
  a fully-populated sim retrieval pool may collapse when the
  real-robot pool is empty (cold-deployment problem already
  flagged in
  [`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)'s
  retrieval-pool-size augmentation). Re-use that brief's
  `K_train ∈ {0, 1, 2, 4, 8}` augmentation here if Option B is
  selected.
- **No-consumption (C)** has no map-conditioning failure modes
  by construction. The v2 mission-success bar IS the cross-room
  number — if it's good enough, ship.

## Approach

Single ablation row added to
[`vla-v2-architecture`](vla-v2-architecture.md)'s eval table:

1. **Train three v2 VLA checkpoints** — Option A, Option B,
   Option C — against the same harness corpus and the same
   held-out scene seeds (per
   [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)'s
   `held_out_seeds` field). Same hyperparameters except the
   conditioning shape.
2. **Eval each against the cross-room subset** of the
   mission-generator queue. Primary metric: per-mission success
   on `wrong_room` and `wrong_instance` cases per the
   five-way label from
   [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md).
3. **Compare with the same CI-width rule** the other v2 briefs
   use (one CI-width over the chosen baseline). The brief picks
   the shape whose lift over the next-lower option is ≥ 1
   CI-width. Ties default to the simpler shape (C > A > B).
4. **Append a §4.7 addendum** to
   [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
   documenting the chosen shape and re-anchoring the
   [`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
   brief's "substrate" framing against the result.

## Acceptance criteria

- [ ] **Conditioning-shape contract written.** A new "Map
      conditioning" section in
      [`vla-v2-architecture`](vla-v2-architecture.md) (added
      in this brief's PR) names the chosen Option (A / B / C),
      the conditioning tensor shape if any, and the
      dataset-pipeline change required. Same PR updates the
      cross-references in
      [`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
      so the "substrate" claim points at the contract section.
- [ ] **Three-row ablation table** in the PR description and
      in the §4.7 addendum, scoring cross-room mission-success
      on the held-out scene seeds for each of A / B / C.
      Numbers, not opinions.
- [ ] **CI-width rule for shape selection.** The brief picks
      A or B only if its cross-room mission-success lift over
      C is at least one bootstrap CI-width on the eval set's
      multi-room scenes. Otherwise C is the contract and the
      hierarchical-graph brief's "substrate" framing is
      downgraded in the same PR.
- [ ] **Cold-deployment carry-over (Option B only).** If
      Option B wins, the training pipeline inherits the
      `K_train ∈ {0, 1, 2, 4, 8}` retrieval-pool-size
      augmentation from
      [`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md);
      the cold-deployment row of the v2 eval table must show
      no regression vs. Option C on empty-map missions.
- [ ] **Runtime-legal inputs only.** A grep of the v2 VLA
      adapter for `scene_metadata`, `scene_labels`,
      `room_adjacency`, `infinigen` returns zero hits.
      Conditioning consumes only `SemanticMapManager` outputs
      (`known_rooms`, `current_room`, `connectivity`,
      `hierarchical_graph` if it ships) and on-board
      perception. Same sim-to-real rule as every other v2
      brief.
- [ ] **§4.7 addendum** to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the chosen shape, the ablation table, and a
      one-paragraph re-anchoring of the hierarchical brief's
      framing.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit. See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- v2 VLA training pipeline (when it exists):
  [`vla-v2-architecture`](vla-v2-architecture.md) §
  "Training pipeline".
- Hierarchical graph API (when it exists):
  [`semantic-graph-object-centric-hierarchical`](../../active/multi-room/semantic-graph-object-centric-hierarchical.md)
  — `hierarchical_graph()`, `known_objects()`, etc.
- Retrieval-augmented training reference recipe:
  [`cotrained-retrieval-augmented`](../../clip-validation/cotrained-retrieval-augmented.md)
  — Step B's RAG-aware training pattern is the direct
  template for Option B here.
- Reference architectures to cross-check against:
  - **GraphPilot** ([arXiv:2511.11266](https://arxiv.org/pdf/2511.11266))
    — scene-graph conditioning for a driving VLA via prompt
    serialization. Direct precedent for Option A.
  - **HOV-SG** ([arXiv:2403.17846](https://arxiv.org/pdf/2403.17846))
    — hierarchical open-vocab 3D scene graph for
    language-grounded robot navigation; cross-attention
    consumer. Direct precedent for Option B.
  - **3D-VLA** (CVPR 2024) — VLA conditioned on a 3D scene
    representation. Single-paper at research-preview maturity;
    informs but does not validate Option B.
  - **OpenVLA** ([arXiv:2406.09246](https://arxiv.org/abs/2406.09246))
    — ships with no map conditioning. The Option C baseline.
  - **VLMaps / CLIP-Fields / LERF** — implicit feature-field
    lineage; Option B's literature spine if B wins.

## Out of scope

- **Real-robot eval.** Sim-only — same rule as every other v2
  brief.
- **A new VLA backbone.** Backbone choice is inherited from
  [`vla-v2-architecture`](vla-v2-architecture.md) and
  [`backbone-bakeoff`](../../clip-validation/backbone-bakeoff.md).
- **Replacing the cascade validator.** If Option B ships, the
  cascade validator's memory bank and the VLA's memory bank
  *can* share infrastructure — but co-tenancy work is a
  follow-up, not this brief.
- **Speaker / caption-corpus changes.** This brief consumes
  whatever
  [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md)
  produces; speaker fine-tuning is its own work.
- **Multi-floor scene graphs.** Single-story constraint
  carries over from every other multi-room brief.
