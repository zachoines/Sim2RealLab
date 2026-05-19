# LLM-guided frontier gain for `explore_until_visible`

**Type:** new feature
**Owner:** DGX agent (new planner-service endpoint + prompt
design; Jetson-side change is a config wire-up to call the
endpoint from the existing skill — same cross-lane shape as
`frontier-exploration-primitive`)
**Priority:** P2 (a measurable-improvement layer on top of the
v1 unguided frontier primitive; ships only after v1's base
success rate is characterized)
**Estimate:** M (~3–5 days; endpoint + prompt + Jetson hook +
A/B eval harness + acceptance run on the mission-generator
queue)
**Branch:** task/llm-guided-frontier-gain

**Pickup gate:** Blocked-on-deps until
[`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md)
ships (no skill to extend) AND
[`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
ships (no language-shaped frontier descriptions to give the
LLM). Un-park by `git mv` per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As an **operator running cold-start cross-room missions where
the v1 unguided frontier-exploration primitive sometimes wastes
visits on the wrong room** ("go to the kitchen table" but the
unguided gain ranks a frontier into the bedroom higher because
the bedroom frontier has more unmapped cells), I want **the
frontier ranking function to multiply in a language-model
prior that biases toward frontiers semantically consistent with
the operator's target**, so that **mission time-to-target and
frontier-visit count on cold-start cross-room missions drop
measurably versus v1, without changing the skill interface or
adding new failure modes**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md)
  — the v1 skill this brief extends. The skill interface,
  detector, navigation loop, and termination conditions are
  unchanged; only the gain function gains an LLM multiplier.
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — provides the nearest-room inference used to turn each
  frontier into a language-shaped description.
- [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  — the cold-start path that invokes `explore_until_visible`.
  The mission shape this brief tunes.
- [`mission-generator`](../../active/harness/mission-generator.md)
  — supplies the canonical cross-room mission queue used in
  the A/B eval.

## Context

### Why this is an extension, not a different direction

The v1 `explore_until_visible(label)` skill in
[`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md)
ranks frontiers by a geometric gain function:

```
score_v1(f) = information_gain(f) / travel_cost(f)
```

`information_gain` is the wavefront cell count (unmapped
neighbors); `travel_cost` is the Nav2 plan length. The skill
walks the ranked frontiers in order until a frontier-visit
grounds the target or the candidate set is exhausted.

This brief introduces a single multiplicative term:

```
score_v2(f) = score_v1(f) × llm_prior(f, target_label, scene_context)
```

`llm_prior` is a unit-interval scalar returned by a new
planner-service endpoint that scores how likely the frontier
is to lead to the operator's target. With
`gain_weights.llm = 0.0`, score_v2 collapses to score_v1
exactly — the v1 behavior is preserved as a feature flag, and
the v2 acceptance bar becomes "measurable improvement over v1
under `llm = 1.0`."

Skill interface, frontier detector, navigation loop,
termination conditions, and failure-handling all stay
identical. Only the gain ranking changes.

### Design: scalar prior (LFG-style), not state machine (CogNav-style)

The state-of-the-art literature presents two patterns:

- **Scalar prior** (Language Frontier Guide,
  [arXiv:2310.10103](https://arxiv.org/abs/2310.10103)): the
  LLM emits a softmax-shaped distribution over frontier
  candidates given the operator's command. Multiplied onto the
  geometric gain. Drop-in, easy to A/B test, easy to disable.
- **Cognitive-state machine** (CogNav, ICCV 2025): the LLM
  reasons over agent states (broad exploration → contextual
  exploration → identification → verification) and emits
  different behaviors per state. More expressive, more
  controllable on hard missions, more prompt-engineering
  surface, harder to roll back.

Recommended for v2: scalar prior. The state-machine variant
is a logical v3 once the scalar-prior baseline is
characterized. Reasons:

- Scalar prior preserves the v1 fallback by construction
  (`weight = 0` → unguided). State-machine swaps the whole
  control flow, so falling back is a code path, not a knob.
- Scalar prior's prompt is a single short call per
  exploration step. State machine implies multiple LLM calls
  per step (state classifier + behavior selector + frontier
  scorer). 3-5× more latency at the same model size.
- The literature (LFG, LOAT — arXiv:2403.09971) shows the
  scalar-prior shape captures most of the gain on
  object-goal-navigation benchmarks. The state-machine gain is
  largest on the longest-horizon missions; the strafer's
  current 6 m SLAM horizon does not yet stress that regime.

### How the LLM gets enough context to score

The LLM cannot read a costmap. Each candidate frontier needs a
language-shaped description that the LLM can reason over. The
v1 wavefront detector emits a frontier set with `(x, y,
cell_count, plan_cost_m)` per candidate; this brief enriches
each entry with metadata already available elsewhere in the
runtime:

| Field | Source | Example |
|---|---|---|
| `nearest_known_room` | [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)'s `current_room` at the frontier's projected pose | `"living_room"` |
| `boundary_with` | room-state's `connectivity` lookup — what room the frontier opens into if it leaves a known room | `"unknown"` or `"kitchen"` |
| `nearest_object_in_map` | semantic-map `query_nearest` on the frontier's pose; top-1 label + distance | `("lamp", 1.4 m)` |
| `bearing_from_robot_deg`, `distance_from_robot_m` | trivial geometry | `(45.0, 3.2)` |
| `leads_into_unknown` | wavefront detector already knows this — surface it | `true` |

The endpoint receives the target label + the current scene
summary (a one-paragraph string from the existing
`describe_scene` VLM endpoint, cached per exploration episode)
+ this enriched frontier list, and returns one unit-interval
score per frontier. Prompt sketch:

```
You are helping a mobile robot decide which unexplored
direction to investigate next.

Target: "kitchen table"
Current scene: "The robot is in a living room with a sofa and
a TV. A hallway opens to the east."

Frontiers:
- F0: 45° east, 3.2 m, leads into unknown region from
  living_room boundary. Nearest mapped object: lamp (1.4 m).
- F1: 180° west, 2.8 m, between living_room and bedroom.
  Nearest mapped object: bed_frame (0.9 m).
- F2: 90° south, 4.1 m, leads into unknown region from
  living_room boundary. No nearby mapped objects.

Score each frontier from 0.0 to 1.0 by how likely it is to
lead to the target. Output a JSON object with one score per
frontier id.
```

### Endpoint

`POST /rank_frontiers` on the DGX planner service (port
8200, not the VLM service — this is text-only reasoning, and
the planner backend already owns the LLM that does intent
classification). Existing schema patterns under
[`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../../source/strafer_autonomy/strafer_autonomy/schemas/)
extend naturally:

```python
class FrontierRankRequest(BaseModel):
    target_label: str
    scene_summary: str
    frontiers: list[FrontierDescription]

class FrontierRankResponse(BaseModel):
    scores: dict[str, float]  # frontier_id → [0, 1]
    reasoning: str | None  # optional, for logs / debugging
```

If image input proves necessary for ranking quality (filed
follow-up), the schema can grow a `current_observation_rgb`
field — but v2 is text-only by design.

### Latency budget

Frontier ranking fires once per exploration step. Each
exploration step is ~10–30 s of robot driving + grounding, so
1–2 s of LLM latency is tolerable; >5 s starts to bottleneck.
Measure before committing to a heavier prompt; the existing
intent classifier round-trips in well under 2 s, so the
baseline budget is in range.

### Failure handling

Every failure mode falls back to v1 geometric-only ranking:

- Endpoint timeout / 5xx → log warning, use v1 scores.
- LLM returns malformed JSON → log warning, use v1 scores.
- LLM scores all frontiers ≈ 0 (no opinion) → use v1 scores.
- LLM scores reference a frontier id we didn't ask about →
  drop the unknown ids, renormalize, blend with v1 weights.

The `gain_weights.llm = 0.0` knob is the operator-side
emergency fallback if the LLM is consistently making things
worse.

## Acceptance criteria

- [ ] **Endpoint shipped.** `POST /rank_frontiers` is
      registered on the DGX planner service and documented
      under [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      with input/output schema.
- [ ] **Schema additions.** `FrontierRankRequest`,
      `FrontierRankResponse`, `FrontierDescription` added to
      [`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../../source/strafer_autonomy/strafer_autonomy/schemas/).
- [ ] **Wavefront detector enriches frontiers** with
      `nearest_known_room`, `boundary_with`,
      `nearest_object_in_map`, `bearing_from_robot_deg`,
      `distance_from_robot_m`, `leads_into_unknown` per the
      table above. Reuses
      [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
      and semantic-map APIs — does not duplicate room
      inference.
- [ ] **`explore_until_visible` calls the endpoint** and
      multiplies the returned scores onto the geometric gain
      per the `gain_weights = {geometric: 1.0, llm: 1.0}`
      knob.
- [ ] **`gain_weights.llm = 0.0` recovers v1 exactly.** A
      regression test pins this: with the LLM disabled, plan
      outputs and frontier-visit order are byte-identical to
      v1's recorded fixture.
- [ ] **Latency budget.** Mean endpoint latency on the
      mission-generator eval set is ≤ 2 s; p95 ≤ 4 s.
      Measured and reported in the PR description. If
      exceeded, document the trade and either trim the prompt
      or downgrade the brief to a measurement-only PR with no
      runtime flip.
- [ ] **A/B eval.** On at least 20 cold-start cross-room
      missions from
      [`mission-generator`](../../active/harness/mission-generator.md)'s
      output queue, the v2 run shows: ≥ 15% reduction in mean
      time-to-target, OR ≥ 1 fewer frontier-visit on average,
      relative to v1 on the same missions. (Pre-register
      whichever metric is primary before running the eval.)
      No regression on near-target / single-room missions.
- [ ] **No regression** in v1's failure modes. With the
      endpoint disabled (any failure mode listed under
      "Failure handling"), behaviour is identical to v1.
- [ ] **Runtime-legal inputs only.** A grep of the new code
      for `scene_metadata`, `scene_labels`, `room_adjacency`
      returns zero hits. Same sim-to-real rule as
      `frontier-exploration-primitive`.
- [ ] **Doc surface updates.**
  - [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md):
    endpoint table gains a `POST /rank_frontiers` row.
  - [`docs/STRAFER_AUTONOMY_NEXT.md` §1.10.1](../../../STRAFER_AUTONOMY_NEXT.md#1101-known-limitation-multi-room-navigation):
    add a one-line note that frontier-exploration has both
    unguided (v1) and LLM-guided (v2) variants once this
    ships.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- v1 skill structure:
  [`frontier-exploration-primitive`](../../completed/frontier-exploration-primitive.md)
  — read its skill shape + frontier-source choice first.
  This brief's wire-up depends on whether v1 vendored
  `m-explore-ros2` or pulled the wavefront detector inline.
- Existing planner endpoint patterns (intent classification):
  [`source/strafer_autonomy/strafer_autonomy/planner/app.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/app.py).
  The new endpoint follows the same FastAPI shape.
- Semantic-map nearest-object lookup:
  [`SemanticMapManager.query_nearest`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py).
- VLM `describe_scene` endpoint (for the scene_summary
  string): existing endpoint in the VLM service. The
  exploration episode caches the description across frontier
  ranks so the call is amortized.
- Reference architectures (read at least two before
  designing the prompt):
  - Language Frontier Guide,
    [arXiv:2310.10103](https://arxiv.org/abs/2310.10103) —
    the scalar-prior pattern this brief follows.
  - CogNav, ICCV 2025
    ([PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_CogNav_Cognitive_Process_Modeling_for_Object_Goal_Navigation_with_LLMs_ICCV_2025_paper.pdf))
    — the state-machine variant; useful for understanding
    where the scalar prior leaves performance on the table.
  - LOAT,
    [arXiv:2403.09971](https://arxiv.org/abs/2403.09971) —
    object-affinity prior; informs the
    `nearest_object_in_map` field choice.
  - FSR-VLN,
    [arXiv:2509.13733](https://arxiv.org/html/2509.13733v1)
    — hierarchical scene-graph + LLM reasoning; informs the
    nearest-room field choice.

## Out of scope

- **Cognitive-state-machine variant (CogNav-style).** Filed
  as a v3 follow-up only if the scalar-prior baseline
  plateaus on hard missions. Adds prompt complexity, latency,
  and a second LLM call per step.
- **Vision-language frontier ranking.** Passing an image
  observation to the LLM at rank time is a natural extension
  but doubles input tokens and complicates latency. Defer to
  a v3 brief filed if text-only scoring proves insufficient.
- **Replacing the planner LLM backend.** Whichever model
  serves intent classification today also serves
  `/rank_frontiers`. A separate routing decision (one model
  for intent, a different model for frontier ranking) is
  filed separately if pursued.
- **Online LLM fine-tuning on mission outcomes.** Closing the
  loop between "the LLM ranked F0 highest, F0 grounded the
  target" and a learned per-scene prior is filed under the
  experimental epic's VLA work; not a runtime brief.
- **Training-time `scene_metadata.json` consumption.**
  Explicitly out of scope and explicitly forbidden, same
  sim-to-real rule as `frontier-exploration-primitive`.
