# Investigate mid-mission validation: CLIP today, alternatives next

**Status:** Shipped 2026-05-05 in `26b97bb` (Either). The audit landed
as [`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
— current-state map of CLIP usage with file:line pointers, structural
+ measurement-required limitations, alternatives survey grounded
against published references, and a staged recommendation with
falsifiable TPR/FPR gates. The investigation surfaced that the
semantic-map package is complete-but-orphaned in production
(`executor/main.py` constructs the runner with `semantic_map=None`
and `background_mapper=None`), which moved the framing from "is CLIP
useful?" to "wire it in, measure it, then decide."
**PR:** https://github.com/zachoines/Sim2RealLab/pull/19
**Follow-ups:** [`validator-evaluation`](../active/clip-validation/validator-evaluation.md)
— P1 wire-and-measure brief (the gating prerequisite). The
originally-listed `learned-mid-mission-validator` follow-up was
retired during PR review in favor of
[`cotrained-retrieval-augmented`](../parked/clip-validation/cotrained-retrieval-augmented.md)
(cascade-improvements path) and
[`vla-v2-architecture`](../parked/experimental/vla-v2-architecture.md)
(end-to-end VLA research arm); see
[`learned-mid-mission-validator`](learned-mid-mission-validator.md)
in this same `completed/` directory for the retirement
rationale.

**Type:** investigation / architecture
**Owner:** Either (the audit half spans `strafer_lab` + `strafer_vlm` +
`strafer_autonomy`; the alternatives half is mostly DGX-led but the
deployment-cost analysis is Jetson-led)
**Priority:** P2 (architecture work; informs the next iteration of
the autonomy stack but doesn't block any in-flight feature)
**Estimate:** L (~multi-day; a real audit + a literature/practice
survey + a defensible recommendation, not a 2-hour skim)
**Branch:** task/mid-mission-validation-investigation

## Story

As an **operator looking at the two-tiered architecture (low-level RL
in `strafer_lab` + autonomy/VLM in `strafer_autonomy` +
`strafer_vlm`) and worrying about what happens *between* the
expensive planner+grounding round-trips**, I want **a written audit
of how CLIP is wired into the system today, where it falls short,
and how state-of-the-art systems handle real-time mission validation
and self-correction**, so that **the next round of architecture work
is grounded in evidence — what actually works on Jetson-class
compute, what's wishful thinking, and which option (better CLIP, a
small VLA, a learned validator, or something else) is worth
prototyping next**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md) — DGX
  vs Jetson hosts, conda envs, where each package lives.
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
  — this brief touches all three packages; mind the lane lines for
  any code-level changes that might surface (the audit phase is
  read-only and can roam freely).
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
  — branch off `main`, one brief → one branch → one PR.
- [`context/conventions.md`](../context/conventions.md) — write-up
  rules, no transient references in source.

## Context

### Why now

The current architecture is a deliberate two-tier split:

- **Low-level continuous control.** `strafer_lab` trains an RL
  policy on Isaac Lab's `Isaac-Strafer-Nav-Real-*` envs; the
  deployable artifact is consumed by Nav2 (or, eventually, by the
  `strafer_direct` and `hybrid_nav2_strafer` backends in
  [`inference-package`](../active/trained-policy/inference-package.md)). At
  steady state this layer runs at ~30 Hz on the Jetson.
- **High-level autonomy.** `strafer_autonomy` (planner + executor)
  + `strafer_vlm` (Qwen2.5-VL grounding + description) translate a
  natural-language mission into a skill sequence and ground
  semantic targets ("the door", "the couch") to map-frame goals.
  Each planner call is ~hundreds of ms; each VLM grounding call is
  ~2-3 s for single-object, ~1.5-4 s for multi-object (per
  [`source/strafer_vlm/README.md`](../../../source/strafer_vlm/README.md)
  "Deferred / known limitations"). `scan_for_target` runs 6
  grounding calls per rotation — ~18 s before navigation even
  begins.

The architecture made sense for a quick path to generalist-like
mission execution on Jetson-Orin compute. But it has a
**verification gap**:

- Between the moment the executor commits to a goal pose and the
  moment Nav2 (or the RL policy) reports `arrived`, the robot
  navigates with no semantic check on whether it's still heading
  somewhere the user actually intended.
- A bad VLM grounding result (the wrong "couch" picked, off by one
  room) is invisible to the lower tier. The robot dutifully
  navigates to the wrong place and only `verify_arrival` (CLIP
  cosine similarity at the goal) catches it — *after* the
  expensive nav leg.
- The original CLIP idea was to close that gap with embedding-drift
  monitoring during transit: as the robot moves, CLIP embeddings of
  the live camera should trend more "kitchen-like" than
  "living-room-like" if the goal is the kitchen. A divergence
  pattern flags an off-course leg early and triggers re-planning or
  a stop-and-re-ground.

### What's already in the repo

The "mid-mission validation" idea is **already partially scaffolded**.
The audit phase needs to inventory exactly how built-out it is and
how much of it actually runs end-to-end:

- [`source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
  — OpenCLIP ViT-B/32 ONNX wrapper (visual + text, 512-dim, 224×224).
- [`semantic_map/manager.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — NetworkX graph + ChromaDB vector store; entry point is
  `add_observation` / `query_by_embedding` / `query_by_text`.
- [`semantic_map/background_mapper.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/background_mapper.py)
  — background tick that ingests live camera frames + detections
  into the map.
- [`semantic_map/transit_monitor.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py)
  — `TransitMonitor` class that "tracks nearest-neighbor region
  drift during navigation" using the semantic map's ANN index.
  Activated when `navigate_to_pose` starts, classifies the robot's
  current view, reports divergence if top-k matches consistently
  fall outside the goal region. **This is the closest thing to the
  "kitchen vs. living-room" idea the user described.**
- [`source/strafer_lab/scripts/finetune_clip.py`](../../../source/strafer_lab/scripts/finetune_clip.py)
  — OpenCLIP ViT-B/32 contrastive fine-tune; exports
  `clip_visual.onnx` + `clip_text.onnx` for the Jetson semantic
  map.
- [`source/strafer_lab/scripts/prepare_vlm_finetune_data.py`](../../../source/strafer_lab/scripts/prepare_vlm_finetune_data.py)
  — produces the CLIP CSV + VLM SFT JSONL from harness-captured
  frames. CLIP gets `(image, text)` positive pairs from
  `dataset_export.export_clip_csv`.
- [`semantic_map/models.py`](../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py)
  — `SemanticNode`, `SemanticEdge`, `DetectedObjectEntry`, `Pose2D`.
- The data path: Infinigen scene gen
  → `extract_scene_metadata.py` (per-scene `scene_metadata.json`)
  → `generate_scenes_metadata.py` (combined `scenes_metadata.json`
  with `floor_top_z`)
  → harness capture (`run_sim_in_the_loop.py --mode harness` writes
  `frames.jsonl` + `frame_*.jpg` per episode)
  → `prepare_vlm_finetune_data.py` (CLIP CSV + VLM JSONL)
  → `finetune_clip.py` (ONNX exports). Stage 6 of
  [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
  walks this end-to-end.

`verify_arrival` (CLIP at the goal pose) is wired into
[`executor/mission_runner.py`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
around line 1302 (`_verify_arrival`).

### The skepticism the brief needs to test

The user's stated worry: **embedding-drift monitoring may be too
blunt to be useful in practice.** Frame-to-frame CLIP cosine on a
mecanum-driven first-person camera is high-variance; small
viewpoint changes can flip the top-k retrieval; the
"kitchen-trending" gradient may not emerge cleanly until the robot
is already 80 % of the way there. If that's true, a higher-rate but
shallower validator (CLIP) plus a slower, deeper one (VLM at
arrival) is just two layers of the same blunt instrument.

Plausible alternatives to weigh:

1. **Better CLIP usage.** Different aggregation (rolling window,
   per-room median rather than top-1), different fine-tune target
   (room classification rather than open-vocab grounding),
   different deployment (TRT-EP at higher rate). Cheap to try; may
   give enough signal.
2. **Small learned validator** (fits Jetson-Orin budget). A ~50 M
   param vision-only model trained explicitly on
   "is-this-trajectory-on-course-toward-target?" labelled tuples
   from the harness data path. Could be a tiny ViT, or a head on
   top of frozen CLIP features. Trains in `strafer_lab` envs;
   inference fits in the bridge's spare cycles.
3. **Small VLA.** A vision-language-action model that emits a
   "stay-on-course / stop / re-ground" signal at 5-10 Hz given
   recent frames + the mission text. Heavier compute footprint;
   harder to fit on Orin Nano. State-of-the-art examples (RT-2,
   OpenVLA, OctoVLA, Pi0) need to be sized against this stack.
4. **No new model, smarter scheduling.** Run VLM `/ground` or
   `/describe` at lower rate but mid-mission (say every 5 s)
   instead of only at the boundaries. Trades latency for compute.
5. **Combination.** CLIP transit-monitor as the cheap/fast
   tripwire; small VLA or VLM `/describe` as the reflective check
   when the tripwire fires.

### Where the deliverable lands

Output is **a write-up + follow-up briefs**, not code:

- A new design doc, recommended location
  `docs/MISSION_VALIDATION_ARCHITECTURE.md`, that captures the
  three sections the user named (current state, limitations,
  alternatives) and ends with a recommendation. This is the
  durable artifact reviewers will revisit.
- One or more follow-up `docs/tasks/active/<slug>.md` briefs filed
  in the same PR for whichever direction the recommendation lands
  on (e.g., `clip-transit-monitor-evaluation.md` if the
  recommendation is "instrument what we have and measure", or
  `learned-mid-mission-validator.md` if the recommendation is "ship
  a small validator model").

The investigation does **not** ship code. Any prototype the audit
needs to run (e.g., to measure CLIP retrieval consistency on
harness-captured frames) goes in a scratch notebook or a one-off
script that doesn't get merged.

## Acceptance criteria

- [ ] **Section 1 — Current state audit** of CLIP in this repo, with
      file paths and line numbers, covers:
  - Where CLIP encodes (`semantic_map/clip_encoder.py`).
  - Where embeddings are stored / queried (the ChromaDB collection
    in `semantic_map/manager.py`).
  - Where CLIP is consumed at runtime (the `verify_arrival` skill,
    the `TransitMonitor`, the `BackgroundMapper`).
  - The training pipeline (`finetune_clip.py`) — what loss, what
    pairs, what hyperparams, what artifacts come out.
  - The data-collection pipeline that feeds it
    (`extract_scene_metadata` → harness capture →
    `prepare_vlm_finetune_data` → CSV/JSONL → `finetune_clip`),
    including which inputs come from sim metadata vs. VLM
    description vs. human label.
  - The infrastructure that sits behind it on the DGX side
    (Infinigen scene gen, `strafer_lab` perception env, MLflow
    tracking).
- [ ] **Section 2 — Limitations analysis** of the current setup,
      anchored in measurable observations rather than intuition. At
      minimum:
  - Round-trip cost of the high tier (planner + grounding) measured
    end-to-end against a representative mission, so the
    "verification gap" is quantified rather than asserted.
  - Frame-to-frame CLIP cosine variance on a real harness-captured
    episode (or a representative subset). Does the
    "kitchen-trending" signal actually emerge in practice, or is
    it lost in viewpoint noise?
  - The known categorical limits of CLIP for this use case
    (open-vocab vs. closed-set room classes; sim-to-real domain
    gap; the perception camera's 640×360 res vs. CLIP's 224×224
    input; aggregation granularity).
  - What the current `TransitMonitor` actually catches and what it
    misses on the same episode subset.
  - The bad-grounding failure mode the user named: what fraction
    of missions in the harness output set would have benefited
    from mid-mission cancellation, vs. continuing to
    `verify_arrival` and failing there.
- [ ] **Section 3 — Alternatives survey** covering the five
      directions in the Context above (better CLIP, small learned
      validator, small VLA, smarter VLM scheduling, combination), at
      a level a reviewer can use to make a build-or-defer decision.
      Each direction names:
  - The state-of-the-art reference(s) — papers, repos, or
    deployed systems — at the time of writing. Cite, don't gesture.
  - Compute cost on Jetson-Orin Nano (rough latency / VRAM /
    throughput). Anchor against the existing budget on
    [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
    and the bridge perf reference numbers in
    [`bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md#phase-level-profiler---profile).
  - Training data requirements — does the current Infinigen +
    harness + sim-metadata + VLM/human-label pipeline already
    produce what's needed, or is new infra required?
  - Sim-to-real risk specific to this approach.
- [ ] **Section 4 — Recommendation** picks one direction (or a
      sequenced combination) with **falsifiable success criteria**
      a future brief can be evaluated against. "Try a small VLA"
      is not a recommendation; "Train a 50 M-param vision-only
      validator on (frame, mission-text, on-course?) tuples;
      target ≥ 90 % on-course recall at 10 Hz Jetson inference"
      is.
- [ ] **Follow-up briefs filed** for whatever the recommendation
      proposes, using the format in [`README.md`](../README.md).
      File them under `docs/tasks/active/` with rows in
      `BOARD.md`'s appropriate priority/lane bucket, in the same
      PR that ships this brief.
- [ ] **The investigation itself doesn't merge code.** If a
      measurement requires a one-off script to run against existing
      harness output, the script lives in a scratch dir
      (`/tmp/...` or similar) and only its measured numbers come
      into the write-up. Production code that comes out of the
      recommendation goes through its own brief → branch → PR.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Existing architecture motivation lives in
  [`docs/STRAFER_AUTONOMY_NEXT.md`](../../STRAFER_AUTONOMY_NEXT.md)
  (current-round design master).
- Existing system flows in
  [`docs/SYSTEM_FLOW_DIAGRAMS.md`](../../SYSTEM_FLOW_DIAGRAMS.md) —
  Flow 5 (sim-in-the-loop) is the bridge harness; Flow 6 is
  real-robot execution. The verification gap lives inside Flow 6's
  `navigate_to_pose` skill.
- The harness output for measurement-anchored claims:
  `data/sim_in_the_loop/<scene_name>/episode_NNNN/frames.jsonl` +
  `frame_*.jpg`. Already produced as part of Stage 5/6 of the
  integration runbook.
- VLM grounding cost numbers live in the "Deferred / known
  limitations" section of
  [`source/strafer_vlm/README.md`](../../../source/strafer_vlm/README.md).
- Jetson compute budget: the existing TRT-EP CLIP path lives in
  `clip_encoder.py`; the inference benchmark scaffold is
  [`Scripts/benchmark_policy.py`](../../../Scripts/benchmark_policy.py)
  (extends to ONNX execution-provider preference lists).
- For state-of-the-art, the user explicitly named small VLAs as a
  candidate; the survey should at minimum cover: OpenVLA, Octo /
  OctoVLA, Pi0, RT-2, NaVid / NaviLLM (navigation-flavored
  variants), and any recent "image-grounded mission monitor"
  literature (search terms: "embodied progress estimation",
  "language-conditioned navigation success monitor",
  "vision-language affordance critic").

## Out of scope

- **Implementing any chosen direction.** This brief stops at the
  recommendation + follow-up briefs. The follow-up briefs are
  where implementation work happens.
- **Refactoring the current `semantic_map/` package.** The audit
  is read-only on the existing code. If the recommendation is
  "instrument what we have," the instrumentation lives in a
  follow-up brief.
- **Changing the planner / VLM contract.** The planner ↔ VLM ↔
  executor schemas in
  [`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../source/strafer_autonomy/strafer_autonomy/schemas/)
  are stable boundaries; this brief doesn't touch them.
- **Real-robot data collection.** Measurement-anchored claims
  draw from harness output, which is sim-side. A future brief
  may add real-robot validation but not this one.
- **Long-horizon planning improvements.** Mission-validation
  (mid-mission self-correction) is distinct from the planner's
  ability to compose far-target plans
  ([`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md)).
  Don't conflate them.
