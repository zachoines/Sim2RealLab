# Trajectory-first captioning: post-hoc speaker-model data augmentation

**Type:** new feature
**Owner:** DGX agent (in-process Isaac Lab driver wrapper +
captioner pipeline; no Jetson code)
**Priority:** P2 (corpus multiplier; complementary to teleop /
oracle / bridge regimes; not gating any v1 measurement decision
but unlocks scale for v2 VLA training and case-3 path-shape
coverage)
**Estimate:** M–L (~week+; random-target driver wrapper +
captioner pipeline + speaker-instructive prompt + failure-pair
synthesis + caching)
**Branch:** task/harness-trajectory-first-captioning

## Story

As an **operator who needs to scale the VLA training corpus
beyond what teleop and forward-mission-generation can produce,
and who wants FoV-honest mission-text labels that reflect what
the camera actually saw rather than what scene metadata claims
should have been visible**, I want **a trajectory-first capture
regime where drivers traverse random-but-reachable A→B paths,
and a speaker model post-hoc generates instructive
`mission_text` + paraphrases + synthesized failure-pair
negatives from the captured frames**, so that **the project
gains a third data-collection regime that complements
operator-intent missions, scales near-trivially, naturally
handles egocentric FoV constraints, and reuses the same
canonical schema all other drivers emit**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)

Parent design doc:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.d](../../MISSION_VALIDATION_ARCHITECTURE.md#36d-trajectory-first-captioning-complementary-regime)
— this brief ships the regime; the arch-doc subsection names
the architectural relationship to teleop, bridge, and oracle.

Sibling briefs:
- [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md) —
  defines the canonical schema this brief emits; its
  "Hindsight relabel pass" item is now scoped narrowly to
  wrong-target relabeling and points here for the broader
  pattern.
- [`harness-mission-generator`](harness-mission-generator.md) —
  the *forward-generation* counterpart (mission text → driver
  → trajectory). This brief is the *post-hoc* counterpart
  (driver → trajectory → mission text).
- [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md) —
  produces the connectivity graph used to filter random
  targets to reachable pairs.
- All three drivers ([`harness-teleop-driver`](harness-teleop-driver.md),
  [`harness-oracle-driver`](harness-oracle-driver.md), bridge
  driver in
  [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md))
  — any of them can run in trajectory-first mode and feed this
  brief's captioner.

## Context

### Precedent in the literature

Trajectory-first captioning is a well-established pattern under
several names:

| Pattern | Source | Mechanism |
|---|---|---|
| **Speaker-Follower** | Fried et al., NeurIPS 2018 (canonical VLN data-augmentation paper) | Train a *speaker* model that maps `trajectory → instruction`; use the speaker's outputs as augmented `(instruction, trajectory)` data for the *follower* navigator. |
| **Hindsight Experience Replay (HER)** | Andrychowicz et al., NeurIPS 2017 | Failed RL trajectories get relabeled with the goal that *was* achieved, turning failure into success-for-a-different-goal. |
| **R2R-EnvDrop, EnvBert, NaVid's data path** | VLN follow-ups (2019–2024) | Speaker-style data augmentation on top of human-collected R2R trajectories. |
| **Open X-Embodiment, RT-2 hindsight relabeling, OpenVLA's caption pipeline** | Modern VLA training pipelines | Subsets of trajectories get post-hoc language annotation by a strong VLM as a corpus multiplier. |

This brief implements the pattern for strafer's harness +
canonical schema.

### What this regime gives the project

- **Scale.** Drivers (random A* + RL controller; or biased
  variants) produce thousands of trajectories cheaply. The
  captioner runs offline on the captured frames in batched VLM
  passes — no LLM call per mission at *generation* time.
- **FoV-honest labels by construction.** The captioner sees
  the *frames the camera actually saw*. It can only annotate
  landmarks that are visible in-frame because that's the only
  signal it has. The egocentric-FoV problem that breaks
  forward-generation landmark annotation
  ([`harness-mission-generator`](harness-mission-generator.md)
  has to reason about scene topology and may annotate
  landmarks the camera never sees) goes away.
- **No LLM-waypoint hallucination.** Paths are real; the LLM's
  job is captioning, not planning. Hallucinations get bounded
  to the language side, recoverable via paraphrase ensembling.
- **Naturally diverse trajectory distribution.** Random
  reachable A→B sweeps cover the scene more uniformly than
  mission-driven paths, which cluster around object-rich rooms.
- **Complementary, not replacement.** Operator teleop captures
  human-intent demos; mission-generator captures structured
  forward-generation; trajectory-first captures bulk
  scene-coverage. The three flows together produce a richer
  corpus than any one alone.

### What this regime does NOT replace

- **Operator-intentional demos from teleop.** Real deployment
  serves operator commands — *forward-directed*, *intentional*.
  Teleop captures that distribution; trajectory-first captures
  "trajectory that happened to reach X," which is structurally
  different. Both are needed.
- **Operator-tagged hard negatives.** Teleop's `X` and
  `SELECT` buttons capture specific failure modes
  (wrong-instance, wrong-room, wrong-path-shape) with operator
  intent attached. Trajectory-first has no operator intent —
  failure pairs have to be *synthesized* (caption a
  successful-to-A trajectory with a deliberately mismatched
  goal-B intent). See "Failure-pair synthesis" below.
- **Path-shape demos.** Random A*-shortest paths don't hug
  walls. Path-shape data still needs operator-typed teleop or
  a path-shape-biased oracle. The captioner can describe
  whatever path-shape is *demonstrated*; it can't invent
  path-shape from a non-path-shape trajectory.

### The critical engineering issue: speaker models must generate *instructive* text

This is the gotcha Speaker-Follower's authors flag explicitly.
There's a fundamental difference between:

- **Descriptive caption (wrong for VLA training).** "The robot
  went down the hallway and stopped at the chair."
- **Instructive caption (right for VLA training).** "Go down
  the hallway and stop at the chair."

If the captioner emits descriptions in past tense /
third-person, the trained VLA learns to *describe* trajectories,
not to *execute* operator intent. The captioner's prompt has to
be carefully constructed for **imperative voice + future-or-present
tense + second-person addressing the robot**.

The captioner's prompt template gets a held-out
"instructive-quality" eval — a small batch (~50 captions) gets
inspected for descriptive leakage. Failure mode: VLA trained on
descriptive captions fails at runtime because operator inputs
("go to the chair") look out-of-distribution.

### Failure-pair synthesis (case-1 / case-2 hard negatives without operator intent)

Trajectory-first naturally produces success-only data: the
trajectory ended somewhere; the captioner says "go to wherever
that was." For training case-1 / case-2 hard negatives, the
captioner runs in two modes per trajectory:

- **Positive caption.** "Go to the [actual_end_object]" — the
  trajectory satisfies this instruction.
- **Negative caption.** "Go to the [scene-metadata-sampled
  alternate object]" — deliberately mismatched. Sample the
  alternate from same-room same-label objects (case-2
  wrong-instance) or different-room objects (case-1
  wrong-room). Tag the resulting `(trajectory,
  negative_caption)` tuple as `category: wrong_instance` or
  `wrong_room` in `mission.json`.

This gives the same five-way label distribution the validator
training pipeline expects (per
[`learned-mid-mission-validator`](learned-mid-mission-validator.md)),
without needing operator-tagged failures. Trajectory-first
becomes a strong source of hard negatives at scale.

### Output schema

Same canonical schema as
[`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md);
each captioned trajectory produces *multiple* `mission.json`
variants per trajectory (positive + N negatives + paraphrases):

```yaml
# mission.json (positive)
mission_id: traj_0123_pos
trajectory_id: traj_0123
mission_text: "Go to the green couch in the living room."
paraphrases: ["Approach the green couch in the living room.", ...]
mission_state: succeeded
category_label: on_course
generator_metadata:
  source: "trajectory-first-captioning"
  speaker_model: "Qwen2.5-VL-7B"
  speaker_seed: 42
  caption_mode: "positive"
  end_object_label: "couch"
  end_object_id: "couch_3"

# mission.json (negative — wrong_instance)
mission_id: traj_0123_neg_wi
trajectory_id: traj_0123        # same underlying trajectory
mission_text: "Go to the brown couch in the living room."
paraphrases: ["Approach the brown couch...", ...]
mission_state: failed
category_label: wrong_instance
generator_metadata:
  source: "trajectory-first-captioning"
  caption_mode: "negative"
  intended_object_label: "couch"
  intended_object_id: "couch_5"   # different from actual end
  actual_end_object_id: "couch_3"
```

The trajectory frames + actions + depth are unchanged; the
mission rows are the multipliers. One trajectory typically
becomes 1 positive + 2–3 negatives + N paraphrases per row =
~10–20 training tuples per trajectory.

### Throughput sanity

Reachable random A→B sampling on a multi-room scene + RL
controller execution → ~30 trajectories / hour per env.
Parallel envs (Isaac Lab supports thousands; budget 64 minimum
on the DGX) → ~2k trajectories / hour. Captioner pass: 7B
Qwen2.5-VL at ~3 s / trajectory × 2k trajectories = ~1.5 hours
of GPU time. Cacheable.

Realistic single-day output: ~10k trajectories × ~15 mission
rows each = ~150k training tuples. That's the same order of
magnitude as published wheeled-VLA corpora and roughly matches
what hindsight-augmented R2R looks like.

## Acceptance criteria

- [ ] **Random-target driver wrapper.**
      `Scripts/trajectory_first_capture.py` (DGX-side) wraps
      the oracle driver in a "no-mission, just navigate
      random reachable A→B" mode. Targets sampled from
      `scene_metadata.json`'s navigable mask, filtered by:
      reachable per the connectivity graph, minimum geodesic
      distance > R (default 2 m), trajectory passes ≥ 1 room
      boundary on multi-room scenes when possible.
- [ ] **Captioner pipeline.**
      `source/strafer_lab/strafer_lab/tools/caption_trajectory.py`
      consumes a captured trajectory + scene metadata, calls
      the 7B Qwen2.5-VL with an **instructive-voice prompt**
      (imperative, second-person, future/present tense),
      emits a positive caption + N=2 negative captions
      (sampled hard negatives) + N=3 paraphrases per caption.
- [ ] **Speaker-instructive eval.** Hand-inspect ≥ 50 captions
      from the captioner's first run. Failure rate (descriptive
      vs. instructive) reported in the PR; iterate on prompt
      until < 10% failure rate. Document the final prompt in
      the script's docstring.
- [ ] **Failure-pair synthesis.** Negatives sampled from
      same-room same-label (case-2 `wrong_instance`) and
      different-room (case-1 `wrong_room`) objects per
      `scene_metadata.json`. Tagged correctly in
      `mission.json.category_label`. Smoke-test: load a
      captured trajectory, run the captioner, verify a
      positive + a wrong-instance negative + a wrong-room
      negative are produced with consistent
      `trajectory_id` and distinct `mission_id`.
- [ ] **Caching.** Captioner outputs cached by
      `(trajectory_id, speaker_model, speaker_seed,
      caption_mode)`. Re-runs against cached input are free;
      cache lives at
      `data/trajectory_first_cache/<scene>/<trajectory_id>.json`.
- [ ] **Schema parity.** `mission.json` rows match the canonical
      schema from
      [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md);
      `actions.jsonl` rows tagged
      `source: "oracle"` (or whichever driver actually ran)
      and `generator_metadata.source: "trajectory-first-captioning"`
      so downstream training can filter by either.
- [ ] **End-to-end smoke run.** Capture ≥ 100 random-target
      trajectories on at least 2 multi-room Infinigen scenes.
      Caption pass produces ≥ 100 positive + ≥ 200 negative
      mission rows. Inspect a random subsample of 20 rows for
      caption quality + correct category-label assignment.
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5b/5c gain notes on the trajectory-first regime as
      an alternative to mission-queue-driven runs.
      [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `trajectory_first_capture.py` + `caption_trajectory.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.
- [ ] No regression on forward-generation drivers
      ([`harness-teleop-driver`](harness-teleop-driver.md),
      [`harness-mission-generator`](harness-mission-generator.md),
      bridge harness): they continue to emit the same schema
      unchanged. Smoke this in the PR description.

## Investigation pointers

- Speaker-Follower paper (Fried et al., NeurIPS 2018) is the
  canonical reference for the instructive-prompt design; its
  Appendix has prompt examples for R2R that translate cleanly
  to indoor wheeled navigation.
- 7B Qwen2.5-VL is already in
  [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 2; reuse the model-loading path. Cached at
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`.
- Random-target sampling: reuse the connectivity graph from
  [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md);
  same A* helper for reachability checks.
- The oracle driver brief
  ([`harness-oracle-driver`](harness-oracle-driver.md)) covers
  the parallel-env + RL-controller path-tracking layer this
  brief drives in trajectory-first mode. Coordinate
  `--controller rl` defaults across both briefs.
- Hard-negative sampling rules: same-room-same-label first;
  fall back to different-room-same-label; fall back to
  different-label-same-room. Avoid sampling negatives that are
  the actual end (would invert the label).

## Out of scope

- **Replacing teleop or forward-mission-generation.** This is
  a third complementary regime, not a replacement. All three
  data-flow regimes feed the same downstream training corpus.
- **Path-shape demonstration generation.** Random A*-shortest
  paths don't naturally produce wall-hugging or
  "via the dining room" trajectories. Path-shape data
  continues to come from operator teleop or a
  path-shape-biased oracle (the latter is in
  [`harness-mission-generator`](harness-mission-generator.md)'s
  scope, not this brief's).
- **Fine-grained captioning beyond mission text.** Per-frame
  language captions ("the robot is approaching the doorway")
  are out of scope; the captioner generates *mission-level*
  text, not per-frame narration.
- **Real-robot trajectory captioning.** Sim-only; the speaker
  model needs scene-metadata context that's not available
  real-side. A future brief layers in real-robot captioning.
- **Replacing the existing narrow "Hindsight relabel pass"**
  in
  [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md).
  That item now points here for the broader pattern but stays
  scoped to its narrower wrong-target case for backward
  compatibility.
- **Speaker-model fine-tuning.** This brief uses the deployed
  Qwen2.5-VL-7B with prompt engineering only. Training a
  task-specific speaker is a future brief if prompt-only
  underdelivers on the instructive-quality eval.
