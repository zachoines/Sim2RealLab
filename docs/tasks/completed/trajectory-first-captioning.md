# Trajectory-first captioning: post-hoc speaker-model data augmentation

**Status:** Retired 2026-05-24. Never picked up. Reason: folded
into the consolidated
[`harness-architecture`](../active/harness/harness-architecture.md)
brief as the
[Scripted × captioner](../active/harness/harness-architecture.md#scripted--captioner-trajectory-first-path)
cell of the driver × mission-source matrix. The 2026-05-24 audit
re-framed this brief's "trajectory-first regime" as a mission
source (post-hoc speaker labels) rather than as a separate
driver — it shares the scripted driver with the oracle path,
differing only in how missions/labels are assigned. The
Speaker-Follower context, instructive-voice rubric (the four
binary checks), failure-pair synthesis logic, and captioner
VRAM budget are preserved verbatim in the consolidated brief.

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
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Parent design doc:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.d](../../../MISSION_VALIDATION_ARCHITECTURE.md#36d-trajectory-first-captioning-complementary-regime)
— this brief ships the regime; the arch-doc subsection names
the architectural relationship to teleop, bridge, and oracle.

Sibling briefs:
- [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md) —
  defines the canonical schema this brief emits; its
  "Hindsight relabel pass" item is now scoped narrowly to
  wrong-target relabeling and points here for the broader
  pattern.
- [`mission-generator`](mission-generator.md) —
  the *forward-generation* counterpart (mission text → driver
  → trajectory). This brief is the *post-hoc* counterpart
  (driver → trajectory → mission text).
- [`scene-connectivity-validation`](../multi-room/scene-connectivity-validation.md) —
  produces the connectivity graph used to filter random
  targets to reachable pairs.
- All three drivers ([`teleop-driver`](teleop-driver.md),
  [`oracle-driver`](../../parked/harness/oracle-driver.md), bridge
  driver in
  [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md))
  — any of them can run in trajectory-first mode and feed this
  brief's captioner.
- [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md) —
  filed alongside this brief by the audit; the parallel-env
  count this brief plans against is asserted, not measured.
  Measure before scale-out.

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
  ([`mission-generator`](mission-generator.md)
  has to reason about scene topology and may annotate
  landmarks the camera never sees — the audit also added a
  start-frame VLM grounding pass there to close part of this
  gap from the forward-generation side) goes away.
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

This gives the same five-way label distribution the cascade
improvements brief
([`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md))
expects, without needing operator-tagged failures.
Trajectory-first becomes a strong source of hard negatives at
scale. (The previously-named small-learned-validator brief was
retired; see
[`completed/learned-mid-mission-validator`](../../completed/learned-mid-mission-validator.md).)

### Output schema

Same canonical schema as
[`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md)
(or whatever
[`output-format-alignment`](../../parked/harness/output-format-alignment.md)
lands on);
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

### Throughput sanity (and the asterisk)

Reachable random A→B sampling on a multi-room scene + RL
controller execution → ~30 trajectories / hour per env.

**Parallel-env claim, asterisked.** Earlier drafts of this
brief asserted "Isaac Lab supports thousands; budget 64 minimum
on the DGX." That asserts a number Isaac Lab can support
**in general**, not what the
`Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0` scene
config supports specifically. The Infinigen Perception scene
deliberately disables `replicate_physics` and carries a 640×360
perception camera; the scene config's own docstring
(`strafer_env_cfg.py:335-337`) flags this scene as capping at
**~1–8 parallel envs** versus the 256+ envs the
80×60-policy-cam variants run. Trajectory-first captioning
*needs* the perception cam to caption, so the assertion of
64 envs without measurement is unsafe.

Two paths the brief can take, decided at pickup time once
[`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md)
measures real numbers:

- **Single-config path:** accept 1–8 envs on the perception
  scene, scale `trajectories / hour` accordingly (~30–240 /
  hour, not ~2k), spread captioning over hours-to-days. Still
  faster than teleop's 30 ep/hr per operator at the high end.
- **Two-pass path:** run the *driver* at high parallelism on
  the NoCam config (`Isaac-Strafer-Nav-Real-NoCam-Play-v0`) to
  generate reachable A→B trajectories cheaply; *replay* each
  trajectory under the perception scene afterwards to harvest
  camera frames. Captioner pass is unchanged. This trades
  perception-scene memory pressure for a replay step;
  feasibility depends on deterministic replay under the
  perception config.

Realistic single-day output is therefore **bounded by
measurement, not by the 2k/hr assertion**. The captioner pass
itself: 7B Qwen2.5-VL at ~3 s / trajectory; with the realistic
~240 trajectories / hour (8 envs, single-config) that's ~12
minutes of GPU time per hour of driving. Cacheable.

### Speaker-instructive evaluation rubric

Section "the critical engineering issue" above commits to an
instructive-quality eval at < 10% failure rate. Define the
rubric explicitly so future operators don't have to guess:

- **Sample.** 50 captions drawn uniformly at random from a
  multi-scene first run (≥ 2 scenes, ≥ 25 per scene).
- **Rubric.** Each caption is scored against four binary
  checks; a caption *fails* if any is violated.
  1. **Voice.** Imperative ("Go to the chair") not declarative
     ("The robot went to the chair").
  2. **Tense.** Future / present, not past.
  3. **Perspective.** Second-person addressing the robot, not
     third-person describing it.
  4. **Groundedness.** The named target is visible in at least
     one frame of the captioned trajectory (per the FoV-honest
     claim).
- **Inspectors.** Two human inspectors score independently;
  disagreements are adjudicated to the conservative ("fail")
  call. Inspector identities recorded in the PR for
  reproducibility.
- **Threshold.** Fewer than 5 of 50 captions fail (10%).
  Iterate on the captioner prompt until threshold met, then
  freeze the prompt and document it in the script's docstring.
- **Re-evaluation trigger.** If the captioner prompt or model
  version changes, re-run the rubric before re-batching the
  corpus.

### Captioner VRAM / scheduling budget

The DGX runs concurrent workloads (training, scene generation,
the bridge mainloop). 7B Qwen2.5-VL at FP16 is ~14 GB of
weights; with 32-frame trajectory inputs and a kv-cache
allocated for a few hundred output tokens, total per-call VRAM
sits around ~22 GB. **Implications:**

- Captioner runs as a *batch job, offline*, not interleaved
  with collection. The collection driver writes JSONL +
  frames; the captioner pass consumes them later.
- Captioner job is queued behind any training run; brief PR
  should record the expected wall-clock-to-corpus delay
  (~hours, not minutes).
- Smaller model (Qwen2.5-VL-3B) is an option if VRAM is
  contested. Document the quality trade in the captioner-eval
  rubric outputs.

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
- [ ] **Speaker-instructive eval.** Score 50 captions using the
      four-check rubric in "Speaker-instructive evaluation
      rubric" above (voice, tense, perspective, groundedness),
      two independent inspectors, conservative adjudication.
      Iterate on the prompt until fewer than 5 of 50 fail
      (10%). Document the final prompt in the script's
      docstring + report the per-check failure breakdown in the
      PR.
- [ ] **Failure-pair synthesis.** Negatives sampled from
      same-room same-label (case-2 `wrong_instance`) and
      different-room (case-1 `wrong_room`) objects per
      `scene_metadata.json`. Tagged correctly in
      `mission.json.category_label`. Smoke-test: load a
      captured trajectory, run the captioner, verify a
      positive + a wrong-instance negative + a wrong-room
      negative are produced with consistent
      `trajectory_id` and distinct `mission_id`.
- [ ] **Parallel-env count grounded in measurement.** The
      driver's `num_envs` argument and the resulting trajectories/
      hour are reported against the measurement landed by
      [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md),
      not the legacy 64-env assertion. If the throughput
      brief hasn't shipped yet, smoke at `num_envs=1`, report
      single-env throughput, and file a follow-up to scale up
      once the measurement exists.
- [ ] **Caching.** Captioner outputs cached by
      `(trajectory_id, speaker_model, speaker_seed,
      caption_mode)`. Re-runs against cached input are free;
      cache lives at
      `data/trajectory_first_cache/<scene>/<trajectory_id>.json`.
- [ ] **Schema parity.** `mission.json` rows match the canonical
      schema from
      [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md);
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
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5b/5c gain notes on the trajectory-first regime as
      an alternative to mission-queue-driven runs.
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `trajectory_first_capture.py` + `caption_trajectory.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.
- [ ] No regression on forward-generation drivers
      ([`teleop-driver`](teleop-driver.md),
      [`mission-generator`](mission-generator.md),
      bridge harness): they continue to emit the same schema
      unchanged. Smoke this in the PR description.

## Investigation pointers

- Speaker-Follower paper (Fried et al., NeurIPS 2018) is the
  canonical reference for the instructive-prompt design; its
  Appendix has prompt examples for R2R that translate cleanly
  to indoor wheeled navigation.
- 7B Qwen2.5-VL is already in
  [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 2; reuse the model-loading path. Cached at
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`.
- Random-target sampling: reuse the connectivity graph from
  [`scene-connectivity-validation`](../multi-room/scene-connectivity-validation.md);
  same A* helper for reachability checks.
- The oracle driver brief
  ([`oracle-driver`](../../parked/harness/oracle-driver.md)) covers
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
  [`mission-generator`](mission-generator.md)'s
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
  [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md).
  That item now points here for the broader pattern but stays
  scoped to its narrower wrong-target case for backward
  compatibility.
- **Speaker-model fine-tuning.** This brief uses the deployed
  Qwen2.5-VL-7B with prompt engineering only. Training a
  task-specific speaker is a future brief if prompt-only
  underdelivers on the instructive-quality eval.
