# Enrich ProcRoom generation toward Infinigen-like depth statistics

**Type:** task / investigation (executes a completed design: generator
enrichment + a descriptor instrument that gates every lever)
**Owner:** DGX (`strafer_lab` lane — `mdp/proc_room.py` generator, env
cfgs, scratch descriptor tooling)
**Priority:** P2 — this is the training-lane response to the
provisionally-indicated Infinigen depth domain gap (the 2026-07-16 ad hoc
A/B: ProcRoom tracks paths, Infinigen does not, matched config). It
decides where the next DEPTH retrain's scene distribution comes from.
Design + descriptor phases are pickable now; any training spend is gated
(see Gates).
**Estimate:** M (~2–3 days: descriptor instrument + re-baseline ~1 day,
tier-1 + enclosure stories ~1 day, descriptor validation passes ~half a
day; excludes the retrain itself, which is operator-scheduled)
**Branch:** task/procroom-depth-enrichment

## Story

As a **DGX operator preparing the next DEPTH_SUBGOAL training run after
the Infinigen A/B indicated a scene domain gap**, I want **ProcRoom's
procedural generation enriched so its depth-image statistics move toward
Infinigen-like values, with every enrichment lever validated by a
descriptor measurement on generated scenes before any GPU-time is
spent**, so that **the retrained policy generalizes to Infinigen-class
(and eventually real) rooms instead of overfitting to ProcRoom's
open-top, close-quarters signature — without wrecking the 64-env
training path that makes ProcRoom the only high-parallelism depth
trainer we have**.

## Context bundle

Read these before starting:

- [`env-composition-contract.md`](../../context/env-composition-contract.md)
  — the composition root this work must not structurally extend without a
  consult; leaf-add vs structural-extension is the dividing line.
- [`depth-subgoal-env.md`](depth-subgoal-env.md) — the DEPTH_SUBGOAL
  variant this enrichment retrains; its Phase-5 run is the consumer.
- [`domain-randomization-audit.md`](domain-randomization-audit.md) — owns
  dynamics/sensor DR widening; Q4 below coordinates a boundary with it.
- [`bridge-scene-memory-budget-gb10.md`](../../completed/bridge-scene-memory-budget-gb10.md)
  and [`infinigen-scene-corpus.md`](../../completed/harness/infinigen-scene-corpus.md)
  — the memory constraints that shape the finetune fallback.
- `SUBGOAL_EPIC/PROCROOM_AB_FINDINGS.md` and
  `SUBGOAL_EPIC/INFINIGEN_AB_FINDINGS.md` (workspace, outside the repo) —
  the measured ProcRoom descriptors, the provisional A/B verdict, and the
  pending NX instrumented session.

## Context

The DEPTH_SUBGOAL policy trains exclusively in ProcRoom. At matched
launch config the policy tracks paths in ProcRoom and does not in
Infinigen (provisional: ad hoc, un-instrumented, single leg per scene).
Working hypothesis: a depth-statistics domain gap. The fix direction is
enriching ProcRoom's generator toward Infinigen-like depth statistics,
with Infinigen-finetune-only as the fallback.

**CALIBRATION PENDING: NX-session descriptor deltas.** The quantified
ProcRoom↔Infinigen descriptor deltas arrive with the instrumented paired
session on the Orin NX (`SUBGOAL_EPIC/INFINIGEN_AB_SESSION_PROMPT.md`,
postponed from the Nano). Until then this brief's targets are
*directional*, derived from measured scene geometry (below). When the NX
numbers land, transcribe them into this section, record **which scene USD
the Infinigen leg bound** (default bind is the lightest scene since the
single-scene bind fix — target class changes the D1/D3 targets
materially), and re-rank the stories against the actual deltas before any
training spend.

### Measured baseline (ProcRoom, deployed 80×45 policy camera, meters)

From the 2026-07-09 session's node obs dump (684 frames; obs slice
`[19:3619]`, ×6.0, reshape 45×80 — see `PROCROOM_AB_FINDINGS.md`):

1. **Near-band:** median 1.13 m; 43% of cells ≤ 1 m; 77% ≤ 2 m.
2. **Row profile:** top row ~4.44 m → bottom ~0.55 m; bottom rows a
   stable near-floor band (temporal std 0.07 m); no dead rows.
3. **Top-band far-saturation:** top ~11 rows 42.8% pinned at the 6 m
   clamp.

Provenance caveats the instrument must carry: measured along
policy-driven mission trajectories (not uniform poses), through the
Jetson's 640×360→80×45 downsample (static ~2 cm offset vs the native
policy camera), by a script (`depth_field_stats.py`) that lives only on
the Jetson — the exact thresholds ("top ~11 rows", pin epsilon) must be
re-specified sim-side (Phase 0).

### Current randomization-axis inventory (what the generator can already vary)

`generate_proc_room` runs per episode reset per env (event chain
`randomize_difficulty → generate_room → reset_robot`), teleporting a
fixed 44-object kinematic palette (20 walls, 8 furniture, 16 clutter)
into a fresh layout; a GPU BFS guarantees solvability
([`proc_room.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py)).

| Axis | Today | Ref (`proc_room.py` unless noted) |
|---|---|---|
| Room width/depth | `U[4.0, 7.0]` m, independent, axis-aligned rectangle | :559-560 |
| Difficulty level | **pinned 7/7** (8 furniture + 16 clutter, walls on) — the 0–7 level table exists but is degenerate | `strafer_env_cfg.py:1536-1540`, :536-545 |
| Doorway | exactly 1; side randint(0,4), position U[0,1], width `U[0.8, 1.2]` m | :583-586 |
| Furniture | ≤8 pieces, always wall-flush, yaw wall-locked, 0.5 m min spacing | :667-720 |
| Clutter | ≤16 items, uniform over interior (0.5 m inset), free yaw | :722-758 |
| Robot spawn | uniform over BFS-reachable cells (0.3 m inflation only) | :420-454, `mdp/events.py:494-553` |
| Goal / subgoal path | reachable-set sample, `min_goal_distance=2.0` | `strafer_env_cfg.py:1660-1668` |
| Physics DR | friction / mass / motor strength bands | `strafer_env_cfg.py:1548-1567` |
| D555 mount DR | ±1°/±3° — **IMU observation math only; the rendered camera pose is never perturbed** | `mdp/events.py:450-491` |

Fixed / not randomized (the enrichment surface):

- **No ceiling; walls 1.0 m tall** (`OBJECT_SIZES`, :47-53; z=0.5
  hardcoded). Camera sits ~0.35 m above floor, level, VFOV ≈56.4°: rays
  above the wall silhouette exit to sky → inf → 6.0 m fill. The 42.8%
  top-band pin is **structural**, present every episode.
- **Internal walls are dead code**: the level table promises 1–2 and
  `max_iw` is computed (:548, :553) but never consumed — no internal
  wall is ever placed.
- **Object sizes/shapes fixed**; tallest non-wall object 0.8 m; nothing
  occludes above 1.0 m; all objects floor-standing; identical object
  multiset every episode at pinned difficulty.
- Materials/lighting fixed (irrelevant to depth: `distance_to_image_plane`
  is geometry-only; sim depth noise is content-independent).
- BFS grid caps the world at 8×8 m (`GRID_SIZE=80` × `GRID_RES=0.1`,
  :81-82); env spacing 10 m.

### The gap, directionally (geometric analysis; magnitudes pend NX)

Infinigen scenes are **enclosed** (measured ceiling ≈2.72–2.78 m,
full-height walls), have **53% of objects topping above 1.0 m** (median
top-z 1.05 m, p90 1.89 m — full-height near columns high in the image),
wall-align large furniture leaving the floor center open, and (heavy
corpus only) span 16–22 m with door-chain sightlines. A scratch ray-cast
proxy over the scenes' cached occupancy grids (robot-height slice,
free-cell origins × 48 headings; summarized here, script stays scratch)
puts median first-hit at **0.90 m for a synthetic ProcRoom level-7 room
vs 1.95 m (Infinigen singleroom) / 2.3–2.5 m (heavy corpus)**, with
≤1 m fractions dropping ~2.5–3×. Direction per descriptor:

| Descriptor | ProcRoom today | Expected Infinigen direction | Primary lever |
|---|---|---|---|
| D1 near-band (43% ≤1 m) | too near — uniform clutter scatter in 4–7 m rooms puts obstacles in the robot's face | substantially farther (≤1 m fraction roughly halves) | density widening (T1), spawn clearance (T1), open floor center |
| D2 row profile | monotone floor→far gradient; bottom band stable (flat floor — preserved in Infinigen, keep it) | top/mid reshaped: real surfaces overhead, near columns high in the image, aperture discontinuities | enclosure (F1), tall furniture (F3), internal walls (F2) |
| D3 top-band far-saturation (42.8% pinned) | structural sky-clamp sheet | **collapses toward ~0** in enclosed single-room scenes; partially returns (11–18%, structured) in heavy scenes | enclosure (F1) |
| D4 (new) high-row near fraction | ~0 (nothing above 1 m) | substantial (tall furniture) | F3 |
| D5 (new) edge/gradient density, occlusion frequency | low (few primitives, no apertures) | higher (clutter columns, doorway apertures) | F2, F3, tier 3 if primitives can't reach it |

Constraint framing from the depth pipeline: depth statistics are shaped
only by geometry 0.4–6 m from the camera (below 0.4 m → constant 0.2
fill; ≥6 m, sky, and holes are indistinguishable at 6.0), and the noise
model's variance grows ~z², so far-shifting the distribution also raises
the noise floor the policy sees
([`observations.py:552-617`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py),
[`noise_models.py:458-512`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py)).

### Training-cost frame (the 64-env path must survive)

`_PROCROOM_DEPTH_TRAIN_NUM_ENVS = 64`
([`strafer_env_cfg.py:1163`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py),
consumed at
[`composed_env_cfg.py:354-358`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py))
is the policy-camera-only depth training path; the ~1–8 env cap applies
only to the 640×360 perception camera (bridge/capture, never trains).
Cost budget: **re-ranging existing sampler parameters is free; a handful
of added kinematic primitives per env is cheap** (PhysX buffers are
sized with ~4× headroom at 44 objects ×
up-to-256 envs, `strafer_env_cfg.py:1811-1826`); per-env mesh assets
×64, large palette growth, or `GRID_SIZE` growth are costly/structural.
Reset-time placement logic is amortized (20 s episodes) — placement
enrichment is ~free per step.

## Design questions (must be resolved during the picking-up PR)

### 1. Shared-generator blast radius — depth-only opt-in vs global change

`proc_room.py` also generates the NOCAM (256-env) and NOCAM_SUBGOAL
scenes; the certified NOCAM baseline trains on today's statistics.
Options: **A.** parameterize enrichment (level-table rows / EventTerm
params / palette-builder args), default = today's behavior, enable on
depth cfgs only — NOCAM contract byte-identical. **B.** change globally
and accept a NOCAM distribution shift. Recommendation: A. Either way the
composition-contract golden hash for the depth variants changes — use
the contract's sanctioned snapshot-update process in the pickup PR, and
verify NOCAM cfgs hash-identical under option A.

### 2. Subgoal-for-Infinigen free-space grid (fallback prerequisite only)

The subgoal objective is composed for `"procroom"` only — the tables at
`composed_env_cfg.py:286-311` say extending a source means giving it a
free-space grid first (`SubgoalCommand` reads
`env._proc_room_free_space`). The finetune fallback requires an
Infinigen entry (rasterize the cached `occupancy.npy` seam into a static
per-env grid). That is a **structural extension of the composition
machinery — if the fallback fires, STOP and file a design consult**
(same protocol as the depth-subgoal-env and corridor-margin consults);
do not pre-build it as part of enrichment.

### 3. `GRID_SIZE` growth for longer sightlines

Heavy-corpus scenes have >6 m door-chain sightlines; ProcRoom spans cap
at ~7.7 m inside the 8 m BFS grid. Growing the grid touches a
load-bearing shared constant (the subgoal off-path corridor is sized
from `INFLATION_CELLS*GRID_RES`). Defer unless the NX deltas show the
far-field span (not enclosure) is the operative gap; if pursued, consult
first.

### 4. Rendered-camera extrinsics DR ownership

Mount DR currently never perturbs the rendered image (observation-side
only). Jittering the actual camera prim pose (pitch/height) would move
row-profile variance toward realism, but sensor-DR widening is
[`domain-randomization-audit`](domain-randomization-audit.md)'s lane.
Resolve the boundary with that brief before implementing F4 here.

### 5. Descriptor instrument promotion

Per this brief's filing rules the descriptor scripts stay in scratch
space. If, post-validation, the operator wants them as a durable
pre-training gate (exposure-gate precedent:
`scripts/measure_perception_exposure.py`), that promotion is a separate
decision — flag it, don't fold it in.

## Approach

### Phase 0 — descriptor instrument + re-baseline (before touching the generator)

Two scratch scripts (sketched here; ~150 lines total, zero production
code):

1. **Capture** (`$ISAACLAB -p`, headless, cameras on): loop
   `env.reset()` on the ProcRoom depth task (each reset = one fresh
   generated room + BFS free-space robot pose; 64-env batching makes
   K≈50 rooms minutes of GPU), settle ~10 zero-action steps, then dump
   ~20 steps per pose via
   `strafer_lab.bridge.obs_dump_terms.evaluate_obs_terms` →
   `ObsDumpWriter` PARITY_SCHEMA JSONL. The same schema is what the
   Jetson node dumps and the future NX dumps use — one instrument,
   every leg.
2. **Descriptors** (plain CPU): JSONL → `6.0 × obs[19:3619]` reshaped
   45×80 → D1–D5 above, with explicit thresholds (pin epsilon, band row
   ranges) written down so numbers are comparable across legs.
   Bottom-band "temporal std" is redefined as across-pose std at a
   static-pose harness (the 0.07 m baseline was motion-conditioned —
   state the redefinition next to the number).

Deliverables: re-baselined vanilla-ProcRoom descriptor table under the
harness protocol (the mission-trajectory numbers above are
pose-distribution-shifted; the harness protocol becomes canonical), and
— strongly recommended, cheapest de-risk available before the NX session
— the **same harness run on the Infinigen bridge env** (lightest scene +
tworoom) for provisional sim-frame targets.

### Ranked stories (small → large; each validates via Phase 0 before any training)

**Tier 1 — parameter-range widening (S each; zero step-rate cost):**

- **T1 density widening.** Un-pin difficulty from 7/7 → sample room-mode
  levels (e.g. 4–7) per episode, and/or widen the level table's
  furniture/clutter counts into ranges. Moves: D1 down (fewer in-face
  obstacles), restores episode-to-episode variance in every descriptor.
  Risk note: sparser rooms are *easier* navigation — no reward
  interaction (the depth obstacle penalty ships inert at weight 0).
- **T2 spawn-clearance widening.** Extract spawn points from a
  more-inflated free mask (spawn-only extra margin over the 0.3 m robot
  radius). Moves: D1's spawn-pose bias specifically (Infinigen's
  standable-clearance median is ~0.9–1.3 m vs ProcRoom's ~0.3 m floor).
  Scene statistics untouched.
- **T3 doorway widening.** Width `U[0.8, 1.2]` → `U[0.8, 2.0]` m
  (Infinigen tworoom's opening is 2 m). Moves: D5 aperture structure,
  mildly.
- **T4 room-span nudge.** `U[4, 7]` → `U[4, 7.5]` m (stays inside the
  8 m grid with wall thickness + inflation margin). Moves: mid/far mass,
  marginally. Rider-sized; do not expect much.

**Tier 2 — new generator features (S–M each; cheap prims):**

- **F1 enclosure — the structural lever.** Raise wall height 1.0 →
  ~2.7 m (palette constant; zero new prims; occupancy is XY-only so BFS
  is untouched) and add one ceiling-slab palette slot (~7.6×7.6×0.1 m
  panel teleported to wall-top over room-mode envs; overhang past walls
  is invisible from inside). Moves: D3 toward the enclosed-scene target
  (~0 pin), D2 top rows from a constant-6.0 sheet to real surfaces at
  1.5–4.6 m. **Implementation trap:** the occupancy rasterizer marks
  every active object's XY AABB — the ceiling slab must be excluded
  from the grid (skip slots whose bottom-z clears robot height) or BFS
  reports the room fully blocked. Cost: +1 kinematic prim/env.
- **F2 internal walls + optional second doorway.** Implement the dead
  `max_iw` axis (1–2 interior segments with a guaranteed gap). Slot
  budget note: the 26 m wall-segment budget is nearly exhausted by the
  perimeter of large rooms — either restrict internal walls to smaller
  rooms or add 2–4 dedicated slots. Moves: D5 occlusion/aperture
  frequency, mid-row discontinuities, sightline segmentation.
- **F3 tall furniture.** Re-dimension or add 2–4 shelf/cabinet variants
  at 1.8–2.4 m (Infinigen: 53% of objects top above 1.0 m). Moves: D4
  (near columns high in the image — the spatial near-band structure the
  three scalar descriptors miss), D5 edge density. Keep `OBJECT_SIZES`,
  the palette builder, and slot-range tables in sync (they are
  load-bearing shared constants).
- **F4 rendered-camera extrinsics micro-DR.** Gated on Q4.

**Tier 3 — hybrid ProcRoom + Infinigen-asset injection (L; consult-gated):**

- **H1.** A few real Infinigen meshes as palette slots (×64 per-env
  copies — BVH/memory cost is why this is last). Trigger: NX deltas show
  higher-order statistics (edge/gradient texture) that primitives
  demonstrably cannot reach after F1–F3. Requires a consult (palette
  architecture + memory budget), not a pre-decision.

### Decision rule and the fallback (sketched, not built)

**Gates on any training spend:** (a) the `test_sim` depth-noise
recalibration must land first (it certifies the retrain's test bed);
(b) the NX-session deltas are transcribed into the calibration slot and
the stories re-ranked against them; (c) enriched-generator descriptor
runs demonstrate each story's predicted movement.

**Infinigen-finetune-only fallback.** Shape (from existing machinery):
leaf env variant `SceneSourceCfg(kind="infinigen")` × depth × subgoal
(blocked on Q2's consult), lightest scene(s) only (the heavy 6–11-room
scenes OOM the unified-memory budget for training), `--num_envs 3–8`
(`_INFINIGEN_TRAIN_NUM_ENVS = 3`; the policy-cam ceiling is unmeasured;
co-located envs put ghost robots in each other's depth frames — a
training-only artifact that worsens with count), warm-start
`--resume <ckpt>` with a low-LR cosine schedule, ProcRoom play-variant
regression per checkpoint as the forgetting guard (no anchor machinery
exists). Cost frame: 12–32× fewer samples/iter than the ProcRoom runs
that produced v1; multi-day serial GB10 occupancy.

**Choose the fallback over enrichment iff the NX deltas show the gap is
where enrichment's levers can't reach:**

- the three aggregate descriptors come back ~matching ProcRoom's yet the
  behavioral gap persists (gap lives in higher-order structure a
  primitive palette can't express); or
- Infinigen *destroys* the stable near-floor band (a
  sensor/rendering-interaction effect, not geometry distribution); or
- per-scene Infinigen descriptor variance ≫ the ProcRoom↔Infinigen mean
  delta ("Infinigen-like values" is then ill-posed; per-scene adaptation
  matches deployment better).

Conversely, large clean deltas on D1/D3 (near-band far too high,
saturation sheet absent in the target) are exactly what tiers 1–2 move
at full 64-env parallelism — enrichment dominates. A
watchdog-contamination finding on the NX legs (`stale=['depth']` halts
explaining "not tracking") voids both designs — re-run the A/B first.

## Acceptance criteria

### Instrument

- [ ] Scratch descriptor capture + descriptor scripts exist and run
      headlessly on the DGX; thresholds (pin epsilon, band definitions,
      the temporal→across-pose std redefinition) documented in-script.
- [ ] Vanilla ProcRoom re-baselined under the harness protocol (≥50
      generated rooms), numbers recorded next to the mission-trajectory
      baseline with the pose-distribution caveat.
- [ ] Sim-frame Infinigen descriptor leg captured (lightest + tworoom)
      or explicitly waived in favor of landed NX deltas.

### Generator

- [ ] Tier-1 stories implemented per Q1's resolved option; each shows
      its predicted descriptor movement on generated scenes.
- [ ] F1 enclosure implemented (wall height + ceiling slot, occupancy
      exclusion handled); top-band pin fraction moves to the
      enclosed-scene target on ≥50 generated rooms.
- [ ] The 64-env depth-train cfg constructs and steps; reset-time and
      step-rate within ~10% of the pre-enrichment baseline at
      `num_envs=64`; PhysX buffer overflow warnings absent.
- [ ] Composition-contract snapshot updated via the sanctioned process;
      NOCAM/NOCAM_SUBGOAL cfgs unchanged under Q1 option A.

### Calibration + maintenance

- [ ] "CALIBRATION PENDING" slot filled from the NX session (deltas +
      bound-scene identity) and stories re-ranked against it before any
      retrain is scheduled.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`proc_room.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py)
  — palette :32-76, room sampling :559-560, doorway :583-586, furniture
  :667-720, clutter :722-758, dead `max_iw` :548, occupancy rasterizer
  :227-313, spawn extraction :420-454.
- [`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  — difficulty pin :1536-1540, ProcRoom scene/events :1352-1595, env
  counts :1155-1169, PhysX buffers :1811-1826.
- [`composed_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)
  — scene selection :330-390, subgoal source tables :286-311.
- [`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  :552-617 and
  [`noise_models.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py)
  :458-512 — the clamp/fill/noise pipeline descriptor scripts must
  replicate.
- [`obs_dump_terms.py`](../../../../source/strafer_lab/strafer_lab/bridge/obs_dump_terms.py)
  + [`obs_dump.py`](../../../../source/strafer_lab/strafer_lab/bridge/obs_dump.py)
  — the capture entry points.
- `SUBGOAL_EPIC/INFINIGEN_AB_SESSION_PROMPT.md` — the NX instrument that
  produces the calibration deltas.

## Out of scope

- **The retrain itself.** Operator-scheduled GPU time; this brief ends
  at "levers implemented + descriptor-validated + calibration slot
  filled."
- **The NX migration / instrumented A/B session.** Separate lane; this
  brief consumes its deltas.
- **The depth-noise recalibration.** Its own shipped work; it gates the
  retrain, not this brief's implementation.
- **Building the Infinigen finetune path.** Sketched here as the
  fallback; its subgoal-grid prerequisite is Q2's consult. Don't
  pre-build.
- **Infinigen corpus changes.** Enrichment targets the ProcRoom
  generator; scene-corpus regeneration is the harness lane's.
- **Reactive-avoidance re-enablement.** That's
  [`depth-subgoal-reactive-avoidance`](../../parked/trained-policy/depth-subgoal-reactive-avoidance.md).
  Don't double-tune.
