# Add a gamepad teleop driver to the harness for behavior-cloning data capture

**Type:** new feature
**Owner:** DGX agent (in-process Isaac Lab entry point; no Jetson
code; reuses `collect_demos.py`'s gamepad mapping)
**Priority:** P1 (unblocks v1 measurement *and* v2 training data
without waiting on bridge perf, MPPI tuning, or Nav2 path quirks
to settle)
**Estimate:** M (~half-week — gamepad code reuse + writer wiring +
mission-queue loader + UX polish + acceptance run)
**Branch:** task/harness-teleop-driver

## Story

As an **operator who needs behavior-cloning-grade trajectories
without depending on a partly-broken MPPI / Nav2 / planner
stack**, I want **a gamepad-driven teleop entry point that runs
in-process Isaac Lab (no ROS, no bridge), captures the same
per-tick `(frame, depth, pose, cmd_vel, mission_text)` schema
the bridge harness produces, and lets me tag episodes with
explicit success / failure / wrong-instance / wrong-room /
wrong-path-shape outcomes**, so that **v1 measurement and v2
VLA training data are unblocked from the bridge stack's current
flakiness, and the data path matches how every published
wheeled VLA is actually trained**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)
  — for context on what the bridge does and why teleop deliberately
  bypasses it

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md),
§3.6 (data-path options for VLA training; teleop is the primary
recommendation).

Sibling brief — schema definition:
[`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md).
This brief reuses that schema verbatim; the only difference is
the driver.

Downstream consumers:
- [`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md)
  (relaxes its hard `Blocked on: next-integration-round` once
  teleop output exists).
- [`learned-mid-mission-validator`](learned-mid-mission-validator.md)
  (teleop's wrong-instance / wrong-room / wrong-path-shape
  buttons are the cleanest hard-negative source).
- [`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md)
  (training data; teleop is the default per §3.6).

## Context

### Why teleop, not the bridge

The bridge harness (`run_sim_in_the_loop.py --mode harness`) was
designed for **end-to-end validation of the v1 stack against
simulated sensors**, not bulk training-data capture. Three real
costs of using it as the training-data factory:

- **Throughput.** Bridge mainloop caps at ~15 FPS headless and
  drops to ~6 FPS when planner + VLM are in the loop (per
  [`bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md#phase-level-profiler---profile)).
  A 30 s mission produces ~180 frames at best.
- **Reliability.** Mission outcome depends on Nav2's local planner
  (currently funky in Infinigen scenes,
  [`nav2-startup-unknown-donut-path-noise`](nav2-startup-unknown-donut-path-noise.md))
  and MPPI's controller (plateaued, motivating the
  [`strafer-inference-package`](strafer-inference-package.md)
  rebuild). Many missions fail for stack reasons unrelated to
  what the training data should reflect.
- **Distribution.** Bridge harness output reflects the v1 stack's
  decision distribution. For VLA training, the canonical literature
  paradigm (RT-2, OpenVLA, π0, NaVid) is human-teleop demos, not
  autonomy-stack imitation.

The bridge harness stays useful as the **validation loop** — it's
how we verify the v1 stack still composes end-to-end. It's the
wrong tool for capturing 10k trajectories.

### What this brief changes

A new entry point at `Scripts/teleop_collect.py` that:

- Loads an Infinigen scene via Isaac Lab's standard env-loading
  path, in-process, no bridge, no ROS.
- Reads gamepad input via the already-tuned mapping in
  [`collect_demos.py`](../../../source/strafer_lab/scripts/collect_demos.py).
  Left stick → `(vx, vy)`; right stick → `ωz`; triggers as
  speed modulators.
- Writes the same schema
  [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
  defines: `frames_tick.jsonl`, `actions.jsonl`, `mission.json`,
  `frame_*.jpg`, `depth_*.png` per episode. The `actions.jsonl`
  rows are tagged `source: "teleop"`.
- Exposes a console + on-screen mission text + episode timer +
  distance-to-target indicator.
- Listens for episode-boundary buttons (see below).
- Persists a `teleop_meta.json` per session: operator handle (env
  var `STRAFER_OPERATOR`), gamepad model, `git rev-parse HEAD`,
  capture wall-clock start, scene seed.

### Mission text — three sources, all first-class

| Source | Mechanism | Volume per scene | Path-shape capable? |
|---|---|---|---|
| **Auto-generated endpoint queue** (default) | A new `tools/build_mission_queue.py` walks `scene_metadata.json`'s `objects[]` + `rooms[]`, emits one mission per object + per room, optionally paraphrased via the existing 7B Qwen2.5-VL pipeline ([`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py) Stage 2). Output: `mission_queue.yaml`. | Hundreds of variants | No — endpoints only |
| **Operator-typed (stdin override)** | Operator presses `X` to swap from queue mode to typed mode; types the mission on stdin; presses Enter; episode begins. Used for path-shape demonstrations like "go to the kitchen by hugging the left wall" or "approach the chair from behind the table." | Operator-throughput-bottlenecked | **Yes — the only v1 source for path-shape data** |
| **Procedural path-shape generator** *(future brief, not v1)* | A scene-graph annotator walks objects + rooms + wall geometries, computes A* candidate paths, annotates with descriptors ("passes near wall W," "stays in room R until step N," "via doorway D"), synthesizes path-shape mission text from the annotations. | Volume; minutes per scene after generator built | Yes |

The procedural generator is staked out in
[`harness-procedural-path-shape-generator`](harness-procedural-path-shape-generator.md)
and only fires when path-shape volume becomes the bottleneck.
This brief ships sources 1 + 2.

### Episode-end button mapping

The operator's per-episode loop ends with one of:

| Button | Outcome tag in `mission.json.outcome` | Episode kept? |
|---|---|---|
| `Y` (triangle) | `succeeded` | Yes |
| `B` (circle) | `failed` (gave up / got stuck / Nav2-style) | Yes |
| `X` (square) | `wrong_instance` or `wrong_room` (operator chooses via secondary D-pad press) — operator deliberately drove to a labelled-failure target | Yes (tagged hard negative) |
| `SELECT` (share) | `trajectory_violation` — operator demonstrated a *path-shape* failure (reached the right endpoint but violated the path constraint, e.g., commanded "hug the wall" but cut through the room center) | Yes (tagged hard negative for case 3) |
| `Back` | discarded | No |

The `X` and `SELECT` buttons are the cleanest hard-negative
source in the project — the operator commits to the specific
failure mode at capture time, no post-hoc inference needed. The
harness brief's `--inject-bad-grounding` flag remains for
bridge-driver hard negatives.

### Operator UX during driving

- **Live view:** Isaac Sim editor viewport, headed (not
  headless). The robot is visible third-person; first-person
  D555 view available via Isaac Lab's camera-toggle.
- **Mission text:** printed to console at episode start; on-screen
  overlay if cheap to add via Isaac Lab's text-render path.
- **Episode timer + distance-to-target:** printed to console once
  per second.
- **Mission queue position:** printed at episode advance ("Mission
  7/40").
- **Replay mode:** **out of scope for v1**; record gamepad events
  alongside `actions.jsonl` so a future replay-with-perturbation
  tool can reconstruct the trajectory under different lighting /
  texture seeds. Building the replay tool itself is a future
  brief.

### Realistic data-volume budget

Honest numbers operators reading this brief should plan against,
**single operator, single session**:

| Episode type | Time per episode | Episodes / hour |
|---|---|---|
| Auto-queue endpoint mission, success | ~1 min (5s read + 30s drive + 25s reset) | ~60 |
| Operator-typed path-shape mission | ~2 min (20s type + 50s drive + 50s reset/think) | ~30 |
| `X`-tagged hard negative (wrong instance / room) | ~1.5 min | ~40 |
| `SELECT`-tagged path-shape negative | ~2 min | ~30 |

Per-target volume the brief expects to support:

| Training target | Volume needed | Estimated operator time |
|---|---|---|
| CLIP fine-tune, 1 scene | ~1k frames (~30 episodes) | ~30 min |
| VLM SFT, 1 scene | ~5k frames (~50 episodes) | ~50 min |
| Learned-validator (frozen-DINOv2 + temporal head), 1 scene | ~5k labeled windows (~30 success + 30 hard-negative) | ~75 min |
| v2 VLA endpoint missions (Octo / TinyVLA frozen-tower fine-tune), 3 scenes total | ~10k trajectories with hindsight relabel + replay-perturbation multipliers (~5×) ⇒ ~2k actual demos | ~30–40 hours |
| v2 VLA path-shape missions (per constraint type) | ~1k path-shape demos | ~30 hours per constraint type — **bottleneck for path-shape; motivates the procedural generator** |

The v1 measurement corpus (3 scenes × 30 success + 30
hard-negative) is one operator-evening of work. The v2 VLA
corpus is a multi-week commitment — the brief expects this and
flags it openly.

## Acceptance criteria

- [ ] **In-process entry point.** `Scripts/teleop_collect.py`
      loads an Infinigen scene via Isaac Lab's standard env-load
      path with no ROS / bridge dependency. Confirmed by
      `nethogs` / `ros2 topic list` showing zero ROS traffic
      during a teleop session.
- [ ] **Gamepad mapping reuse.**
      `source/strafer_lab/strafer_lab/sim/teleop_driver.py`
      imports the gamepad → `cmd_vel` mapping from
      [`collect_demos.py`](../../../source/strafer_lab/scripts/collect_demos.py)
      verbatim (or via a shared module if light refactoring is
      needed). No re-tuning of axis mappings.
- [ ] **Output schema parity.** Per-episode output matches
      [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)'s
      schema bit-for-bit. A unit test loads a single episode and
      confirms `frames_tick.jsonl`, `actions.jsonl`,
      `mission.json`, JPEG, and PNG paths conform.
- [ ] **Three mission-text sources.** Auto-queue loader walks
      `scene_metadata.json` and emits a YAML queue;
      operator stdin override switches to typed mode and back;
      the procedural generator's hook is a no-op call to a
      future module (the generator itself is filed separately).
- [ ] **Episode-end buttons** map to `mission.json.outcome` per
      the table above. `Back` discards the episode without
      writing.
- [ ] **`teleop_meta.json` per session** with operator handle,
      gamepad model, repo SHA, capture wall-clock, scene seed.
- [ ] **Acceptance run.** Capture ≥ 30 missions in ≥ 2 Infinigen
      scenes, with ≥ 5 hard-negative episodes per scene
      (`X`-tagged) and ≥ 1 path-shape demo per scene
      (operator-typed, ending with either `Y` or `SELECT`).
      Resulting tree committed under
      `docs/artifacts/teleop_acceptance/<run_id>/` (a small
      summary; full data lives gitignored under
      `data/teleop/`).
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      gains a **Stage 5b — Teleop data collection** section
      between Stages 5 and 6, covering: prerequisites (scene
      generated, mission queue built), invocation
      (`Scripts/teleop_collect.py`), button mapping, output
      schema reference, troubleshooting (gamepad not
      detected, scene fails to load, button-tag mismatch).
      [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `teleop_collect.py` + `build_mission_queue.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the bridge harness — `--mode bridge` and
      `--mode harness` continue to work unchanged. Smoke this in
      the PR description.

## Investigation pointers

- Existing gamepad code:
  [`source/strafer_lab/scripts/collect_demos.py`](../../../source/strafer_lab/scripts/collect_demos.py).
  The brief reuses its mapping; pgylib detection (`pygame.joystick`)
  may need to be wrapped if Isaac Lab swallows pygame events
  during in-process teleop.
- Isaac Lab's native teleop entry: there are existing
  `teleop_se3` / `Se2Gamepad` interfaces in `isaaclab.devices`
  worth checking — strafer's chassis is mecanum holonomic
  `(vx, vy, ωz)`, so a custom mapping is likely needed regardless,
  but the device class scaffolding may be reusable.
- Mission queue YAML schema: model after the existing
  `scene_metadata.json` consumer pattern in
  [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 1 (programmatic walk) so paraphrase generation can
  share the loader.
- Distance-to-target: compute via straight-line `target_position_3d
  - robot_pos` for v1; A*-on-navigable-mask is over-scoped here
  (lives in the harness brief's `compute_progress.py`
  post-processor).
- The Foxglove visualization referenced in
  [`vlm-bbox-overlay`](vlm-bbox-overlay.md) is *not* needed for
  teleop UX; the Isaac Sim editor viewport is sufficient for v1.

## Out of scope

- **Multi-operator collaborative teleop.** One operator at a
  time. Future brief if needed.
- **Networked teleop** (operator on one machine, sim on another).
  Out of scope; the bridge already covers cross-host operation
  for the v1 stack.
- **Real-robot teleop.** Sim-only. A future brief layers in
  real-robot teleop demos for sim-to-real transfer.
- **Replay-with-perturbation** (record once, replay across
  lighting / texture seeds for data multiplication). Recording
  the gamepad event stream is in scope; the *replay tool* is
  a future brief.
- **Procedural path-shape mission generation.** Filed as
  [`harness-procedural-path-shape-generator`](harness-procedural-path-shape-generator.md)
  for when path-shape volume becomes the bottleneck.
- **Voice mission entry, AR / VR teleop, multi-camera live view.**
  All over-scoped for v1.
- **The `actions.jsonl` writer plumbing** itself.
  [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
  ships the writer; this brief just emits actions through it
  with `source: "teleop"`.
- **Action-tokenization / VLA-side dataset format conversion.**
  The teleop brief writes the canonical JSONL schema; the
  HDF5 / RLDS exporter from the harness brief handles
  downstream-pipeline format conversion.
