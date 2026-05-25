# Add a gamepad teleop driver to the harness for behavior-cloning data capture

**Type:** new feature
**Owner:** DGX agent (in-process Isaac Lab entry point; no Jetson
code; reuses `collect_demos.py`'s gamepad mapping)
**Priority:** P1 (unblocks v1 measurement *and* v2 training data
without waiting on bridge perf, MPPI tuning, or Nav2 path quirks
to settle). Soft-blocks on
[`mission-generator`](mission-generator.md) for
the queue source — for an interim version of this brief, a
hand-authored YAML queue is acceptable; the generator can ship
in parallel.
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
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
  — for context on what the bridge does and why teleop deliberately
  bypasses it

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md),
§3.6 (data-path options for VLA training; teleop is the primary
recommendation).

Sibling brief — schema definition:
[`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md).
This brief reuses that schema verbatim; the only difference is
the driver. Note: an audit pass filed
[`output-format-alignment`](../../parked/harness/output-format-alignment.md)
proposing the canonical schema move to LeRobot v2 / Isaac Lab
`RecorderManager`-compatible HDF5; this brief's "emit the canonical
schema" criterion adopts whatever that brief lands on if it ships
first, otherwise it emits the JSONL schema from
[`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md).

Downstream consumers:
- [`validator-evaluation`](../clip-validation/validator-evaluation.md)
  (relaxes its hard `Blocked on: next-integration-round` once
  teleop output exists; teleop's wrong-instance / wrong-room /
  wrong-path-shape buttons are the cleanest hard-negative
  source for the CLIP cascade eval and the co-trained
  validator
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)).
- [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md)
  (training data; teleop is the default per §3.6).

## Context

### Why teleop, not the bridge

The bridge harness (`run_sim_in_the_loop.py --mode harness`) was
designed for **end-to-end validation of the v1 stack against
simulated sensors**, not bulk training-data capture. Three real
costs of using it as the training-data factory:

- **Throughput.** Bridge mainloop caps at ~15 FPS headless and
  drops to ~6 FPS when planner + VLM are in the loop (per
  [`bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md#phase-level-profiler---profile)).
  A 30 s mission produces ~180 frames at best.
- **Reliability.** Mission outcome depends on Nav2's local planner
  (currently funky in Infinigen scenes,
  [`nav2-startup-unknown-donut-path-noise`](../../completed/nav2-startup-unknown-donut-path-noise.md))
  and MPPI's controller (plateaued, motivating the
  [`inference-package`](../../completed/inference-package.md)
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
  [`collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py).
  Left stick → `(vx, vy)`; right stick → `ωz`; triggers as
  speed modulators.
- Writes the same schema
  [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md)
  defines: `frames_tick.jsonl`, `actions.jsonl`, `mission.json`,
  `frame_*.jpg`, `frame_*.depth.npy` per episode. The
  `actions.jsonl` rows are tagged `source: "teleop"`.
- Exposes a console + on-screen mission text + episode timer +
  distance-to-target indicator.
- Listens for episode-boundary buttons (see below).
- Persists a `teleop_meta.json` per session: operator handle (env
  var `STRAFER_OPERATOR`), gamepad model, `git rev-parse HEAD`,
  capture wall-clock start, scene seed.

### Mission text — two sources, both first-class

| Source | Mechanism | Volume per scene | Path-shape capable? |
|---|---|---|---|
| **Mission queue YAML** (default) | The teleop driver loads `mission_queue.yaml` produced by [`mission-generator`](mission-generator.md). Each row contains `mission_text` (+ paraphrases) and optional `planned_path` (which teleop ignores — operator drives directly). Multi-room missions are included by default. | Tens to hundreds of variants per scene | **Yes** — generator emits endpoint + path-shape + cross-room rows |
| **Operator-typed (stdin override)** | Operator presses `X` to swap from queue mode to typed mode; types the mission on stdin; presses Enter; episode begins. Used for ad-hoc missions the queue didn't cover, including path-shape variations the operator wants to demonstrate that day. | Operator-throughput-bottlenecked | Yes |

Multi-room is the default. Cross-room missions are present in
the queue whenever the scene is multi-room (per the
connectivity graph from
[`scene-connectivity-validation`](../multi-room/scene-connectivity-validation.md)).
This brief does **not** own queue generation; it consumes
whatever
[`mission-generator`](mission-generator.md)
emits.

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

The operator drives third-person but the *training data* the
robot will ever see is first-person. Operator UX has to bridge
that gap and give enough cues that 60 episodes / hour translates
to 60 *useful* episodes, not 60 quick-but-noisy ones.

- **Live third-person view:** Isaac Sim editor viewport, headed
  (not headless). The robot is visible from above / behind;
  the operator gets full spatial awareness of the scene.
- **First-person picture-in-picture (PIP):** a corner overlay
  sourced from the perception camera's render-product
  (`d555_camera_perception`, 640×360). Operator can confirm
  the target is in the *robot's* FoV, not just the editor's.
  Catches "the chair is visible third-person but blocked from
  the D555's view" failures before they pollute the corpus.
- **Suggested-path overlay (operator-only).** The mission
  queue's `planned_path` waypoints (LLM-emitted for path-shape
  missions, A*-shortest for endpoints — see
  [`mission-generator`](mission-generator.md))
  are rendered as a polyline in the editor viewport via Isaac
  Sim debug-draw primitives. **Critically operator-visible
  only — debug-draw renders into the editor viewport, not into
  the camera-output captures the perception camera writes
  to disk.** The acceptance criteria below verify this
  separation.
  - Color: solid for endpoint missions ("follow this exactly");
    dashed-with-constraint-color for path-shape missions
    ("follow this loosely, modify for the constraint").
  - Behavior: traversed segments fade behind the robot; the
    upcoming portion stays salient.
  - **Deviation hint.** When the operator drifts > 1 m from
    the polyline, a subtle color-bleed near the robot's
    position cues the deviation. Useful for noticing
    accidental drift on endpoint missions; expected behavior
    on path-shape missions where the operator deliberately
    differs.
- **HUD mission text.** Persistent on-screen overlay showing
  the current `mission_text` + the active paraphrase (operator
  can cycle paraphrases via D-pad). Console echo stays for
  log scrollback, but the on-screen text is what the operator
  reads while driving.
- **Recording status indicator.** Visual `REC` overlay (red
  dot or similar) when capture is active; gray `PAUSED` when
  not. Catches accidental "I drove for 30 seconds while not
  recording" failures.
- **Episode timer + distance-to-target + queue position:**
  printed to console once per second; cheap, no overlay
  needed.
- **Replay mode:** **out of scope for v1**; record gamepad
  events alongside `actions.jsonl` so a future
  replay-with-perturbation tool can reconstruct the trajectory
  under different lighting / texture seeds. Building the
  replay tool itself is a future brief.

### Operator UX additions filed-on-trigger

The following are deliberately deferred to keep v1 scoped.
Each gets a small follow-up brief if real teleop sessions
surface the need:

- **Constraint visualization** (highlight the south wall when
  mission says "hug south wall"). Would require structured
  constraint info that the renamed mission-generator brief
  explicitly retired in favor of free-text. Don't regress
  that decision; defer.
- **Prev-trajectory ghosts** for replay-with-perturbation.
  Depends on the replay tool itself; future brief.
- **Quick-skip / mission-queue manipulation tools** (delete a
  bad queue row mid-session, re-order, etc.). The `Back`
  button + stdin override cover v1 use cases.
- **Voice mission entry, AR / VR teleop, multi-camera live
  view, multi-operator collaborative teleop.** Out of scope
  for v1; named in the existing Out of scope section.

### Realistic data-volume budget

Honest numbers operators reading this brief should plan against,
**single operator, single session**. Earlier drafts of this brief
quoted ~60 endpoint-episodes/hour; an audit pass calibrated the
table downward against peer data-collection studies (DROID
recorded ~30–40 demos/hr per operator with a comparable UX
overhead). The numbers below reflect the heavier UX surface this
brief specifies (PiP, suggested-path overlay, HUD, REC indicator,
button-tag selection, queue-position console) plus the reset
cost of re-spawning the env at a new start pose. Treat as a soft
upper bound; the acceptance run (≥30 missions across ≥2 scenes)
is the calibration measurement that should update this table:

| Episode type | Time per episode | Episodes / hour |
|---|---|---|
| Auto-queue endpoint mission, success | ~1.5–2 min (10s read + 30–45s drive + 30–45s reset) | ~30–40 |
| Operator-typed path-shape mission | ~2.5–3 min (20s type + 50s drive + 60s reset/think) | ~20–25 |
| `X`-tagged hard negative (wrong instance / room) | ~2 min | ~25–30 |
| `SELECT`-tagged path-shape negative | ~2.5–3 min | ~20–25 |

Per-target volume the brief expects to support, **using the
calibrated ~30 ep/hr endpoint rate above**:

| Training target | Volume needed | Estimated operator time |
|---|---|---|
| CLIP fine-tune, 1 scene | ~1k frames (~30 episodes) | ~1 hour |
| VLM SFT, 1 scene | ~5k frames (~50 episodes) | ~1.5–2 hours |
| CLIP cascade eval + co-trained validator hard negatives, 1 scene | ~30 success + 30 `X`-tagged hard-negative episodes | ~2 hours |
| v2 VLA endpoint missions (Octo / TinyVLA frozen-tower fine-tune), 3 scenes total | ~10k trajectories with hindsight relabel + replay-perturbation multipliers (~5×) ⇒ ~2k actual demos | ~60–80 hours |
| v2 VLA path-shape missions (per constraint type) | ~1k path-shape demos | ~50 hours per constraint type — **bottleneck for path-shape; motivates the procedural generator** |

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
      [`collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py)
      verbatim (or via a shared module if light refactoring is
      needed). No re-tuning of axis mappings.
- [ ] **Output schema parity.** Per-episode output matches
      [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md)'s
      schema bit-for-bit (or the format
      [`output-format-alignment`](../../parked/harness/output-format-alignment.md)
      lands on, whichever is canonical when this brief is
      picked up). A unit test loads a single episode and
      confirms the per-tick rows, action rows, mission record,
      and image / depth paths conform.
- [ ] **Two mission-text sources.** Mission queue YAML loader
      consumes the canonical
      `mission_queue.yaml` produced by
      [`mission-generator`](mission-generator.md);
      operator stdin override switches to typed mode and back.
      Multi-room queue rows are accepted by default — no
      teleop-side filtering needed; the generator owns
      reachability gating.
- [ ] **Episode-end buttons** map to `mission.json.outcome` per
      the table above. `Back` discards the episode without
      writing.
- [ ] **Suggested-path overlay leak check.** With the path
      overlay rendering in the editor viewport, run one episode
      and inspect the saved `frame_*.jpg` captures: the polyline
      must **not** appear in any captured frame. Verified by a
      pixel-difference check between the overlay-on and
      overlay-off variants of the same pose, OR by visual
      inspection of N=10 captured frames. Failure here means the
      perception camera is picking up debug-draw — the
      implementation must move the overlay to an editor-only
      render layer before this brief ships.
- [ ] **First-person PIP, HUD mission text, REC indicator.**
      All three render in the operator's editor viewport, not
      in the perception camera's output. PIP shows live
      `d555_camera_perception` output. HUD shows
      `mission_text` + active paraphrase. REC shows capture
      state.
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
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      gains a **Stage 5b — Teleop data collection** section
      between Stages 5 and 6, covering: prerequisites (scene
      generated, mission queue built), invocation
      (`Scripts/teleop_collect.py`), button mapping, output
      schema reference, troubleshooting (gamepad not
      detected, scene fails to load, button-tag mismatch).
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains
      `teleop_collect.py`. (`build_mission_queue.py` is owned
      by [`mission-generator`](mission-generator.md).)
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the bridge harness — `--mode bridge` and
      `--mode harness` continue to work unchanged. Smoke this in
      the PR description.

## Investigation pointers

- Existing gamepad code:
  [`source/strafer_lab/scripts/collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py).
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
  [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 1 (programmatic walk) so paraphrase generation can
  share the loader.
- Distance-to-target: compute via straight-line `target_position_3d
  - robot_pos` for v1; A*-on-navigable-mask is over-scoped here
  (lives in the harness brief's `compute_progress.py`
  post-processor).
- The Foxglove visualization referenced in
  [`completed/vlm-bbox-overlay.md`](../../completed/vlm-bbox-overlay.md)
  is *not* needed for teleop UX; the Isaac Sim editor viewport
  is sufficient for v1.

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
- **Mission queue generation.** Filed as
  [`mission-generator`](mission-generator.md);
  this brief consumes the queue, doesn't author it.
- **Voice mission entry, AR / VR teleop, multi-camera live view.**
  All over-scoped for v1.
- **The `actions.jsonl` writer plumbing** itself.
  [`behavior-cloning-data-expansion`](behavior-cloning-data-expansion.md)
  ships the writer; this brief just emits actions through it
  with `source: "teleop"`.
- **Action-tokenization / VLA-side dataset format conversion.**
  The teleop brief writes the canonical JSONL schema; the
  HDF5 / RLDS exporter from the harness brief handles
  downstream-pipeline format conversion.
