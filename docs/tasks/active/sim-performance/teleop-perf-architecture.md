# Teleop capture perf architecture — drop perception render, decouple env.step, per-variant cameras

**Type:** architecture + refactor (teleop driver + env cfg variants + writer)
**Owner:** DGX agent (teleop-lane)
**Priority:** P2 — teleop-blocking, not training-blocking. Operator
session at 5–6 FPS makes the harness ergonomically unusable on
high-density Infinigen scenes; lifts the corpus-collection blocker
without touching the bridge or RL paths.
**Estimate:** L — three independent levers, each S–M; total ~3–5 days
of work across them depending on which compose.
**Branch:** `task/teleop-perf-architecture`

## Story

As a **DGX operator running `Scripts/capture.py --driver teleop` on a
high-density Infinigen scene** I want **the gamepad-driven session to
sustain ≥ 25 FPS viewport smoothness while the writer keeps capturing
at 8 Hz** so that **a 30-episode session is operator-paced, not
RTX-render-bound, and the harness corpus brief's ≥ 30 ep × ≥ 2 scene
acceptance becomes a one-evening run instead of a half-day grind**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
- [`context/conventions.md`](../../context/conventions.md)
- Sibling perf brief: [`bridge-throughput-toward-25hz`](bridge-throughput-toward-25hz.md) —
  bridge-driver-scoped levers. **This brief intentionally avoids
  overlapping** — teleop-side levers can ship independently because
  the consumers differ (operator viewport + LeRobot writer vs. ROS
  topic publishers + autonomy executor).
- Sibling renderer brief: [`isaac-sim-rt-2-default-renderer`](isaac-sim-rt-2-default-renderer.md) —
  global renderer-mode flip; orthogonal lever that compounds with all
  three here.

Consumer briefs whose camera schema this brief must not break:
- [`harness-architecture` → Output format](../harness/harness-architecture.md#output-format--lerobot-v3) —
  the per-variant camera toggle proposed below MUST emit a LeRobot v3
  `features` dict that consumers can read without a custom adapter
  per (driver, mission-source) cell.
- [`infinigen-scene-corpus`](../harness/infinigen-scene-corpus.md) —
  the corpus this brief unblocks.
- [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md) —
  downstream consumer of the teleop corpus; needs RGB full-res for the
  VLA, doesn't need depth.

## Motivation

PR #63 shipped a targeted `update_period` change that lowered the
sensor `_is_outdated` flag's set-rate (sensor_base.py:186-205), so the
perception camera's `_renderer.render(self._render_data)` (camera.py:450)
only fires every `round(env_step_hz / capture_rate_hz)` ticks instead
of every step. Measured: ~5.5 FPS, not the predicted 12+. Gap is
real — the update_period lever works as advertised, but the operator's
viewport smoothness depends on a second render that the lever doesn't
touch.

### The 20 FPS observation revisited

Prior investigation dismissed the operator's "20 FPS smooth while
everything stalled" observation as a compositor mirage. The operator
pushed back: *they were flying the perspective camera around the
scene smoothly while feeling the rest of the system was stalled, and
moving the camera forces a re-render (cached framebuffers can't
update on camera transform changes).* That's correct. The mirage
verdict was wrong on the cached-framebuffer point.

What actually happens (verdict: **20 FPS was real**, but for a
mechanism the prior investigation missed, NOT because Kit has an
independent background render thread):

1. **Kit's editor app loop runs whenever the Python script is NOT
   inside `env.step()`.** Kit is a Qt-based desktop app with its own
   event loop. When the script is blocked on `input()` at the
   mission picker, or sitting in the `time.sleep(0.3)` debounce
   after `end_episode`, Kit's event loop services viewport
   interaction (perspective camera move + re-render) at whatever
   rate the X compositor + RTX renderer can sustain — typically
   20–30 Hz for the 640×360 perception scene at high_quality_dgx
   density on the DGX. The "everything stalled" the operator felt
   was the Python script blocked on a synchronization boundary, not
   Kit's render pipeline.

2. **When env.step IS running, Kit's app loop is serialized through
   `KitVisualizer.step()` calling `app.update()` synchronously**
   (kit_visualizer.py:108-128). That synchronous pump runs *with
   the env.step thread holding the GIL*. There is no async render
   thread — `apps/isaaclab.python.kit:245` explicitly sets
   `app.asyncRendering = false`. The viewport re-render cost is
   paid every env.step regardless of camera `update_period` (which
   only gates the writer-facing sensor read in camera.py:188-193).

So the 5.5 FPS during teleop and the 20 FPS during the picker pause
are the SAME render pipeline at the SAME RTX cost — the difference
is whether the env.step + manager-loop + writer chain is also
running in series with each frame. The 20 FPS observation tells us
the upper bound of what the RTX render pipeline alone can deliver on
this scene; the 5.5 FPS tells us how much env.step adds on top.

Implication: `update_period` is the wrong lever for operator
smoothness on this scene. The viewport render is the dominant cost
and it fires regardless of update_period. The three heavier-hitting
levers below attack the actual binding constraints.

### Three heavier-hitting levers

1. **Drop the perception camera render entirely on non-capture
   steps.** The teleop driver only needs the perception camera's
   render output every `ticks_per_capture` (default 8) steps when
   the writer samples; the other 7/8 steps render it for no
   downstream consumer. The operator's viewport doesn't need it
   either — the editor uses `/OmniverseKit_Persp`
   (teleop_capture.py:347-348), not the d555 perception camera.
   The `update_period` lever already gates the sensor-side read,
   but the renderer is still wired into the env's `scene.update`
   pass. We can additionally skip the render-product creation
   itself on non-capture steps via either (a) removing the camera
   from the env cfg on the teleop path and re-attaching it only on
   capture ticks, or (b) the same `update_period` trick combined
   with confirmation that `_update_outdated_buffers` is truly the
   only render trigger on the teleop path (not the bridge's
   OmniGraph RunOnce node).

2. **Run env.step on a background thread; render UI + writer on
   the main thread.** Gamepad-driven teleop is naturally async —
   the operator inputs at human speed (~10 Hz, sometimes burstier),
   the writer samples at 8 Hz, env.step runs at 30 Hz. Today these
   are serial in one Python loop: gamepad → env.step → render →
   writer → repeat. With env.step on its own thread, the main
   thread can service the viewport at Kit's native rate while
   env.step ticks at whatever pace the GPU sustains. The handoff
   surfaces several concurrency hazards enumerated in
   [Operator FAQ → D](#d-background-envstep-thread--what-are-the-concurrency-hazards).

3. **Lower `env_step_hz` from 30 → 15.** Teleop doesn't need 30 Hz
   physics — the operator's gamepad inputs and the captured
   action trajectory don't have 30 Hz structure. Halving the env
   step rate doubles every tick's compute budget. The capture
   contract (capture_rate_hz=8 Hz) still has 1.875 ticks per
   capture at 15 Hz, plenty of headroom. Open question covered
   in [Operator FAQ → E](#e-lower-env_step_hz-from-30-to-15--clip--vla-training-implications):
   does this hurt CLIP / VLA training?

4. **Per-env-variant camera configuration.** Different consumers
   need different camera outputs. Today the InfinigenPerception env
   instantiates both the 640×360 perception camera AND the 80×60
   policy camera unconditionally (strafer_env_cfg.py:373-378),
   even when teleop is only writing the perception RGB. Bridge
   sim-in-the-loop needs both at full resolution PLUS the
   80×60 depth for the continuous-control policy. A per-variant
   camera toggle would let the teleop env declare
   `cameras_required = ["rgb_full"]` and skip the depth + policy
   camera renders entirely. Covered in
   [Operator FAQ → C](#c-per-env-variant-camera-configuration-design).

## Operator FAQ

The operator's review of the prior perf investigation surfaced five
concrete questions. The five answers below are the reasoning trail —
read them in order; each builds on the previous.

### A. Was the 20 FPS observation real?

**Yes, it was real.** The verdict above (Motivation → "The 20 FPS
observation revisited") supersedes the prior "compositor mirage"
claim. The mechanism is:

- Kit's editor app loop runs independently whenever the Python
  script is between env.step calls (mission picker, debounce
  sleeps, etc.). The Qt event loop services viewport interaction
  at whatever rate the X compositor + RTX renderer sustain on the
  current scene.
- The render IS real (cached framebuffers can't update on camera
  transform changes — the operator's load-bearing observation).
- When env.step IS running, Kit's pump is serialized through
  `KitVisualizer.step()` → `app.update()` (kit_visualizer.py:108-128)
  with `app.asyncRendering = false` (isaaclab.python.kit:245).
  No background render thread exists.

So the 20 FPS picker observation and the 5.5 FPS teleop observation
are the same render pipeline at the same RTX cost. The teleop number
is lower because env.step + manager loop + writer add ~135 ms on top
of the ~50 ms render. Not a mirage; not a hidden async render
thread either. The operator's intuition that "rendering was actually
fine, something else was stalled" was correct — and the something
else was the synchronous env.step + writer chain.

This matters because it focuses the remediation: the binding
constraint during teleop is the serialization of env.step + render
+ writer, not the RTX render cost per se. Lever 2 (background
env.step) attacks the serialization directly.

### B. The Infinigen cameras in the USDC — what are they?

The operator confirmed seed=1's USDC contains four camera prims:
`/World/Room/camera_0_0`, `/World/Room/camera_0_0/camera_0_0`,
`/World/Room/camera_0_1`, `/World/Room/camera_0_1/camera_0_1`.
Disabling them in the editor produced no measurable perf change.

**What they are:** Infinigen's `generate_indoors` pipeline authors
stage-shot cameras (one per room, named `camera_<room>_<index>`)
for its own offline rendering passes — Infinigen ships its own
Cycles render path that needs explicit cameras to produce the
preview / dataset images Infinigen itself publishes. The
inner-named twin (`/World/Room/camera_0_0/camera_0_0`) is
Infinigen's Xform-with-same-name-Mesh-child export pattern (same
as the `bedroom_0_0_floor` pattern handled in postprocess_scene_usd.py:60).
These cameras are not bound to any Isaac Sim render product, so
no RTX render product evaluates them per-tick — the lack of perf
change when disabled is consistent with that.

**Why this is useful as a data point:** confirms RTX render cost
is NOT in idle/unbound cameras. It's in the perception camera's
render product (the one the writer reads). Lever 1 (drop the
perception render on non-capture steps) is therefore the correct
attack surface.

**Cleanup:** Add a `postprocess_scene_usd.py` pass that strips
these Infinigen authoring cameras during the scene-prep bake. They
add tiny per-prim overhead (USD traversal, transform updates) and
clutter `usdview` / Omniverse Composer when an operator opens the
scene to debug. They serve no role in the strafer runtime. See
the matching acceptance bullet below.

### C. Per-env-variant camera configuration design

The operator's observation: different consumers need different
camera outputs. Concrete shapes:

| Consumer | RGB full (640×360) | Depth full (640×360) | Depth policy (80×60) | Policy RGB (80×60) |
|---|:-:|:-:|:-:|:-:|
| Bridge sim-in-the-loop + real autonomy | yes (VLM) | yes (RTAB) | yes (RL policy) | no (deployment doesn't need it) |
| Teleop via capture.py (current proposal) | yes | no | no | no |
| Trajectory-first captioning (future scripted driver + captioner mission) | yes | no | no | no |
| Room-state eval (future scripted driver + coverage mission) | yes | yes (for VPR depth back-projection) | no | no |
| RL training (parallel scripted driver) | no (NOCAM variant) | no | no | no |

The current `_BaseInfinigenPerceptionNavEnvCfg`
(strafer_env_cfg.py:1295-1316) instantiates ALL of them
unconditionally because it was designed around "one env cfg = one
deployed config." That conflates two concerns: **what robot
dynamics the env simulates** (the cfg's true purpose) and **what
data products the env exposes** (a per-consumer concern).

**Proposed shape — `cameras_required` env_cfg field:**

```python
@configclass
class _BaseInfinigenPerceptionNavEnvCfg(_BaseStraferNavEnvCfg):
    # Which camera outputs this env produces. Drives both:
    #   (a) which TiledCameraCfg objects get registered into the scene
    #   (b) which features the writer declares + which add_frame args
    #       are required vs. optional.
    cameras_required: tuple[str, ...] = (
        "rgb_full",
        "depth_full",
        "rgb_policy",
        "depth_policy",
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_infinigen_scene_setup(self)
        self._apply_camera_required_pruning()

    def _apply_camera_required_pruning(self):
        # Strip the TiledCameraCfg attrs not in cameras_required so
        # Isaac Lab's scene builder never instantiates them.
        if "rgb_full" not in self.cameras_required and \
           "depth_full" not in self.cameras_required:
            self.scene.d555_camera_perception = None
        # ... similar for the policy camera
        # rgb_full + depth_full share the perception camera (single
        # render product, two channels), so the toggle is over data_types
        # not over the camera itself when both are off
```

Concrete env variants this enables:

| Variant | `cameras_required` | Consumer |
|---|---|---|
| `_Teleop` (this brief) | `("rgb_full",)` | `capture.py --driver teleop` |
| `_BridgeAutonomy` (existing default) | `("rgb_full", "depth_full", "rgb_policy", "depth_policy")` | bridge sim-in-the-loop |
| `_Coverage` (future) | `("rgb_full", "depth_full")` | scripted driver + coverage mission (room-state eval) |
| `_Captioner` (future) | `("rgb_full",)` | scripted driver + captioner mission (trajectory-first) |

Matching writer-side toggle in `lerobot_writer.py`: extend
`build_features` (lerobot_writer.py:251-287) so it takes the same
`cameras_required` tuple and conditionally adds each feature
column. `add_frame` then validates that args match the declared
schema (no zero-padding for absent cameras — frames that weren't
captured aren't authored at all).

**Boundary respected:** the camera toggle is **what the env
exposes**, not **how the env simulates** — the underlying robot
dynamics, action space, observation pipeline (for the trained
policy paths) stay identical. So a `_Teleop` variant with only
rgb_full is still distributionally close to the RL-deployment
config for the trajectory data it produces.

### D. Background env.step thread — concurrency hazards

The prior investigation hand-waved "concurrency hazards." Concrete
list, each with a specific mitigation:

| # | Hazard | Mitigation |
|---|---|---|
| 1 | **Action handoff between threads.** Gamepad reader thread writes `(vx, vy, omega)`; env.step thread reads on each tick. Race: env.step reads a half-updated 3-tuple → wrong velocity for that step. | Use a single `threading.Lock` around action write + read, OR an atomic-replace pattern: gamepad thread builds a fresh tensor + assigns to a single shared reference; env.step reads the reference + atomically swaps. The atomic-replace pattern is lock-free and Python's GIL guarantees reference assignment is atomic. |
| 2 | **Torch tensor / CUDA stream affinity.** Tensors created on one thread under one CUDA stream that another thread tries to read can deadlock or produce garbage if the producing op hasn't completed. Most relevant for: env.step thread builds the action tensor that gamepad-thread input data flowed into; writer thread reads camera output tensors. | Pin all torch ops to the env.step thread's CUDA stream. Cross-thread tensor handoffs are `.cpu().numpy()` boundaries — the bridge async-camera-publisher pattern (the GPU→CPU readback on a dedicated worker stream) is the proven precedent. The writer's `add_frame` already does the cpu().numpy() conversion before storing, so the tensor lifecycle is bounded to one thread. |
| 3 | **Shared `env.scene` access during env.step.** Sensors (camera, IMU), articulations (robot pose), and the manager loop all mutate scene state during env.step. If the main thread reads `scene["robot"].data.root_pos_w` for the viewport follow-cam or HUD distance display while env.step is mid-tick, the read either sees a stale pose or a torn pose (mid-write). | Designate the env.step thread as the sole writer of scene state. Main-thread reads happen only between env.step ticks via a posted snapshot: env.step thread, at end of each tick, atomically publishes `(pose, achieved_vel, mission_distance)` to a shared dict that main-thread consumers (HUD, PIP, arcade follow tick) read from. The `_arcade_follow_tick` (teleop_capture.py:414-450) and HUD code (teleop_capture.py:1241-1261) both go through the snapshot, not through `unwrapped.scene`. |
| 4 | **`env.reset()` boundary.** Resets touch the full scene graph (sensor buffers, articulation states, USD stage events). If a reset starts on the env.step thread while the writer is mid-`add_frame` for the previous episode, the camera output tensor the writer is reading can race against the reset's zero-out. | Drive resets through a single-threaded barrier: env.step thread signals "reset requested" → finishes current tick → waits on a barrier for the writer to flush + close the episode → performs the reset → releases the barrier → new episode begins. The existing `_begin_next_episode` (teleop_capture.py:1009-1076) already has this sequential structure; the threaded version just adds the barrier wait. |
| 5 | **Kit's USD stage notifications crossing thread boundaries.** Kit's USD notice handlers (`/clock` update, render product completion, etc.) are dispatched on Kit's app thread — which is the main thread in the current single-thread design. If env.step runs on a worker thread and triggers USD writes (rebuilding the render product on reset, debug-draw target marker updates), the notice handlers fire on the worker — Kit's threading model assumes the app thread is the USD writer, so this can wedge Kit's editor. | Keep all USD-touching operations (`_TargetMarker.set_target`, `_hide_overhead_prims`, viewport possession, set_camera_view) on the main thread. The env.step thread only touches torch tensors, sensor data buffers, and manager-loop state — all of which are PhysX-side / Fabric-side. USD writes are funneled through main-thread callbacks (the snapshot mechanism in hazard 3 above, extended with "USD ops to dispatch" queue). |
| 6 | **`lerobot_writer.add_frame` thread safety.** The writer holds an open Parquet shard + an in-flight episode buffer. Concurrent `add_frame` calls from different threads would corrupt the buffer. | Single-thread the writer: run it on the main thread, called from the snapshot consumer. The env.step thread queues `(pose, achieved_vel, action, rgb_arr, depth_arr)` snapshots; the writer thread (== main thread) drains the queue at capture_rate_hz. The `AsyncImageWriter` already does PIL encode off-thread (lerobot_writer.py:308-321); this brief just extends the pattern one layer up — `add_frame` itself is single-thread, but it's called from the main thread, not the env.step thread. |
| 7 | **Python GIL contention.** Even with threads, only one Python thread executes Python bytecode at a time. If env.step is GIL-heavy (Python-side manager loop dispatch ~20 ms per tick per `bridge-throughput-toward-25hz` measurements), the main thread can't run the viewport/HUD/writer drain in parallel. | Lever 2's win is concretely from: (a) the env.step thread blocking on PhysX (no GIL held; PhysX is C++) for ~22 ms of each ~42 ms env.step, freeing the main thread to render the viewport, drive HUD, drain writer queue. (b) The render's CUDA submission is also C++/non-GIL. So the realized parallelism is ~30-50% — not 2× — but enough to lift the operator-felt FPS from 5.5 to ~12–15. **This is the brief's win-per-risk uncertainty** — if measurement shows < 20% win, fall back to Lever 1 + Lever 3 only and shelve Lever 2. |

The pattern across all six is the same: **one writer per shared
resource, snapshots for cross-thread reads, USD ops stay on main**.
This is the conventional Qt/desktop pattern, not bespoke
concurrency design — it's what Kit itself does internally for its
own UI thread vs. worker threads.

### E. Lower `env_step_hz` from 30 to 15 — CLIP / VLA training implications

The prior analysis flagged this as a contract change with training
implications but only addressed RL continuous-control policy
training. CLIP and VLA training have different sensitivities:

**CLIP training.** Contrastive on (image, text) pairs with no
temporal structure inside a single training example. The training
sees one frame + one caption at a time; the corpus generates
many such pairs per trajectory. Lowering env_step_hz halves the
upper bound on (image, caption) pairs per trajectory of fixed
wall-clock duration, but the **diversity per pair** is unaffected.
The corpus quality concern is: do consecutive frames at 15 Hz
provide enough scene-state diversity that random-frame sampling
for contrastive batches still produces non-trivially different
positives/negatives? Answer: yes — at 8 Hz capture rate (the
default `--capture-rate-hz`), 15 Hz env_step_hz still gives 1.875
env ticks per captured frame, so consecutive captures are ~125 ms
apart in sim time. At typical operator velocities (0.5–1.5 m/s),
that's 6–19 cm of scene-frame translation per capture, more than
enough for the image content to differ. **CLIP training:
unaffected.**

**VLA training.** Action prediction on (image, text, action)
tuples. The action target is the per-tick gamepad command. Two
sub-concerns:

1. **Temporal context compression.** A VLA conditioned on an
   N-frame history at 15 Hz sees half the temporal context (in
   wall-clock) as one conditioned at 30 Hz. For an N=8 history
   window, that's 1.07 s vs 0.53 s. If the policy needs to
   reason about acceleration / deceleration profiles over
   ~0.5–1 s windows, halving the rate compresses that into the
   stride space — recoverable by doubling N, at the cost of
   memory.
2. **Action-rate quantization.** Real-robot deployment runs the
   trained VLA at whatever rate the deployment loop sustains
   (typically 10–20 Hz). If training data captured at 15 Hz is
   replayed at 20 Hz inference, the action sequence is slightly
   sped up; if at 10 Hz, slowed down. The VLA is robust to this
   if the training corpus includes rate variation (which it will,
   given the operator's varying gamepad pace), but a single-rate
   training corpus + cross-rate deployment is a known sim-to-real
   gap.

**Recommendation:** different env_step_hz per launch is already
configurable (the env_cfg is per-task, and the teleop driver
loads it via `load_cfg_from_registry`). Don't change the global
default; expose `--env-step-hz` on `Scripts/capture.py` with a
default of 30 for back-compat. Operator chooses 15 for teleop
sessions where perf matters more than action-rate fidelity, 30
for sessions that feed VLA training corpora where action-rate
matters. The writer's `capture_rate_hz=8` already decouples
the dataset's effective sample rate from env_step_hz; the
**dataset shape doesn't change** regardless of env_step_hz,
only the action signal's quantization within each captured
frame's gap.

**Action-fidelity caveat:** at env_step_hz=15, the LeRobot
`action` column at each captured frame is the gamepad command
at that env tick — but the operator's gamepad inputs between
captures (those happening on the 1.875 env ticks per capture)
are dropped. If the VLA needs the action trajectory at
capture_rate_hz fidelity (not env_step_hz), the writer should
record an **action sequence** per capture, not a single
action. Out of scope for this brief but worth flagging — file
as a follow-up if VLA training picks up and shows symptoms.

## Acceptance

Ship one or more of the four levers above, each verified
independently. Brief is shipped when ≥ 2 of the 4 land OR when
operator-measured teleop FPS sustains ≥ 15 on `high_quality_dgx`
Infinigen scenes at `--capture-rate-hz 8`.

**Lever 1 — Drop perception render on non-capture steps:**
- [ ] On non-capture env.step ticks, the perception camera
      produces zero render work (verifiable via `nvidia-smi
      dmon -s u` or RTX call counts inside `--profile` mode).
- [ ] The captured LeRobot dataset is identical (bit-for-bit
      on the perception RGB column) to a baseline run with the
      lever off.
- [ ] No regression in viewport smoothness (Kit persp camera
      still renders the scene at the same rate it did pre-lever).

**Lever 2 — Background env.step thread:**
- [ ] env.step runs on a dedicated worker thread; main thread
      services viewport + HUD + writer drain.
- [ ] All six hazards from FAQ → D have a concrete mitigation
      landed (action handoff via atomic-replace, snapshot for
      scene reads, single-thread USD ops, barrier on reset,
      single-thread writer, etc.).
- [ ] Operator-measured FPS ≥ 12 sustained on
      `scene_high_quality_dgx_000_seed1` at default settings,
      vs. today's ~5.5.
- [ ] No deadlock or hang in a 30-episode session; clean exit
      via Start-held quit.

**Lever 3 — Lower env_step_hz for teleop default:**
- [ ] `Scripts/capture.py --driver teleop` exposes
      `--env-step-hz` with default 30 (back-compat). Operator
      can pass `--env-step-hz 15` for a perf-prioritized
      session.
- [ ] `docs/HARNESS_DATA_CAPTURE.md` documents the trade-off
      (action-rate fidelity vs. operator FPS), citing the
      FAQ → E reasoning.
- [ ] `harness-architecture.md` is updated to mention
      `env_step_hz` as a per-launch knob, with the existing
      `capture_rate_hz` framing extended (the dataset's
      effective sample rate is unchanged; only the action
      signal's quantization within each captured-frame gap
      varies).

**Lever 4 — Per-env-variant camera configuration:**
- [ ] `_BaseInfinigenPerceptionNavEnvCfg` (or a sibling
      class) gains a `cameras_required` field. Teleop driver
      registers a `_Teleop` variant with
      `cameras_required=("rgb_full",)`; the existing default
      (used by bridge + scripted) keeps the full quartet.
- [ ] `lerobot_writer.build_features` accepts a
      `cameras_required` arg and conditionally emits feature
      columns; `add_frame` validates kwargs against the declared
      schema (clean error, not silent zero-padding, when an
      undeclared camera arg is passed).
- [ ] Teleop session's writer no longer encodes the policy
      camera MP4 stream when only RGB full is requested
      (verifiable: `videos/observation.images.policy/` dir
      absent in the output).
- [ ] Bridge sim-in-the-loop env path is unchanged
      (regression check: a `run_sim_in_the_loop.py` smoke
      against the existing default cfg produces the same
      camera publications).

**Cleanup acceptance (independent of levers above):**
- [ ] `postprocess_scene_usd.py` is extended with an
      Infinigen stage-camera stripper: walks the USDC for
      `Camera`-type prims whose parent is `/World/Room/*`
      (the Infinigen authoring-camera pattern) and removes
      them. Naming reflects the camera's origin (avoid the
      transient PR-#63 reference per the project's
      no-transient-doc-refs convention).
- [ ] Re-running the postprocess pass on the existing
      `scene_high_quality_dgx_000_seed1.usdc` removes the
      four cameras the operator identified.
- [ ] No measurable perf change is the expected outcome
      (matches the operator's observation); document this in
      the postprocess CLI help so a future operator doesn't
      assume the cleanup is a perf lever.

**Cross-cutting:**
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No change to the `Scripts/capture.py` CLI surface beyond
      the optional `--env-step-hz` flag (Lever 3). The
      `(driver, mission_source)` matrix is unchanged.

## Approach

Implementation order, picked for win-per-risk:

1. **Cleanup first** (postprocess stage-camera stripper) — small,
   low-risk, gives the operator a clean baseline scene for measuring
   the other levers.

2. **Lever 4** (per-variant camera config) — pure refactor; doesn't
   change behavior on existing call paths, just enables a leaner
   teleop variant. Unblocks Lever 1's measurement (Lever 1 against
   a 4-camera env is harder to attribute than against a 1-camera env).

3. **Lever 3** (env_step_hz CLI flag) — single-line config knob with
   docs; operator gets a perf dial immediately. Useful as a
   short-term workaround while Levers 1 + 2 are being designed.

4. **Lever 1** (drop perception render on non-capture steps) —
   medium-complexity refactor. The `update_period` path already does
   most of the work; this lever extends it with confirmation that
   no other render trigger fires + the env removes the camera from
   the scene on the non-capture path (vs. just leaving it idle).

5. **Lever 2** (background env.step thread) — highest complexity,
   highest potential win. Land only after Levers 1 + 3 + 4 are
   shipped and the perf gap remaining justifies the concurrency
   surface area. The brief explicitly allows shipping without
   Lever 2 if the other three close the gap to ≥ 15 FPS.

Each lever ships as its own PR within the brief's branch
(`task/teleop-perf-architecture/<lever-N>`). Brief is closed when ≥
2 levers land AND the operator-measured FPS hits the bar.

## Out of scope

- **Bridge-driver perf.** The sibling
  [`bridge-throughput-toward-25hz`](bridge-throughput-toward-25hz.md)
  brief owns this. The levers there (Kit pump cadence, OmniGraph
  deletion, lean bridge env cfg) are orthogonal to the teleop
  levers here and can ship in parallel.
- **RL training throughput.** Separate path. See
  [`training-throughput-profile-and-investigate`](../investigations/training-throughput-profile-and-investigate.md).
- **Renderer flip to RT 2.0.** Owned by
  [`isaac-sim-rt-2-default-renderer`](isaac-sim-rt-2-default-renderer.md).
  Will compound with the four levers here when it lands; don't
  combine the work into one brief.
- **Lowering perception camera resolution below 640×360.** This
  creates a sim-to-real gap for the deployed perception camera and
  belongs in a dedicated resolution brief if it ever becomes the
  next-tier lever. The four levers here close the gap without
  touching resolution.
- **Writer-side video-codec changes.** H.264 vs SVT-AV1 is decided
  in `harness-architecture.md`; not a perf lever for the operator's
  felt FPS (the writer runs off-thread via AsyncImageWriter
  already).
- **Real-robot capture rates.** This brief is sim-only. Any
  decision about real-robot capture cadence belongs in the
  Jetson lane.
- **An action-sequence-per-capture column** (the FAQ → E
  action-fidelity caveat). File as follow-up if VLA training
  surfaces the symptom; not gating the four levers here.

## Triggered by

PR #63 (Tier 1 harness teleop driver) operator-measured ~5.5 FPS
on `scene_high_quality_dgx_000_seed1` at default settings — below
the predicted 12+ FPS the `update_period` lever was sized for.
Operator review of the prior perf investigation surfaced the five
questions in [Operator FAQ](#operator-faq); this brief is the
follow-up that answers them and articulates the heavier-hitting
levers as concrete acceptance items.
