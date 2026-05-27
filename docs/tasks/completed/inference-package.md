# Build the `strafer_inference` Jetson runtime for trained-DEPTH policy execution

**Status:** Shipped 2026-05-25 in `4b5315e` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/55
**Follow-ups:** [`strafer-direct-sim-validation`](../active/trained-policy/strafer-direct-sim-validation.md) — operator-driven sim validation (rosbag parity + TRT-EP latency + architectural-win mission); runs when the sim-in-the-loop rig + a deployable DEPTH checkpoint are both in hand. [`policy-rate-shared-constants`](../active/trained-policy/policy-rate-shared-constants.md) — DGX-side delegation of `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` to the new shared constants this brief added.

**Type:** task / feature
**Owner:** Jetson (new ROS package lives in `strafer_ros/`)
**Priority:** P1 — the trained-policy backend is the architectural answer
to MPPI's plateau. **DEPTH variant is the MVP** because (a) the
ProcRoom-Depth env in `strafer_lab` is already trainable today and (b)
NOCAM in `strafer_direct` would be unsafe — see Story.
**Estimate:** L (~1.5 weeks: package skeleton + DEPTH observation
pipeline including the depth-downsample stage + TRT inference path +
backend dispatch)
**Branch:** task/strafer-inference-package

## Story

As a **mission operator running navigate_to_pose missions on the real
chassis or in sim-in-the-loop**, I want **a `strafer_direct` execution
backend that runs the Isaac-Lab-trained DEPTH navigation policy
through the same observation / action contract the policy was trained
on**, so that **robot motion converges to the velocity envelope the
policy demonstrates (~1.4 m/s sustained) while the policy uses depth
perception to plan its own obstacle-avoidance — instead of plateauing
at ~63 % of that ceiling under MPPI's critic landscape**.

### Why DEPTH and not NOCAM for `strafer_direct`

The first version of this brief proposed `PolicyVariant.NOCAM` as the
direct-mode MVP. **That would be unsafe.** NOCAM observations are
IMU + encoders + goal-relative + body-velocity + last-action. Zero
perception. A NOCAM policy in pure-RL `strafer_direct` mode drives
toward the goal pose with no awareness of walls, furniture, or
anything else the chassis might run into. On a real robot, that's a
collision. Even in sim it produces unproductive episodes.

`PolicyVariant.DEPTH` adds 4800 dims of flattened 80×60 depth from the
D555. With the right reward shaping (which the existing ProcRoomDepth
env has been training against), the policy learns obstacle-aware
behavior. That's what makes `strafer_direct` actually deployable.

NOCAM's place in the system is **hybrid mode** — Nav2 supplies the
obstacle-aware global path, the NOCAM_SUBGOAL policy follows rolling
subgoals along it. That's
[`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md)
+ the prerequisite training-env work in
[`subgoal-env`](subgoal-env.md).

### Why DEPTH-MVP is counter-intuitively the smaller path

| Path | Env work | Training | Jetson inference |
|---|---|---|---|
| **DEPTH + `strafer_direct`** | done (`Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` + `Robust` variant) | done / in flight | this brief |
| NOCAM_SUBGOAL + hybrid | new env + path planner + subgoal command + reward shaping + termination tuning | full training run from scratch | `strafer-inference-hybrid-mode.md` |

The DEPTH ProcRoom env has been training for months; a deployable
checkpoint is gated mostly on training-time + the
[`policy-export-tooling.md`](../../completed/policy-export-tooling.md) brief landing.
The hybrid-mode path requires a brand-new env that doesn't exist yet.

So this brief gets the trained-policy backend deployed first, on the
existing-env path; hybrid mode follows once the new training env is
built.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
  — particularly the "Camera resolutions (sim mirrors real)" section,
  which is load-bearing for Phase 2's depth-downsample stage.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
  — the canonical spec for hidden-state shape, reset semantics,
  per-tick state threading, determinism, and thread-safety across
  the train -> export -> inference chain. Read before writing the
  Phase 3 `reset()` call sites; do not redefine the trigger set in
  this brief — consume it from the contract (point 4).
- [observation-contract-cleanup.md](../../completed/observation-contract-cleanup.md)
  — load-bearing predecessor for the Phase 2 NOCAM-fields obs-parity
  acceptance (≤ 1e-5 max abs delta). `body_velocity_xy` now does
  encoder-derived FK over the same joint-velocity tensor the Jetson
  reads from `/strafer/joint_states.velocity`, so this brief's
  parity test is meaningful.
- [completed/mppi-critic-tuning-for-sim-envelope.md](../../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the predecessor whose validation surfaced the MPPI plateau and
  motivated this work as the architectural alternative.

## Context

### What already exists

The shared sim-to-real contract is in place — policy export and
deployment plumbing are the missing pieces, not the contract:

[`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py)
ships:

| API | Purpose |
|-----|---------|
| `PolicyVariant.NOCAM` (19 dims) | IMU + encoders + goal-relative + body-vel + last-action |
| `PolicyVariant.DEPTH` (4819 dims) | NOCAM + 4800-dim flattened 80×60 depth |
| `assemble_observation(raw, variant)` | Normalize + concatenate raw sensor dict → flat float32 obs |
| `interpret_action(action_normalized)` | Denormalize `[-1, 1]` policy output → `(vx, vy, omega)` in m/s, rad/s |
| `action_to_wheel_ticks(action_normalized)` | Convenience: action → wheel ticks/sec via `mecanum_kinematics` |
| `load_policy(path, variant)` | Load `.pt` (TorchScript) or `.onnx` (ONNX Runtime, including TRT EP); returns a `LoadedPolicy` (callable `(obs) → action`, with `.reset()` for episode boundaries and `.is_recurrent`) |
| `benchmark_policy(policy, variant, n_iters)` | Inference-latency stats |

The Isaac Lab side keeps the env aligned via
[`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
and the registered task IDs in
[`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py)
— specifically `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` and
`Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0` are the deployment
targets. The depth observation is produced by
[`mdp/observations.py:depth_image`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
with documented preprocessing (RAW depth → nearfield fill →
`scale = 1/max_depth` to `[0, 1]`).

### What's missing

1. **`source/strafer_ros/strafer_inference/`** — the ROS 2 ament_python
   package that holds the inference node, launch file, config, tests,
   and entry point. Empty today.
2. **`Scripts/export_policy.py` (TorchScript + ONNX + TRT EP path).**
   Filed as
   [`policy-export-tooling.md`](../../completed/policy-export-tooling.md). Hard
   dependency in DGX lane: DEPTH inference latency on a 4819-dim
   observation through a conv-aware policy is NOT sub-millisecond on
   CPU. The TRT execution provider is required for deployment, not
   optional. The export brief now covers both formats as MVP.
3. **`execution_backend` dispatch in
   [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)**
   `JetsonRosClient.navigate_to_pose` — currently routes
   unconditionally through Nav2's `/navigate_to_pose` action. Needs
   to honor an `execution_backend` field (env-var
   `STRAFER_NAV_BACKEND` or per-step) with values `nav2` (default)
   and `strafer_direct`.
4. **Depth downsample pipeline.** The policy was trained on 80×60
   depth from `d555_camera` (the policy camera). The bridge invariants
   doc states the policy camera is sim-only and **NOT bridged to
   ROS**. What the Jetson gets via `/d555/depth/image_rect_raw` is
   the 640×360 `d555_camera_perception` stream. So the inference
   node must downsample 640×360 → 80×60 before feeding
   `assemble_observation`. This applies in both sim-in-the-loop and
   real-robot lanes (the policy camera doesn't exist on the real
   D555 either — there's one physical sensor at 640×360 native).
5. **A deployable DEPTH checkpoint** from `strafer_lab` PPO training
   against `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` (or `-Robust-`
   variant). Out of scope for this brief — gates the validation
   steps but not the package skeleton.

### Two execution modes

`hybrid_nav2_strafer` is filed separately
([`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md))
and depends on a different `PolicyVariant` (`NOCAM_SUBGOAL`) plus
new training env work
([`subgoal-env`](subgoal-env.md)).

| Mode | Global plan | Local control | Filed as |
|------|-------------|---------------|----------|
| `nav2` (default) | Nav2 GridBased | Nav2 MPPI | Shipped today |
| `strafer_direct` | none — direct goal | trained DEPTH policy via `strafer_inference` | **This brief** |
| `hybrid_nav2_strafer` | Nav2 GridBased | trained `NOCAM_SUBGOAL` policy via `strafer_inference` | [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md) |

The mode is a *robot-side* execution choice; the autonomy / planner
layer doesn't change. Operator-facing surface stays
`navigate_to_pose(goal_pose=...)`; backend choice is a config knob.

## Approach

Five tightly-scoped phases. Each phase gets its own review-able
commit on the same branch — no cross-phase intermixing (makes
bisection trivial when the trained policy lands and validation
surfaces an integration bug).

### Phase 1 — Package skeleton (½ day)

Create `source/strafer_ros/strafer_inference/`:
- `package.xml` (ament_python), `setup.py`, `setup.cfg`,
  `resource/strafer_inference`
- `strafer_inference/__init__.py`
- `strafer_inference/inference_node.py` — empty rclpy `Node` subclass
  that subscribes to the contract inputs, publishes
  `/strafer/cmd_vel`, and exposes a `/navigate_to_pose` action server
  (parallel to Nav2's, same action type) but does nothing yet.
- `launch/inference.launch.py` — brings up the node with a config
  pointing at a model artifact path.
- `config/inference.yaml` — `model_path`, `policy_variant` (default
  `DEPTH`), `infer_period_s`, `goal_topic`, `cmd_vel_topic`,
  `depth_topic`, `tf_max_age_s`, etc.
- `test/test_inference_config.py` — at least confirms launch import
  + config file presence + entry point exists. Mirrors
  `strafer_navigation/test/test_nav_config.py` patterns.

This phase ships *no* RL code. It must build clean in
`~/strafer_ws` via `colcon build --packages-select strafer_inference`.

### Phase 2 — DEPTH observation pipeline (3–4 days)

Wire the observation side end-to-end against `PolicyVariant.DEPTH`:

- Subscribe to:
  - `/d555/imu/filtered` — IMU 6-axis
  - `/strafer/joint_states` — encoder velocities
  - `/strafer/odom` — body-frame velocity
  - `/d555/depth/image_rect_raw` (640×360 `sensor_msgs/Image`,
    32FC1 meters) — raw bridged depth
  - Goal pose: `/strafer/goal` or via the action server, in `map`
    frame (consistent with Nav2's input)

- **Inference rate is derived, not hardcoded:**
  The Jetson cannot import `strafer_lab` (Isaac Lab dep). Promote
  the two relevant constants to `strafer_shared.constants` in the
  same commit (additive-only — per the narrow strafer_shared
  exception in the ownership-boundaries module) and read from
  there on both sides:
  ```python
  # In strafer_shared.constants
  POLICY_SIM_DT = 1.0 / 120.0
  POLICY_DECIMATION = 4
  POLICY_PERIOD_S = POLICY_SIM_DT * POLICY_DECIMATION  # 1/30 s = 30 Hz
  ```
  In `strafer_env_cfg.py` re-export the same module-level constants
  so `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` are now
  delegations rather than originals. The inference node imports
  `POLICY_PERIOD_S` directly. Either way, do not duplicate the
  constant — sim and real read from one source of truth.

- **Depth downsample stage (load-bearing for sim-real parity):**

  The policy was trained on 80×60 depth from the `d555_camera`
  policy camera, which is *not bridged*. The bridged stream is the
  640×360 `d555_camera_perception` camera. So:

  ```python
  # Pipeline: 640x360 raw depth → resize to 80x60 → match training preprocessing
  #
  # Training uses mdp/observations.py:depth_image which:
  #   1. Takes RAW meters
  #   2. Replaces values < nearfield_clip (0.4 m) with nearfield_fill (0.2 m)
  #   3. Applies noise (sim only)
  #   4. Scales by 1/max_depth (default 6.0) to [0, 1]
  #
  # Inference must mirror the deterministic parts (steps 1, 2, 4); skip
  # the noise model (real robot has its own noise; sim-in-the-loop too).
  ```

  Use `cv_bridge` or numpy to resize 640×360 → 80×60 with
  area-averaging (`cv2.INTER_AREA`) — `INTER_LINEAR` introduces
  artifacts on sharp depth edges. Profile the resize cost on
  Jetson; it should be ~1 ms.

- Raw dict construction (each field's coordinate frame matters for
  contract parity):
  - `imu_accel`, `imu_gyro` from `/d555/imu/filtered`. IMU frame —
    `assemble_observation` applies `IMU_ACCEL_SCALE` /
    `IMU_GYRO_SCALE`; do not pre-rotate.
  - `encoder_vels_ticks` from `joint_states.velocity` *via
    `mecanum_kinematics.wheel_vels_to_ticks_per_sec`* — do NOT
    recompute encoder geometry locally; that's `strafer_shared` lane.
  - `goal_relative`, `goal_distance`, `goal_heading_to_goal` —
    **body-frame**. Compute by transforming the `map`-frame goal
    pose through TF (`map → base_link`) and taking `(dx, dy)` in
    body frame, then `norm` and `atan2(dy, dx)`. The training env
    does the equivalent transform; if inference computes them in
    `map` frame, the policy turns the wrong way on real-robot.
  - `body_velocity_xy` from `odom.twist.linear.{x,y}` — `odom`
    frame (body-convention). Pass-through.
  - `last_action` — **the raw [-1, 1]³ policy output from the
    previous tick**, NOT the post-`interpret_action` velocity. The
    training env caches `env.action_manager.action`
    ([`mdp/observations.py:499`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)),
    which is the raw policy tensor before
    `MecanumWheelAction.process_actions` clamps and scales. Cache
    the same. Zero on first tick.
  - `depth_image` — flattened 80×60 = 4800 floats in `[0, 1]`,
    matching the training env's `depth_image` term output exactly.

- Pass through `assemble_observation(raw, PolicyVariant.DEPTH)` and
  emit the resulting 4819-dim vector at debug level. **Also log
  every `(obs_summary, action_output, t_inference_ns)` tuple** at
  debug level — `obs_summary` should be a hash + a few summary
  stats (depth mean, depth min, etc.) since logging the full 4819
  vector per tick is too verbose. Needed for post-hoc
  distribution-shift analysis vs. the training set.

No inference yet. Output `cmd_vel = (0, 0, 0)` while the obs
pipeline is validated against a recorded sim-in-the-loop rosbag.

Phase 2 acceptance: the assembled obs vector matches what
`strafer_lab`'s gym environment produces for the same simulated
state, within float32 noise (≤ 1e-5 max abs delta on the NOCAM 19
dims; ≤ 1e-3 max abs delta on the 4800-dim depth, since the
640×360→80×60 resize introduces small numerical differences vs.
the sim's native-80×60 render). This is the sim-to-real contract
guarantee — if it doesn't hold, the policy will not transfer.

### Phase 3 — Inference + action interpretation + safety (2 days)

- Load the model via `load_policy(path, PolicyVariant.DEPTH)` at
  node startup. **Model-load failure is fatal:** if `model_path`
  doesn't exist or `load_policy` raises, the node logs a clear
  error and refuses to start the action server. Operator gets a
  launch-time failure, not silent degradation. The autonomy-side
  `JetsonRosClient` then sees the action server as unavailable
  and falls back to `nav2`.
- **Determinism contract:** per
  [`context/recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
  point 5 — for a recurrent artifact (the DEPTH variant), the
  determinism assertion is "two same-obs calls with `reset()` between
  them are byte-identical." Do NOT assert byte-identity between
  consecutive calls without reset; that asserts the model is
  stateless and would force a false-positive failure on the
  recurrent path. PPO trained via rsl_rl produces a Gaussian policy
  (mean + std); the export step
  ([`policy-export-tooling.md`](../../completed/policy-export-tooling.md))
  freezes the deterministic head, so on top of the reset+same-obs
  case, the action stream is fully reproducible.
- **TRT execution provider is required** (DEPTH inference on the
  Jetson Orin Nano is too slow on CPU/CUDA-EP alone). The export
  brief produces `.onnx` + (optionally) a pre-built `.engine`
  alongside; this node loads via ONNX Runtime with provider
  preference `['TensorrtExecutionProvider',
  'CUDAExecutionProvider', 'CPUExecutionProvider']`. The fallback
  chain handles missing TRT on a dev workstation; production must
  load via TRT and the launch-time benchmark surfaces a warning
  if not.

  **Engine cold-start.** Without a pre-built `.engine` shipped
  alongside (the export brief's optional `tensorrt_engine_path`),
  ONNX Runtime's TRT EP builds the engine on first inference. On
  Jetson Orin Nano this takes 10–30 s for EfficientNet-B0-class
  graphs. During that build the node *appears* to hang to any
  external observer (no logs, no twist publish). Surface the build
  with a startup log line ("Building TensorRT engine, may take
  ~30 s…") and a `node.declare_parameter("ready", False)` that
  flips to `True` after the first successful inference, so
  operator-side health checks can distinguish "still warming up"
  from "wedged."
- On each tick, feed the assembled obs to the loaded callable,
  get back a 3-vector action. Cache it as `last_action` for the
  next tick (raw, pre-`interpret_action` — see Phase 2).
- `interpret_action(action)` → `(vx, vy, omega)` in m/s, rad/s.
- **Output magnitude clamp (safety):** the policy can output
  `(0.99, 0.99, 0.99)` which `interpret_action` denormalizes to
  `(1.55 m/s, 1.55 m/s, 4.15 rad/s)` — but the chassis can't
  physically achieve max forward + max lateral + max spin
  simultaneously (per-wheel motor cap). Apply an L1 ceiling:

  ```python
  l1 = abs(vx) + abs(vy)
  cap = STRAFER_NAV_VEL_SCALE * MAX_LINEAR_VEL  # 1.5683 sim, 0.78 real
  if l1 > cap:
      scale = cap / l1
      vx *= scale; vy *= scale
  # omega clamp is independent: |omega| <= NAV_VEL_SCALE * MAX_ANGULAR_VEL
  ```

  (Post-ship: this math now lives in
  [`strafer_shared.mecanum_kinematics.l1_clamp_twist`](../../../source/strafer_shared/strafer_shared/mecanum_kinematics.py)
  with a torch-batched sibling `l1_clamp_twist_batched` that the sim
  action term consumes. `strafer_inference.obs_pipeline` re-exports
  the scalar form under the original `l1_clamp_velocity` name via an
  alias.)

- Publish to `/strafer/cmd_vel` (real-robot path — driver remaps
  it) or `/cmd_vel` (sim-in-the-loop path matches Nav2's
  publisher contract).
- **Watchdog (6-source for DEPTH):** publish zero twist + log a
  warning if ANY of:
  - No goal received for `goal_timeout_s` (default 1.0 s).
  - No IMU sample for `obs_timeout_s` (default 0.2 s).
  - No `/strafer/joint_states` for `obs_timeout_s` (encoder
    feedback drives the `encoder_vels_ticks` obs term).
  - No `/strafer/odom` for `obs_timeout_s`.
  - No `/d555/depth/image_rect_raw` for `depth_timeout_s` (default
    0.5 s — depth publish rate is slower than IMU).
  - TF lookup `map → base_link` is older than `tf_max_age_s`
    (default 0.5 s).

  The earlier draft of this brief listed five sources and folded
  joint_states under "encoder_vels comes from joint_states." That's
  correct upstream but joint_states is published by the chassis
  driver on its own topic with potentially different rate / latency
  from odom — they're correlated but not the same. A frozen
  joint_states with a fresh odom would zero the encoder field on
  one tick while leaving the body-velocity field fresh, which is
  exactly the kind of half-fresh obs that produces silent
  inference garbage. Independent watchdog source.

Phase 3 acceptance: with a hand-exported `.onnx` test artifact
(any small DEPTH-shaped network — e.g. 4819-dim input → 3-dim
output through a conv stub), the node publishes non-zero
`/cmd_vel` at the derived `infer_period_s` whenever the goal
subscription is active and all five watchdog inputs are fresh.
Zero twist when any watchdog trips. Latency target: end-to-end
(obs receive → cmd_vel publish) < 10 ms p95 on Jetson Orin Nano
via TRT EP. CPU-only fallback path is allowed to be slower; the
launch-time benchmark warns operators if they're running without
TRT.

### Phase 4 — Backend dispatch in `JetsonRosClient` (½ day)

Update
[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
`navigate_to_pose` to honor `execution_backend`:

- Read default from `STRAFER_NAV_BACKEND` env var (default `"nav2"`).
- Per-step override via `step.execution_backend` if the schema
  gains the field (separate brief if so).
- For `"nav2"`: keep current `/navigate_to_pose` action client
  path.
- For `"strafer_direct"`: send to the new `strafer_inference`
  action server (same action type, different node namespace).
  **Server-unavailable fallback:** if the operator selected
  `strafer_direct` but the strafer_inference action server is not
  advertised (Phase 3 makes model-load failure fatal at the
  inference node, so the action server never starts → the autonomy
  client sees no server within the
  `wait_for_server(timeout_sec=10.0)` window), log a clear error
  and fall back to `nav2` for *this mission*. Subsequent missions
  re-attempt the lookup in case the operator manually restarted the
  inference node. This is the same pattern as the existing
  `nav2_unavailable` error path in
  [`ros_client.py`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  `navigate_to_pose`.
- Unknown values: log a clear error naming the value, fall back to
  `"nav2"`. Includes `"hybrid_nav2_strafer"` (filed in
  [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md))
  — the fallback prevents typos from silently failing.

The env-var default is the conservative choice — real-robot
bringup keeps Nav2 unchanged unless the operator explicitly opts
in. Mirrors the `STRAFER_NAV_VEL_SCALE` pattern.

### Phase 5 — sim-in-the-loop validation (extracted)

The end-to-end sim validation (parity bounds, latency benchmark,
architectural-win mission) was originally a Phase 5 here. It now
lives in [`strafer-direct-sim-validation.md`](strafer-direct-sim-validation.md)
so the inference-package PR can merge with all unit-testable
acceptance closed; the operator-driven sim validation runs as a
follow-up once a deployable checkpoint and the sim-in-the-loop rig
are both in hand. Real-robot DEPTH validation gates on the
follow-up's sim validation passing and lives in a brief filed at
that point.

### What's intentionally NOT in scope here

(See the top-level **Out of scope** section below for the full
list. This subsection only flags items that would be tempting to
do as part of the five phases above and shouldn't be.)

- **Replacing Nav2 as the default.** Default stays `nav2`. The
  trained policy is opt-in via `execution_backend` until it has
  validated end-to-end on real hardware (separate brief).
- **Folding inference into `strafer_autonomy`.** The package
  separation is load-bearing per
  [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md):
  `strafer_inference` is robot-local execution, `strafer_autonomy`
  is mission planning. Don't conflate.
- **Re-applying motor sign flips.** `wheel_axis_signs` lives in
  `strafer_shared.mecanum_kinematics`. Do not add a second sign
  correction layer in `strafer_inference`.
- **Deploying NOCAM in `strafer_direct` mode.** The whole reason
  this brief targets DEPTH is that NOCAM can't see obstacles. A
  hand-built NOCAM dummy is fine for plumbing-only smoke tests in
  Phases 1–3 (any deterministic 19-dim → 3-dim mapping); shipping
  it as a real backend is not.

## Acceptance criteria

### Build / structure

- [ ] `colcon build --packages-select strafer_inference` succeeds
      in `~/strafer_ws` on the Jetson; `colcon test` passes the
      package's smoke tests.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. Particular
      attention to
      [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
      (Jetson lane gains a new package) and
      [`context/repo-topology.md`](../../context/repo-topology.md)
      ("Workspace layout" section).

### Contract parity (sim-to-real load-bearing)

The two rosbag-driven parity bounds (≤ 1e-5 NOCAM, ≤ 1e-3 DEPTH)
were extracted into
[`strafer-direct-sim-validation.md`](strafer-direct-sim-validation.md)
alongside Phase 5's other operator-validation items so the
inference-package PR can merge with the unit-testable contract-
parity acceptance below closed and the rig-gated acceptance tracked
in a single follow-up.

- [ ] **`infer_period_s` derived, not hardcoded**: anchored by a
      unit test that asserts the value changes when
      `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` change
      (mock-patched).
- [ ] **`last_action` cache semantics**: unit test feeds a sequence
      of synthetic policy outputs and asserts the *raw [-1, 1]³
      output* (not the post-`interpret_action` velocity) appears
      in the next tick's obs vector at the `last_action` field
      offset.
- [ ] **`goal_relative` body-frame**: unit test sets
      `map → base_link` to a known yaw rotation, places a known
      goal in `map`, asserts the obs's `goal_relative` matches the
      body-frame transform.
- [ ] **Deterministic inference (recurrent-aware)**: per
      [`context/recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
      point 5, two same-obs calls produce byte-identical actions
      *iff* `policy.reset()` is called between them; without an
      intervening `reset()`, the actions differ by construction
      (hidden state evolved). The unit test asserts both prongs at
      the inference-node-fake-policy seam; cross-format parity is
      pinned at
      [`source/strafer_lab/tests/test_recurrent_contract_e2e.py`](../../../../source/strafer_lab/tests/test_recurrent_contract_e2e.py).
      The naive "two consecutive calls byte-identical" assertion is
      wrong for a recurrent artifact and would silently force the
      scripted module into a stateless mode.

### Safety / robustness

- [ ] **Output L1 clamp**: unit test feeds a policy output of
      `(0.99, 0.99, 0.99)` and asserts the published `/cmd_vel`
      satisfies `|vx| + |vy| ≤ NAV_VEL_SCALE * MAX_LINEAR_VEL` and
      `|wz| ≤ NAV_VEL_SCALE * MAX_ANGULAR_VEL`.
- [ ] **Watchdog (6-source)**: unit test using mocked clocks
      asserts zero twist on each independent failure: stale goal,
      stale IMU, stale joint_states, stale odom, stale depth, stale
      `map → base_link` TF.
- [ ] **Model-load failure**: unit test points `model_path` at a
      non-existent file; node logs a clear error and the action
      server is not advertised. The autonomy client must observe
      `wait_for_server` timeout and fall back to `nav2` for that
      mission per the Phase 4 fallback rule.
- [ ] **TRT-EP cold-start surfacing**: with a fresh ONNX (no
      pre-built engine), the node logs a "Building TensorRT engine"
      line at startup and the `ready` parameter flips to `True`
      only after the first successful inference. Operator health
      check can distinguish "warming up" from "wedged."
- [ ] **`(obs_summary, action)` debug logging**: with
      `--ros-args --log-level debug` enabled, every tick emits a
      structured log line containing the obs summary
      (hash + depth-mean + depth-min), the 3-dim action, and
      inference timestamp.

### Integration

- [ ] `JetsonRosClient.navigate_to_pose` routes to
      `strafer_inference` when
      `STRAFER_NAV_BACKEND=strafer_direct`, to Nav2 when unset or
      `nav2`, and falls back to Nav2 with a logged error on
      unknown values.
- [ ] **Real-robot bringup is unaffected** when
      `STRAFER_NAV_BACKEND` is unset.

### Operator-driven sim validation (extracted)

The TRT-EP latency benchmark and the architectural-win sim mission
acceptance (sustained ≥ 1.0 m/s; reach-without-colliding with a
single obstacle) were extracted into
[`strafer-direct-sim-validation.md`](strafer-direct-sim-validation.md).
That brief gates on the sim-in-the-loop rig (rosbag parity, latency
benchmark) and on a deployable DEPTH checkpoint (architectural-win
mission); none of those items are unit-testable, so they merge as a
follow-up rather than blocking this brief's PR.

## Investigation pointers

- [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py)
  — the contract. Already complete; the inference node consumes it.
- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  — gym-env observation groups + normalization scales. Use this to
  author the obs-parity test in Phase 2.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `depth_image` term documents the exact preprocessing pipeline
  the Phase 2 downsample stage must mirror.
- [`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py)
  — registered task IDs, including
  `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` (the deployment
  target).
- [`source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py)
  — clarifies the policy-camera-vs-perception-camera split. The
  `d555_camera` (80×60) is sim-only; the `d555_camera_perception`
  (640×360) is bridged. Phase 2's downsample exists because of
  this split.
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
  "Camera resolutions (sim mirrors real)" section — confirms the
  policy camera is intentionally not bridged (and why).
- [`source/strafer_ros/strafer_navigation/`](../../../../source/strafer_ros/strafer_navigation/)
  — closest existing example of a `strafer_ros/*` ament_python
  package with launch + config + tests. Mirror this layout.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  `JetsonRosClient.navigate_to_pose` — Phase 4's edit site.

## Out of scope

- **Policy export tooling.** Filed as
  [`policy-export-tooling.md`](../../completed/policy-export-tooling.md) (DGX-lane).
  Now MVP-required for both TorchScript and ONNX-with-TRT-EP paths.
- **Pre-deployment training with goal-position noise.** Filed as
  [`goal-noise-training`](goal-noise-training.md)
  (DGX-lane). Required for VLM-grounded mission quality;
  deployment-prep training pass on top of a converged DEPTH
  baseline checkpoint.
- **MPPI critic re-tuning.** That's
  [`completed/mppi-critic-tuning-for-sim-envelope.md`](../../completed/mppi-critic-tuning-for-sim-envelope.md);
  this brief's existence presupposes that ceiling.
- **The planner-side wavy-path issue.** That's
  [`nav2-startup-unknown-donut-path-noise.md`](../../completed/nav2-startup-unknown-donut-path-noise.md).
  Affects the `nav2` backend's startup behavior; orthogonal to
  RL policy execution.
- **`hybrid_nav2_strafer` mode.** Filed as
  [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md)
  (cross-lane). Requires a new `PolicyVariant.NOCAM_SUBGOAL`
  trained against a subgoal-following env that doesn't exist yet
  ([`subgoal-env`](subgoal-env.md)).
  Hybrid coexists with `strafer_direct`; this brief doesn't block
  on it.
- **Sim-in-the-loop validation against a trained checkpoint.** Lives
  in [`strafer-direct-sim-validation.md`](strafer-direct-sim-validation.md);
  carries the rosbag parity bounds, the TRT-EP latency benchmark,
  and the architectural-win mission acceptance.
- **Real-robot DEPTH validation.** File as a separate brief
  (`strafer-inference-real-robot-validation.md`) once
  [`strafer-direct-sim-validation.md`](strafer-direct-sim-validation.md)
  ships. Real-robot DEPTH introduces sensor-noise distribution
  shift, lighting variance, and dynamic obstacles that warrant
  their own scope.
- **NOCAM in `strafer_direct` as a deployable mode.** Hand-built
  NOCAM dummy artifacts are fine for plumbing-only smoke tests
  during Phases 1–3 (the action server / dispatch / watchdog
  plumbing doesn't depend on which variant is loaded). Shipping
  NOCAM as the actual `strafer_direct` backend is not safe and
  not in scope.
- **Removing or downgrading Nav2.** Nav2 is the default and the
  fallback. Even after the trained policy is deployed, the
  `nav2` backend should remain available for missions where
  classical planning is preferred (long-horizon paths in known
  maps, recovery from policy wedges, etc.).
