# Build the `strafer_inference` Jetson runtime for trained-policy execution

**Type:** task / feature
**Owner:** Jetson (new ROS package lives in `strafer_ros/`)
**Priority:** P1 — the trained-policy backend is the architectural answer to MPPI's plateau (see Story).
**Estimate:** L (~1 week initial NOCAM path + integration). TensorRT optimization moves to the future DEPTH-variant brief — NOCAM is a 19-dim MLP and runs sub-millisecond on CPU, so TRT's payoff is in the depth-conv path, not here.
**Branch:** task/strafer-inference-package

## Story

As a **mission operator running navigate_to_pose missions on the real
chassis or in sim-in-the-loop**, I want **an `execution_backend` that
runs the Isaac-Lab-trained RL navigation policy through the same
observation / action contract the policy was trained on**, so that
**robot motion converges to the velocity envelope the policy and
joystick teleop already demonstrate (~1.4 m/s sustained), instead of
plateauing at ~63 % of that ceiling under MPPI's critic landscape**.

The MPPI critic-tuning sweep
([`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md))
moved sustained median odom vx from 0.001 m/s (bridge unit-mismatch +
narrow exploration) to 0.632 m/s through a converged set of
sim-only critic / sampling overrides — but five tuning passes after
the v2 critic rebalance hit a plateau. Three regressions in a row
(`GoalCritic` weight up, `iteration_count` up, `temperature` down)
exhausted the brief's full single-knob candidate list. The remaining
gap to the ≥1.0 m/s acceptance threshold isn't reachable by further
tuning of this controller at this critic landscape; on the Jetson
Orin Nano, MPPI's CPU budget caps at `iteration_count=1` (B-pass 4
showed deadline misses at 2). The trained policy was trained on the
full envelope, so it's the architectural lift past the plateau —
exactly what `STRAFER_AUTONOMY_ROS.md` (archived) called out as the
intended end-state when MPPI was first wired in as the backup.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [mppi-critic-tuning-for-sim-envelope.md](../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the predecessor whose validation surfaced the MPPI plateau and
  motivated this work as the architectural alternative.

## Context

### What already exists

The shared sim-to-real contract is already in place — policy export
is the missing dependency, not the contract:

[`source/strafer_shared/strafer_shared/policy_interface.py`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
ships:

| API | Purpose |
|-----|---------|
| `PolicyVariant.NOCAM` (19 dims) | IMU + encoders + goal-relative + body-vel + last-action |
| `PolicyVariant.DEPTH` (4819 dims) | NOCAM + 4800-dim flattened depth |
| `assemble_observation(raw, variant)` | Normalize + concatenate raw sensor dict → flat float32 obs |
| `interpret_action(action_normalized)` | Denormalize `[-1, 1]` policy output → `(vx, vy, omega)` in m/s, rad/s |
| `action_to_wheel_ticks(action_normalized)` | Convenience: action → wheel ticks/sec via `mecanum_kinematics` |
| `load_policy(path, variant)` | Load `.pt` (TorchScript) or `.onnx` (ONNX Runtime); returns `(obs) → action` callable |
| `benchmark_policy(policy, variant, n_iters)` | Inference-latency stats |

The Isaac Lab side keeps the env aligned to this contract via
`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`
(observation groups L249–258, normalization scales L225–234). Both
lanes already reference this module — the contract has lived in the
codebase since before the Nav2 backup was wired.

### What's missing

1. **`source/strafer_ros/strafer_inference/`** — the ROS 2 ament_python
   package that holds the inference node, launch file, config, tests,
   and entry point. Empty today.
2. **`Scripts/export_policy.py`** — the DGX-side checkpoint → `.pt` /
   `.onnx` conversion tool. Filed as
   [`policy-export-tooling.md`](policy-export-tooling.md). Hard
   dependency in DGX lane; this brief blocks on it for end-to-end
   validation but can land the runtime plumbing against a
   hand-exported test artifact (any deterministic 19-dim → 3-dim
   mapping).
3. **`execution_backend` dispatch in
   [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)**
   `JetsonRosClient.navigate_to_pose` — currently routes unconditionally
   through Nav2's `/navigate_to_pose` action. Needs to honor an
   `execution_backend` field (env-var `STRAFER_NAV_BACKEND` or per-step)
   with values `nav2` (default) and `strafer_direct`.
4. **A deployable checkpoint** from `strafer_lab` PPO training that
   has actually been trained with the current observation contract.
   Out of scope for this brief — gates the validation steps but not
   the package skeleton. **The brief targets the
   `strafer_navigation` env config in
   [`strafer_env_cfg.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
   with `PolicyVariant.NOCAM`, `_DEFAULT_NAV_DECIMATION = 4`,
   `_DEFAULT_NAV_SIM_DT = 1/120`** — pin this in the deployment
   metadata so a future env change forces a co-update of the
   inference node's rate calculation (see Phase 2).

### Two execution modes

The archived design doc proposed three modes. The third
(`hybrid_nav2_strafer`: Nav2 emits subgoals, RL drives between them)
requires a *different trained policy* — one trained on
subgoal-following, not the goal-directed policy `strafer_lab`
currently produces — so it gets its own brief at
[`strafer-inference-hybrid-mode.md`](strafer-inference-hybrid-mode.md).
Shipping it as "Phase X maybe later" inside this brief would land a
half-finished mode; separate brief keeps both scopes clean.

| Mode | Global plan | Local control | Filed as |
|------|-------------|---------------|----------|
| `nav2` (default) | Nav2 GridBased | Nav2 MPPI | Shipped today |
| `strafer_direct` | none — direct goal | trained policy via `strafer_inference` | **This brief** |
| `hybrid_nav2_strafer` | Nav2 GridBased | trained `NOCAM_SUBGOAL` policy via `strafer_inference` | [`strafer-inference-hybrid-mode.md`](strafer-inference-hybrid-mode.md) |

The mode is a *robot-side* execution choice; the autonomy / planner
layer doesn't change. Operator-facing surface stays
`navigate_to_pose(goal_pose=...)`; backend choice is a config knob.

## Approach

Work proceeds in four tightly-scoped phases. Each phase gets its own
review-able commit on the same branch — no cross-phase intermixing
(makes bisection trivial when the trained policy lands and validation
surfaces an integration bug). TensorRT integration is deliberately
*not* a phase here — see the DEPTH variant follow-up in Out of scope.

### Phase 1 — Package skeleton (½ day)

Create `source/strafer_ros/strafer_inference/`:
- `package.xml` (ament_python), `setup.py`, `setup.cfg`, `resource/strafer_inference`
- `strafer_inference/__init__.py`
- `strafer_inference/inference_node.py` — empty rclpy `Node` subclass
  that subscribes to the contract inputs, publishes `/strafer/cmd_vel`,
  and exposes a `/navigate_to_pose` action server (parallel to Nav2's,
  same action type) but does nothing yet.
- `launch/inference.launch.py` — brings up the node with a config
  pointing at a model artifact path.
- `config/inference.yaml` — `model_path`, `policy_variant`, `infer_rate_hz`,
  `goal_topic`, `cmd_vel_topic`, etc.
- `test/test_inference_config.py` — at least confirms launch import +
  config file presence + entry point exists. Mirrors
  `strafer_navigation/test/test_nav_config.py` patterns.

This phase ships *no* RL code — just the ROS plumbing that future
phases hang work off. It must build clean in `~/strafer_ws` via
`colcon build --packages-select strafer_inference`.

### Phase 2 — NOCAM observation pipeline (1–2 days)

Wire the observation side end-to-end against `PolicyVariant.NOCAM`:

- Subscribe to `/d555/imu/filtered`, `/strafer/joint_states`,
  `/strafer/odom` (matching the
  [`bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)
  contract in both sim and real).
- Subscribe to `/strafer/goal` (or accept goal pose via the action
  server) — pose in `map` frame (consistent with Nav2's input).
- On a fixed-rate timer, assemble the raw dict expected by
  `assemble_observation`. **Inference rate is derived, not
  hardcoded:**

  ```python
  # Match the training env exactly. Hardcoding 30 Hz couples to the
  # current env config and silently breaks when training rate changes.
  from strafer_lab.tasks.navigation.strafer_env_cfg import (
      _DEFAULT_NAV_SIM_DT, _DEFAULT_NAV_DECIMATION,
  )
  infer_period_s = _DEFAULT_NAV_SIM_DT * _DEFAULT_NAV_DECIMATION
  # Currently 1/120 * 4 = 1/30 s = 30 Hz.
  ```

  If `strafer_lab` isn't installed on the Jetson (it isn't), import
  the constants via `strafer_shared` re-export — add the re-export
  in the same commit. Either way, do not duplicate the constant.

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
    pose through TF (`map → base_link`) and taking
    `(dx, dy)` in body frame, then `norm` and
    `atan2(dy, dx)`. The training env (`mdp/observations.py`) does
    the equivalent transform; if inference computes them in `map`
    frame, the policy turns the wrong way on real-robot.
  - `body_velocity_xy` from `odom.twist.linear.{x,y}` —
    `odom` frame, which `strafer_driver`'s odom publisher emits in
    body convention (X forward, Y left). Pass-through.
  - `last_action` — **the raw [-1, 1]³ policy output from the
    previous tick**, NOT the post-`interpret_action` velocity. The
    training env caches `env.action_manager.action`
    ([`mdp/observations.py:499`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)),
    which is the raw policy tensor before `MecanumWheelAction.process_actions`
    clamps and scales. Cache the same. Zero on first tick (matches
    env reset).

- Pass through `assemble_observation(raw, variant)` and emit the
  resulting 19-dim vector at debug level. **Also log every
  `(obs_vector, action_output, t_inference_ns)` tuple** at debug
  level — needed for post-hoc distribution-shift analysis vs. the
  training set, and for diagnosing real-vs-sim divergence.

No inference yet. Output `cmd_vel = (0, 0, 0)` while the obs pipeline
is validated against a recorded sim rosbag.

Phase 2 acceptance: the assembled obs vector matches what
`strafer_lab`'s gym environment produces for the same simulated
state, within float32 noise (≤ 1e-5 max abs delta on the 19-dim
vector). This is the sim-to-real contract guarantee — if it
doesn't hold, the policy will not transfer.

### Phase 3 — Inference + action interpretation + safety (1 day)

- Load the model via `load_policy(path, PolicyVariant.NOCAM)` at node
  startup. **Model-load failure is fatal:** if `model_path` doesn't
  exist or `load_policy` raises, the node logs a clear error and
  refuses to start the action server. Operator gets a launch-time
  failure, not silent degradation. The autonomy-side
  `JetsonRosClient` then sees the action server as unavailable and
  falls back to `nav2` — same surface as if the inference node
  weren't launched at all.
- **Determinism contract:** the loaded callable must be
  deterministic — same observation → same action across calls.
  PPO trained via rsl_rl produces a Gaussian policy (mean + std);
  the export step (DGX-lane) must freeze the deterministic head
  (mean only, no sampling). Phase 3 asserts this with a unit test:
  feed the same obs vector twice, assert byte-identical action
  outputs. If the test fails, the export tooling shipped a
  stochastic head and needs to be fixed before deployment — surface
  the failure early.
- On each tick, feed the assembled obs to the loaded callable, get
  back a 3-vector action. Cache it as `last_action` for the next
  tick (raw, pre-`interpret_action` — see Phase 2).
- `interpret_action(action)` → `(vx, vy, omega)` in m/s, rad/s.
- **Output magnitude clamp (safety):** the policy can output
  `(0.99, 0.99, 0.99)` which `interpret_action` denormalizes to
  `(1.55 m/s, 1.55 m/s, 4.15 rad/s)` — but the chassis can't
  physically achieve max forward + max lateral + max spin
  simultaneously (per-wheel motor cap). After `interpret_action`,
  apply an L1 ceiling:

  ```python
  l1 = abs(vx) + abs(vy)
  cap = STRAFER_NAV_VEL_SCALE * MAX_LINEAR_VEL  # 1.5683 sim, 0.78 real
  if l1 > cap:
      scale = cap / l1
      vx *= scale; vy *= scale
  # omega clamp is independent: |omega| <= NAV_VEL_SCALE * MAX_ANGULAR_VEL
  ```

  Documented as "safety, not contract" — the bridge does similar
  clamping in sim, but the real chassis has no per-wheel safety
  fallback. One line, real protection.
- Publish to `/strafer/cmd_vel` (real-robot path — driver remaps it
  to the chassis) or `/cmd_vel` (sim-in-the-loop path matches Nav2's
  publisher contract).
- **Watchdog (expanded):** publish zero twist + log a warning if
  ANY of the following:
  - No goal received for `goal_timeout_s` (default 1.0 s).
  - No IMU sample for `obs_timeout_s` (default 0.2 s).
  - No `/strafer/odom` for `obs_timeout_s`.
  - **TF lookup `map → base_link` is older than
    `tf_max_age_s` (default 0.5 s)** — `goal_relative` depends on
    this transform; stale TF = wrong observation = policy gets
    confused. SLAM stalls produce this; it's a real failure mode.

Phase 3 acceptance: with a hand-exported `.pt` artifact (any small
model that produces valid `[-1, 1]^3` outputs is acceptable for
plumbing validation — semantic correctness comes in real-world
validation), the node publishes non-zero `/cmd_vel` at the derived
`infer_period_s` whenever the goal subscription is active and all
four watchdog inputs are fresh. Zero twist when any watchdog trips.
Latency target: end-to-end (obs receive → cmd_vel publish) < 5 ms
p95 on Jetson Orin Nano. `benchmark_policy` gives the inference-only
number (sub-millisecond expected for NOCAM); the wrapping
infrastructure shouldn't add more than ~3 ms.

### Phase 4 — Backend dispatch in `JetsonRosClient` (½ day)

Update
[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
`navigate_to_pose` to honor `execution_backend`:

- Read default from `STRAFER_NAV_BACKEND` env var (default `"nav2"`).
- Per-step override via `step.execution_backend` if the schema gains
  the field (separate brief if so — small change).
- For `"nav2"`: keep current `/navigate_to_pose` action client path.
- For `"strafer_direct"`: send to the new `strafer_inference`
  action server (same action type, different node namespace).
- Unknown values: log a clear error naming the value, fall back to
  `"nav2"`. Includes `"hybrid_nav2_strafer"` (filed as a follow-up
  brief) — the fallback prevents typos from silently failing.

The env-var default is the conservative choice — real-robot bringup
keeps Nav2 unchanged unless the operator explicitly opts in. Mirrors
the `STRAFER_NAV_VEL_SCALE` pattern.

### What's intentionally NOT in scope here

(See the top-level **Out of scope** section below for the full list.
This subsection only flags items that would be tempting to do as
part of the four phases above and shouldn't be.)

- **Replacing Nav2 as the default.** Default stays `nav2`. The
  trained policy is opt-in via `execution_backend` until it has
  validated end-to-end on real hardware.
- **Folding inference into `strafer_autonomy`.** The package
  separation is load-bearing per
  [`context/ownership-boundaries.md`](../context/ownership-boundaries.md):
  `strafer_inference` is robot-local execution, `strafer_autonomy` is
  mission planning. Don't conflate.
- **Re-applying motor sign flips.** `wheel_axis_signs` lives in
  `strafer_shared.mecanum_kinematics`. Do not add a second sign
  correction layer in `strafer_inference` — same trap the archived
  ROS architecture doc explicitly called out.

## Acceptance criteria

### Build / structure

- [ ] `colcon build --packages-select strafer_inference` succeeds in
      `~/strafer_ws` on the Jetson; `colcon test` passes the
      package's smoke tests.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. Particular
      attention to
      [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
      (Jetson lane gains a new package) and
      [`context/repo-topology.md`](../context/repo-topology.md)
      ("Workspace layout" section).

### Contract parity (sim-to-real load-bearing)

- [ ] **Obs vector parity**: with a recorded sim-in-the-loop rosbag,
      the inference node's assembled obs vector at each tick matches
      the gym-env obs at the same sim timestamp within float32 noise
      (≤ 1e-5 max abs delta on the 19-dim vector). If this drifts,
      the policy will not transfer.
- [ ] **`infer_period_s` derived, not hardcoded**: the node
      computes its rate from
      `_DEFAULT_NAV_SIM_DT * _DEFAULT_NAV_DECIMATION` (re-exported
      via `strafer_shared` if `strafer_lab` isn't a Jetson
      dependency). Anchored by a unit test that asserts the value
      changes when those constants change (mock-patched).
- [ ] **`last_action` cache semantics**: unit test feeds the node a
      sequence of synthetic policy outputs and asserts the *raw
      [-1, 1]³ output* (not the post-`interpret_action` velocity)
      appears in the next tick's obs vector at the
      `last_action` field offset.
- [ ] **`goal_relative` body-frame**: unit test sets
      `map → base_link` to a known yaw rotation, places a known
      goal in `map`, asserts the obs's `goal_relative` matches the
      body-frame transform.
- [ ] **Deterministic inference**: feed the loaded policy the same
      obs vector twice; assert byte-identical action outputs. If
      this fails, the export tooling shipped a stochastic head and
      it must be fixed before deployment.

### Safety / robustness

- [ ] **Output L1 clamp**: unit test feeds a policy output of
      `(0.99, 0.99, 0.99)` and asserts the published `/cmd_vel`
      satisfies `|vx| + |vy| ≤ NAV_VEL_SCALE * MAX_LINEAR_VEL` and
      `|wz| ≤ NAV_VEL_SCALE * MAX_ANGULAR_VEL`.
- [ ] **Watchdog (4-source)**: unit test using mocked clocks
      asserts zero twist on each independent failure: stale goal,
      stale IMU, stale odom, stale `map → base_link` TF.
- [ ] **Model-load failure**: unit test points
      `model_path` at a non-existent file; node logs a clear error
      and the action server is not advertised. Operator gets a
      launch-time failure, not silent degradation.
- [ ] **`(obs, action)` debug logging**: with `--ros-args --log-level
      debug` enabled, every tick emits a structured log line
      containing the 19-dim obs vector, 3-dim action, and inference
      timestamp. Verifiable via `ros2 log list` or rqt_console.

### Integration

- [ ] `JetsonRosClient.navigate_to_pose` routes to
      `strafer_inference` when `STRAFER_NAV_BACKEND=strafer_direct`,
      to Nav2 when unset or `nav2`, and falls back to Nav2 with a
      logged error on unknown values.
- [ ] **Real-robot bringup is unaffected** when `STRAFER_NAV_BACKEND`
      is unset — same controller, same launch graph as before this
      brief.
- [ ] **Latency p95 < 5 ms** (obs receive → cmd_vel publish) on
      Jetson Orin Nano. Recorded in the PR via the
      [`tune_capture.py`](../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
      harness extended to also capture inference-side timestamps,
      OR a dedicated `benchmark_inference_node.py`. NOCAM on CPU
      should clear this comfortably; the budget exists for the
      future DEPTH variant.

### End-to-end (gates on the deployable checkpoint dependency)

- [ ] On a `translate forward 3 m` sim mission with
      `STRAFER_NAV_BACKEND=strafer_direct`, observed
      `/strafer/odom.linear.x` 1 s sustained median ≥ 1.0 m/s. This
      is the architectural-win acceptance — the metric the MPPI
      brief plateaued under at 0.632 m/s.

## Investigation pointers

- `source/strafer_shared/strafer_shared/policy_interface.py` — the
  contract. `assemble_observation`, `interpret_action`, `load_policy`,
  `benchmark_policy`, `PolicyVariant`. Already complete; the
  inference node consumes it.
- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`
  L225–258 — the matching gym-env observation groups and normalization
  scales. Use this to author the obs-parity test in Phase 2.
- `source/strafer_ros/strafer_navigation/{launch,test,config,setup.py}`
  — closest existing example of a `strafer_ros/*` ament_python
  package with launch + config + tests. Mirror this layout.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`
  `JetsonRosClient.navigate_to_pose` — Phase 4's edit site.
- Archived design (still useful, not authoritative): `git show
  695e8c0a:docs/archive/STRAFER_AUTONOMY_ROS.md` and
  `git show 695e8c0a:docs/archive/SIM_TO_REAL_PLAN.md` spell out the
  three-mode design + the validation sequence (export → Jetson
  runtime → motion → sim-vs-real comparison) that this brief's
  acceptance criteria operationalize.
- [`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the run-table evolution under MPPI documents the plateau this
  package's trained-policy backend is expected to break past.

## Out of scope

- **Policy export tooling (`Scripts/export_policy.py`).** Filed as
  [`policy-export-tooling.md`](policy-export-tooling.md) (DGX-lane).
  Jetson cannot land it. This brief blocks on it for end-to-end
  validation but can complete Phases 1–4 against a hand-exported
  test artifact.
- **Pre-deployment training with goal-position noise.** Filed as
  [`policy-goal-noise-training.md`](policy-goal-noise-training.md)
  (DGX-lane). Required for the policy to be robust to the
  ±0.2–0.5 m localization error of VLM-grounded goals; deployment-
  prep training pass on top of a converged baseline checkpoint.
- **MPPI critic re-tuning.** That's
  [`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md);
  this brief's existence presupposes that ceiling.
- **The planner-side wavy-path issue.** That's
  [`nav2-startup-unknown-donut-path-noise.md`](nav2-startup-unknown-donut-path-noise.md).
  Affects the `nav2` backend's startup behavior; orthogonal to RL
  policy execution (the trained policy doesn't consume Nav2 paths in
  `strafer_direct` mode).
- **DEPTH variant inference.** First transfer target is NOCAM;
  follow-up brief once NOCAM validates. **The DEPTH variant brief
  also owns the TensorRT integration** (`onnxruntime-gpu` + TRT
  execution provider on the Jetson; pre-built `.engine` if benchmarks
  demand it). NOCAM is a 19-dim MLP and runs sub-millisecond on CPU,
  so TRT's payoff is in the depth-conv path. Adding TRT support to
  `load_policy` here would be premature — install footguns
  (TRT EP version pinned to JetPack version, etc.) earn their
  maintenance cost only when something actually needs them.
- **`hybrid_nav2_strafer` mode.** Filed as
  [`strafer-inference-hybrid-mode.md`](strafer-inference-hybrid-mode.md)
  (cross-lane). Requires a new `PolicyVariant.NOCAM_SUBGOAL` and a
  subgoal-following policy trained against it; the hybrid backend
  consumes Nav2's `/plan` for global geometry while the trained
  policy does local control. Depends on this brief shipping
  `strafer_direct` first.
- **Removing or downgrading Nav2.** Nav2 is the default and the
  fallback. Even after the trained policy is deployed, the
  `nav2` backend should remain available for missions where
  classical planning is preferred (long-horizon paths in known
  maps, recovery from policy wedges, etc.).
