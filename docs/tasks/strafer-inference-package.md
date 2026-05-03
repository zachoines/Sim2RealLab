# Build the `strafer_inference` Jetson runtime for trained-policy execution

**Type:** task / feature
**Owner:** Jetson (new ROS package lives in `strafer_ros/`)
**Priority:** P1 — the trained-policy backend is the architectural answer to MPPI's plateau (see Story).
**Estimate:** L (~1 week initial NOCAM path + integration; +0.5–1 week TensorRT optimization once a deployable checkpoint exists)
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
([`mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md))
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
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
- [mppi-critic-tuning-for-sim-envelope.md](mppi-critic-tuning-for-sim-envelope.md)
  — the predecessor whose validation surfaced the MPPI plateau and
  motivated this work as the architectural alternative.

## Context

### What already exists

The shared sim-to-real contract is already in place — policy export
is the missing dependency, not the contract:

[`source/strafer_shared/strafer_shared/policy_interface.py`](../../source/strafer_shared/strafer_shared/policy_interface.py)
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
   `.onnx` conversion tool. Tracked in
   [`DEFERRED_WORK.md`](DEFERRED_WORK.md) under `strafer_lab` →
   "Policy export tooling". Hard dependency in DGX lane; this brief
   blocks on it for end-to-end validation but can land the runtime
   plumbing against a hand-exported test artifact.
3. **`execution_backend` dispatch in
   [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)**
   `JetsonRosClient.navigate_to_pose` — currently routes unconditionally
   through Nav2's `/navigate_to_pose` action. Needs to honor an
   `execution_backend` field (env-var `STRAFER_NAV_BACKEND` or per-step)
   with values `nav2` (default), `strafer_direct`, `hybrid_nav2_strafer`.
4. **A deployable checkpoint** from `strafer_lab` PPO training that
   has actually been trained with the current observation contract.
   Out of scope for this brief — gates the validation steps but not
   the package skeleton.

### Three execution modes

The archived design doc spelled out three modes. They share the same
ROS contract (`navigate_to_pose` skill) but differ in what runs the
local-control loop:

| Mode | Global plan | Local control | Status |
|------|-------------|---------------|--------|
| `nav2` (MVP, default) | Nav2 GridBased | Nav2 MPPI | Shipped |
| `strafer_direct` | none | trained policy via `strafer_inference` | This brief |
| `hybrid_nav2_strafer` | Nav2 GridBased | trained policy via `strafer_inference`, fed Nav2 path subgoals | Follow-up after `strafer_direct` validates |

The mode is a *robot-side* execution choice; the autonomy / planner
layer doesn't change. Operator-facing surface stays
`navigate_to_pose(goal_pose=...)`; backend choice is a config knob.

## Approach

Work proceeds in five tightly-scoped phases. Each phase gets its own
review-able commit on the same branch — no cross-phase intermixing
(makes bisection trivial when the trained policy lands and validation
surfaces an integration bug).

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
  [`bridge-runtime-invariants.md`](context/bridge-runtime-invariants.md)
  contract in both sim and real).
- Subscribe to `/strafer/goal` (or accept goal pose via the action
  server) — robot-frame target.
- On a fixed rate (`infer_rate_hz`, default 30 Hz to match training
  decimation), assemble the raw dict expected by
  `assemble_observation`:
  - `imu_accel`, `imu_gyro` from IMU
  - `encoder_vels_ticks` from `joint_states.velocity` *via
    `mecanum_kinematics.wheel_vels_to_ticks_per_sec`* — do NOT recompute
    encoder geometry locally; that's `strafer_shared` lane.
  - `goal_relative`, `goal_distance`, `goal_heading_to_goal` from goal
    pose - current pose, in body frame.
  - `body_velocity_xy` from `odom.twist.linear.{x,y}`.
  - `last_action` cached from the previous tick (zero on first tick).
- Pass through `assemble_observation(raw, variant)` and log the
  resulting 19-dim vector at debug level.

No inference yet. Output `cmd_vel = (0, 0, 0)` while the obs pipeline
is validated against a recorded sim rosbag. Acceptance for this
phase: the assembled obs vector matches what `strafer_lab`'s gym
environment produces for the same simulated state, within float32
noise. This is the sim-to-real contract guarantee — if it doesn't
hold, the policy will not transfer.

### Phase 3 — Inference + action interpretation (1 day)

- Load the model via `load_policy(path, PolicyVariant.NOCAM)` at node
  startup.
- On each tick, feed the assembled obs to the loaded callable, get
  back a 3-vector action.
- `interpret_action(action)` → `(vx, vy, omega)`.
- Publish to `/strafer/cmd_vel` (real-robot path — driver remaps it
  to the chassis) or `/cmd_vel` (sim-in-the-loop path matches Nav2's
  publisher contract).
- Watchdog: if no goal received for `goal_timeout_s` (default 1.0)
  OR no IMU/odom for `obs_timeout_s` (default 0.2), publish zero
  twist and log a warning. Hard requirement — autonomy / safety
  invariant.

Phase 3 acceptance: with a hand-exported `.pt` artifact (any small
model that produces valid `[-1, 1]^3` outputs is acceptable for
plumbing validation — semantic correctness comes in Phase 5), the
node publishes non-zero `/cmd_vel` at `infer_rate_hz` whenever the
goal subscription is active and watchdog inputs are fresh. Latency
target: end-to-end (obs receive → cmd_vel publish) < 5 ms p95 on
Jetson Orin Nano. `benchmark_policy` should already give us the
inference-only number; the wrapping infrastructure shouldn't add
more than ~2 ms.

### Phase 4 — Backend dispatch in `JetsonRosClient` (½ day)

Update
[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
`navigate_to_pose` to honor `execution_backend`:

- Read default from `STRAFER_NAV_BACKEND` env var (default `"nav2"`).
- Per-step override via `step.execution_backend` if the schema gains
  the field (separate brief if so — small change).
- For `"nav2"`: keep current `/navigate_to_pose` action client path.
- For `"strafer_direct"`: send to the new `strafer_inference`
  action server (same action type, different node namespace).
- For `"hybrid_nav2_strafer"`: out of scope this brief — log a
  "not yet implemented" error and fall back to `"nav2"`.

The env-var default is the conservative choice — real-robot bringup
keeps Nav2 unchanged unless the operator explicitly opts in. Mirrors
the `STRAFER_NAV_VEL_SCALE` pattern.

### Phase 5 — TensorRT path + Jetson optimization (½–1 week, after a deployable checkpoint exists)

`load_policy` already supports `.onnx` via ONNX Runtime. ONNX Runtime
on the Jetson can use the TensorRT execution provider transparently
(no API change in the runtime; just a build/install detail). Concrete
work:

- Verify `onnxruntime-gpu` + TensorRT are installed in the
  Jetson Python environment; if not, document the install path
  (typically `python3 -m pip install onnxruntime-gpu` from NVIDIA's
  Jetson wheel index).
- Add a `providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
  hint to `load_policy` for `.onnx` models (or a separate
  `load_policy_trt` path if the API divergence gets ugly). The
  fallback chain handles missing TRT gracefully.
- Run `benchmark_policy` on Jetson with each provider in the
  fallback chain. Record the latency table in the PR.
- If a true `.engine` (pre-built TRT plan) outperforms ONNX+TRT,
  consider extending `load_policy` to a third format. Defer that
  decision until benchmarks demand it — adding format support
  incurs maintenance cost permanently.

### What's intentionally NOT in scope here

- **Replacing Nav2 as the default.** Default stays `nav2`. The
  trained policy is opt-in via `execution_backend` until it has
  validated end-to-end on real hardware.
- **Folding inference into `strafer_autonomy`.** The package
  separation is load-bearing per
  [`context/ownership-boundaries.md`](context/ownership-boundaries.md):
  `strafer_inference` is robot-local execution, `strafer_autonomy` is
  mission planning. Don't conflate.
- **Re-applying motor sign flips.** `wheel_axis_signs` lives in
  `strafer_shared.mecanum_kinematics`. Do not add a second sign
  correction layer in `strafer_inference` — same trap the archived
  ROS architecture doc explicitly called out.
- **DEPTH variant.** First transfer target is NOCAM; depth-based
  policy adds a 4800-dim depth observation that's harder to validate
  end-to-end. Add as a follow-up brief once NOCAM transfers cleanly.
- **`hybrid_nav2_strafer`.** Out of scope per Phase 4. Filed as a
  follow-up brief if `strafer_direct` validates and the operator
  wants Nav2 path-planning + RL local control.

## Acceptance criteria

- [ ] `colcon build --packages-select strafer_inference` succeeds in
      `~/strafer_ws` on the Jetson; `colcon test` passes the
      package's smoke tests (Phase 1 unit tests + Phase 3 latency
      assertion).
- [ ] `assemble_observation` round-trip parity: with a recorded
      sim-in-the-loop rosbag, the inference node's assembled obs
      vector at each tick matches the gym-env obs at the same sim
      timestamp within float32 noise (≤ 1e-5 max abs delta).
      Anchors the sim-to-real contract — if this drifts, the policy
      will not transfer.
- [ ] With a hand-exported test `.pt` artifact, the node publishes
      non-zero `/cmd_vel` at `infer_rate_hz` while the goal /
      observation watchdogs are fresh, and zero twist when either
      times out. Watchdog test is a unit test using mocked clocks.
- [ ] End-to-end latency p95 < 5 ms (obs receive → cmd_vel publish)
      on Jetson Orin Nano. Recorded in the PR via the harness from
      [`mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md)
      (`tune_capture.py`) extended to also capture inference-side
      timestamps, OR a dedicated `benchmark_inference_node.py`.
- [ ] `JetsonRosClient.navigate_to_pose` routes to
      `strafer_inference` when `STRAFER_NAV_BACKEND=strafer_direct`
      and to Nav2 otherwise. Default unset = `nav2` — anchored by a
      unit test asserting the default path is byte-identical.
- [ ] Real-robot bringup is unaffected when `STRAFER_NAV_BACKEND` is
      unset — same controller, same launch graph as before this
      brief.
- [ ] Once a deployable checkpoint lands (separate dependency): on
      a `translate forward 3 m` sim mission with
      `STRAFER_NAV_BACKEND=strafer_direct`, observed
      `/strafer/odom.linear.x` 1 s sustained median ≥ 1.0 m/s. This
      is the architectural-win acceptance — the same metric the MPPI
      brief plateaued under.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. Particular
      attention to
      [`context/ownership-boundaries.md`](context/ownership-boundaries.md)
      (Jetson lane gains a new package) and
      [`context/repo-topology.md`](context/repo-topology.md)
      ("Workspace layout" section).

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
- [`mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md)
  — the run-table evolution under MPPI documents the plateau this
  package's trained-policy backend is expected to break past.

## Out of scope

- **Policy export tooling (`Scripts/export_policy.py`).** Tracked in
  [`DEFERRED_WORK.md`](DEFERRED_WORK.md) → `strafer_lab`. DGX-lane
  brief; Jetson cannot land it. This brief blocks on it for end-to-end
  validation but can complete Phases 1–4 against a hand-exported test
  artifact.
- **Pre-deployment training with goal-position noise.** Tracked in
  [`DEFERRED_WORK.md`](DEFERRED_WORK.md) → `strafer_lab`. Required for
  the policy to be robust to the ±0.2–0.5 m localization error of
  VLM-grounded goals; DGX-lane.
- **MPPI critic re-tuning.** That's
  [`mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md);
  this brief's existence presupposes that ceiling.
- **The planner-side wavy-path issue.** That's
  [`nav2-startup-unknown-donut-path-noise.md`](nav2-startup-unknown-donut-path-noise.md).
  Affects the `nav2` backend's startup behavior; orthogonal to RL
  policy execution (the trained policy doesn't consume Nav2 paths in
  `strafer_direct` mode).
- **DEPTH variant inference.** First transfer target is NOCAM;
  follow-up brief once NOCAM validates.
- **`hybrid_nav2_strafer`.** Follow-up brief once `strafer_direct`
  validates.
- **Removing or downgrading Nav2.** Nav2 is the default and the
  fallback. Even after the trained policy is deployed, the
  `nav2` backend should remain available for missions where
  classical planning is preferred (long-horizon paths in known
  maps, recovery from policy wedges, etc.).
