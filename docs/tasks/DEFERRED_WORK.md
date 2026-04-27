# Deferred work

Open items that have not shipped yet and will likely roll into the next
design round. Extracted from the task breakdowns before archival so the
next planning pass has a single source of truth without spelunking
through `docs/archive/`.

Every item has:

- **What** — one-sentence description.
- **Why deferred** — effort / priority / blocker.
- **Package(s)** — which README(s) will need to be updated when the item ships.

Grouped by package so "what's pending for my package" is easy to scan.
Items that cross multiple packages are in the final section.

---

## `strafer_autonomy`

### `orient_relative_to_target` skill

**What**: Rotate the robot to face toward, away from, or perpendicular to a
known target pose. Handler is drafted in
[`executor/mission_runner.py`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
but the skill name is commented out of `DEFAULT_AVAILABLE_SKILLS`. The
corresponding `strafer_msgs/action/OrientRelativeToTarget.action` does not
yet exist.

**Why deferred**: Not required for the current "go to X" / "wait by X"
mission set. MVP operator flows work without explicit facing behavior.

**Packages**: `strafer_autonomy`, `strafer_ros` (new action definition + implementation).

### Parallel health checks in `build_command_server`

**Status**: **Shipped** — `check_vlm_health=True` runs parallel probes in
[`command_server.py`](../source/strafer_autonomy/strafer_autonomy/executor/command_server.py).
Retained here only as a cross-reference — remove on next revision.

### Agentic combined planning endpoint (broader than `/plan_with_grounding`)

**What**: Extend the co-located planner/VLM pattern beyond single-target
pre-grounding. One endpoint that can optionally include scene description,
multi-object detection, or clarifying questions inside the plan response,
reducing N LAN round-trips to 1 for agentic mission flows.

**Why deferred**: Natural follow-up after `/plan_with_grounding` is
validated in practice. Medium effort, low priority; `/plan_with_grounding`
already proves the co-location pattern.

**Packages**: `strafer_autonomy` (planner service), `strafer_vlm` (if new
endpoints are needed).

### Databricks Model Serving infrastructure setup

**What**: Register pyfunc wrappers and stand up actual Databricks serving
endpoints for the planner and VLM. Client code (`DatabricksServingPlannerClient`,
`DatabricksServingGroundingClient`) and the registration CLI ship today;
endpoint provisioning does not.

**Why deferred**: Infrastructure-side setup depending on workspace access;
optional deployment path. LAN HTTP remains the primary transport.

**Packages**: `strafer_autonomy` (deployment + docs).

### Plan repair on failure

**What**: When a step fails, the planner proposes a revised plan rather
than terminating the mission. Currently failures end the mission with a
`failed` final state and the operator re-submits.

**Why deferred**: Large effort, low priority. Requires richer failure
taxonomy + repair-prompt design + validated re-plan logic.

**Packages**: `strafer_autonomy` (planner + executor).

### Multi-room navigation

**What**: `scan_for_target` currently fails when the target is in a
different room (not visible from the robot's current position). Supported
today: single-room targets and repeat visits to previously-seen targets
via the semantic map.

**Why deferred**: Medium effort, medium priority. Short-term mitigation is
to navigate to the stored map pose on scan failure; full cross-room
planning is a larger rework.

**Packages**: `strafer_autonomy` (scan_for_target + planner), `strafer_lab`
(possibly training on multi-room navigation).

---

## `strafer_vlm`

### Containerization / deployment automation

**What**: No Docker / compose / systemd unit files. Services run directly
in a venv with manual `make serve-vlm` / `make serve-planner`.

**Why deferred**: Operational convenience, not functionality. Works today
with the NVRTC fix + `make` targets.

**Packages**: `strafer_vlm`, `strafer_autonomy` (planner service has the
same gap).

### Postman collection completeness

**What**: [`source/SImToRealLab.postman_collection.json`](../source/SImToRealLab.postman_collection.json)
covers `/health` and `/ground` only. `/describe`, `/detect_objects`, and
the planner's `/plan` + `/plan_with_grounding` are missing.

**Why deferred**: Low priority; all endpoints work via `curl` and Swagger UI.

**Packages**: `strafer_vlm` (update the collection).

---

## `strafer_ros`

### `strafer_inference` Jetson package

**What**: The Jetson-side RL policy execution runtime. Once present, it
becomes the backend for `execution_backend="strafer_direct"` (pure-RL)
and `"hybrid_nav2_strafer"` (Nav2 global + RL local) on the
`navigate_to_pose` skill.

**Why deferred**: Depends on a deployable policy checkpoint from
[`strafer_lab`](../source/strafer_lab/) + policy export tooling (also
deferred; see below). Default `navigate_to_pose` remains `nav2`.

**Packages**: `strafer_ros` (new package), `strafer_shared` (load_policy
already exists), `strafer_autonomy` (dispatch in `JetsonRosClient.navigate_to_pose`).

### `OrientRelativeToTarget.action` definition

**What**: Action-type file in `strafer_msgs/action/`. Currently only
`ExecuteMission.action` lives there. The drafted executor handler (see
above) has nothing to dispatch to.

**Why deferred**: Ships with the `orient_relative_to_target` skill.

**Packages**: `strafer_ros`.

### Redundant static TF in `perception.launch.py`

**What**: Remove the duplicate `base_link → d555_link` static transform
that `perception.launch.py` publishes. The URDF transform via
`robot_state_publisher` is authoritative.

**Why deferred**: Cosmetic; TF consumers already prefer the URDF transform
because it wins the `last publisher wins` race under normal conditions.
Small cleanup.

**Packages**: `strafer_ros`.

### `MissionStatus.msg` topic

**What**: Topic-based mission status for dashboards / logging, in addition
to the existing `get_mission_status` service.

**Why deferred**: Low priority; action feedback plus the status service
cover the CLI flow. Only needed if a UI layer wants streaming status.

**Packages**: `strafer_ros` (message definition), `strafer_autonomy`
(publisher).

---

## `strafer_lab`

### Electronics masses in the USD

**What**: Add rigid-body meshes + catalog masses for 2× RoboClaw motor
controllers, Jetson Orin Nano, buck converter, and Intel RealSense D555 to
`Scripts/setup_physics.py`. Each should be positioned in the USD and given
its datasheet mass.

**Why deferred**: Without these, the simulated chassis inertia
underestimates the real robot — a systematic sim-to-real gap that grows
more important once the policy gets tuned for fine control. Not a blocker
for current PPO training.

**Packages**: `strafer_lab`.

### Policy export tooling (`export_policy_as_jit`)

**What**: Wrapper script that converts a trained PPO checkpoint into a
TorchScript `.pt` artifact loadable by `strafer_shared.policy_interface.load_policy()`.
Target: `python Scripts/export_policy.py --checkpoint logs/best_model/model_*.pt --output model.pt`.
Paired with a `benchmark_policy()` helper to validate Jetson inference
latency (<5 ms target).

**Why deferred**: Gates `strafer_inference` Jetson deployment.

**Packages**: `strafer_lab` (export), `strafer_shared` (loader already
exists), `strafer_ros` (future `strafer_inference` consumer).

### Pre-deployment training with goal-position noise

**What**: Before exporting the checkpoint intended for VLM-sourced goals,
retrain with `goal_position_noise_std: 0.2-0.3 m` in
[`tasks/navigation/mdp/commands.py`](../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
to match the ±0.2-0.5 m localization error of Qwen2.5-VL-3B grounding.
Without this, the policy oscillates at deployment when given imprecise
VLM-generated goals.

**Why deferred**: Requires an otherwise-converged baseline policy first;
goal-noise training is a targeted final pass, not a fresh train from
scratch.

**Packages**: `strafer_lab`.

---

## Hardware / cross-package

### `rotate_in_place` PID tuning on real hardware

**What**: `JetsonRosClient.rotate_in_place()` uses open-loop `cmd_vel`
angular-Z with odom yaw feedback. Tolerance + gain tuning on actual
hardware is pending.

**Why deferred**: Behavior works; tuning is an incremental improvement.
Needs bench time on the real robot.

**Packages**: `strafer_autonomy` (client), `strafer_ros` (driver response
characteristics).

---

## Failure-to-sim feedback pipeline

**What**: Capture real-world mission failures on the Jetson, extract the
failing scene + command + failure mode, and automatically generate
targeted training scenarios in simulation (Infinigen scene + mission
script + failure-mode label). Produces a closed-loop "real failures
become sim regression tests" pipeline.

**Why deferred**: Large effort, low priority. Depends on:
- Jetson-side structured failure logging (not yet shipped).
- Synthetic data infrastructure (most of it ships today, but the
  scenario-generation wrapper `gen_failure_scenarios.py` doesn't).
- DGX-side Infinigen scene library (builds out organically as the
  synthetic data pipeline runs).

**Packages**: spans `strafer_autonomy` (failure logging), `strafer_lab`
(scenario generation), `strafer_ros` (failure classification hooks).

This is the largest deferred item and the one most likely to anchor the
next design round.

---

## How this list gets maintained

- When a deferred item ships, **delete the entry from this file in the
  same commit** and update the corresponding package README (move from
  "Deferred / known limitations" to "What ships today").
- When a new deferred item is identified (e.g., in the next design
  round), add it here with the three-field template above.
- Do not let entries drift. An item that has been silently completed but
  is still listed here is worse than no list.
