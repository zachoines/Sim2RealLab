# Audit nav-side deadlines for wall-clock leakage under sub-unity RTF

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/executor/`, `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`, `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`)
**Priority:** P2
**Estimate:** M (~1–2 days; grep + audit + targeted conversions + retest at low RTF)
**Branch:** task/nav-deadline-sim-time-audit

## Story

As a **mission operator running sim-in-the-loop at ~15 fps on DGX (RTF ≈ 0.05–0.1)**, I want **every deadline / watchdog / safety cap in the nav stack to either tick on the sim clock or use a stall detector that distinguishes "no progress" from "slow progress"**, so that **rotations and translations are not aborted partway through just because the wall-clock window expired while sim time was still advancing legitimately**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout" section.
- [completed/rotate-in-place-sim-clock-deadline.md](../../completed/rotate-in-place-sim-clock-deadline.md)
- [completed/progress-aware-nav-timeouts.md](../../completed/progress-aware-nav-timeouts.md)

## Context

Operator observation (2026-05-11): rotations issued via the executor's `rotate_by_degrees` skill (and equivalents) terminate before the robot has completed the requested arc when running on the sim bridge. The bridge currently runs at ~15 fps on the DGX in headless mode, which means RTF ≈ 0.05–0.1 (see [bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md) "Phase-level profiler" reference numbers).

Two prior briefs already converted the most obvious deadlines to sim-time (`rotate-in-place-sim-clock-deadline`, `progress-aware-nav-timeouts`). The bridge-runtime-invariants doc explicitly calls out the convention:

> `navigate_to_pose` (via `_wait_for_future` / `_wait_for_nav_result`) and `rotate_in_place` both compute their deadline as `clock.now() + Duration(seconds=timeout)` and bound the wait with a `2 * timeout` wall-clock safety cap so a stalled `/clock` cannot wedge the executor.

The `2 * timeout` wall-clock safety cap is the most likely culprit: at RTF=0.05, a 90 s sim-time deadline corresponds to 1800 s of wall time, but the safety cap fires at `2 * 90 = 180` wall seconds. That cap was designed to bound a *stalled* `/clock`; it's now firing in a *slow* `/clock` regime. The recent `donut_warmup` brief on this same PR hit and fixed the same class of bug (`5939a03` and predecessors) by replacing the absolute wall-clock cap with a sim-time stall detector.

The same pattern almost certainly leaks through any other deadline-with-wall-cap pair in the stack:
- Executor `_wait_for_future`, `_wait_for_nav_result`, `rotate_in_place` wall-clock safety caps.
- Nav2 Spin behavior `time_allowance` (parametrized per-goal; default 10 s is sim-time but the Spin behavior internally may use wall clock for some bounds).
- `behavior_server` `transform_tolerance` and `cycle_frequency`.
- `controller_server` `failure_tolerance` window.
- Plan-compiler skill timeouts (recently widened — verify they're sim-clock).
- RTAB-Map `wait_for_transform` (likely wall, but probably tolerable).

## Approach

### A. Inventory deadlines

Grep the executor + ros_client + nav2_params + any custom Jetson nodes for:
- `time.monotonic()`, `time.time()`, `time.sleep()`.
- `Duration(seconds=` followed by a value coming from anywhere other than `node.get_clock().now()`.
- `timeout`, `deadline`, `cap`, `watchdog` identifiers — broad sweep to catch any place a wall-clock bound has been added.
- Specifically the `2 * timeout` pattern called out in the bridge-runtime-invariants doc.

### B. Convert per-site

For each wall-clock bound that's there as a stalled-`/clock` safety:
- **Replace** the absolute wall-clock cap with a **sim-time stall detector**: track the most recent wall-clock instant at which sim time was seen advancing, and bail only if wall time runs `clock_stall_bail_wall_s` (default ~15 s) past that mark without sim-time progress.
- Mirror the pattern shipped in `donut_warmup` (`source/strafer_ros/strafer_bringup/strafer_bringup/donut_warmup.py`'s `run()` loop) — it's the reference implementation.

For bounds that are intentionally wall-clock (e.g., DDS connectivity timeouts, HTTP client deadlines to DGX services), leave them alone — wall-clock is correct there.

### C. Retest at low RTF

Stand up the bridge in headless mode (target RTF ~0.05–0.1) and run the smoke missions documented in `INTEGRATION_SIM_IN_THE_LOOP.md`. Confirm rotations and translations complete in sim time as expected.

## Acceptance criteria

- [ ] All wall-clock safety caps in the Jetson executor / ros_client / Nav2 config that exist to guard against a stalled `/clock` have been replaced by sim-time-progress stall detectors (or documented why an absolute wall cap is correct at that site).
- [ ] An audit table in the PR description lists every deadline site touched, its previous bound, its new bound, and the test that exercises it at low RTF.
- [ ] A `rotate_by_degrees 360` skill issued against the bridge at RTF ≤ 0.1 completes the full arc without spurious termination.
- [ ] A `translate forward 3 m` mission at RTF ≤ 0.1 completes without spurious termination.
- [ ] No regression on cold-start without a DB (`make clean-map && make launch-sim`): smoke missions still complete end-to-end.
- [ ] Real-robot bringup unaffected: at RTF = 1.0 the stall detector and the previous absolute cap fire at indistinguishable times, modulo the configured `clock_stall_bail_wall_s` slack. Unit-test this where feasible.
- [ ] Update the "Sim-time-aware navigation timeout" section of [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md) to describe the stall-detector pattern instead of the `2 * timeout` cap.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` — `_wait_for_future`, `_wait_for_nav_result`, `rotate_in_place`, `navigate_to_pose`. The `2 * timeout` wall cap should live here.
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py` — per-step budgets, stall watchdog wiring.
- `source/strafer_ros/strafer_bringup/strafer_bringup/donut_warmup.py` — reference stall-detector implementation. Mirror the pattern.
- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml` — `controller_server.failure_tolerance`, `behavior_server.cycle_frequency`, etc.
- Use the bridge `--profile` mode to confirm the RTF you're testing against; numbers in `bridge-runtime-invariants.md` are the reference.

## Out of scope

- **Speeding up the bridge.** Sub-unity RTF is a perf reality (camera-publish OmniGraph dominates per the bridge-runtime-invariants doc); this brief is about tolerating low RTF, not eliminating it. Bridge perf work tracks separately under `async-camera-publishers.md`.
- **Real-robot deadline tuning.** Real-robot runs at RTF = 1.0; current bounds are appropriate. If real-robot regressions surface from a stall-detector substitution, file a follow-up.
