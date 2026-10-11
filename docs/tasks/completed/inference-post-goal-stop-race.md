# Keep the inference node's mission-end stop the last command it sends for a goal

**Status:** Shipped 2026-10-10 in `2af556b` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/238

**Type:** bug (safety: `/cmd_vel` ordering at the end of a trained-policy goal)
**Owner:** Jetson (`strafer_inference`)
**Priority:** P1 — before the real-lane gate. The base driver repeats the last `/cmd_vel` until
0.5 s pass without one, so a policy command published after the stop drives the chassis for up to
half a second after the goal has ended, beside whatever the goal was next to.
**Estimate:** S
**Branch:** `task/inference-post-goal-stop-race`

## Story

As **the base driver and the sim bridge, which both hold the last `/cmd_vel` they received**, I
want **the stop the inference node publishes when a goal ends to be the last command it sends for
that goal**, so that **a goal that has ended cannot leave the robot moving.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [measurements/goal-a-cli-video-2026-10-09](../../measurements/goal-a-cli-video-2026-10-09/README.md)
  — the mission that showed it (L2).

## Context

**The symptom.** In the video set's mission L2, the observer's `/cmd_vel` series
(`record-files/logs/runs/container_gate_all/L2.json`, `series.cmd`) ends with these two
messages:

| t_sim (s) | t_wall (s) | vx | vy | wz |
|---|---|---|---|---|
| 330.600 | 1791568700.356 | 0 | 0 | 0 |
| 330.600 | 1791568700.359 | −0.027 | +0.352 | +0.239 |

- **What the series records.** Each entry carries its receive time at the observer, on the sim
  clock and on the robot's wall clock, and its values. It records no publisher.
- **Timing.** The goal's SUCCEEDED status reached the observer at wall .356, 0.6 ms after the
  zero.
- **The second message is the mission's last.**
  - The observer listened for another 0.5 s of sim time and received nothing more.
  - The bridge held it until its sim-time watchdog zeroed it: "No cmd_vel for 0.51s (sim) --
    zeroing bridge action." (`record-files/logs/dgx/goal_a_video_bridge.log` line 274).
  - Under it, the robot moved 0.079 m, almost entirely to its left, and turned 5.8°
    anticlockwise between the result at t_sim 330.600 and the settled pose at 331.100
    (`L2.progress.jsonl`, the `final_tf` and `settled_tf` records).
- **Who published it is inferred, not recorded.**
  - `/cmd_vel`'s publishers at the end were `behavior_server` (5), `jetson_ros_client`,
    `strafer_inference` and `velocity_smoother`.
  - Only the inference node was commanding: the policy's twists before it arrived at 330.500,
    330.533 and 330.567, and this one at 330.600 is the next instant on that 30 Hz grid.
- **How often it happens.**
  - G1 in the same set ends on the zero.
  - So do the eleven missions of the v3 gate record (`goal-a-rig-gate-v3-2026-09-25`, nine on v3
    and two on v2) and the four CLI confirmation missions with a command series
    (`goal-a-cli-confirmation-2026-10-03`).
  - L2 is the one end of seventeen that fell inside a tick.

**The mechanism** (`strafer_inference/inference_node.py`, lines at `f4d9cb2`):
- **Two threads.**
  - The tick runs in the default mutually exclusive callback group (timer, 577–580).
  - The action server's execute callback runs in its own `ReentrantCallbackGroup` (469, 572), on
    a `MultiThreadedExecutor` of five threads (1543).
- **The tick.**
  - It reads `_goal_active` once, in its watchdog check (`goal_active=self._goal_active`, 1097),
    and an idle tick publishes nothing.
  - It then assembles the observation and calls the policy under `_policy_lock` (1156–1157).
  - It publishes the twist (1196) without looking at the goal again.
- **The goal's end.**
  - The execute loop ends a goal with `succeed()` (995), `abort()` at `mission_timeout_s` (1018),
    `canceled()` (980), or `abort()` on preemption (987).
  - Its `finally` decrements the goal count under `_goal_count_lock` (1026–1027). It then
    publishes the stop, outside the lock, unless a newer goal superseded this one (1032–1033).
- **The race.**
  - A tick that passed its watchdog check before the goal ended publishes its action after the
    stop.
  - The window is the tick's observation assembly and policy call.
  - Every way a goal ends goes through the same `finally`, so success, the time-out abort and a
    cancel are all exposed.
- **A second window.**
  - Goal presence is a count, and a preempted goal stays in it until its loop next wakes, up to
    50 ms later (`time.sleep(0.05)`, 1021).
  - If its successor ends inside that time, the successor publishes the stop while the count is
    still 1.
  - Ticks keep driving until the predecessor leaves, and it leaves without a stop because it was
    superseded.
  - It takes a successor that ends on its first poll, inside its arrival radius, or one cancelled
    as it starts.
- **Preemption is different by design.** A superseded goal publishes no stop, because its
  successor owns `/cmd_vel` and a stop would fight the successor's commands. A tick in flight
  across a preemption publishing its action is that same rule, not the race.

**Why it matters on the robot.**
- `strafer_driver/roboclaw_node.py` stores each `/cmd_vel` and re-sends it at 50 Hz until 0.5 s
  pass without one, then sends zero (`WATCHDOG_TIMEOUT_SEC = 0.5`, line 65; 270–279).
- After a policy goal ends, nothing else publishes `/cmd_vel` until the executor starts a rotate
  or a Nav2 goal.
- The sim bridge does the same in sim time (`cmd_watchdog_sim_s`, 0.5 s), which is how L2 ended.

**The fix.**
- The goal's stop and the tick's publish are serialised under `_goal_count_lock`.
- A tick whose goal published its stop after the tick's watchdog check publishes a stop in place
  of its action.
- To tell, the node advances a counter with each mission-end stop. The tick reads it together
  with goal presence before its watchdog check, and compares it again at its publish.
- Once the most recently started goal has ended, the tick treats the node as idle, even while
  goals it preempted are still leaving their loops.
- Preemption keeps its rule.

## Acceptance criteria

- [x] **The race, as a test.** A test in `strafer_inference/test/test_inference_runtime.py`
      starts a goal on its own thread, as the action server does. It ends the goal from inside
      the policy call, between the tick's watchdog check and its publish, and asserts that the
      last `/cmd_vel` is a stop. It fails on `main`, and the PR shows that output.
- [x] **A plain end.** A goal that ends between ticks publishes exactly one stop, after the
      policy's action, and the idle tick after it publishes nothing.
- [x] **Preemption.** A tick in flight across a preemption publishes its action and no stop.
- [x] **A successor that ends first.** A goal that ends while the goal it preempted is still in
      its loop leaves its stop as the only command; a tick in between publishes nothing. The test
      fails on `main`.
- [x] `make test-ros` passes, read from each package's JUnit XML.
- [x] **On the sim-bridge lane.** One CLI mission is run with the fixed node, and its last
      `/cmd_vel` in the observer's `series.cmd` is the zero twist. The check is descriptive: one
      mission cannot show the race is gone, which is the test's job. It is recorded on the video
      record or in a sibling record, with its files deposited.
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the policy backends' goal handling: the existing goal-flag, reset and
      preemption tests in `test_inference_runtime.py` pass with their assertions unchanged.

## Investigation pointers

- `strafer_inference/inference_node.py`: `_execute_callback` and its `finally`; `_on_tick` from
  the watchdog check to the policy publish.
- `strafer_inference/test/test_inference_runtime.py`:
  - `_ready_depth_node` gives a tick that reaches the policy.
  - `TestGoalActiveFlag` runs `_execute_callback` on a thread.
  - The ready node's stub policy returns a zero action on a zero observation, so the race test
    needs a policy whose action is not zero.
- `strafer_driver/roboclaw_node.py`:65, 270–279 — the driver's hold.
- `strafer_lab/bridge/cmd_watchdog.py`, `bridge/async_publisher.py` — the bridge's hold.

## Out of scope

- Recording each `/cmd_vel` message's publisher in the mission observer.
- The executor's own zero on its cancel path, and Nav2's `/cmd_vel` publishers.
- Whether a new goal should zero `_last_action`, which the node carries from one mission into the
  next one's first observation.
