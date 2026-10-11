# Keep the inference node's mission-end stop the last command it sends for a goal

**Status:** Shipped 2026-10-10 in `83b5647` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/238
**Follow-ups:** [`inference-finished-successor-stop`](../active/reliability/inference-finished-successor-stop.md) — a goal that ends while the goal it preempted is still leaving its loop;
[`inference-last-action-mission-reset`](../active/trained-policy/inference-last-action-mission-reset.md) — `_last_action` at a mission boundary

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
- **A second window, left for its own brief.**
  - Goal presence is a count, and a preempted goal stays in it until its loop next wakes, up to
    50 ms later (`time.sleep(0.05)`, 1021).
  - If its successor ends inside that time, the successor publishes the stop while the count is
    still 1. Ticks keep driving until the predecessor leaves, and it leaves without a stop
    because it was superseded.
  - It takes a successor that ends on its first poll, inside its arrival radius, or one cancelled
    as it starts. The executor's replacement path cancels, zeroes and joins the old mission before
    it starts a new one, so the deployed flow does not appear to produce it.
  - It is the same on `main` and under this fix:
    [`inference-finished-successor-stop`](../active/reliability/inference-finished-successor-stop.md).
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
- A counter counts mission-end stops. The tick reads it together with goal presence before the
  TF age read and the watchdog check, and compares it again at its publish.
- A tick whose goal published its stop in between publishes nothing in place of its action.
- Preemption keeps its rule.

## Acceptance criteria

The new tests are in `TestMissionEndStopIsLast`, `strafer_inference/test/test_inference_runtime.py`.
Each runs a goal on its own thread, as the action server does.

- [x] **The race, as a test.**
  - The goal ends from inside the policy call, between the tick's watchdog check and its publish.
  - The test asserts:
    - the `/cmd_vel` publishes are exactly the stop;
    - the tick counts one `skip_goal_ended`;
    - `_last_action` is unchanged.
  - It fails on `main`, and the PR shows that output.
- [x] **The stop goes out inside the goal lock, with the epoch already advanced**
      (`test_mission_end_stop_goes_out_inside_the_goal_lock`). It fails on `main`.
- [x] **The read comes before the TF age read and the watchdog check**
      (`test_goal_ending_before_the_watchdog_check_still_skips`).
  - The goal ends inside the tick's TF read. The publishes are exactly the stop, and one skip is
    counted.
  - It fails on `main`, and when the epoch is read after goal presence or the watchdog is handed
    a lockless presence read.
- [x] **The stale branch reads goal presence live**
      (`test_goal_ending_before_a_stale_watchdog_check_stays_idle`).
  - The goal ends inside the TF read with another source stale. The publishes are exactly the
    stop.
  - It fails if the stale branch is handed the snapshot instead.
- [x] **Preemption** (`test_tick_spanning_a_preemption_publishes_its_action`). A tick in flight
      across a preemption publishes its action, compared by value, and no stop. It fails if the
      epoch advances on every goal end.
- [x] **One stop on a plain end.** `test_mission_end_publishes_final_stop` counts exactly one
      publish.
- [x] `make test-ros` passes, read from each package's JUnit XML.
- [x] **On the sim-bridge lane.** One CLI mission is run with the fixed node. Its last `/cmd_vel`
      in the observer's `series.cmd`, ordered by the wall stamp, is the zero twist.
  - The check is descriptive: one mission cannot show the race is gone, which is the tests' job.
  - It is recorded on the video record, with its files deposited.
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the policy backends' goal handling. The existing goal-flag, reset and
      preemption tests in `test_inference_runtime.py` pass, and their assertions are unchanged
      apart from the one-publish count above. The cadence-line test gains `goal_ended=`.

## Outcome

**What the counter counts.**
- `_stop_epoch` counts mission-end stops. It advances in the `finally`'s single hold of
  `_goal_count_lock`, together with:
  - the decrement;
  - the superseded check, which on `main` read `_current_goal_handle` outside the lock;
  - the stop's publish.
- A changed counter at the tick's publish therefore means the stop is already on the channel.

**Skip, not a second zero.**
- On a changed counter the tick publishes nothing.
- It counts the skip as `skip_goal_ended`, which shows as `goal_ended=` on the cadence line.
- It writes `_last_action` only together with a publish.
- Why skip:
  - Between missions the node publishes nothing (the idle rule).
  - The gate records' "one zero twist at the result" stays literally true only under a skip.
  - Neither downstream watchdog needs a second zero: both reset on any command.

**Where the tick reads.**
- Goal presence and the counter are read in one hold, before the TF age read and the watchdog
  check, and that presence is what the watchdog sees.
- The stale-source branch still reads goal presence live, as on `main`. A goal ending between the
  two reads, with another source stale, therefore takes the idle branch, not a mid-mission zero.

**Residuals, by construction.**
- A tick that a mission-end stop overtakes has run one inference and publishes nothing.
- If a successor started inside that window, its first command comes one tick later: one period
  under the deployed `timer_reuse`.
- The stop is published while the lock is held. Under DDS back-pressure (RELIABLE, depth 10) one
  publish can stall a tick or a goal's preamble for that publish: bounded, with no lock nesting.
- Cyclone's `max_blocking_time` on the robot has not been verified.

**What the tests pin, and what they cannot.**
- Each new test fails on at least one version of the node that breaks a rule:

  | version | failing tests |
  |---|---|
  | `main` | the race; the stop inside the lock; the read before the watchdog |
  | the counter advanced on every goal end | preemption |
  | a zero published in place of the skip | the race; the stop inside the lock; the read before the watchdog |
  | the counter read after goal presence | the read before the watchdog |
  | a lockless presence read handed to the watchdog | the read before the watchdog |
  | the snapshot handed to the stale branch | the live read |
  | the stop published outside the hold | the stop inside the lock |

  The runs and their JUnit XML are deposited with the sim-bridge check, under `tests/`.

- `Lock.locked()` has no owner. The single hold of decrement, check, counter and stop is
  therefore pinned by review, not by the lock test.

**Left as on `main`, filed or noted.**
- [`inference-finished-successor-stop`](../active/reliability/inference-finished-successor-stop.md):
  the second window above.
- [`inference-last-action-mission-reset`](../active/trained-policy/inference-last-action-mission-reset.md):
  `_last_action` is not zeroed at a mission boundary, while training zeroes the action at every
  reset.
- Noted in that brief: a successor's count bump comes before its hidden-state reset, so one tick
  can infer with the predecessor's hidden state.

**On the sim-bridge lane.**
- One CLI mission to G1's goal ran with images built at `f4d9cb2` and this fix's
  `inference_node.py` (`83b5647`) mounted over the image's installed copy. It SUCCEEDED.
- Ordered by the wall stamp, the last of its 381 `/cmd_vel` messages is the zero twist, and
  nothing followed in 0.5 s of sim time. The order is by wall stamp because a stop and a
  straddling twist can share a sim time, as L2's did at 330.6. Here the stop and the last policy
  twist are 8 ms of sim time and 54 ms of wall time apart.
- No tick straddled that end (`goal_ended=0`).
- Recorded in
  [`goal-a-cli-video-2026-10-09`](../../measurements/goal-a-cli-video-2026-10-09/README.md),
  "Later check: the stop after a goal". The files are in the sibling deposit
  `goal-a-cli-video-2026-10-09/stop-race-check/` (Sim2RealLab-Artifacts `d2bb6d50`).

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
- `_last_action` at a mission boundary, and the finished-successor ordering (Follow-ups).
