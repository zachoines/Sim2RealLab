# Keep the inference node idle after a goal that ended before the goal it preempted left

**Type:** bug (`/cmd_vel` ordering between overlapping trained-policy goals)
**Owner:** Jetson (`strafer_inference`)
**Priority:** P2 — the same defect class as the post-goal stop race, a policy command left as the
last `/cmd_vel` after a goal has ended. The ordering is reachable in the deployed flow only under
a condition (the reachability box): two replacement submissions racing, or the executor restarted
mid-goal, and a newer goal that ends at once.
**Estimate:** S
**Branch:** `task/inference-finished-successor-stop`

## Story

As **the base driver, which holds the last `/cmd_vel` it received for 0.5 s**, I want **the inference
node not to leave a policy command after the newest goal's stop, even while a goal it preempted is
still leaving its loop**, so that **a stop is the last command.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`inference-post-goal-stop-race`](../../completed/inference-post-goal-stop-race.md) — the stop
  epoch this builds on, and the ordering it left as on `main`.

## Context

**The ordering** (`strafer_inference/inference_node.py`, after #238):
- **A is live.** Goal A is executing, and the count of executing goals is 1.
- **B ends inside A's drain.** Goal B is accepted, and `_handle_accepted` makes it the current goal.
  B's execute gate raises the count to 2. B then ends at once: its first distance poll is already
  inside the 0.30 m arrival radius, or it is cancelled as it starts. B's `finally` decrements the
  count to 1, advances the stop epoch and publishes the stop.
- **A is still counted.** A has not yet seen that it was superseded: it checks once per loop, after
  a `time.sleep(0.05)`, so for up to about 50 ms it stays in the count.
- **The tick drives.** A tick whose snapshot falls inside that time reads goal presence as true
  (count 1) and the epoch as already advanced. It runs the policy and publishes its action, and
  the epoch compare does not stop it.
- **A leaves without a stop.** When A wakes, it aborts as superseded. Its `finally` decrements to 0
  and, being superseded, publishes no stop.
- **The result.** A policy command follows B's stop and is the last `/cmd_vel`. The base driver
  holds it for 0.5 s.

**Why it is P2, not P1.**
- **What the executor does on a replacement.** It does not overlap missions this way.
  `MissionRunner.start_mission` (`executor/mission_runner.py`, 262–307 at `f4d9cb2`) refuses a
  second mission unless `replace_active_mission` is set. On a replacement it:
  - cancels the active navigation, waiting for the cancel response;
  - publishes a zero `/cmd_vel` (`_cancel_robot_actions`, 2329–2343 at `6cd6fd7`);
  - joins the old mission's thread, for up to `replace_join_timeout_s` (5 s);
  - only then starts the new mission.
- **The join outlasts the cancelled goal's loop** (read 2026-10-14; the reachability box). A single
  replacement cannot meet a draining A. Two paths that are not a single replacement can, and both
  need an operator action as well as a newer goal that ends at once.

**Candidate fixes**, both small:
1. **A flag for the newest goal's end.** It is set with the mission-end stop and cleared at the next
   execute gate. The tick's goal presence becomes "count > 0 and the newest goal has not ended".
   - This was implemented and tested once, in #238's history (`2af556b`), before it was split out
     to this brief.
   - Its test, `test_goal_ending_before_its_preempted_predecessor_leaves` at `2af556b`, fails on
     `main` and on an epoch-only node from before #238's final form, which publishes a zero in place
     of the action rather than skipping it. It passes with the flag.
   - The runs are deposited in Sim2RealLab-Artifacts `cf1da2d8`,
     `goal-a-cli-video-2026-10-09/stop-race-check/tests/finished_successor/`. The deposit holds a
     re-run of `2af556b`'s test module against four nodes: `main`, that epoch-only node, `2af556b`'s
     node with the flag, and that node with the epoch advanced on every goal end. It also holds the
     first runs, made on the working tree before the work commit.
   - The stale-source branch's live read of goal presence would need the same condition. Otherwise
     a stale source during A's drain logs a mid-mission trip and publishes a zero.
2. **Advance and stop when the decrement reaches zero.** A superseded goal whose exit leaves no goal
   executing advances the epoch and publishes a stop.
   - This closes the window by the stop that A sends.
   - It also sends a stop when a superseded goal leaves before its queued successor's execute gate
     has run. That is a stop while a successor exists, which the preemption rule avoids.
   - Ticks during A's drain still publish actions before A's stop. The last command is then A's
     stop, so it meets the test below, which asks for a stop last; candidate 1 also leaves no
     action after B's stop.

## Acceptance criteria

- [x] **Reachability, stated.** The brief records whether the executor's replacement path can
      produce this ordering. That means reading whether the old mission's thread returns only
      after the node's result for the cancelled goal, and whether any other client sends
      overlapping goals.
      *2026-10-14, read at `6cd6fd7`: reachable under a condition, though not through a single
      replacement.*
  - **One replacement: no.**
    - The old mission thread runs the blocking navigate itself. `background_mapper` is never
      passed (`executor/main.py`:284-289), so there is no transit monitor
      (`mission_runner.py`:1292-1294).
    - The thread waits on goal A's result future (`clients/ros_client.py`:1025-1031; hybrid
      1171-1177), and only the result, the deadline or a `/clock` stall ends that wait
      (1811-1827).
    - rclpy Humble's action server sets the result after the execute callback returns
      (`rclpy/action/server.py`:337, 357, 422-423 in `strafer-gpu:humble`), so after A's
      `finally` (`inference_node.py`:1033-1044).
    - `start_mission` joins the old thread (`mission_runner.py`:298), or refuses with
      `mission_replace_timeout` (299-307). The new mission plans (371-381) before its first goal.
  - **Two submissions racing `start_mission`: yes.**
    - The active check (274-292) and the install (323-325) are separate lock sections, and only
      `_finish_runtime` clears the active mission (2260-2266).
    - The ExecuteMission server runs a reentrant group on four threads (`command_server.py`:54,
      166-188, 204-209). Two goals with `replace_active_mission` set that arrive during one
      join both join the same mission (`mission_runner.py`:298), and both start a mission.
    - `_nav_lock` guards only the handle slot (`ros_client.py`:331, 1025-1026), so the two
      missions' goals overlap on the node.
  - **The executor restarted mid-goal: yes.**
    - The executor cancels nothing when its process ends (`executor/main.py`:292-298).
    - The node's loop has no client-liveness check, so A runs on for up to its 60 s bound
      (`inference_node.py`:983-1029), and the new executor's first policy goal supersedes it.
    - A crash alone does not restart the executor. The launch keeps `goal_projection` and runs
      the executor as a plain process with no respawn (`autonomy_nodes.launch.py`:44-63), so
      the trigger is an operator restart or recreate of `autonomy`.
  - **On either path,** B must end inside A's drain. It may be within 0.30 m on its first poll
    (`inference_node.py`:998-1004), as a cardinal translate of 0.30 m or less can be. Or it may
    be cancelled as it starts (984-989). Both paths also need a policy backend in the
    executor's env (`executor/main.py`:269-274); committed canon is `nav2`
    (`deploy/compose/autonomy.env`:7).
  - **Not a source:**
    - Within one mission, navigate and translate compile with `retry_limit=0`
      (`planner/plan_compiler.py`:103, 173), and a non-success ends the mission.
    - The subgoal generator sends only ComputePathToPose (`subgoal_generator_node.py`:339-341).
    - The sim harness submits with `replace_active_mission` false (`runtime_mission.py`:143).
    - A manual `ros2 action send_goal` can overlap, as an operator path.
  - **Read on the way, not fixed here:**
    - The executor leaves its goal driving under the policy when its process ends.
    - The staging loop never checks the mission's cancel event.
    - `cancel_active_navigation` misses a goal whose handle is not yet stored
      (`ros_client.py`:1025-1026, 1287-1291).
- [x] **The ordering, as a test** in `strafer_inference/test/test_inference_runtime.py`:
  - A runs on its own thread, its distance stub blocked so it cannot see its supersession.
  - B is accepted and ends on its first poll, on the test thread.
  - A tick runs, then A is released.
  - The test asserts that, in publish order, the last `/cmd_vel` is a stop and no policy action
    follows the final stop.
  - It fails on `main`.
  - *2026-10-14:* written as
    `TestMissionEndStopIsLast::test_goal_ending_before_its_preempted_predecessor_leaves`, marked
    strict xfail; the fix removes the marker. With the marker ignored (`--runxfail`) on `main`'s
    node it fails ten runs in ten: the publishes are B's stop, then A's policy action
    (`x=0.392, y=0.392, z=2.093`). With the marker, the ROS suite passes: 840 JUnit cases, 828
    passed, 11 skipped, this one xfailed.
- [ ] **One fix chosen** from the candidates above, or another, with the reason. The test above
      passes, and the preemption test of `inference-post-goal-stop-race` (a tick in flight across
      a preemption publishes its action) still passes.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-ros` passes.

## Out of scope

- The post-goal stop race itself, shipped in `inference-post-goal-stop-race`.
- The executor's replacement path, unless the reachability box finds it can overlap goals.
- `_last_action` at a mission boundary
  ([`inference-last-action-mission-reset`](../trained-policy/inference-last-action-mission-reset.md)).
