# Keep the inference node idle after a goal that ended before the goal it preempted left

**Type:** bug (`/cmd_vel` ordering between overlapping trained-policy goals)
**Owner:** Jetson (`strafer_inference`)
**Priority:** P2 — the same defect class as the post-goal stop race, a policy command left as the
last `/cmd_vel` after a goal has ended. But the ordering it needs does not appear to arise in the
deployed flow (see Context).
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
  - publishes a zero `/cmd_vel` (`_cancel_robot_actions`, 2329–2341);
  - joins the old mission's thread, for up to `replace_join_timeout_s` (5 s);
  - only then starts the new mission.
- **What has not been read.** Whether that join always outlasts the node's own execute loop for
  the cancelled goal. If the mission thread waits for the navigate result, it does, and B can never
  meet a draining A.

**Candidate fixes**, both small:
1. **A flag for the newest goal's end.** It is set with the mission-end stop and cleared at the next
   execute gate. The tick's goal presence becomes "count > 0 and the newest goal has not ended".
   - This was implemented and tested once, in #238's history (`2af556b`), before it was split out
     to this brief.
   - Its test failed on `main` and on the epoch-only fix, and passed with the flag.
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

- [ ] **Reachability, stated.** The brief records whether the executor's replacement path can
      produce this ordering. That means reading whether the old mission's thread returns only
      after the node's result for the cancelled goal, and whether any other client sends
      overlapping goals.
- [ ] **The ordering, as a test** in `strafer_inference/test/test_inference_runtime.py`:
  - A runs on its own thread, its distance stub blocked so it cannot see its supersession.
  - B is accepted and ends on its first poll, on the test thread.
  - A tick runs, then A is released.
  - The test asserts that, in publish order, the last `/cmd_vel` is a stop and no policy action
    follows the final stop.
  - It fails on `main`.
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
