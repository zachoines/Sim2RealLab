# Preempt on new `navigate_to_pose` goals; retire the `/strafer/goal` topic

**Type:** refactor
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** M
**Branch:** task/inference-goal-preemption

## Story

As a **mission operator**, I want **the action to be the single,
preemptible goal channel into the inference node**, so that **goal
updates use the action model's own semantics and the watchdog's goal
source needs no dual (presence OR receive-time) logic**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/conventions.md](../../context/conventions.md)

## Context

Follow-up to `inference-goal-watchdog-latched` (PR #130). That fix left
the goal with two delivery paths: the `navigate_to_pose` action (real
missions, presence-keyed freshness) and the dormant `/strafer/goal`
topic (live goal updates + the distance-threshold mid-mission
hidden-state reset — nothing in the stack publishes it). The dual path
forced dual watchdog semantics (`goal_active` OR fresh rx-time) and
left the action server accepting unconditionally, so overlapping goals
were handled only by the active-goal counter, not by any explicit
policy.

The action model already has a first-class answer to "the goal moved":
preemption. Adopting it collapses the goal channel to one path.

## The change

- **Newest-goal-wins preemption** on the action server: a
  `handle_accepted_callback` records the accepted handle as the owner;
  a superseded execute loop notices it lost ownership and aborts
  (`navigate_to_pose goal preempted by a newer goal.`). Ownership is
  **never cleared, only replaced** — a successor that already finished
  still supersedes its predecessors (no zombie resume when a
  short-lived successor exits inside the old loop's 50 ms sleep
  window) — and the ownership check also **gates execute entry**, so a
  superseded goal whose execute task starts late aborts without
  touching the live mission's goal pose or hidden state. Each accepted
  goal resets the policy hidden state at its own mission start, so a
  preempting goal update is a recurrent-contract mission boundary —
  this subsumes the topic path's distance-threshold reset.
- **Retire `/strafer/goal`**: drop the subscriber, `_on_goal`, the
  `goal_topic` / `goal_timeout_s` / `is_mid_mission_reset` /
  `mid_mission_reset_distance_m` parameters, and the watchdog's
  receive-time path. The `goal` watchdog source becomes purely
  presence-keyed: fresh exactly while a goal executes; idle trips it
  (safe zero-twist between missions — unchanged live behavior, since
  nothing ever published the topic).
- `WatchdogTimeouts` loses its `goal` threshold; `stale_sources` loses
  `last_goal_rx_t`. The active-goal counter from PR #130 stays — a
  preempted goal still briefly overlaps its successor while draining.

## Acceptance criteria

- [x] Preemption: accepting goal B while goal A executes aborts A,
      resets hidden state for B's mission start, and keeps the watchdog
      `goal` source fresh across the boundary (node-level threaded
      test + live-node smoke).
- [x] The `goal` watchdog source is presence-keyed only; idle trips it
      even with all streams fresh (watchdog tests).
- [x] No `/strafer/goal` references remain in package code, config, or
      the subgoal-generator docs that contrasted against it.
- [x] `strafer_inference` green via colcon on the Jetson.
- [ ] Operator sim missions re-validated (rides the same gate as
      PR #130 — `strafer_direct` + `hybrid_nav2_strafer`); confirm the
      autonomy client's cancel→new-goal retry path now lands as a clean
      preemption.
- [x] If the work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit.

## Out of scope

- Hybrid replan ownership (the client's 0.5 s `ComputePathToPose`
  cadence) — filed separately as
  [`hybrid-replan-ownership`](hybrid-replan-ownership.md).
- The `/plan` → subgoal chain and the subgoal watchdog source.
- Autonomy-client (`ros_client.py`) changes — its cancel + re-send flow
  now lands as a preemption on the server side with no client change.
