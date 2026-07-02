# Fix the inference `goal` watchdog latched-vs-streamed mismatch

**Type:** bug
**Owner:** Jetson agent
**Priority:** P1
**Estimate:** S
**Branch:** task/inference-goal-watchdog-latched

## Story

As a **mission operator**, I want **the inference watchdog to model the
mission goal as active-goal presence instead of stream freshness**, so
that **`strafer_direct` / `hybrid_nav2_strafer` missions are not
zero-twisted ~1 s after goal accept**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/conventions.md](../../context/conventions.md)

## Context

A `navigate_to_pose` action goal is **latched for the whole mission**,
but the watchdog modeled `goal` as a **streamed** freshness source:

- `inference_node.py`'s `_execute_callback` stamps `_last_goal_rx_t`
  once at goal accept, then blocks in its polling loop (distance /
  feedback / timeout checks) without ever refreshing the stamp.
- Nothing publishes `/strafer/goal` — the node only *subscribes* to it;
  missions arrive through the action — so `_on_goal` (which would
  refresh the stamp) never fires during a mission.
- `watchdog.stale_sources` marks `goal` stale once the stamp ages past
  `goal_timeout_s` (1.0 s), and `_on_tick` publishes a zero `Twist`
  whenever anything is stale.

⇒ ~1 s into every mission `goal` went stale and the policy zero-twisted
for the rest of the mission. Backend-agnostic: `strafer_direct` and
`hybrid_nav2_strafer` both hit it. The `/plan` → subgoal chain was
healthy (1.9 Hz / 35 Hz) and is untouched.

## Fix (Option B — model the goal as active-goal presence)

- `_active_goal_count` on the node (with a `_goal_active` property =
  count > 0): incremented at goal accept, decremented in a `try/finally`
  on every mission exit path (succeed / abort / cancel / timeout /
  exception); `_on_tick` passes the property to `stale_sources`.
- A **counter behind a small lock, not the brief's plain bool**:
  adversarial review confirmed the action server accepts and executes
  goals concurrently (unconditional ACCEPT + default `handle_accepted`
  + `ReentrantCallbackGroup`), and the autonomy client's cancel is
  fire-and-forget — so a retry's goal B can overlap goal A's drain, and
  a bool's `finally` would clear freshness under the surviving mission,
  silently reinstating the bug. The counter makes overlap safe.
- `stale_sources(..., goal_active=False)`: an active action goal **or**
  a fresh `/strafer/goal` topic message keeps `goal` fresh. The topic
  path (live goal updates / mid-mission reset) is preserved, and the
  defaulted keyword keeps existing callers unchanged.

## Acceptance criteria

- [x] Watchdog-level tests: an active goal keeps `goal` fresh with a
      missing or stale topic stamp; idle (default `goal_active=False`)
      still trips `goal` — the safe idle zero-twist is preserved;
      existing staleness cases pass unchanged (`test_watchdog.py`).
- [x] Node-level tests: `_goal_active` is `True` across the mission
      loop and `False` after the succeed / cancel / timeout-abort /
      exception / no-policy-abort exits; an overlapping goal's exit
      does not clear freshness under the survivor; and the
      `_on_tick → stale_sources(goal_active=...)` wiring is pinned
      (mutation-tested — dropping the kwarg fails the test)
      (`test_inference_runtime.py`).
- [x] `strafer_inference` green via colcon on the Jetson.
- [x] Live-node smoke (standalone, no sim): idle → `goal` in
      `stale=[...]`; during a 60 s action goal → `goal` absent the
      whole mission; after the timeout-abort → `goal` stale again.
- [ ] `strafer_direct` sim mission drives to the goal (no ~1 s cutoff),
      sustained non-zero `/strafer/cmd_vel` — operator, needs the
      sim-in-the-loop rig.
- [ ] `hybrid_nav2_strafer` sim mission follows the rolling subgoal —
      operator, needs the sim-in-the-loop rig.
- [x] If the work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`,
      update those in the same commit.

## Out of scope

- `ros_client.py` dispatch and the `/plan` + subgoal chain (healthy).
- The sim bridge and the subgoal generator.
- `goal_timeout_s` (stays 1.0 s — it still governs the topic path; the
  fix is the source *semantics*, not the timeout).
- `_last_goal_map` semantics (mid-mission reset behavior unchanged).

Unblocks
[`strafer-direct-sim-validation`](strafer-direct-sim-validation.md) and
[`strafer-hybrid-sim-validation`](../../parked/trained-policy/strafer-hybrid-sim-validation.md)
— this is the runtime bug those validations were expected to surface.
