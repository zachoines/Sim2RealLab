# Replace hand-rolled tick counters with node-clock deadlines

**Type:** task / tooling (code health, ROS idiom)
**Owner:** Jetson (the nodes are `strafer_ros`; the suites run in `strafer-cpu:humble`)
**Priority:** P3 (behaviour-preserving; the payoff is legibility and one fewer way to be wrong)
**Estimate:** M (~1 day: audit, convert, re-pin the affected tests)
**Branch:** `task/ros-node-clock-deadlines`

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- Sibling: [`deploy-hardening`](https://github.com/zachoines/Sim2RealLab/pull/169), which added the
  tick counters this brief converts

## The problem

Several `strafer_ros` nodes track elapsed time by counting control-loop ticks
and converting a seconds-valued parameter into a tick budget. The rolling-subgoal
generator is the newest instance:

```python
self._hold_ticks_budget      = round(self._starvation_hold_s / tick_period)
self._stationary_ticks_required = round(starvation_stationary_s / tick_period)
```

then decremented per `_on_tick`. The reason was sound — those windows must be
**sim-time** budgets, because at a sub-unity RTF a wall-clock window buys a
fraction of the robot motion it implies — but the mechanism is not idiomatic.
`self.get_clock()` already **is** sim time under `use_sim_time`, so a deadline
expresses the same thing directly:

```python
self._hold_deadline   = self.get_clock().now() + Duration(seconds=self._starvation_hold_s)
self._stationary_since = self.get_clock().now()      # reset when the robot moves
```

That removes both derived tick budgets, both counters, and the dependence on
`update_period_s`. Behaviour is equivalent including under a stalled `/clock`:
a ROS-time timer stops firing, so tick counting stalls too, and neither deadline
expires.

## The distinction this brief must preserve

**Node-clock timestamps, not more timers.** A one-shot `create_timer` would need
cancel/recreate on every state transition and fires on the executor rather than
in the tick that makes the decision, adding concurrency to state machines that
are currently single-threaded and easy to reason about. "Use timers" and "use the
node clock" are different changes; only the second is clearly right here.

**Some wall-clock uses are deliberate and must not move.** The generator's
plan-freshness window (`path_timeout_s`, `time.monotonic`) guards against a dead
planner — a wall-clock event that must fire even if `/clock` stops entirely.
Converting it to sim time would mean a stalled bridge never trips the staleness
guard, which is the opposite of what it is for. The same reasoning covers the
replan cadence, which already runs on a `STEADY_TIME` clock with a comment
explaining why. **Audit output must classify each site as sim-time or
wall-clock and justify wall-clock ones**, not convert uniformly.

## Acceptance

- [ ] Every `strafer_ros` node audited for hand-rolled tick/counter time state;
      the inventory recorded here with a sim-time / wall-clock ruling per site.
- [ ] Sites whose semantics are a sim-time timeout converted to node-clock
      deadlines; wall-clock sites left alone with a one-line comment saying why.
- [ ] No new `create_timer` introduced by the conversion.
- [ ] Behaviour pinned before and after: for the generator specifically, the
      guard sequence (engage after N refusals, release after the travel
      threshold) must still hold — its unit suite covers this, and the rig
      evidence from the deploy PR is the reference.
- [ ] Any parameter whose unit changes meaning (e.g. a value that was only ever
      read as ticks) is called out in the PR, since operators tune these.

## Out of scope

- Changing any timeout's *value*.
- The wall-clock plan-freshness and replan-cadence budgets described above.
- Multi-threaded executor work; this brief must not change concurrency.

## Triggered by

PR review of `deploy-hardening` (2026-07-29): "we manually tick and manage these
counters — is there a more ROS2-native way using timers? Manual tick management
is something we have done in many of our ROS source files."
