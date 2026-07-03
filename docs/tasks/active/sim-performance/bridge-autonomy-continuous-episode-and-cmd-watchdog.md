# Bridge autonomy mode: continuous episode (no reset-on-collision) + cmd_vel stop-on-silence watchdog

**Type:** bug
**Owner:** DGX agent
**Priority:** P2
**Estimate:** S–M
**Branch:** task/bridge-autonomy-continuous-episode-and-cmd-watchdog

## Story

As a **mission operator running sim-in-the-loop autonomy validation**, I
want **the bridge to behave like the real robot for the duration of a
mission — one continuous episode, and motors that stop when commands
stop**, so that **a policy mission plays out end-to-end instead of the
robot teleport-resetting on contact and coasting on stale commands**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`bridge-publish-rate-decouple`](bridge-publish-rate-decouple.md) — sibling bridge-fidelity work.

## Context

Two bridge behaviors — correct for *training*, wrong for *deployment
validation* — surfaced during the 2026-07-02/03 NOCAM_SUBGOAL
sim-in-the-loop runs. They make a hybrid mission unplayable regardless
of policy quality.

### 1. Reset-on-collision (episode terminations leak into bridge mode)

The bridge env inherits the training terminations
(`composed_env_cfg.py:450`, `_TERMINATIONS_BY_SOURCE_SUBGOAL[kind]()` →
`TerminationsCfg_ProcRoom_Subgoal`), which include a collision/contact
termination (`strafer_env_cfg.py` contact sensor on `body_link`) and
episode-length / off-path terminations. On the rig the operator sees:
the policy drives toward the target, **collides with it, the env resets
the robot to a random spawn, and the mission re-chases from there —
repeating until the mission times out.** The real robot does not
teleport on contact; for autonomy validation the episode must be
**continuous** (a bump is a bump, the mission continues or fails, no
reset).

Bridge/autonomy mode should run with the reset-driving terminations
**disabled** (or an effectively infinite episode) while training keeps
them. Confirm which terms fire (collision, time_out, off_path,
goal-reached) and gate them off for the bridge-autonomy variant only.

### 2. cmd_vel held indefinitely (no stop-on-silence watchdog)

`async_publisher.get_cmd_vel()` returns the last received `(vx, vy, wz)`
with no staleness gate (a `last_cmd_monotonic()` accessor exists but the
getter ignores it). So when `/cmd_vel` goes quiet, the bridge keeps
applying the last command — a robot commanded to circle keeps circling
indefinitely, and the last policy velocity coasts the robot past the
goal. The real RoboClaw driver (`roboclaw_node.py`) zeroes the motors
when no `cmd_vel` arrives within its watchdog window; the bridge must
mirror that: **if `now - last_cmd_monotonic() > watchdog_s`, apply a
zero action.** (The inference node now also emits an explicit stop on
mission end — `strafer_inference` PR #132 — but the bridge watchdog is
the correct floor and matches real-robot semantics.)

## Acceptance criteria

- [ ] Bridge/autonomy mode runs a single continuous episode: a collision
      does **not** reset/teleport the robot; verify via `/strafer/odom`
      continuity across a deliberate bump (no >0.5 m position jump).
- [ ] Training runs are unaffected — terminations/resets still active in
      the training env variants (bridge-only override, not a global
      change).
- [ ] The bridge zeroes its applied action when `/cmd_vel` is silent
      longer than a watchdog window (default aligned with the RoboClaw
      driver's); verify a drifting robot halts when the command stream
      stops, without an external zero-twist.
- [ ] Decide + document whether goal-reached / episode-length terms
      should still fire in bridge mode (they may be useful as a
      mission-end signal, or may belong purely to the autonomy executor).
- [ ] If this invalidates a fact in any context module / README / guide
      under `docs/`, update it in the same commit.

## Out of scope

- Bridge publish throughput (that is
  [`bridge-publish-rate-decouple`](bridge-publish-rate-decouple.md)).
- Policy convergence / goal-approach behavior (that is the
  `strafer-nocam-subgoal-singleroom-sim-validation` play-episode sanity
  check — a separate axis: does the policy stop at the goal at all).

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py:450`
  (terminations selection), `strafer_env_cfg.py` (contact sensor + the
  `TerminationsCfg*` classes).
- `source/strafer_lab/strafer_lab/bridge/async_publisher.py`
  `get_cmd_vel` / `last_cmd_monotonic` (the missing watchdog).
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`
  (the real-robot command-watchdog reference to mirror).
- Filed off the 2026-07-03 sim-in-the-loop debugging (operator-confirmed
  circle-drift = held command, collide→reset = env term).
