# Make `start_mission`'s check and install one step

**Type:** bug (overlapping submissions can both start a mission, and one runs on beyond reach)
**Owner:** Jetson (`strafer_autonomy` executor)
**Priority:** P2. It takes two submissions inside one replacement's join, or a submission inside
another's install gap: a double submit from the CLI, or two clients. The sim harness never sets
`replace_active_mission`.
**Estimate:** S
**Branch:** `task/executor-start-mission-atomic`

## Story

As **an operator or client submitting a mission while another is starting or being replaced**, I
want **exactly one mission to win**, so that **every running mission can be cancelled and none
drives on as an orphan.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`inference-finished-successor-stop`](inference-finished-successor-stop.md), its reachability
  box: where this was read, as one of the two paths that overlap goals on the inference node.

## Context

Read at `6cd6fd7`. Paths are under `source/strafer_autonomy/strafer_autonomy/`; a bare
`mission_runner.py` or `command_server.py` is in its `executor/`.

- **Two lock sections.** `MissionRunner.start_mission` (`mission_runner.py`:262-331) checks
  under the lock at 274-292 and installs under it at 323-325.
  - The check reads the active runtime (275-276), refuses without `replace_active_mission`
    (277-284), and otherwise requests the old mission's cancel and takes its thread (285-291).
  - Between the two sections nothing is locked:
    - `_cancel_robot_actions` (294-295), whose cancel waits up to 5 s for its response
      (`clients/ros_client.py`:1293-1294);
    - the join (298), up to `replace_join_timeout_s`, 5 s;
    - the timeout refusal (299-307);
    - building the runtime and its thread (309-322).
- **The check passes twice.** `_request_cancel_locked` (2269-2280) sets the cancel event, state
  and message but leaves `active` true. Only `_finish_runtime` clears it (2260-2267). A second
  replacement that arrives during the first one's join therefore passes 276 as well and joins the
  same thread (298). Both build a runtime, both install (323-325) and both start (326), and the
  later install overwrites `_active_runtime` (324).
- **The server lets submissions overlap.** The ExecuteMission action server accepts every goal
  (`command_server.py`:190-193) and runs its callbacks on a reentrant group over a four-thread
  executor (54, 164-188). Each execute callback calls `start_mission` (204-209).
- **The orphan cannot be cancelled.**
  - The overwritten mission keeps its daemon thread (316-326) and its own cancel event (223).
  - `cancel_active_mission` (342-367) reads only `_active_runtime` (347), and a later
    replacement reads only 275, so neither can set that event.
  - The orphan's own cancel step cancels the other mission (`_cancel_step`, 2212-2214).
  - Its ExecuteMission goal mirrors the other mission, because the server polls runner-wide
    status (214, 221). A cancel on that goal cancels the other mission (216-217).
  - The ros client holds a single goal-handle slot (`clients/ros_client.py`:332, set at 899-900,
    1025-1026 and 1171-1172), so `_cancel_robot_actions` reaches at most one of the two
    missions' goals.
- **The window is wider than two replacements.**
  - A non-replacing submission that arrives after the old mission's `_finish_runtime`
    (2265-2266) and before the replacer's install (324) sees no active mission. It is accepted
    with no join, and one install then overwrites the other.
  - With no mission active, two plain submissions can both pass in the gap at 293-322.
- **Who submits.** The CLI sets `replace_active_mission` from `--replace-active`
  (`cli.py`:121). The sim harness sends it false (`strafer_lab`'s `runtime_mission.py`:143).
- **What it does downstream.** Two missions' policy goals overlap on the inference node, and the
  newer one supersedes the older. That is one of the paths in `inference-finished-successor-stop`.

## Acceptance criteria

- [ ] **A test with two concurrent submissions, as the four-thread server delivers them**,
      failing on `main`.
  - Two threads call `start_mission` with `replace_active_mission` set while an old mission is
    active. The test asserts that exactly one is accepted, that `_active_runtime` is the accepted
    one, and that at most one mission plans.
  - A deterministic interleaving needs no real mission thread, because `start_mission` only calls
    `join` and `is_alive` on it (298-299). One way: a fake old thread whose `join` records each
    arrival and blocks on a test-owned event, plus a planner that blocks on a gate.
  - The waits are bounded, so a fix that serializes submissions cannot hang the test.
  - A second case covers the non-replacing leak. The old mission finishes inside the first
    replacer's join, and a submission without `replace_active_mission` is then refused with
    `mission_active`.
- [ ] **One lock section from check to install.** A submission that passes the check reserves the
      slot under the same lock, for example a pending runtime or a replacing token, so a second
      submission sees a mission as active. The join stays outside the lock. A submission whose
      join times out releases the reservation.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-autonomy` passes.

## Investigation pointers

- `executor/mission_runner.py`: `start_mission` (:262-331), `cancel_active_mission` (:342-367),
  `_finish_runtime` (:2252-2267) and `_request_cancel_locked` (:2269-2280).
- `executor/command_server.py`: the server's construction (:164-188) and its execute callback
  (:200-234).
- Runner helpers with `MagicMock` clients to start from: `tests/test_describe_scene.py`'s
  `_make_runner` and `_make_runtime`. No test calls `start_mission` today.

## Out of scope

- The executor's exit path: [`executor-exit-cancels-active-goal`](executor-exit-cancels-active-goal.md).
- The ros client's single goal-handle slot, and `cancel_active_navigation` missing a goal whose
  handle is not yet stored (`clients/ros_client.py`:1025-1026, 1287-1291).
- The stop ordering on the inference node:
  [`inference-finished-successor-stop`](inference-finished-successor-stop.md).
