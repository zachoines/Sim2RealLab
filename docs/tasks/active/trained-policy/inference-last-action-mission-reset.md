# Zero the inference node's last action at a mission boundary, as training does at a reset

**Type:** bug (train/deploy observation parity at the start of a trained-policy goal)
**Owner:** Jetson (`strafer_inference`)
**Priority:** P3. It is one observation term on one tick per goal, but a recurrent policy carries
that tick in its state. Raise it if [the episode-start stale-depth question](../../../measurements/vfov-gap-v3-2026-10-03/README.md#4-what-tick-0-shows-and-when-the-room-arrives) shows that a goal's first ticks matter on the rig.
**Estimate:** S
**Branch:** `task/inference-last-action-mission-reset`

## Story

As **the trained policy at the first tick of a goal**, I want **the `last_action` term to be zero, as
it was at the start of every training episode**, so that **a goal's first observation does not carry
the previous goal's final command.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`inference-post-goal-stop-race`](../../completed/inference-post-goal-stop-race.md) — after it,
  `_last_action` is written only together with a publish.

## Context

**Deploy** (`strafer_inference/inference_node.py`):
- **One write.** `_last_action` is initialised to zeros in `__init__`. Its only write is at the
  policy publish, and after #238 it is written only when the action is published.
- **A new goal does not clear it.** The execute preamble resets the policy's hidden state under
  `_policy_lock`, and nothing more.
- **The result.** A goal's first observation carries the previous goal's final applied action
  through `build_raw_obs_dict(last_action=...)`.

**Training** (Isaac Lab `v3.0.0-beta2.patch1`):
- `ActionManager.reset` zeroes `_prev_action` and `_action` for the reset environments
  (`isaaclab/managers/action_manager.py`, 358–359).
- The `last_action` observation term returns `env.action_manager.action`
  (`strafer_lab/tasks/navigation/mdp/observations.py`, 656–664).
- So every training episode's first observation has a zero `last_action`.
- The sim-in-the-loop env adapter also zeroes its own last action on reset
  (`strafer_lab/sim_in_the_loop/runtime_env.py`, 191).

**Related, not this brief.** A successor goal's execute gate raises the goal count before its
preamble resets the policy's hidden state. A tick that runs between the two infers with the
predecessor's hidden state, for one tick.

## Acceptance criteria

- [ ] At the mission boundary, `_last_action` is zeroed together with the hidden-state reset, under
      the same lock.
- [ ] A test in `strafer_inference/test/test_inference_runtime.py`: after a goal publishes a
      non-zero action and ends, the next goal's first observation carries a zero `last_action`.
      The test fails on `main`.
- [ ] The straddle case of `inference-post-goal-stop-race` still holds. A tick that a mission-end
      stop overtakes neither publishes nor writes `_last_action`.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-ros` passes.

## Out of scope

- The successor's hidden-state reset coming after its count bump (noted above).
- [The episode-start stale-depth question](../../../measurements/vfov-gap-v3-2026-10-03/README.md#4-what-tick-0-shows-and-when-the-room-arrives), which this brief is to be read alongside.
- [`inference-finished-successor-stop`](../reliability/inference-finished-successor-stop.md).
