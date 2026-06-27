# `--debug-overhead-cam` for scripted / coverage capture runs

**Type:** small feature (capture QA / observability)
**Owner:** DGX agent
**Priority:** P3 — does not block any acceptance bar; the headless capture runs
produce correct corpora today. This is a visual-QA convenience that shortens the
loop when a capture run *misbehaves* and the cause is in the scene, not the data.
**Estimate:** S (reuse `teleop_capture.py`'s existing overhead-viewport rig —
`--hide-overhead` / `--overhead-regex` / the `world_arcade` top-down viewport —
behind a `--debug-overhead-cam` flag on the scripted/coverage capture entry; a
periodic top-down video writer like `train_strafer_navigation.py`'s already has).
**Branch:** `task/capture-debug-overhead-cam`

**Blocked on / trigger:** Filed-on-trigger. Un-park when **a scripted or coverage
capture run shows unexplained robot behavior** (stuck, drifting, colliding with
nothing visible, 0 usable episodes) whose cause needs eyes on the scene rather
than log/metric inspection. The motivating class: a runtime collider that is
invisible in the rendered RGB — e.g. the `skirtingboard_support` convex-hull
phantom-slab collider fixed in `occupancy-interior-fidelity` would have wedged the
robot on an invisible floor slab at capture time; an overhead view makes that
obvious, where the headless frame stream does not. (That specific collider is now
fixed; this brief is for the *next* one.)

## Story

As a **harness operator debugging a misbehaving scripted/coverage capture run** I
want **an optional overhead following camera (and periodic top-down video), the
same rig teleop already has** so that **I can watch the robot in the scene and
spot a physical cause — an invisible collider, a bad spawn, a wedge — instead of
inferring it from logs and discarded-episode counts.**

## Context bundle

Read before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- `source/strafer_lab/scripts/teleop_capture.py` — the overhead-viewport rig to
  reuse (`world_arcade` top-down viewport, `--hide-overhead` to drop
  ceilings/roof/exterior hull, `--overhead-regex` override). Visibility is reset
  per launch (it does not mutate the USDC).
- `source/strafer_lab/scripts/train_strafer_navigation.py` — the periodic
  overhead-view **video writer** pattern (an existing precedent for recording a
  top-down view off a non-teleop run).
- `source/strafer_lab/scripts/coverage_capture.py` — the scripted/coverage entry
  the flag is added to.

## Sketch

- Add `--debug-overhead-cam` (operator-only) to the scripted/coverage capture
  entry: parent the teleop overhead viewport/camera to follow the active robot,
  reusing the `--hide-overhead` / `--overhead-regex` structure-prim hiding so the
  top-down view is unobstructed.
- Optionally a periodic top-down video (gated, off by default; mirror
  `train_strafer_navigation.py`) so a long headless run leaves a reviewable
  artifact without an attached GUI session.
- Operator-only: no effect on the captured LeRobot corpus, the episode contract,
  or throughput when the flag is off. Does not modify the scene USDC on disk.

## Acceptance (sketch)

- [ ] `--debug-overhead-cam` on a coverage/scripted run shows the robot following
      from overhead with structure prims hidden; off by default, zero corpus/throughput
      impact when unset.
- [ ] (optional) periodic top-down video written to the run's output dir.
- [ ] Reuses the teleop rig — no second overhead-cam implementation.

## Out of scope

- The captured-corpus camera streams (this is an operator debug view, not a
  recorded observation).
- Fixing whatever misbehavior triggered the un-park — this brief only adds the
  lens.
