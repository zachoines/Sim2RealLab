# Modularize harness capture out of run_sim_in_the_loop.py

**Type:** refactor (no behavior change)
**Owner:** DGX agent
**Priority:** P3 — organizational; the Tier 2 bridge driver works, this
makes the seam clean.
**Estimate:** M (~1–2 days; extraction + arg-group + tick-loop dedup +
keep tests green).
**Branch:** task/harness-mode-modularization

**Blocked on / trigger:** the Tier 2 bridge driver PR (#88) lands first.
Do **not** refactor `run_sim_in_the_loop.py` while that PR is in flight —
this brief picks up once it merges.

## Story

As a **maintainer of the sim↔ROS bridge**, I want
**`run_sim_in_the_loop.py` to stay the thin mechanical bridge between
`strafer_ros` and `strafer_lab`, with the harness data-capture concern
factored cleanly behind a config object and a session module**, so that
**the bridge script's argv surface and tick loop aren't progressively
bloated by capture-specific orchestration, and harness capture is a
separable subsystem.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)

## Context

The Tier 2 bridge driver migrated `--mode harness` onto the LeRobot v3
writer. It already moved most orchestration into
`strafer_lab/sim_in_the_loop/` (harness, lerobot_recorder, extras), but
two seams stayed in `run_sim_in_the_loop.py`:

1. **Argv bloat.** A block of `--inject-bad-grounding*`,
   `--repo-id`, `--session-id`, `--mission-queue`, etc. capture-only
   arguments now live on the bridge script's parser alongside the
   mechanical bridge args.
2. **Tick-loop overlap.** Harness mode runs "the same per-tick protocol
   as bridge mode" (its own code comment) without sharing the loop body.

The operator's framing: `run_sim_in_the_loop.py` should be the
mechanical sim↔ROS connection; data capture is a separable concern that
shouldn't accrete onto it.

### Target shape

- **Group the capture args** behind a single `add_harness_args(parser)`
  and resolve them into a `HarnessCaptureConfig` dataclass
  (`from_args(args)`), so the bridge parser carries one cohesive harness
  group instead of a scatter of flags.
- **Extract a `HarnessCaptureSession`** (in `strafer_lab/sim_in_the_loop/`)
  that owns the capture tick loop and takes the constructed env +
  config; `run_sim_in_the_loop.py --mode harness` becomes a thin
  construct-and-hand-off.
- **Dedupe the tick loop:** factor the shared per-tick bridge protocol
  (env.step / publisher pump / `/cmd_vel` read) so bridge and harness
  modes call one implementation, with capture as an injected per-tick
  hook rather than a parallel loop.

### The deeper fork (decide at pickup)

Whether harness capture should remain `run_sim_in_the_loop.py --mode
harness` at all, vs. become a dedicated `harness_bridge_capture.py`
driver script (sibling to `teleop_capture.py`) that imports the bridge
core as a **library**. The library route is the cleanest end-state and
best matches "run_sim = pure bridge," but requires the bridge core to be
import-safe (no `__main__`-only setup). Evaluate at pickup; the
incremental extraction above is valuable regardless of which way this
goes, so it is not gated on resolving the fork.

## Acceptance criteria

- [ ] `run_sim_in_the_loop.py`'s top-level argparse carries the
      mechanical bridge args + one `add_harness_args` group, not a
      scattered set of capture flags.
- [ ] `HarnessCaptureConfig` + `HarnessCaptureSession` own the capture
      orchestration; `--mode harness` is a thin construct-and-run.
- [ ] Bridge and harness modes share one per-tick loop implementation
      (capture is an injected hook), no duplicated loop body.
- [ ] **No behavior change** — the Tier 2 acceptance run reproduces
      byte-identical dataset structure before/after this refactor (or,
      if a dataset isn't on hand, the harness unit + smoke tests stay
      green and the capture lifecycle is unchanged).
- [ ] `make test-lab-pure` green; harness tests unchanged in intent.
- [ ] Brief shipped to `completed/` per `conventions.md` in the
      shipping PR.

## Out of scope

- Any schema / writer change (`lerobot_writer.py`, `lerobot_detections.py`,
  `lerobot_depth.py`).
- The hard-negative taxonomy expansion — that is
  [`grounding-negative-taxonomy`](grounding-negative-taxonomy.md).
- New `(driver, mission-source)` cells.
