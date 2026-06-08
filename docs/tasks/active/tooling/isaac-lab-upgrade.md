# Upgrade the pinned Isaac Lab / Isaac Sim

**Type:** task / tooling (dependency + sim-stack upgrade)
**Owner:** DGX (lane: `env_isaaclab3`, the Isaac Sim/Lab install, `env_setup.sh`)
**Priority:** P3 — nothing is blocked today; bumps to P2 if a needed upstream fix/feature (or a torch the deploy chain needs) is gated behind the bump.
**Estimate:** M–L — the bump itself is small; the risk is the whole sim stack (training, bridge, rendering, the Kit test suite), so the cost is the full re-validation, not the upgrade.
**Branch:** task/isaac-lab-upgrade

## Story

As **the DGX agent**, I want **the pinned Isaac Lab / Isaac Sim moved to a
current release**, so that **we pick up upstream fixes + perf, stop drifting
from the develop line, and learn whether a newer Isaac Sim reaches torch 2.11
(the lever that would narrow the `.venv_vlm` env split).**

## Context bundle

- [`context/repo-topology.md`](../../context/repo-topology.md) — the conda env set + `env_setup.sh`.
- [`context/conventions.md`](../../context/conventions.md)
- Related: [`install-docs-consolidation`](../../active/tooling/install-docs-consolidation.md)
  (env topology — this bump feeds its env map) and
  [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md) (`make test-lab`
  is the re-validation gate).

## Context (measured)

The DGX runs Isaac Lab **develop @ `ae41e2aca68` (2026-04-23)** — VERSION
`3.0.0`, `isaaclab` ext `4.6.12` — in `env_isaaclab3` on torch
**`2.10.0+cu130`**. As of filing that is ~6 weeks behind the develop line.

Two reasons to move:

1. **Drift + upstream fixes/perf.** Isaac Lab develop moves fast; the longer
   we pin an old commit the harder the eventual jump and the more upstream
   physics/rendering fixes we forgo. The teleop-perf / roller-bounce work
   fought PhysX behavior that a newer release may have changed.
2. **The env-consolidation lever.** Isaac Sim's *compiled* torch is the hard
   floor for `env_isaaclab3` (currently 2.10). `.venv_vlm` is kept separate
   by design (it wants torch 2.11 + transformers 5.x). A newer Isaac Sim that
   ships torch 2.11 wouldn't eliminate that cadence split, but it would narrow
   the gap — worth measuring as part of the bump.

**Risk:** this touches everything that boots Kit — training
(`train_strafer_navigation.py`), the sim bridge (`run_sim_in_the_loop.py`),
rendering/video, and the entire `run_tests.py` Kit suite. A bad bump has a
wide blast radius, so the deliverable is dominated by re-validation.

## Approach

- Bump the pinned Isaac Lab commit/release on a branch; recreate
  `env_isaaclab3` and update its recreate command + any pinned versions in
  `env_setup.sh` / `repo-topology.md`.
- Record the **torch version** the new release ships (the input to the
  `.venv_vlm` consolidation question).
- Re-validate the sim stack: `make test-lab` (Kit suite + pure-Python), a
  training smoke, and a `make sim-bridge` smoke. Confirm no regression to the
  physics fixes (roller-bounce / teleop-perf shared cfg) or the headless
  `--video` render path.
- Confirm the legacy policy-export path still works — or, if the new torch
  removes it, that triggers
  [`policy-export-deprecation-migration`](../trained-policy/policy-export-deprecation-migration.md).

## Acceptance

- [ ] Pinned Isaac Lab version bumped; `env_isaaclab3` recreate command + any
      pinned versions in `env_setup.sh` / `repo-topology.md` updated; the new
      torch version recorded.
- [ ] `make test-lab` green from a fresh `env_isaaclab3` (modulo the known
      [`collision-imu-signal-flaky`](../investigations/collision-imu-signal-flaky.md)
      flake).
- [ ] Training smoke + `make sim-bridge` smoke pass; no regression to the
      roller-bounce / teleop-perf physics or the `--video` render path.
- [ ] The torch-version delta is recorded against the `.venv_vlm`
      consolidation question in
      [`install-docs-consolidation`](../../active/tooling/install-docs-consolidation.md);
      if the bump changes any env fact, update `repo-topology.md` in the same
      commit.

## Out of scope

- The `.venv_vlm` fold itself (its own cadence call; this brief only measures
  the torch delta).
- CI for the Kit suite (owned by
  [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md)).
- Migrating policy export off deprecated APIs
  ([`policy-export-deprecation-migration`](../trained-policy/policy-export-deprecation-migration.md))
  — unless the new torch *removes* the legacy path, in which case that brief
  becomes a hard dependency.

## Triggered by

Env-topology thread (test-tree-unification PR): the pinned Isaac Lab is ~6
weeks stale, and the env-consolidation analysis flagged Isaac Sim's torch as
the floor that gates further venv consolidation. Filed to keep the sim stack
current and to measure the torch-2.11 lever.
