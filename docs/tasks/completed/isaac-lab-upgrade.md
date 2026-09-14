# Upgrade the pinned Isaac Lab / Isaac Sim

**Status:** Shipped 2026-09-13 in `99c6c83` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/218

**Type:** task / tooling (dependency + sim-stack upgrade)
**Owner:** DGX (lane: `env_isaaclab3`, the Isaac Sim/Lab install, `env_setup.sh`)
**Priority:** P3 — nothing is blocked today; bumps to P2 if a needed upstream fix/feature (or a torch the deploy chain needs) is gated behind the bump.
**Estimate:** M–L — the bump itself is small; the risk is the whole sim stack (training, bridge, rendering, the Kit test suite), so the cost is the full re-validation, not the upgrade.
**Branch:** task/isaac-lab-upgrade-* — one standing brief, several PRs (the
upgrade lands in stages; each stage is its own `task/isaac-lab-upgrade-<stage>`
branch off `main`).

## Story

As **the DGX agent**, I want **the pinned Isaac Lab / Isaac Sim moved to a
current release**, so that **we pick up upstream fixes + perf, stop drifting
from the develop line, and learn whether a newer Isaac Sim reaches torch 2.11
(the lever that would narrow the `.venv_vlm` env split).**

## Context bundle

- [`context/repo-topology.md`](../context/repo-topology.md) — the conda env set + `env_setup.sh`.
- [`context/conventions.md`](../context/conventions.md)
- Related: [`install-docs-consolidation`](install-docs-consolidation.md)
  (env topology — this bump feeds its env map) and
  [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md) (`make test-lab`
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

**Deprecated-extension relocation.** Isaac Sim 6.0 deprecates
`isaacsim.core.utils` and the 6.0.x line moves it to `isaacsim/extsDeprecated/`.
Isaac Lab's kit apps registered that directory at `ae41e2aca68` and no longer do
at `3.0.0-beta2`, so the module is present on disk but unimportable under
`isaaclab.sh -p` and any direct import dies at runtime inside a booted Kit —
`run_sim_in_the_loop.py` among them. The eight call sites route through
`strafer_lab.isaacsim_compat`, which prefers the documented replacements
(`isaacsim.core.experimental.utils`, `isaacsim.core.rendering_manager`) and keeps
the deprecated location as a fallback. A pure-suite guard test fails if anything
imports the deprecated surface directly again.

## Approach

- **Build a second env + clone pair alongside the existing one.** The new Isaac
  Lab clone is checked out at the target release and installed into a new conda
  env; the old pair stays on disk, untouched and still working, for the whole
  migration.
- The new env is created with **`rsl-rl-lib==5.4.2`** and
  **`onnxscript>=0.7.1`** from the start, rather than flipping either pin in
  place afterwards. Install the Isaac Lab editables *without* the `rsl-rl`
  extra — its `==5.0.1` pin would trip `pip check` — then install
  `rsl-rl-lib==5.4.2` directly.
- **Cutover is a rename, not a reinstall**: `STRAFER_ISAACLAB_PYTHON`,
  `CONDA_ENV`, and `ISAACLAB` in `.env` select which pair the repo uses, but
  because the tagged pair is promoted *into* the canonical names (below), those
  three strings are identical before and after and there is nothing to flip.
  Update the recreate command + pinned versions in `.env.example` /
  `repo-topology.md` (the recipe itself lives in
  [`source/strafer_lab/README.md` → Install](../../../source/strafer_lab/README.md#install))
  to describe the new pair in the same PR as the flip. **Rollback is the rename,
  the clone move and the editable re-link, all run in reverse** — a pointer
  flip-back would be a no-op.
- **One canonical name, renamed at the flip.** Documentation names a single
  Isaac Lab environment by role; only `.env` and the Makefile defaults carry the
  concrete name. The flip renames the tagged environment to `env_isaaclab3` and
  the retired one to `env_isaaclab3-retired`, moves the clones to match, and
  re-links the `isaaclab_*` editables in **both** environments against their moved
  paths, so afterwards only canonical names appear anywhere. Re-linking the retired
  environment is not optional: its editables name the canonical clone path, which
  after the move holds the *other* pair, and it resolves there silently. The retired
  pair stays on disk until the first rig gate passes on the new stack.
- **The old pair is a preserved artifact, not scratch space.** It is no longer
  rebuildable from the notes that produced it, and it is the only way to
  recompute pre-bump config hashes when a golden moves and the attribution
  needs both trees. Do not delete it, reinstall over it, or upgrade anything
  inside it.
- Record the **torch version** the new release ships (the input to the
  `.venv_vlm` consolidation question).
- **The recipe is reconstructed from the build's own pip logs and then proved
  by rebuilding into a third, throwaway env** — the pair was built before any
  recipe was written down, so transcribing was not an option and a freeze
  cannot recover ordering.
- **Every Kit launch goes through a boot watchdog.** Isaac Sim 6.0.1
  intermittently stops during a Kit boot on this host, so a launch that never
  starts is relaunched and counted rather than recorded as a failed measurement.
  It is a no-op on the earlier pin. The stall was located to a deadlock inside
  the carb plugin registry and neither candidate avoidance works, so the
  wrapper is the mitigation rather than a placeholder — see
  `docs/measurements/kit-boot-hang-2026-09-11`.
- Re-validate the sim stack: `make test-lab` (Kit suite + pure-Python), a
  training smoke, and a `make sim-bridge` smoke. Confirm no regression to the
  physics fixes (roller-bounce / teleop-perf shared cfg) or the headless
  `--video` render path.
- Confirm the legacy policy-export path still works — or, if the new torch
  removes it, that triggers
  [`policy-export-deprecation-migration`](../active/trained-policy/policy-export-deprecation-migration.md).

## Acceptance

- [x] Pinned Isaac Lab version bumped in a **new** env + clone pair built
      alongside the old one; the recreate command + pinned versions (including
      `rsl-rl-lib==5.4.2` and `onnxscript>=0.7.1`) live in
      [`source/strafer_lab/README.md` → Install](../../../source/strafer_lab/README.md#install),
      with `.env.example` / `repo-topology.md` describing the pair by role; the
      new torch version recorded (2.11.0+cu130, against 2.10.0+cu130 retired).
      `env_setup.sh` is a pointer-exporter and carries no pins, so it is not a
      recipe home — the recipe was relocated to the package README.
- [x] The `.env` pointers select the new pair, the old pair is still intact on
      disk, and rollback is demonstrated to restore the old behavior. Rehearsed
      both ways on 2026-09-13. Backwards: the pre-flip pair comes back at Isaac
      Sim 6.0.0.0 / torch 2.10.0+cu130, and it recomputes **22 of 22** pre-flip
      contract hashes — the property this brief preserves the old pair for.
      Forwards: the canonical pair returns and the contract gate reads 148 / 148.
      Rollback is the rename, the clone move and the editable re-link run in
      reverse; flipping the pointers is not part of it, because after the
      promotion to canonical names they read the same either way.
- [x] `make test-lab` green on the new pair (modulo the known
      [`collision-imu-signal-flaky`](../active/investigations/collision-imu-signal-flaky.md)
      flake). Kit **487 / 487**, pure **1252 passed / 1 skipped**, `exit=0`. The flake
      did not appear; imu read 4/4.
- [x] Training smoke + `make sim-bridge` smoke pass; no regression to the
      roller-bounce / teleop-perf physics or the `--video` render path. Both smokes
      pass, and the `--video` path records on both pins. **With one measured
      exception**: the recorded image is photometrically darker on the new pin
      (×0.715), which is not a break in the path but is a real change to every RGB
      calibration downstream of it. It is tracked in
      [`render-photometric-shift-isaacsim6`](../active/reliability/render-photometric-shift-isaacsim6.md);
      the depth observation, which the policy lane consumes, is unmoved.
- [x] The torch-version delta is recorded against the `.venv_vlm`
      consolidation question; if the bump changes any env fact, update
      `repo-topology.md` in the same commit. The lever the brief was filed to
      measure has moved: the Isaac Lab env now carries **torch 2.11.0+cu130**
      against `.venv_vlm`'s **2.11 +cu128**, so the two share a minor and differ
      only in the CUDA build. `repo-topology.md` already states this and needs
      no edit. The delta is *not* written into
      [`install-docs-consolidation`](install-docs-consolidation.md)
      — that brief is completed, and completed briefs are records of what was
      true when they shipped. Whether the narrowed gap is enough to fold
      `.venv_vlm` remains its own cadence call.

## Out of scope

- The `.venv_vlm` fold itself (its own cadence call; this brief only measures
  the torch delta).
- CI for the Kit suite (owned by
  [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md)).
- Migrating policy export off deprecated APIs
  ([`policy-export-deprecation-migration`](../active/trained-policy/policy-export-deprecation-migration.md))
  — unless the new torch *removes* the legacy path, in which case that brief
  becomes a hard dependency.

## Triggered by

Env-topology thread (test-tree-unification PR): the pinned Isaac Lab is ~6
weeks stale, and the env-consolidation analysis flagged Isaac Sim's torch as
the floor that gates further venv consolidation. Filed to keep the sim stack
current and to measure the torch-2.11 lever.
