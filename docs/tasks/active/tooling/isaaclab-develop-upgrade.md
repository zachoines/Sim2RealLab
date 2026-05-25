# Upgrade Isaac Lab from `ae41e2aca68` (VERSION 3.0.0 pre-release) to current develop

**Type:** investigation + task
**Owner:** DGX agent (lane: `source/strafer_lab/`)
**Priority:** P3
**Estimate:** M (~2-3 days; bisect + fix import drift, validate bridge + training smoke, re-publish pin)
**Branch:** task/isaaclab-develop-upgrade

## Story

As a **DGX/sim developer**, I want **strafer_lab to import cleanly against the current Isaac Lab `develop` tip (and ideally a tagged Isaac Lab 3.x release)**, so that **new clones on fresh hosts don't have to checkout a months-old commit hash and the Windows-bringup runbook can stop pinning by SHA**.

## Context

Today (May 2026) `env_isaaclab3` on the DGX is pinned to commit
`ae41e2aca68` (`VERSION = 3.0.0`, pre-release line of Isaac Lab 3.x).
This pin was the implicit baseline strafer_lab was developed against.

Since `ae41e2aca68`, the Isaac Lab develop branch has merged several
API-breaking refactors. The Windows-workstation bringup (commit
`fbd7c80`, brief `windows-workstation-bringup.md`) tried to clone
`develop` tip and surfaced the first of these breaks:

- **`isaaclab.utils.configclass` is now a lazy-loaded submodule, not
  a callable function.** PR #4741 "Lazifies IsaacLab export system"
  (commit `5001364ce`) replaced the eager re-export in
  `isaaclab/utils/__init__.py` with `lazy_loader.attach_stub(...)`.
  The stub does not promote the `configclass` *function* up to the
  `isaaclab.utils` namespace, so existing `from isaaclab.utils import
  configclass` resolves to the submodule object. Affected file:
  `source/strafer_lab/strafer_lab/tasks/navigation/agents/distributions.py:22`
  and any other `from isaaclab.utils import <name>` in `strafer_lab/`.

The Windows bringup pinned IsaacLab to `ae41e2aca68` to ship; this
brief is the follow-up that actually catches up.

## Acceptance criteria

- [ ] strafer_lab imports cleanly against the current Isaac Lab
      `develop` tip with `./isaaclab.sh --install rl`. No
      `TypeError: 'module' object is not callable` or analogous
      `from isaaclab.X import Y` failures.
- [ ] `make sim-bridge` runs to the steady-state per-phase reference
      numbers in `docs/tasks/context/bridge-runtime-invariants.md` on
      the DGX with the new Isaac Lab.
- [ ] `make sim-harness` completes a multi-mission sweep against a
      stock Infinigen scene with no regressions in reachability
      labeling.
- [ ] `Scripts/train_strafer_navigation.py` reaches the first PPO
      iteration on `Isaac-Strafer-Nav-Real-NoCam-v0` with no
      Isaac-Lab-API errors (smoke test only — no convergence check).
- [ ] The runbook update is committed in the same PR:
      `docs/INTEGRATION_WINDOWS_WORKSTATION.md` Step 7 drops the
      SHA pin and points at the new known-good tag/branch, AND
      `docs/DGX_SPARK_SETUP.md` Step 5 (`Isaac Lab 3.0 (Develop
      Branch)` section) updates to the new install token / commit.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Approach

### Phase 1 — Map the drift (½ day)

- `cd ~/Documents/repos/IsaacLab && git log ae41e2aca68..origin/develop
  --oneline -- source/isaaclab/isaaclab/utils/` — enumerate the
  utils-layer refactors landed since the pin.
- Grep strafer_lab for every `from isaaclab` / `from isaaclab_*`
  import; cross-reference with the diff above. Build a short list
  of the API surfaces affected.
- Decide pin target: latest stable Isaac Lab tag (if 3.x is out by
  then) OR a known-good commit on develop. Capture rationale in
  the PR.

### Phase 2 — Fix the imports (1 day)

- Mechanical changes: `from isaaclab.utils import X` →
  `from isaaclab.utils.X import X` for any X now lazy-loaded as a
  submodule. Probably `configclass`, possibly `string_utils`,
  `math_utils`, `dict_utils`.
- Run `python -c "import strafer_lab"` until it imports cleanly.

### Phase 3 — Validate (1 day)

- `make sim-bridge` end-to-end (cross-host with the Jetson).
- `make sim-harness` for one scene.
- `Scripts/train_strafer_navigation.py --num_envs 16` for one iter.
- Re-derive bridge per-phase numbers, update
  `bridge-runtime-invariants.md` if anything drifted.

## Out of scope

- **Switching off rsl_rl.** If a newer Isaac Lab release renames its
  RL bundle (`rsl_rl` token already became `rl` between the pin and
  develop tip), update the install command — but don't migrate
  strafer_lab to a different RL framework.
- **Isaac Sim binary version upgrade beyond 6.0.x.** Isaac Sim 6.x
  is the validated runtime; this brief is about Isaac Lab only.
- **Re-architecting `env_isaaclab3` as a non-conda venv.** That's a
  separate "do we still need conda on Linux" discussion.

## Investigation pointers

- IsaacLab PR #4741 (commit `5001364ce`, "Lazifies IsaacLab export
  system") for the import-pattern change.
- The Windows-bringup brief's decision log (commit `fbd7c80`) names
  `ae41e2aca68` as the pre-upgrade pin.
- IsaacLab releases page — Isaac Lab 3.x tag if shipped by upgrade time.
