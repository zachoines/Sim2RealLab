# Jetson README Install/Run fixes from the install-docs audit

**Type:** documentation refresh (Jetson lane)
**Owner:** Jetson (`source/strafer_ros/README.md`, the executor side of `source/strafer_autonomy/README.md`)
**Priority:** P2 (the two path/pip blockers make the documented install a silent no-op on a fresh Jetson)
**Estimate:** S (~half day; verified doc edits, no code)
**Branch:** `task/jetson-readme-install-run-fixes`

## Story

As **a new contributor on a fresh Jetson Orin Nano**, I want **the
`strafer_ros` and `strafer_autonomy`-executor README Install/Run sections to
match what actually works on the Jetson**, so that **following them verbatim
builds and runs the stack instead of silently producing zero packages.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md) — the Jetson repo path is `~/workspaces/Sim2RealLab` (lowercase, plural), Python 3.10 + colcon.
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md) — these READMEs are Jetson lane.
- [context/conventions.md](../../context/conventions.md) — user-facing documentation maintenance.

## Context

This is the **Jetson-lane half** of [`install-docs-consolidation`](../../completed/install-docs-consolidation.md)
(DGX coordinator slice shipped in PR #83). That brief's Jetson audit
(2026-06-08 on `jetson-desktop`: Ubuntu 22.04.5 / L4T R36.5.0, ROS Humble,
Python 3.10.12, pip 22.0.2) verified the discrepancies below from a clean
rebuild. The DGX coordinator already fixed the **same** path/pip/source-ROS
bugs in the top-level `Readme.md` Jetson block (PR #83); this brief mirrors
them into the Jetson-lane READMEs the coordinator could not touch.

Follow the shape + conventions `install-docs-consolidation` established:
the `## Install / ### Linux / ### Run` template, audited against a live
Jetson, **no `_Last verified` footers**, de-dup via links.

## Acceptance — verified discrepancies to fix

### `source/strafer_ros/README.md`

- [ ] **L133 [blocker]** — `ln -s ~/Workspace/Sim2RealLab/...` → `~/workspaces/...` (lowercase, plural). The wrong path leaves a dangling `*` symlink and `colcon build` reports "0 packages finished" — a silent no-op install.
- [ ] **L146 [blocker]** — `pip install -e source/strafer_shared source/strafer_autonomy` fails on stock pip 22.0.2 (PEP 660, missing `build_editable` hook). Add `--no-build-isolation`, and use `-e A -e B` (a single `-e` only makes `strafer_shared` editable).
- [ ] **L130-138 [major]** — the install block never `source /opt/ros/humble/setup.bash` before `colcon build` (works only because `.bashrc` auto-sources ROS; breaks under `--noprofile` / CI / sudo / cron).
- [ ] **L260 [major]** — broken link `docs/DEFERRED_WORK.md` → `docs/tasks/DEFERRED_WORK.md`.
- [ ] **L253 [major]** — documents `make test-unit` (a deprecated alias) → `make test-driver`.
- [ ] **L252 [major]** — `make test  # all colcon tests` is wrong; on the Jetson `make test` auto-dispatches to `test-jetson`. The colcon-only path is `make test-ros`.
- [ ] **L9 [minor]** — "six runtime ROS 2 packages" is stale → seven runtime (incl. `strafer_inference`) + `strafer_msgs` = 8.
- [ ] **L245 [minor]** — the hand-rolled `colcon test --packages-select` list (5 pkgs) skips `strafer_msgs` / `description` / `inference` → use `make test-ros`.

### `source/strafer_autonomy/README.md` (executor side)

- [ ] **L123 [blocker]** — "logs a warning if unreachable" is **false**: an unreachable VLM/planner propagates out of `build_command_server` and the executor exits 1 (`/execute_mission` never registers). Reconcile with the code's own docstring. *(The behavioral question — should there be an operator skip flag — is owned by [`executor-startup-health-check-contract`](../reliability/executor-startup-health-check-contract.md); this brief is the doc-wording reconcile.)*
- [ ] **L256 [major]** — "fails fast if reachable but `model_loaded=false`" is true but, by omission, implies an unreachable service is tolerated; it isn't.
- [ ] **L217-218 [major]** — `pip install -e source/strafer_autonomy` fails on pip 22.0.2 → add `--no-build-isolation`.
- [ ] **L283 [major]** — the CLI common-flag list is wrong: only `--node-name` and `--wait-timeout` are common to submit/status/cancel; `--action-name` is submit+cancel only, `--service-name` is status only.

### Cross-check

- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Out of scope

- **Behavior changes** to the executor health check — [`executor-startup-health-check-contract`](../reliability/executor-startup-health-check-contract.md) owns the design question (fail-fast docs vs. an advisory flag); this brief only makes the README describe today's behavior.
- **The `make test-jetson` green-gate fix** — owned by [`jetson-test-gate-cross-lane-deps`](jetson-test-gate-cross-lane-deps.md).
- **DGX-side or Windows install docs** — DGX shipped in PR #83; Windows is owned by [`windows-workstation-bringup`](windows-workstation-bringup.md).

## Triggered by

The Jetson-side audit for `install-docs-consolidation` (2026-06-08): a clean rebuild on `jetson-desktop` surfaced two install blockers (wrong repo path, PEP 660 pip) plus several stale Run/Test references in the Jetson-lane READMEs the DGX coordinator could not edit.
