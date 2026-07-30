# `make test-jetson` cross-lane test deps break the Jetson gate

**Status:** Shipped 2026-07-29 in `c36a714` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/170

**Type:** task / tooling (test gating)
**Owner:** Either (the markers live in `source/strafer_autonomy/tests/`, shared; the symptom is Jetson-only)
**Priority:** P3 (tooling polish; bumps to P2 once CI actually gates on `make test-jetson`)
**Estimate:** S (~half day: import-guarded skips + a marker, then re-run on a clean Jetson)
**Branch:** `task/jetson-test-gate-cross-lane-deps`

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md) — the 3 DGX envs; the Jetson uses system Python 3.10 + colcon, none of them.
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- Sibling: [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md) (the `make test-*` interface) and [`test-ci-workflow`](../active/tooling/test-ci-workflow.md) (the CI matrix that will gate on it).

## The problem (measured)

Filed off the Jetson-side audit for [`install-docs-consolidation`](install-docs-consolidation.md) (2026-06-08, `jetson-desktop`). `make test-jetson` runs `test-autonomy` + `test-ros` + `test-driver`. On a clean Jetson:

- `test-ros` — **332/332 pass**.
- `test-driver` — **60/60 pass**.
- `test-autonomy` — **15 failures**, all `ModuleNotFoundError` for deps the Jetson legitimately does not install:
  - 4× `strafer_vlm` (in `test_databricks_models.py`) — `strafer_vlm` is a DGX-only package; the Jetson consumes the VLM over HTTP.
  - 11× `shapely` while importing `source/strafer_lab/.../tools` (in `test_scene_labels.py`, `test_spatial_description.py`) — `strafer_lab` + `shapely` are DGX/sim-only.

So the Jetson umbrella **cannot exit 0 on a correctly-provisioned Jetson**, which blocks using `make test-jetson` as a CI/PR gate there.

These tests pass on the DGX (where `.venv_vlm` / `env_isaaclab3` carry the deps), so the fix is to make them **skip when their cross-lane deps are absent**, not to install the deps on the Jetson.

## Acceptance

- [x] `test_databricks_models.py` and the `strafer_lab.tools`-importing tests (`test_scene_labels.py`, `test_spatial_description.py`, and any sibling) **skip cleanly** when `strafer_vlm` / `strafer_lab` / `shapely` are not importable (e.g. `pytest.importorskip` at module top, or a `requires_dgx_deps` marker deselected on hosts without them).
- [x] `make test-autonomy` exits 0 on a clean Jetson (skips reported, no errors); unchanged on the DGX (all still run and pass).
- [x] `make test-jetson` exits 0 on `jetson-desktop` with `strafer_ros` built.
- [x] No new marker was introduced (`pytest.importorskip` at each seam is self-maintaining where a marker needs every new test tagged), so there was nothing to register; the mechanism and the dependency list are documented in the package README's Testing section. If a new marker is introduced, register it in `source/strafer_autonomy/pyproject.toml` next to `requires_ros`, and note it in the package README's Testing section.

## Out of scope

- Installing `strafer_vlm` / `strafer_lab` / `shapely` on the Jetson — they don't belong there.
- The CI workflow itself — owned by [`test-ci-workflow`](../active/tooling/test-ci-workflow.md); this brief just makes the local gate green so CI can adopt it.

## Triggered by

Jetson-side audit during `install-docs-consolidation` (2026-06-08): `make test-jetson` reported 15 `test-autonomy` failures, all missing-cross-lane-dep import errors, none in the executor core path.

## Outcome

Shipped wider than filed. The brief's subject — cross-lane deps — was one of
three causes:

1. **Cross-lane deps.** `importorskip` at the narrowest seam covering each
   family (a fixture, a helper, or the module top). Robot image: 32 failed +
   47 errors -> 608 passed, 79 skipped, exit 0, with the same 608 passing as
   before. Installing a gated dep restores its runs (networkx: 13 passed + 14
   skipped -> 27 passed), so the workstation lane is unchanged.
   The dep list had grown since filing: `chromadb`, `networkx` and `fastapi`
   joined the `strafer_vlm` / `strafer_lab` / `shapely` originally recorded.

2. **No host toolchain.** A container-primary robot host has no bare-metal ROS,
   colcon or pytest, so `test-ros` / `test-driver` could not run at all. Both
   route through `tools/run_ros_tests.sh` — native when available, else
   `strafer-cpu:humble`.

3. **The gate was vacuous.** `test-ros` drove `colcon test`, whose
   ament_python task invokes `python3 -m unittest -v` with no discovery
   arguments: 0 collected, `OK`, exit 0. It drives pytest per package now.

**Open, for whoever next runs a provisioned Jetson:** this brief recorded
`test-ros` at 332/332 on `jetson-desktop`, which cannot be reconciled with a
runner that collects nothing. Most likely apt colcon (in the image) versus pip
colcon (on that host). The native branch of `run_ros_tests.sh` is also untested
— this host has no ROS toolchain to exercise it.
