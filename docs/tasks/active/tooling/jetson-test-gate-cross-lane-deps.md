# `make test-jetson` cross-lane test deps break the Jetson gate

**Type:** task / tooling (test gating)
**Owner:** Either (the markers live in `source/strafer_autonomy/tests/`, shared; the symptom is Jetson-only)
**Priority:** P3 (tooling polish; bumps to P2 once CI actually gates on `make test-jetson`)
**Estimate:** S (~half day: import-guarded skips + a marker, then re-run on a clean Jetson)
**Branch:** `task/jetson-test-gate-cross-lane-deps`

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md) — the 3 DGX envs; the Jetson uses system Python 3.10 + colcon, none of them.
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- Sibling: [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md) (the `make test-*` interface) and [`test-ci-workflow`](test-ci-workflow.md) (the CI matrix that will gate on it).

## The problem (measured)

Filed off the Jetson-side audit for [`install-docs-consolidation`](../../completed/install-docs-consolidation.md) (2026-06-08, `jetson-desktop`). `make test-jetson` runs `test-autonomy` + `test-ros` + `test-driver`. On a clean Jetson:

- `test-ros` — **332/332 pass**.
- `test-driver` — **60/60 pass**.
- `test-autonomy` — **15 failures**, all `ModuleNotFoundError` for deps the Jetson legitimately does not install:
  - 4× `strafer_vlm` (in `test_databricks_models.py`) — `strafer_vlm` is a DGX-only package; the Jetson consumes the VLM over HTTP.
  - 11× `shapely` while importing `source/strafer_lab/.../tools` (in `test_scene_labels.py`, `test_spatial_description.py`) — `strafer_lab` + `shapely` are DGX/sim-only.

So the Jetson umbrella **cannot exit 0 on a correctly-provisioned Jetson**, which blocks using `make test-jetson` as a CI/PR gate there.

These tests pass on the DGX (where `.venv_vlm` / `env_isaaclab3` carry the deps), so the fix is to make them **skip when their cross-lane deps are absent**, not to install the deps on the Jetson.

## Acceptance

- [ ] `test_databricks_models.py` and the `strafer_lab.tools`-importing tests (`test_scene_labels.py`, `test_spatial_description.py`, and any sibling) **skip cleanly** when `strafer_vlm` / `strafer_lab` / `shapely` are not importable (e.g. `pytest.importorskip` at module top, or a `requires_dgx_deps` marker deselected on hosts without them).
- [ ] `make test-autonomy` exits 0 on a clean Jetson (skips reported, no errors); unchanged on the DGX (all still run and pass).
- [ ] `make test-jetson` exits 0 on `jetson-desktop` with `strafer_ros` built.
- [ ] If a new marker is introduced, register it in `source/strafer_autonomy/pyproject.toml` next to `requires_ros`, and note it in the package README's Testing section.

## Out of scope

- Installing `strafer_vlm` / `strafer_lab` / `shapely` on the Jetson — they don't belong there.
- The CI workflow itself — owned by [`test-ci-workflow`](test-ci-workflow.md); this brief just makes the local gate green so CI can adopt it.

## Triggered by

Jetson-side audit during `install-docs-consolidation` (2026-06-08): `make test-jetson` reported 15 `test-autonomy` failures, all missing-cross-lane-dep import errors, none in the executor core path.
