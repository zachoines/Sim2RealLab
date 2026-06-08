# CI test workflow — GitHub Actions matrix over `make test-*`

**Type:** task / tooling (CI/CD)
**Owner:** Either (the YAML is host-agnostic; the `lab` matrix entry leans DGX since it needs a self-hosted runner)
**Priority:** P3 — no feature blocked; bumps to P2 once two+ agents are landing changes in the same week and "whoever remembered to run the suite" stops scaling.
**Estimate:** M — the workflow YAML + the repo's Actions-permission setup; +1 day if the `ros` container matrix entry is included.
**Branch:** task/test-ci-workflow

## Story

As **an agent landing changes in either lane**, I want **PRs to get
automatic test feedback from the same `make test-*` targets I run locally**,
so that **drift doesn't rely on whoever remembers to run the suite, and both
hosts validate changes the same way.**

## Context bundle

- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/conventions.md`](../../context/conventions.md)
- **Consumes:** [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md)
  (shipped) — the `make test-*` interface this workflow invokes. This brief
  is the **CI half (Part 2)** carved out of that one; the Makefile half
  shipped already.

## Context

The `make test-*` interface now exists (per the completed brief above):
`test-autonomy` (host-agnostic), `test-vlm` (`.venv_vlm`), `test-ros`
(colcon), `test-driver`, `test-lab` / `test-lab-pure`, and the `test-dgx` /
`test-jetson` umbrellas. CI just has to invoke the right targets per runner.

There is **no `.github/workflows/` directory yet** — this brief introduces
it. **Check the repo's per-repo Actions permissions before assuming the YAML
will Just Work** (Actions enabled, default `GITHUB_TOKEN` scope, whether
self-hosted runners are allowed).

The runner already fails loud on a crashed/zero-collected suite (the
`run_tests.py` no-XML fix shipped with Part 1), but the **pytest-based matrix
entries should still start from a clean env** so a vendored-plugin autoload
(the `launch_testing` → `lark` crash documented in the parent brief) can't
abort collection — set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` or run with
`PYTHONPATH=` as the Makefile targets already do.

## Approach

A matrix workflow under `.github/workflows/test.yml`:

| Matrix entry | Runner | Runs | Gating |
|---|---|---|---|
| `autonomy` | `ubuntu-latest` | `pip install -e source/strafer_autonomy` + `make test-autonomy` | per-PR (always) |
| `vlm` | `ubuntu-latest` | `pip install -e source/strafer_vlm` + `make test-vlm` (skip/mark any GPU-only tests) | per-PR (always) |
| `ros` | container `osrf/ros:humble-desktop` | `colcon build` + `make test-ros` | per-PR; long but tractable |
| `lab` | self-hosted DGX label | `make test-lab` | **not** per-PR — `schedule: cron` + `workflow_dispatch`; informative, not blocking, until cost/flakiness (and the `collision-imu-signal-flaky` flake) are characterized |

**Land whichever subset is cleanly reviewable.** `autonomy` on every PR is
the minimal safe first gate (no container, no self-hosted runner). `vlm` and
`ros` are the next rungs; `lab` waits on a self-hosted DGX runner actually
existing and should stay informative-only given the known Kit-suite flake.

## Acceptance

- [ ] `.github/workflows/test.yml` runs `autonomy` on every PR and is green
      on a clean tree.
- [ ] `vlm` and `ros` added where cleanly reviewable (wiring `ros` is a
      stretch-within-stretch; land the subset that reviews cleanly).
- [ ] `lab` (if added) runs on `schedule` + `workflow_dispatch` only, marked
      informative/non-blocking, and gated so the `collision-imu-signal-flaky`
      flake doesn't red-wall the nightly.
- [ ] CI invokes the `make test-*` targets (not hand-rolled pytest lines), so
      the local and CI invocations can't drift.
- [ ] If your work invalidates a fact in any context module, README, or
      guide, update it in the same commit.

## Out of scope

- The `make test-*` targets themselves — shipped in
  [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md).
- Provisioning the self-hosted DGX runner as a hard prerequisite — the `lab`
  entry is the only piece that needs it and is explicitly allowed to ship
  later (or never).

## Triggered by

Carved out of [`unify-test-targets-and-ci`](../../completed/unify-test-targets-and-ci.md)
when its Part 1 (Makefile unification) shipped: CI is outward-facing (runs on
the repo, consumes Actions minutes, gates PRs) and needs the repo's Actions
permissions + a self-hosted DGX runner for `lab`, so it was split to land on
its own review rather than block the Makefile work.
