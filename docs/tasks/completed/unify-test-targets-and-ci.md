# Unify test invocation paradigms across Makefile + add CI/CD entry points

**Status:** Shipped 2026-06-08 in `a0b3780` (DGX) — **Part 1 (Makefile unification) only.** Part 2 (CI/CD) was extracted to the follow-up below and is **not** shipped here.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/82
**Follow-ups:** [`test-ci-workflow`](../active/tooling/test-ci-workflow.md) — the GitHub Actions matrix over `make test-*` (Part 2, carved out because it is outward-facing and needs the repo's Actions permissions + a self-hosted DGX runner).

**Type:** task / tooling
**Owner:** Either (Makefile + repo-root tooling lean DGX; `source/strafer_ros/*/test/` discovery touches Jetson lane; new GitHub Actions YAML is host-agnostic — primary owner picks based on which half is heavier when picked up)
**Priority:** P3 (tooling polish; doesn't block features. Bumps to P2 once two or more independent test-paradigm drifts have been observed in flight.)
**Estimate:** M (~1–2 days for Makefile unification + brief author CI YAML; +1 day if the `ros:humble` matrix entry is included in scope)
**Branch:** task/unify-test-targets-and-ci

## Story

As **an agent working in either lane**, I want **a single, consistent
`make test-*` interface plus a CI workflow that exercises the same
targets**, so that **the four current invocation paradigms don't
drift further, both hosts validate changes the same way, and PRs get
automatic test feedback instead of relying on whichever agent
remembers to run the suite.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)

## Context

The repo currently has four ways to invoke tests, each documented
implicitly rather than by the Makefile:

| Target | Scope | Host | Backend |
|---|---|---|---|
| `make test` | colcon test over all `strafer_ros/*` packages | Jetson | colcon |
| `make test-unit` | `source/strafer_ros/strafer_driver/test/` only | Jetson | direct pytest |
| `make test-dgx` | `source/strafer_autonomy/tests/` + `source/strafer_vlm/tests/` minus `requires_ros` | DGX (uses `.venv_vlm`) | pytest |
| `source/strafer_lab/run_tests.py` | strafer_lab Isaac Sim suites (`terminations`, `events`, `rewards`, ...) | DGX (under `isaaclab.sh -p`) | bespoke subprocess + JUnit XML parser |

Drift observations that motivated this brief:

1. **No Jetson-side equivalent of `test-dgx`.** When the
   `align-after-scan-grounding` change was validated on Jetson, the
   agent ran
   `python -m pytest source/strafer_autonomy/tests/test_*.py` by
   hand. That command is not memorialized anywhere — a different
   agent picking up the next autonomy change will type a slightly
   different incantation, and the diff between "what I ran" and
   "what the DGX agent ran" widens silently.
2. **`make test-dgx` requires `.venv_vlm`.** The target works on
   the DGX but cannot run on Jetson without bootstrapping a venv
   the Jetson otherwise doesn't need. The autonomy tests themselves
   are venv-independent — they just need `pip install -e
   source/strafer_autonomy`.
3. **`run_tests.py` is shell-invoked rather than make-invoked.**
   This exists for real reasons (Isaac Sim's `os._exit` kills
   pytest's summary before it prints; init floods stdout) — the
   wrapper is intentional. But there's no `make test-lab`, so
   operators (and CI later) have to know the bespoke command line.
4. **No CI gating.** Every check today is "the agent who landed the
   change ran the tests they thought were relevant." That's been
   adequate at single-agent throughput; it won't scale once two
   agents land changes in the same week.

The point of this brief is *unification of invocation*, not
unification of the *backends*. `colcon test` for ROS packages and
`run_tests.py` for Isaac-Sim-loaded environments are correct for
their domains. The Makefile should hide that distinction behind a
predictable `make test-<thing>` interface.

## Finding: `run_tests.py` can false-green on a missing `lark` dep

*Surfaced while trying to run the Isaac-Sim `test/` contract suite during
the [`roller-contact-high-omega-bounce`](roller-contact-high-omega-bounce.md)
work (PR #76); filed here because this brief owns runner + CI invocation.*

`pytest` auto-loads every installed plugin via entry points. Isaac Sim's
vendored ROS 2 (humble, under `~/.cache/packman/chk/nv_ros2/...`) ships
`launch_testing`, which registers a pytest plugin. Loading it imports
`launch` → `launch.frontend.parse_substitution` → `from lark import Lark`,
and `lark` is **not installed** in `env_isaaclab3`. So when that vendored
ROS 2 path is on `sys.path` at pytest startup, **collection aborts before
any `strafer_lab` test runs**:

```
launch_testing/__init__.py → tools → process → import launch
  → launch/frontend/__init__.py → .parser → parse_substitution.py
  → from lark import Lark   ← ModuleNotFoundError: No module named 'lark'
```

Two properties make this a CI concern, not just an annoyance:

1. **It can present as a false green.** A `run_tests.py <suite>` run was
   observed reporting `0/0 passed … XML not generated … ALL PASSED` — the
   shape you get when the pytest subprocess crashes (here, on the `lark`
   import) before writing its JUnit XML and the wrapper reads the absence
   of failures as success. A real CI gate must treat "no XML / 0 collected"
   as an **error**, not a pass.
2. **It is environment-path-dependent.** It fires only when the vendored
   ROS 2 path is active at pytest startup (reproduced via direct `pytest`
   from a worktree; did not fire from a `--collect-only` in the primary
   checkout), so it surfaces intermittently — exactly the kind of flake CI
   should pin down.

Fix is small and external to `strafer_lab`: install `lark` into
`env_isaaclab3`, **or** disable plugin autoload for the lab runs
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, or `-p no:launch_testing`) in
`run_tests.py` / the `make test-lab` wrapper this brief adds. Whichever
path is chosen, the runner should also **fail loudly on zero collected
tests** so a future autoload break can't masquerade as green.

## Approach

Land in two parts. **Part 1 is the locked scope; Part 2 is the
stretch.**

### Part 1: Makefile unification (locked scope)

Introduce the targets below. Existing targets remain (back-compat)
but are redefined to call the new building blocks so the entry-point
list and the documented behavior converge:

| Target | What it runs |
|---|---|
| `make test-autonomy` | `python -m pytest source/strafer_autonomy/tests/ -m "not requires_ros"` — host-agnostic, no venv requirement beyond `pip install -e source/strafer_autonomy` |
| `make test-vlm` | `$(VENV_VLM)/bin/python -m pytest source/strafer_vlm/tests/` — DGX-only (gated on `.venv_vlm` presence with a clear "run `make install-tools` first" error if missing) |
| `make test-ros` | `colcon test` over `strafer_ros/*` packages (alias of the current `make test`, but renamed so its scope is self-documenting) |
| `make test-driver` | direct-pytest path for `strafer_driver/test/` (alias of the current `make test-unit`, renamed to reflect what it is) |
| `make test-lab` | **All strafer_lab tests in `env_isaaclab3`** — `isaaclab -p source/strafer_lab/run_tests.py all` (Kit suites) **+** `pytest source/strafer_lab/tests/` (pure-Python: harness + policy_tooling + contracts). DGX-only (gated on `$ISAACLAB`). Already shipped (test-tree-unification PR). It **absorbed the former `test-harness`**: the `.venv_harness` fold is confirmed + executed (lerobot coexists with `env_isaaclab3`'s CUDA torch; the harness suite also gained `pxr`). `make test-lab-pure` runs just the pure-Python half for fast iteration. |
| `make test-dgx` | composite of `test-autonomy` + `test-vlm` + `test-lab` — the **repo-wide DGX e2e umbrella** (each suite in its env: `.venv_vlm` for vlm, `env_isaaclab3` for the lab umbrella, autonomy host-agnostic) |
| `make test-jetson` | composite of `test-autonomy` + `test-ros` + `test-driver` — Jetson-friendly umbrella |
| `make test` | dispatches to `test-jetson` or `test-dgx` based on host detection (preferred) or is left as today's colcon alias with a deprecation echo (acceptable) |

Back-compat aliases (`test-unit`) kept; new names are the preferred
forms. The `help` target's auto-generated comment lines should make
the per-host umbrellas the visible defaults.

**On "a single `make test-all` orchestrator":** with the strafer_lab suites
unified under `test-lab` and folded into the `test-dgx` composite above, `make test-dgx` (and the host-dispatched
`make test`) *is* the one-command repo-wide e2e — every suite runs in its
correct env — that the env-topology discussion raised. No separate
`test-all` target is warranted; the full cross-host suite is `test-dgx` +
`test-jetson`. **Which env each suite runs in** is documented by
[`install-docs-consolidation`](../active/tooling/install-docs-consolidation.md)
(env topology) + `repo-topology.md`; this brief consumes that map rather than
re-deriving it.

### Part 2: CI/CD on GitHub Actions (stretch)

> **Extracted, not shipped here.** Part 2 moved to its own brief
> [`test-ci-workflow`](../active/tooling/test-ci-workflow.md) when Part 1
> shipped. The matrix below is preserved as the original plan; the follow-up
> brief is the live home for it.

A matrix workflow under `.github/workflows/test.yml`:

| Matrix entry | Runner | What it runs |
|---|---|---|
| `autonomy` | `ubuntu-latest` | `pip install -e source/strafer_autonomy` + `make test-autonomy` |
| `vlm` | `ubuntu-latest` | `pip install -e source/strafer_vlm` + `make test-vlm` (skip if GPU-only tests are present and unmocked) |
| `ros` | container: `osrf/ros:humble-desktop` | `colcon build` + `make test-ros`. Reasonable on hosted runners; long but tractable. |
| `lab` | self-hosted DGX label | nightly schedule, not per-PR. `make test-lab`. |

Per-PR gating: `autonomy` always; `vlm` always; `ros` always; `lab`
on a separate `schedule: cron: '0 7 * * *'` workflow plus an
optional `workflow_dispatch` for on-demand. Treat `lab` as informative,
not blocking, until cost/flakiness are characterized.

Ship Part 2 in a follow-up PR if the YAML touches too many
permission/secrets considerations to land in the same review. The
locked-scope deliverable is Part 1.

## Acceptance criteria

- [ ] `make test-autonomy` exists, runs on **both DGX and Jetson**
      with no venv-bootstrapping prerequisite beyond `pip install -e
      source/strafer_autonomy`, and exits 0 on a green tree.
- [ ] `make test-jetson` runs successfully on Jetson and covers
      autonomy + ROS-package tests.
- [ ] `make test-dgx` runs successfully on DGX and covers
      autonomy + VLM + lab tests (with `lab` gateable behind an
      env flag if the operator wants to skip the heavy Kit suite) — i.e. it is
      the single repo-wide DGX e2e command. Only two DGX test envs now:
      `.venv_vlm` (vlm) and `env_isaaclab3` (the lab umbrella).
- [ ] `make test-lab` (already shipped) is in the composite — it runs
      `run_tests.py all` (Kit) + `pytest tests/` (pure-Python) in
      `env_isaaclab3`. The former `make test-harness` is gone (folded in);
      `make test-lab-pure` is the fast pure-Python-only iteration target.
- [ ] `make help` lists the new targets with one-line descriptions;
      the deprecated `test-unit` either remains aliased or prints a
      one-line note pointing at `test-driver`.
- [ ] No regression on `make test-dgx`'s current behavior for
      operators who already use it — the externally observable
      output (suites run, PYTHONPATH handling, exit code) is the
      same modulo additionally running `lab` (which the env flag
      can suppress).
- [ ] A short note in [`example_commands_cheatsheet.md`](../../example_commands_cheatsheet.md)
      under a new **Testing** section documents the per-host
      umbrella commands.
- [ ] (Stretch) `.github/workflows/test.yml` runs `autonomy` on
      every PR. Wiring `ros` is a stretch within the stretch — land
      whichever subset is cleanly reviewable.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit.

## Investigation pointers

- `Makefile` lines 33–39, 120–128: today's `test`, `test-unit`,
  `test-dgx` definitions. `VENV_VLM` is the local var to thread.
- `source/strafer_lab/run_tests.py`: the bespoke wrapper. Treat as
  a black box invoked via `isaaclab -p`.
- `source/strafer_autonomy/pyproject.toml`: `requires_ros` marker
  is registered here.
- `source/strafer_ros/strafer_driver/test/`: candidate for the
  renamed `test-driver` target.
- Recent CI-adjacent precedent: there is no existing
  `.github/workflows/` directory; this brief introduces it. Check
  GitHub's per-repo Actions permissions before assuming the YAML
  will Just Work.

## Out of scope

- **Folding `run_tests.py` into pure pytest.** Isaac Sim's init
  constraints make this unproductive; keep the wrapper.
- **Adding tests that don't currently exist.** This brief is about
  invocation, not coverage.
- **Self-hosted DGX runner setup as a hard prerequisite.** The
  `lab` CI matrix entry is the only piece that needs it, and the
  brief explicitly lets that ship later (or never).
- **Migrating from colcon to plain pytest for ROS packages.** The
  colcon path is correct for packages that link ROS interfaces;
  this brief preserves it.
- **Linting/format target rework.** `make lint` / `make format` are
  out of scope. If a future tooling sweep wants to unify those too,
  file separately.
