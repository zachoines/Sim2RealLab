# Unify the strafer_lab test tree layout (test/ vs tests/, and the under-organized tests/ root)

**Type:** task / tooling (test-tree reorganization)
**Owner:** DGX agent (lane: `source/strafer_lab/test/`, `source/strafer_lab/tests/`, `source/strafer_lab/run_tests.py`, top-level `Makefile`)
**Priority:** P3 — tooling polish; doesn't block features. Bumps to P2 if a test file gets lost / silently un-run a second time (one such gap already exists — see Context).
**Estimate:** S–M (~1 day: a mechanical move + import/conftest fixups + runner/Makefile wiring; the taxonomy call is the only judgement part).
**Branch:** task/strafer-lab-test-tree-unification

## Story

As **an agent adding or running a strafer_lab test**, I want **one
test tree whose subfolders say what kind of test lives there (and
which interpreter runs it)**, so that **I stop tripping over two
near-identically-named folders (`test/` and `tests/`), every test file
is actually run by some `make` target, and a new test has one obvious
home.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/conventions.md`](../../context/conventions.md)
- **Sibling brief — coordinate, do not overlap:**
  [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md). That brief
  unifies **test *invocation*** (`make test-*` wrappers + CI) and treats
  `run_tests.py` as a black box; it explicitly excludes coverage/layout.
  This brief unifies the **physical *layout*** the wrappers point at. They
  meet at the Makefile: this brief should land **first** (or they
  coordinate), so `unify-test-targets-and-ci`'s `make test-lab` /
  harness paths reference the final directory names. If both are in
  flight, the layout names here are the contract the invocation brief
  consumes.

## The problem (measured)

`source/strafer_lab/` has two sibling test directories whose names
differ only by an `s`:

| Tree | What it actually contains | Interpreter | Run by |
|---|---|---|---|
| `test/` (singular) | **Isaac-Sim-required** suites — env / sensors / rewards / terminations / actions / observations / curriculums / noise_models / depth_noise / imu. Each boots Kit (`AppLauncher`, `ManagerBasedRLEnv`, the `SimulationContext` singleton). | `$ISAACLAB -p` (Kit) | `run_tests.py` (bespoke subprocess + JUnit-XML wrapper, because Isaac Sim's `os._exit` kills pytest's summary) |
| `tests/` (plural) | **Pure-Python** tests — no Kit. Its own `conftest.py` says so: *"tests in this tree exercise pure-Python helpers ... intentionally isolated from [the sim] conftest."* | `.venv_harness` (or any torch venv) | `make test-harness` — **but only the `tests/harness/` subdir** |

Two concrete defects, not just an eyesore:

1. **`test/` vs `tests/` is a genuine foot-gun.** The names encode a
   real and important distinction (needs-Kit vs pure-Python), but the
   *naming* communicates none of it — they look like a typo of each
   other. New tests land in the wrong one; readers can't tell which is
   which without opening files.

2. **Five `tests/` root files are run by NO `make` target.** `make
   test-harness` points only at `tests/harness/`, and `run_tests.py`
   only knows the `test/` (singular) suites. So these pure-Python tests
   currently run in **neither** runner:
   - `tests/test_action_clamp.py` — L1 body-velocity clamp vs the
     deployment `torch.clamp` contract (the Jetson/sim mecanum boundary)
   - `tests/test_export_policy.py` — `Scripts/export_policy.py` TorchScript/ONNX round-trip
   - `tests/test_load_policy.py` — stateless + recurrent `.pt`/`.onnx` loader round-trip
   - `tests/test_obs_contract_parity.py` — encoder-FK observation parity (the sim→real obs contract; imports `warp` but does not boot Kit)
   - `tests/test_recurrent_contract_e2e.py` — recurrent hidden-state contract across the export pipeline

   These are exactly the **sim-to-real boundary contracts** — the
   highest-value tests to *not* silently stop running.

## The taxonomy (proposal — the one judgement call)

The deciding axis is **does it need Isaac Sim / Kit**, because that
fixes the interpreter and the runner. Within the pure-Python tree,
group by *what the test guards*. Proposed shape:

```
source/strafer_lab/
  test_sim/                 # RENAMED from test/  — needs Kit; run via run_tests.py under $ISAACLAB
    env/ sensors/ rewards/ terminations/ actions/ observations/
    curriculums/ noise_models/ ...                (unchanged contents)
  tests/                    # pure-Python; no Kit; run in .venv_harness
    harness/                # writer / capture / postprocess / picker / profiler  (exists)
    policy_tooling/         # test_export_policy.py, test_load_policy.py          (NEW)
    contracts/              # test_action_clamp.py, test_obs_contract_parity.py,  (NEW)
                            #   test_recurrent_contract_e2e.py
                            #   — the sim↔real boundary guards (action/obs/recurrent)
```

Naming is the operator's call; the **invariant** is:
- one obviously-Kit tree whose name says "sim" (the singular `test/` →
  `test_sim/` so it can never again read as a typo of `tests/`), and
- the pure-Python `tests/` tree fully foldered by intent, with **no
  loose files at its root** and **every subdir run by a `make` target**.

`policy_tooling` vs `contracts` split (per the operator's instinct):
**export/load plumbing** (does the `.pt`/`.onnx` round-trip work) is a
different concern from a **deploy-boundary contract** (does the sim
emit the byte-identical obs/action the real robot's signal chain
expects). The action-clamp, obs-parity, and recurrent-contract tests
are the latter — they fail when the sim↔real seam drifts, regardless of
export mechanics — so they get their own `contracts/` home.

## Acceptance

- [ ] The singular `test/` tree is renamed to a name that unambiguously
      reads as the **Isaac-Sim** tree (e.g. `test_sim/`); `run_tests.py`'s
      `TEST_ROOT` + suite paths updated to match; `isaaclab -p
      run_tests.py all` still green.
- [ ] The five loose files at `tests/` root are moved into intent-named
      subdirs (`policy_tooling/`, `contracts/` or the operator's chosen
      names); `tests/` root has **no** stray `test_*.py`.
- [ ] Every pure-Python subdir under `tests/` is run by a `make` target.
      Minimum: extend the harness target (or add a sibling) so
      `policy_tooling/` + `contracts/` are no longer orphaned — the five
      currently-unrun files execute and pass. (Coordinate the exact
      target name with [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md).)
- [ ] Imports / `conftest.py` / `pyproject` test-path globs updated so
      no test breaks on the move; `git mv` preserves history.
- [ ] `make test-harness` (or its renamed successor) stays green at its
      prior count **plus** the five newly-included files.
- [ ] If the move invalidates a path in any context module, README,
      `run_tests.py` docstring, or `docs/` guide, update it in the same
      commit.

## Out of scope

- **Test *invocation* unification + CI** — owned by
  [`unify-test-targets-and-ci`](unify-test-targets-and-ci.md). This brief
  only moves files and ensures each tree is *reachable*; the polished
  `make test-<host>` umbrellas + GitHub Actions are that brief's.
- **Folding `run_tests.py` into plain pytest.** Isaac Sim's `os._exit` /
  init-flood constraints make the wrapper intentional; keep it (same
  call as the sibling brief).
- **Adding new test coverage.** Pure reorganization; do not write new
  tests (beyond wiring the orphaned ones in).
- **The `strafer_ros/*/test/` or `strafer_autonomy/tests/` trees.**
  Other packages, other lanes; this brief is `strafer_lab`-only.

## Triggered by

Observed while validating the `teleop-perf-architecture` cleanup: the
`test/` vs `tests/` split is an eyesore, and inspection showed the
`tests/` root additionally carries policy-export/load and
action/obs/recurrent contract tests that no `make` target runs. Filed
to give each test kind one obvious, reachable home.
