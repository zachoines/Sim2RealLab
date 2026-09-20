# Make a failing Kit test suite exit non-zero

**Type:** task (test-infrastructure defect with a candidate one-line fix)
**Owner:** DGX
**Priority:** P1 — every Kit gate invoked as raw `pytest` reports success
regardless of failures, so any CI step, script or session that reads the exit
code has been reading a constant.
**Estimate:** S (one kwarg, plus the proof and a regression guard)
**Branch:** `task/kit-suite-exit-code-false-green`

## Story

As **anyone gating on the Isaac Sim test suites**, I want **a failing suite to
exit non-zero**, so that **a green exit code means the tests passed rather than
that the process reached its teardown.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [measurements/deploy-resolution-depth-2026-09-19](../../../measurements/deploy-resolution-depth-2026-09-19/README.md)
  — where the symptom was isolated and the fix arm run.

## Context

`source/strafer_lab/test_sim/conftest.py` ends the session with
`simulation_app.close()` and then `os._exit(exitstatus)`. The second call never
runs: `SimulationApp.close` takes `exit_code: int = 0`, and with
`/app/fastShutdown` enabled — the default, which Isaac Lab passes — Kit
terminates the process itself. Isaac Sim's own docstring names the case:
"Process exit status to preserve when fast shutdown terminates the process.
Nonzero values flush stdio and exit with the supplied status before Kit's
fast-shutdown path can replace it with 0."

Measured on `Isaac-Strafer` `test_sim/env/test_composition_contract.py` with one
golden deliberately broken, in a detached worktree:

| arm | conftest | junit failures | process exit |
|---|---|---|---|
| A | `simulation_app.close()` | 1 | **0** |
| B | `simulation_app.close(exit_code=exitstatus)` | 1 | **1** |

So the fix is one kwarg, and it is already proven on the failing path.

Two things are not the cause and should not be changed for it. The
`os._exit(exitstatus)` call passes pytest's own status and is correct; a
Kit-free mimic of the same teardown exits 1. And `tools/kit_boot_watchdog.sh`
propagates faithfully — it returns whatever the wrapped command returned,
including the false 0.

The truncated terminal summary is a **separate** effect of the same teardown and
survives the fix: the session's `pytest_sessionfinish` runs before the terminal
reporter's, so the failure lines and the count are never printed. Anyone reading
output rather than an exit code needs the junit XML either way.

`source/strafer_lab/run_tests.py` is unaffected and always has been: it decides
pass or fail from the junit XML (`failures = int(suite.get("failures", 0))`),
which is what [`#214`](https://github.com/zachoines/Sim2RealLab/pull/214)
hardened it to do. The exposure is every invocation that is *not* `run_tests.py`.

The symptom has been on record since 2026-08-23: the Isaac Lab stage-3 record
logged a run showing 22 failures followed by `### pytest exit=0`, and read that
exit code as evidence the bare invocation was healthy — listing "the invocation"
among the causes it excluded.

## Acceptance criteria

- [ ] `test_sim/conftest.py` passes the session status through the Kit shutdown
      so a failing suite exits non-zero. The candidate is
      `simulation_app.close(exit_code=exitstatus)`; if it regresses teardown on
      any suite, say so and take the alternative rather than reverting silently.
- [ ] Proven on the failing path, not only the passing one: a broken golden in a
      worktree must make the suite exit non-zero, and the same suite unbroken
      must still exit zero. Both arms deposited.
- [ ] Every `test_sim` suite still tears down cleanly through `run_tests.py`,
      with no new relaunches attributable to the change.
- [ ] A regression guard that fails if the exit code stops tracking the junit
      result, so this cannot silently return.
- [ ] The truncated terminal summary is either fixed alongside or recorded as a
      known separate effect, with the reason it is separate.
- [ ] The 2026-08-23 stage-3 record gets a dated amendment noting that the
      `exit=0` it read as healthy was this defect.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- `run_tests.py`, which reads the XML and is already correct.
- `tools/kit_boot_watchdog.sh`, which propagates correctly.
- Re-auditing past records whose gates ran through `run_tests.py`; their results
  were read from the XML.
