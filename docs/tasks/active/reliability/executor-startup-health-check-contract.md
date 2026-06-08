# Executor startup health-check: docs say advisory, code fails fast

**Type:** task / reliability (executor startup contract + docs)
**Owner:** Jetson (`source/strafer_autonomy/strafer_autonomy/executor/`)
**Priority:** P3 (correctness-of-docs now; the skip-flag half is an ergonomics call)
**Estimate:** S (doc reconcile is minutes; the optional advisory-mode flag is ~half day)
**Branch:** `task/executor-startup-health-check-contract`

## Context bundle

- [context/ownership-boundaries.md](../../context/ownership-boundaries.md) — executor is Jetson lane.
- [context/conventions.md](../../context/conventions.md)
- `source/strafer_autonomy/strafer_autonomy/executor/command_server.py` (`build_command_server`, the `future.result(timeout=10.0)` call ~L313-315 + its docstring ~L294-297) and `executor/main.py` (~L237, the unguarded `build_command_server` call).

## The problem

Filed off the Jetson-side audit for [`install-docs-consolidation`](../tooling/install-docs-consolidation.md) (2026-06-08). The README and the code disagree on what the executor does when the VLM/planner service is **unreachable** at startup:

- `source/strafer_autonomy/README.md` L123: "runs parallel health checks … and **logs a warning if unreachable**." L256 implies only the `model_loaded=false` case aborts.
- Reality (verified both directions on `jetson-desktop`): an unreachable service raises `GroundingServiceUnavailable` / `PlannerServiceUnavailable` out of `build_command_server`, which is **not** wrapped in a try/except in `main.py`, so the executor **exits 1** and `/execute_mission` never registers. The code's own docstring already says *"Unreachable services propagate their original exception."* A reachable-but-`model_loaded=false` service also fails fast.

So: the **code is internally consistent (fail-fast on unreachable or unloaded)**; the **README overstates tolerance**. There is also no operator-facing env/launch flag to make the check advisory — only the internal `check_vlm_health=False` kwarg used by tests.

## Decision + acceptance

Decide and implement one of:

- **A — fail-fast is the contract (doc-only).** Update `README.md` L123/L256 (and `strafer_ros/README.md` L225 "services are advisory…") to state the executor requires both services reachable **and** `model_loaded=true` at startup, and exits non-zero otherwise. Smallest change; matches today's behavior.
- **B — add an advisory mode.** Add an operator-facing flag/env (e.g. `STRAFER_REQUIRE_SERVICES=0`) that downgrades unreachable/unloaded to a logged warning and lets `/execute_mission` register anyway (missions then fail cleanly per skill). Update the same docs to describe both modes.

Acceptance (whichever path):

- [ ] `strafer_autonomy/README.md` L123/L256 and `strafer_ros/README.md` L225 no longer claim "advisory / warning" when the behavior is fail-fast.
- [ ] If B is chosen, the flag is documented in the README Run section and has a test for both modes.
- [ ] No change to the existing fail-fast default unless B is explicitly adopted.

## Out of scope

- The VLM/planner services themselves (DGX lane).
- Reconnect/retry-after-startup behavior — that's a larger reliability item, file separately if wanted.

## Triggered by

Jetson-side audit during `install-docs-consolidation` (2026-06-08): the documented `make launch-autonomy` smoke test does not pass without reachable DGX services — the executor crashes at startup rather than warning, contradicting the README.
