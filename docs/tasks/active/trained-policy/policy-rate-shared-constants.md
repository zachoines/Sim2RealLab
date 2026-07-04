# Delegate `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` (+ the cmd-vel watchdog window) to `strafer_shared.constants`

**Type:** task / refactor
**Owner:** DGX (touches `strafer_lab/`; plus one Jetson one-liner in
`strafer_driver` for the watchdog item below)
**Priority:** P2 — sim/real drift hazard, but the values already happen
to agree numerically. The risk is the next time someone changes either
side.
**Estimate:** S (~1 hr — two files, a handful of lines, two tests)
**Branch:** task/policy-rate-shared-constants

## Story

As a **trained-policy backend developer (Jetson) and Isaac Lab env
maintainer (DGX)**, I want **the policy step rate to live in one
place — `strafer_shared.constants` — and for both the gym env config
and the Jetson inference node to read from there**, so that **the
30 Hz contract the policy was trained on cannot silently diverge
between sim and real**.

## Context

The Jetson-side trained-policy backend (the
[`inference-package`](../../completed/inference-package.md) brief, Phase 2) needs the
policy step period to drive its inference loop at the same rate the
policy was trained on. The brief asks for the value to be **derived,
not hardcoded**, by promoting the two underlying constants to
`strafer_shared.constants` and consuming them on both the sim and real
sides.

That promotion has already shipped — additive only — on the
`task/strafer-inference-package` branch:

```python
# source/strafer_shared/strafer_shared/constants.py
POLICY_SIM_DT = 1.0 / 120.0
POLICY_DECIMATION = 4
POLICY_PERIOD_S = POLICY_SIM_DT * POLICY_DECIMATION  # 1/30 s = 30 Hz
```

The Jetson side already consumes these. The DGX side still carries the
originals:

```python
# source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py:1132-1134
_DEFAULT_NAV_SIM_DT = 1.0 / 120.0
_DEFAULT_NAV_RENDER_INTERVAL = 4
_DEFAULT_NAV_DECIMATION = 4
```

Today the values agree numerically. Tomorrow, when training experiments
want to try a 15 Hz or 60 Hz variant, the natural impulse is to edit
the env config — and the inference node would silently keep running at
the old 30 Hz. That's the kind of sim/real drift the whole
`strafer_shared` module exists to prevent.

## Approach

One file, three lines:

```python
# source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py
from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

_DEFAULT_NAV_SIM_DT = POLICY_SIM_DT
_DEFAULT_NAV_DECIMATION = POLICY_DECIMATION
# _DEFAULT_NAV_RENDER_INTERVAL stays as-is; it's a renderer tuning knob
# (how often the sim emits camera frames), not the policy step rate.
```

That preserves the existing `_apply_default_nav_runtime` wiring and
keeps every env using the same numerical values — but the source of
truth is now the shared module, so changing `POLICY_SIM_DT` once
re-propagates to every consumer.

The Jetson side already has the matching unit test
(`test_infer_period_derived_from_shared_constants` in
`source/strafer_ros/strafer_inference/test/test_inference_config.py`)
that mock-patches the shared constants and asserts the inference node
picks up the new period. After this refactor lands, an analogous DGX-
side test that asserts `_DEFAULT_NAV_SIM_DT is POLICY_SIM_DT` (identity
check on module-level binding) anchors the delegation so a future PR
can't accidentally re-introduce a literal.

## Also in scope (added 2026-07-03): shared cmd-vel watchdog window

Same drift hazard, second instance — created by PR #134. The stop-on-silence
command watchdog now exists in **two mirrored literals**:

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`
  `WATCHDOG_TIMEOUT_SEC = 0.5` (real robot — motors zeroed after 0.5
  **wall**-seconds of `/cmd_vel` silence), and
- `source/strafer_lab/strafer_lab/bridge/config.py`
  `BridgeConfig.cmd_watchdog_sim_s = 0.5` (sim bridge — held action zeroed
  after 0.5 **sim**-seconds of silence; PR #134).

Parity between them is currently **by convention** (cross-referencing
comments). Make it structural: add one shared constant

```python
# source/strafer_shared/strafer_shared/constants.py
CMD_WATCHDOG_TIMEOUT_S = 0.5
```

and have both sides default from it. The constant's comment must state the
clock-domain contract, because it is the whole point: the window is the
**stream-relative** silence budget, denominated in each side's own clock
domain — wall on the real robot, sim time in the bridge (both the policy
tick and the physics live in that domain per side). That is what keeps the
invariant identical on both paths: the watchdog trips after the same ~15
missed 30 Hz policy ticks, and the robot coasts the same in-world distance
on a stale command before halting. It is deliberately **not** "the same
wall-clock duration" — a wall window in the bridge would false-trip between
healthy commands at sub-unity RTF (see the PR #134 rationale in
`bridge/cmd_watchdog.py`).

## Acceptance criteria

- [ ] `_DEFAULT_NAV_SIM_DT` and `_DEFAULT_NAV_DECIMATION` are
      re-exports of `strafer_shared.constants.POLICY_SIM_DT` /
      `POLICY_DECIMATION`, not literals.
- [ ] `strafer_shared.constants.CMD_WATCHDOG_TIMEOUT_S` exists with the
      clock-domain contract documented; `roboclaw_node.WATCHDOG_TIMEOUT_SEC`
      and `BridgeConfig.cmd_watchdog_sim_s` both default from it (no
      mirrored literals remain — grep for `0.5` near both sites).
- [ ] A test on each side anchors the delegation (driver: equality/identity
      against the shared constant in the `strafer_driver` suite; bridge: the
      pxr-free autonomy suite asserts `BridgeConfig().cmd_watchdog_sim_s ==
      CMD_WATCHDOG_TIMEOUT_S` — `bridge/config.py` already imports from
      `strafer_shared`, so no new dependency).
- [ ] A unit test in `source/strafer_lab/tests/` mock-patches
      `strafer_shared.constants.POLICY_SIM_DT` and asserts the env-cfg
      module's resolved sim dt follows. (Or asserts the identity
      binding at import time — either form is acceptable as long as a
      future literal in `strafer_env_cfg.py` fails the test.)
- [ ] Existing `colcon test --packages-select strafer_lab` and any
      env-build smoke test pass unchanged — values agree numerically
      today, so behavior is identical.
- [ ] `_DEFAULT_NAV_RENDER_INTERVAL` is intentionally left as-is and
      that intent is documented inline (one line).
- [ ] If your work invalidates a fact in any referenced context module
      or in `strafer_shared/constants.py`'s "Policy step rate" comment
      header, update it in the same commit.

## Out of scope

- **Adding any new policy-rate variants** (15 Hz, 60 Hz, etc.). This
  brief is pure source-of-truth consolidation; no behavior change.
- **`_DEFAULT_NAV_RENDER_INTERVAL`** — renderer-tuning knob, not part
  of the policy step contract. Leave it where it lives.
- **Jetson-side changes beyond the watchdog one-liner.** The inference
  node already reads from the shared constants; the only Jetson touch is
  `roboclaw_node.WATCHDOG_TIMEOUT_SEC` defaulting from
  `CMD_WATCHDOG_TIMEOUT_S` (added scope above).
- **Retuning the watchdog value.** 0.5 s stays; this is source-of-truth
  consolidation only.
