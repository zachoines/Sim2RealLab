# Delegate `_DEFAULT_NAV_SIM_DT` / `_DEFAULT_NAV_DECIMATION` to `strafer_shared.constants`

**Type:** task / refactor
**Owner:** DGX (touches `strafer_lab/`)
**Priority:** P2 — sim/real drift hazard, but the values already happen
to agree numerically. The risk is the next time someone changes either
side.
**Estimate:** S (~1 hr — one file, three lines, one test)
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
[`inference-package`](inference-package.md) brief, Phase 2) needs the
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

## Acceptance criteria

- [ ] `_DEFAULT_NAV_SIM_DT` and `_DEFAULT_NAV_DECIMATION` are
      re-exports of `strafer_shared.constants.POLICY_SIM_DT` /
      `POLICY_DECIMATION`, not literals.
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
- **Jetson-side changes.** The inference node already reads from the
  shared constants; nothing further on that side is needed.
