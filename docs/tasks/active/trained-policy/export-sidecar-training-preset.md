# Sidecar `training_preset` records the class name, not the rsl_rl preset variable

**Type:** bug
**Owner:** DGX (`strafer_lab` lane)
**Priority:** P3 — cosmetic; the inference node doesn't gate on
`training_preset`. Worth fixing because the field is what an operator
reads to verify "did this artifact come from the right runner config";
right now it reads `RslRlOnPolicyRunnerCfg` (the class name common to
all variants) instead of `STRAFER_PPO_DEPTH_RUNNER_CFG` /
`STRAFER_PPO_RUNNER_CFG` (the per-variant preset variable). Filed off
[`export-onnx-depth`](export-onnx-depth.md)'s real-checkpoint smoke
test.
**Estimate:** S (~1–2 hours: read the registry entry instead of
`type(agent_cfg).__name__`, update existing test fixture's literal,
re-run a one-shot export to verify).
**Branch:** task/export-sidecar-training-preset

## Story

As an **operator inspecting a deployed policy artifact's sidecar**, I
want **`training_preset` to record the rsl_rl preset variable name
(e.g. `STRAFER_PPO_DEPTH_RUNNER_CFG`)**, so that **the field actually
disambiguates which runner config produced the checkpoint, instead of
recording the same class name for every variant**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [policy-export-tooling.md](../../completed/policy-export-tooling.md)
  — the brief that established the sidecar contract;
  `training_preset` is documented there as "which rsl_rl preset this
  checkpoint came from."

## Context

### Symptom

`Scripts/export_policy.py --variant DEPTH ...` writes a sidecar with:

```json
{
  ...
  "training_preset": "RslRlOnPolicyRunnerCfg",
  ...
}
```

The same field for `--variant NOCAM` also yields
`"RslRlOnPolicyRunnerCfg"` — both variants use the same parent class,
so the field carries no per-variant information.

### Root cause

[`Scripts/export_policy.py`](../../../../Scripts/export_policy.py)
`main()` computes:

```python
training_preset = type(agent_cfg).__name__
```

`type(agent_cfg).__name__` returns the configclass name. The intent
(per the original brief) was the *variable* name registered under
`rsl_rl_cfg_entry_point` — e.g.
`"strafer_lab.tasks.navigation.agents.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG"`,
whose tail (`STRAFER_PPO_DEPTH_RUNNER_CFG`) is the human-meaningful
preset identifier.

### Why the test missed it

[`source/strafer_lab/tests/test_export_policy.py`](../../../../source/strafer_lab/tests/test_export_policy.py)
`test_metadata_sidecar_records_documented_fields` passes the
preset name as a literal:

```python
training_preset="STRAFER_PPO_RUNNER_CFG",
```

so the *helper* (`write_metadata_sidecar`) is correct — the gap is in
`main()`'s call site, which never exercises the registry-lookup path
the test pretended to.

## Approach

In [`Scripts/export_policy.py`](../../../../Scripts/export_policy.py)
`main()`, replace:

```python
training_preset = type(agent_cfg).__name__
```

with a lookup that walks `gym.spec(args.env).kwargs["rsl_rl_cfg_entry_point"]`
or equivalent and extracts the variable name after the `:`. Fall back
to `type(agent_cfg).__name__` only when the registry returns no
entry-point string (defensive; should not happen for registered envs).

Add a test that exercises the new helper against a fake registry
entry shaped like the production strings.

## Acceptance criteria

- [ ] A fresh `Scripts/export_policy.py --variant NOCAM` writes
      `"training_preset": "STRAFER_PPO_RUNNER_CFG"` in the sidecar.
- [ ] `--variant DEPTH` writes
      `"training_preset": "STRAFER_PPO_DEPTH_RUNNER_CFG"`.
- [ ] `python -m pytest source/strafer_lab/tests/test_export_policy.py`
      passes — including a new test exercising the registry-derived
      preset name path (the existing test's literal stays as a
      consumer-side contract pin).
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`Scripts/export_policy.py`](../../../../Scripts/export_policy.py)
  `main()` — the call site that builds `training_preset`.
- [`source/strafer_lab/tests/test_export_policy.py`](../../../../source/strafer_lab/tests/test_export_policy.py)
  `test_metadata_sidecar_records_documented_fields` — the existing
  contract pin; keep, augment.
- `source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`
  (env registration) — where the `rsl_rl_cfg_entry_point` strings
  are wired.

## Out of scope

- **Backfilling old artifacts.** Already-shipped sidecars in
  `models/` (if any) stay as they are; this brief fixes the
  generator, not the historical record.
- **Sidecar schema changes.** Field name + value type stay
  identical; only the value changes from class-name to variable-name.
