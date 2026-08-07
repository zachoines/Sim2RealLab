# Carry the trained step period on the policy artifact

**Status:** Shipped 2026-08-06 in `c7d282b` (Jetson). Both halves landed in one
PR — the DGX half did not need to wait for the provenance-manifest work.
**The `strafer_lab` suites covering that half are unrun:** neither torch nor
onnxruntime is present on the Jetson host or in `strafer-cpu:humble`, and both
test modules import torch at module scope, so
`tests/policy_tooling/test_export_policy.py` and `test_load_policy.py` need a
run on an Isaac Sim host before merge. In their place the same functions were
driven directly with torch stubbed out — sidecar round-trip at full float64
precision, the write-side validation, the loader's absent/valid/unusable
branches — and the CLI's period derivation checked by parsing `main`'s AST. The
Jetson half is covered by `make test-ros` (735 passed) and its three behavioural
cases were mutation-checked against the pre-change lookup.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/194

**Type:** task (train↔deploy contract)
**Owner:** Jetson + DGX
**Priority:** P1 — it is the prerequisite that makes a per-run cadence choice
safe. Nothing is broken while exactly one global cadence exists; the day that
stops being true, a mismatch is silent.
**Estimate:** S (one sidecar field, one loader accessor, one node preference +
a disagreement log)
**Branch:** `task/policy-artifact-cadence-contract`

## Story

As the **inference node loading an exported policy**, I want **the artifact to
state the step period it was trained at**, so that **a policy trained at a
different cadence than the deploy constant is caught at load instead of running
its recurrent state at the wrong rate in silence.**

## Context bundle

- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

The trained step period has exactly one home today,
[`strafer_shared/constants.py`](../../../source/strafer_shared/strafer_shared/constants.py):

```python
POLICY_SIM_DT = 1.0 / 120.0
POLICY_DECIMATION = 4
POLICY_PERIOD_S = POLICY_SIM_DT * POLICY_DECIMATION  # 1/30 s = 30 Hz
```

Its comment names it the single source of truth, and both consumers honour
that: the Isaac Lab task config takes `cfg.sim.dt` / `cfg.decimation` from it,
and the inference node derives its tick from it through
`_default_infer_period()`. `inference.yaml` deliberately omits `infer_period_s`
so the deploy rate cannot drift from training via a config literal.

**That protection holds only while there is exactly one global cadence.** It is
a *compile-time* coupling: the node reads whatever the constant says at the
moment it starts, not what the policy in its hands was trained against. A
checkpoint exported before a `POLICY_DECIMATION` change and deployed after it
runs at the new rate with no error raised.

**The artifact cannot currently say otherwise.** `write_metadata_sidecar` in
[`export_policy.py`](../../../source/strafer_lab/scripts/export_policy.py)
records `policy_variant`, `obs_dim`, `action_dim`, `env_id`, `training_preset`,
`source_checkpoint`, `formats`, `is_recurrent`, `git_commit`,
`export_timestamp`, `onnx_opset` and the TensorRT fields. There is no cadence,
period, or decimation field, and `policy_interface` never consults one.

**Why this is a recurrent-policy hazard specifically.** A GRU's dynamics are
indexed by step count, not wall time. A policy trained to integrate at 30
steps per second of world time and run at 20 has a different effective time
constant on every gate in the recurrent cell — with correct-looking
observations, a correct-looking action shape, and nothing to log. It is the
same class of silent failure as a skipped ONNX hidden-state write-back, which
[`recurrent-policy-contract.md`](../context/recurrent-policy-contract.md)
point 3 exists to prevent. Obs-dim mismatch is caught at load; cadence mismatch
is not.

**Why now.** The cadence setpoint is under a pre-registered decision rule
recorded in
[`depth-qos-reliable-flip`](depth-qos-reliable-flip.md): if the
sustained achievable rate lands below 20 Hz, the setpoint moves via
`POLICY_DECIMATION` and training matches. The measurement that supplies that
rate now belongs to
[`depth-receiver-host-capacity`](../active/reliability/depth-receiver-host-capacity.md),
after the QoS re-measure that was going to produce it turned out to be measuring
the wrong thing. The moment that branch is taken — or any per-run cadence
configurability lands on the training side — the constant stops being a
sufficient contract. This brief lands **before** that, not after.

The branch is not expected to open: the host already receives ~28.5 Hz sim with
the inference node stopped, above the rule's 27 Hz threshold. That is a reason
to land this cheaply and early, not a reason to skip it — the hazard is silent
when it does fire.

## Acceptance criteria

- [x] The export sidecar records the trained step period — `trained_period_s`,
      seconds of world time per policy step. Landed standalone rather than
      riding the provenance manifest. Taken off the env config the checkpoint
      was reconstructed against (`cfg.sim.dt * cfg.decimation`), **not** off
      `constants.py`, so an env whose cadence is set per-run records what it
      actually stepped at. A required keyword argument: an export that records
      no cadence is the artifact the field exists to stop producing.
- [x] `load_policy` surfaces it as `LoadedPolicy.trained_period_s` — the same
      shape as `is_recurrent` / `active_providers`, a class attribute defaulting
      to `None`.
- [x] The inference node prefers it in `_resolve_infer_period`, and warns with
      both periods, both Hz figures, and which one it took.
- [x] An artifact with no period recorded keeps today's behaviour exactly.
      Absence is normal and means "no opinion"; only a *present* value that is
      not a positive number of seconds raises, and that is a broken artifact
      rather than a legacy one.
- [x] The resolved period is what `_infer_period_s` holds, so the `cadence:`
      line's target, the shortfall percentage and `timer_deadline_missed` all
      follow it.
- [x] Tests. Jetson side: six cases in `TestTrainedPeriodPreference` — the
      artifact's period reaches the rclpy timer, the disagreement log names both
      values, an absent field takes the constant silently, an agreeing field is
      not a disagreement, no policy falls back to the parameter, and the cadence
      line's target follows the artifact. The three behavioural ones were
      mutation-checked against the pre-change lookup. They derive their
      off-cadence period from `_default_infer_period()` rather than a `30 Hz`
      literal, so they keep describing a disagreement after a setpoint move.
      DGX side: round-trip, write-side validation and required-argument cases in
      `test_export_policy.py`, and surfacing / absent / unusable cases for both
      formats in `test_load_policy.py` — **written but unrun here**, see the
      status stamp.
- [x] Docs swept: the `infer_period_s` comment in `inference.yaml`,
      `source/strafer_ros/README.md`, `source/strafer_lab/README.md`,
      `docs/example_commands_cheatsheet.md`, and a seventh pinned point in
      [`recurrent-policy-contract.md`](../context/recurrent-policy-contract.md)
      plus its in-code mirror.
- [x] No regression: `make test-ros` 735 passed / 11 skipped across all seven
      `strafer_ros` packages, against 729 before.

## Out of scope

- **Changing the setpoint.** This brief makes a change *safe*; it does not make
  one. `POLICY_DECIMATION` stays 4 and the deploy cadence stays 30 Hz.
- **Per-run cadence configurability on the training side.** Whether the period
  becomes a training-run parameter rather than a constant is a separate
  question; this brief only ensures the answer is recorded on the artifact
  whenever it is taken.
- **The hold / duplicate randomization work.** A different axis entirely —
  degradation *within* a cadence, not the cadence itself.
- The two-way obs/action contract, which `policy_interface` already pins.

## Investigation pointers

- `write_metadata_sidecar` / `read_metadata_sidecar` in
  [`export_policy.py`](../../../source/strafer_lab/scripts/export_policy.py)
  — both importable, so the sidecar shape has one owner.
- `_default_infer_period()` in
  [`inference_node.py`](../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py)
  is indirected through a function specifically so tests can patch the
  constants; the artifact preference belongs at the same seam.
- The `infer_period_s` comment at the top of
  [`inference.yaml`](../../../source/strafer_ros/strafer_inference/config/inference.yaml)
  states the current contract and is what this brief revises.
