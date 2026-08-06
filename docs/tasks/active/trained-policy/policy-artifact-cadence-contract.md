# Carry the trained step period on the policy artifact

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

- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

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
[`recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
point 3 exists to prevent. Obs-dim mismatch is caught at load; cadence mismatch
is not.

**Why now.** The cadence setpoint is under a pre-registered decision rule in
[`depth-qos-reliable-flip`](../reliability/depth-qos-reliable-flip.md): if the
post-flip rig re-measure lands below 20 Hz sustained, the setpoint moves via
`POLICY_DECIMATION` and training matches. The moment that branch is taken —
or any per-run cadence configurability lands on the training side — the
constant stops being a sufficient contract. This brief lands **before** that,
not after.

## Acceptance criteria

- [ ] The export sidecar records the trained step period. Name and units are
      the implementer's call, but the value must be recoverable without
      re-deriving it from `constants.py` — that is the coupling being removed.
      **DGX half** (`export_policy.py`); it can ride the provenance-manifest
      work rather than landing standalone.
- [ ] `load_policy` surfaces the recorded period on the loaded artifact, in the
      same shape as the other sidecar-derived metadata it already exposes.
- [ ] The inference node **prefers the artifact's value** over
      `_default_infer_period()` when the artifact carries one, and logs loudly
      on disagreement — naming both values and which one it took. **Jetson
      half.**
- [ ] An artifact with **no** period recorded keeps today's behaviour exactly:
      derive from the constant, no warning. Every exported policy predates this
      field, so a hard failure would brick the deploy path on legacy artifacts.
- [ ] Whatever the node ends up ticking at is what the `cadence:` line reports
      as its target, so the achieved-vs-target figure and the sub-90% shortfall
      warning stay meaningful after a setpoint change.
- [ ] Tests: sidecar round-trip on the DGX side; on the Jetson side, a node
      built against an artifact whose period disagrees with the constant takes
      the artifact's and logs, and one with the field absent takes the
      constant silently.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

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
