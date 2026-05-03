# Policy export tooling: PPO checkpoint → deployable `.pt` / `.onnx` artifact

**Type:** task / feature
**Owner:** DGX (`strafer_lab` lane — training + export tooling)
**Priority:** P1 — hard dependency for
[`strafer-inference-package.md`](strafer-inference-package.md)
end-to-end validation.
**Estimate:** M (~2–3 days; tooling + tests + bench against the
existing `load_policy()` round-trip)
**Branch:** task/policy-export-tooling

## Story

As an **operator promoting a trained PPO checkpoint to robot
deployment**, I want **a single export script that converts an
rsl_rl checkpoint into a TorchScript `.pt` (later ONNX `.onnx`)
artifact loadable by
[`strafer_shared.policy_interface.load_policy()`](../../../source/strafer_shared/strafer_shared/policy_interface.py)**,
so that **the Jetson inference node consumes the exact policy
contract the env trained against without per-deployment ad-hoc
conversion code, and the inference node's deterministic-output
contract is anchored at export time rather than discovered on
robot**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [strafer-inference-package.md](strafer-inference-package.md) — the
  Jetson-side consumer that gates on this brief. The brief's Phase 3
  asserts byte-identical action outputs across two same-obs calls;
  that assertion is only meaningful if the export tooling actually
  freezes a deterministic head, which is this brief's job.

## Context

### What already exists

- [`strafer_shared.policy_interface.load_policy(path, variant)`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
  supports both `.pt` (TorchScript via `torch.jit.load`) and `.onnx`
  (ONNX Runtime). Returns a `(obs) → action` callable. The loading
  side is *done*; this brief produces artifacts that load cleanly.
- [`benchmark_policy(policy, variant, n_iters)`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
  measures inference latency. The right validation tool for this brief.
- Training entry point:
  [`Scripts/train_strafer_navigation.py`](../../../Scripts/train_strafer_navigation.py)
  produces rsl_rl checkpoints under
  `logs/rsl_rl/strafer_navigation/<run>/best_model/model_*.pt`.

### What's missing

`Scripts/export_policy.py` doesn't exist. Today there's no defined
path from a converged training checkpoint to a deployable artifact.

### Why deterministic export is load-bearing

PPO trained via rsl_rl produces a Gaussian actor: `action ~ N(μ, σ)`.
At training time the σ enables exploration. At deployment time we
want the **mean only** — same obs in, same action out, every time.

If the export ships the stochastic head intact, two consecutive
inference calls with the same observation produce *different* actions,
and the strafer-inference Phase 3 determinism test fails. Worse, on
robot the policy makes non-reproducible decisions and the silently-flaky
behavior is hard to bisect.

This brief is where the deterministic-head freeze happens. The
inference node assumes it's done and asserts it; this brief is the
producer of that contract.

## Approach

### Phase 1 — TorchScript export (primary path)

Create `Scripts/export_policy.py`:

```
python Scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/<run>/best_model/model_<step>.pt \
    --output models/strafer_navigation_v0.pt \
    --variant NOCAM
```

Implementation:

1. Load the rsl_rl checkpoint (extract `policy.actor_critic.actor` —
   the actor network).
2. Wrap in a deterministic-mean module:
   - rsl_rl's actor returns `(mean, std)` or distribution; export
     wraps it as `lambda obs: actor.mean(obs)` only.
   - No sampling. No `torch.distributions` calls. The traced /
     scripted graph contains only the deterministic forward path.
3. Export via `torch.jit.script` (preferred — preserves control flow)
   or `torch.jit.trace` (fallback if the actor module isn't
   script-compatible).
4. Round-trip verify:
   - Reload via `strafer_shared.policy_interface.load_policy()`.
   - Sample a synthetic observation vector (uniform random, shape
     `(variant.obs_dim,)`).
   - Call the loaded policy twice with the same observation. Assert
     byte-identical output. **Hard fail** if not — the export is bad.
5. Write a `<output>.json` metadata sidecar with:
   - `policy_variant` (string, e.g. `"NOCAM"`)
   - `obs_dim` (int)
   - `action_dim` (int, expected 3)
   - `env_id` (training task name)
   - `training_preset` (which rsl_rl preset this checkpoint came from)
   - `source_checkpoint` (path)
   - `git_commit` (the repo SHA at export time)
   - `export_timestamp` (ISO 8601)

Tests in `source/strafer_lab/tests/test_export_policy.py`:
- Export a tiny dummy actor (built in the test, not loaded from disk),
  load via `load_policy()`, assert output dimensions match
  `variant.obs_dim → 3`, assert determinism.
- Assert `<output>.json` exists alongside `<output>.pt` with the
  documented fields.
- Assert metadata `obs_dim` matches `PolicyVariant.<variant>.obs_dim`
  (catches a checkpoint-variant mismatch at export time, not at
  Jetson-load time).

### Phase 2 — ONNX export (defer to DEPTH-variant brief)

ONNX export is the path TensorRT consumes. NOCAM is a 19-dim MLP and
runs sub-millisecond on CPU per
[`strafer-inference-package.md`](strafer-inference-package.md);
adding ONNX support to `export_policy.py` is dead weight until the
DEPTH variant brief actually uses it. **Out of scope here.**

When the DEPTH brief picks this up, the ONNX path is `torch.onnx.export`
on the same deterministic-mean wrapper. Same metadata sidecar,
different `--format onnx` flag.

### Phase 3 — Benchmark hook

`Scripts/benchmark_policy.py`:

```
python Scripts/benchmark_policy.py --model models/strafer_navigation_v0.pt --iters 1000
```

Thin wrapper: load via `load_policy()`, run `benchmark_policy()`, print
median / p95 / p99 latency. Used by the Jetson side to verify
deployment-time inference latency after `rsync`-ing the artifact, and
by this brief's CI to catch latency regressions in the export
toolchain.

## Acceptance criteria

### Build / structure

- [ ] `python Scripts/export_policy.py --help` documents the CLI:
      `--checkpoint`, `--output`, `--variant`.
- [ ] `python -m pytest source/strafer_lab/tests/test_export_policy.py`
      passes — exports a tiny dummy actor, round-trips through
      `load_policy()`, asserts dimensions + determinism + metadata.

### Determinism contract (load-bearing for [`strafer-inference-package.md`](strafer-inference-package.md))

- [ ] Exported `.pt` produces byte-identical output for the same input
      across two consecutive `load_policy()` calls. The export script
      asserts this *before* writing the artifact — fails fast if the
      stochastic head wasn't frozen properly.
- [ ] Round-trip dimension check: loaded policy on a randomly-sampled
      `(variant.obs_dim,)` observation returns `(action_dim,)` with
      `action_dim == 3`.

### Metadata sidecar

- [ ] `<output>.json` exists alongside `<output>.pt` with all fields
      listed in Phase 1. Variant + obs_dim are validated against
      `PolicyVariant.<variant>` at export time.
- [ ] Loading code on the Jetson side
      ([`strafer-inference-package.md`](strafer-inference-package.md))
      reads the sidecar and asserts it matches the configured
      `PolicyVariant`. Mismatch is fatal at startup. *(Cross-brief
      invariant — owned by the inference brief but enforced here at
      export time.)*

### Performance

- [ ] `benchmark_policy()` on the exported NOCAM `.pt` returns median
      latency < 1 ms on Jetson Orin Nano (CPU, no CUDA), measured via
      `Scripts/benchmark_policy.py --iters 1000`.

### Sim-side smoke test

- [ ] Run `Scripts/play_strafer_navigation.py` with
      `--policy models/<exported>.pt`. Verify the rollout looks sane
      (robot navigates toward goal, no NaN actions, no diverging
      trajectories). This is a "did the export break the policy"
      sanity check, not a quantitative regression test.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
  — `load_policy` (lines 155–202), `benchmark_policy`
  (lines 211+). The export must produce artifacts these consume
  without modification.
- rsl_rl source (installed via pip): the
  `ActorCritic.act_inference()` method is typically the deterministic-
  mean entry point; that's what should get traced/scripted.
- [`Scripts/train_strafer_navigation.py`](../../../Scripts/train_strafer_navigation.py)
  — training entry point; checkpoint paths configured here.
- [`Scripts/play_strafer_navigation.py`](../../../Scripts/play_strafer_navigation.py)
  — closest existing example of loading and running a trained policy
  on the env. Useful reference for the smoke test.
- The
  [`docs/archive/SIM_TO_REAL_PLAN.md`](https://github.com/zachoines/Sim2RealLab/blob/695e8c0a/docs/archive/SIM_TO_REAL_PLAN.md)
  archived plan (pre-deletion at commit `695e8c0a`) sketched the
  TorchScript-first / ONNX-later sequence this brief implements.

## Out of scope

- **Producing a converged checkpoint.** Training is ongoing, separate
  operator-side work. This brief takes whatever checkpoint is provided
  as input and converts it.
- **TensorRT engine generation.** Lives in the future DEPTH-variant
  brief per
  [`strafer-inference-package.md`](strafer-inference-package.md).
- **ONNX export.** Same — the DEPTH brief picks it up when it actually
  needs it; adding ONNX here without a consumer is YAGNI.
- **Goal-position noise training.** That's
  [`policy-goal-noise-training.md`](policy-goal-noise-training.md).
  Orthogonal — happens *before* export, produces the checkpoint this
  brief consumes.
- **Model registry / Databricks Model Serving.** Separate operator-side
  deployment work; not blocking for the LAN-HTTP / rsync path used by
  this brief.
- **Inference-side integration.** That's
  [`strafer-inference-package.md`](strafer-inference-package.md). This
  brief's output is rsync'd onto the Jetson by the operator.
