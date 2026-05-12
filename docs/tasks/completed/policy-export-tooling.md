# Policy export tooling: PPO checkpoint → deployable `.pt` + `.onnx` artifact

**Status:** Shipped 2026-05-04 in `545aefc` (DGX). NoCam end-to-end
(TorchScript + ONNX) works through `Scripts/export_policy.py`; the
deterministic-mean freeze, sidecar contract, and round-trip determinism
guard are anchored at export time. DEPTH ONNX is a known MVP gap:
`StraferDepthRNNModel.as_onnx` still raises `NotImplementedError`, so
the Jetson TRT-EP path for the DEPTH variant is blocked on the
follow-up brief.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/17
**Follow-ups:**
- [`policy-export-onnx-depth.md`](../active/trained-policy/export-onnx-depth.md)
  — closes the MVP-DEPTH-ONNX gap by authoring `_OnnxDepthGRUModel` in
  `source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`.
- [`policy-loader-recurrent-state.md`](../active/trained-policy/loader-recurrent-state.md)
  — extends `strafer_shared.policy_interface.load_policy()` so recurrent
  artifacts (DEPTH GRU today, NoCam LSTM in the future) expose
  `.reset()` and ONNX hidden state threads across calls. Required for
  the Jetson inference node to drive stateful exports deterministically.

**Type:** task / feature
**Owner:** DGX (`strafer_lab` lane — training + export tooling)
**Priority:** P1 — hard dependency for
[`strafer-inference-package.md`](../active/trained-policy/inference-package.md)
end-to-end validation. **Both TorchScript and ONNX paths are
MVP-required** since the DEPTH variant in the inference brief is
too slow on CPU and depends on the TensorRT execution provider
(consumed via the ONNX path).
**Estimate:** M–L (~3–5 days: TorchScript + ONNX export + TRT-EP
verification on the Jetson + tests + bench against the existing
`load_policy()` round-trip).
**Branch:** task/policy-export-tooling

## Story

As an **operator promoting a trained PPO checkpoint to robot
deployment**, I want **a single export script that converts an
rsl_rl checkpoint into both a TorchScript `.pt` and an ONNX
`.onnx` artifact loadable by
[`strafer_shared.policy_interface.load_policy()`](../../../source/strafer_shared/strafer_shared/policy_interface.py)**,
so that **the Jetson inference node consumes the exact policy
contract the env trained against — through the TensorRT execution
provider for the DEPTH MVP — without per-deployment ad-hoc
conversion code, and the inference node's deterministic-output
contract is anchored at export time rather than discovered on
robot**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [strafer-inference-package.md](../active/trained-policy/inference-package.md) — the
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

### Phase 2 — ONNX export (MVP-required for DEPTH)

The DEPTH variant in
[`strafer-inference-package.md`](../active/trained-policy/inference-package.md) is
too slow on CPU/CUDA-EP alone — the TRT execution provider is
required for the latency target. ONNX is the path TensorRT
consumes, so this phase is part of the MVP, not deferred.

CLI extension:

```
python Scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/<run>/best_model/model_<step>.pt \
    --output models/strafer_depth_v0 \
    --variant DEPTH \
    --formats pt,onnx
```

Implementation (on top of Phase 1's deterministic-mean wrapper):

1. After the TorchScript export succeeds, also run `torch.onnx.export`
   on the same wrapper module.
2. Use opset 17+ (Jetson's onnxruntime-gpu wheel supports it).
3. Round-trip verify via `load_policy(<output>.onnx, variant)` —
   same byte-identical-determinism check as the TorchScript path.
4. The metadata sidecar gains:
   - `formats` (list, e.g. `["pt", "onnx"]`)
   - `onnx_opset` (int)
   - `tensorrt_engine_path` (optional, set if a pre-built engine
     was generated alongside; otherwise omitted)

Pre-built TRT engine (optional, but useful for production):
- ONNX Runtime's TRT EP can build engines at first inference, but
  that adds ~30 s to first-call latency on Jetson. Pre-building
  with `onnxruntime` or NVIDIA's `trtexec` and shipping the
  `.engine` alongside avoids the cold-start cost.
- Pre-build path uses the same JetPack version's TRT runtime — so
  the engine is JetPack-version-pinned. Document that pinning in
  the metadata (`tensorrt_version` field).

Tests in `source/strafer_lab/tests/test_export_policy.py` extend:
- ONNX round-trip determinism (same as TorchScript path).
- Metadata sidecar contains the `formats` field with both `pt` and
  `onnx`.
- ONNX file is loadable by ONNX Runtime's CPU EP (the test runs on
  DGX, not Jetson — TRT EP verification happens on Jetson during
  Phase 4).

### Phase 3 — Benchmark hook

`Scripts/benchmark_policy.py`:

```
python Scripts/benchmark_policy.py --model models/strafer_depth_v0.onnx --iters 1000
```

Thin wrapper: load via `load_policy()`, run `benchmark_policy()`,
print median / p95 / p99 latency. Used by the Jetson side to verify
deployment-time inference latency after `rsync`-ing the artifact,
and by this brief's CI to catch latency regressions in the export
toolchain.

### Phase 4 — Jetson-side TRT-EP verification (1 day)

Verify the ONNX artifact loads through the TensorRT execution
provider on the Jetson Orin Nano. This phase runs *after*
rsync'ing the export to a Jetson, since TRT EP availability is
JetPack-version-coupled and not testable on the DGX.

- Confirm `onnxruntime-gpu` (with TRT EP) is installed in the
  Jetson Python environment. If not, document the install path
  (typically `python3 -m pip install onnxruntime-gpu` from
  NVIDIA's Jetson wheel index, version pinned to the JetPack
  version).
- Run `Scripts/benchmark_policy.py` with provider preference
  `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
  — record the latency table per provider in the PR description.
- For DEPTH variant, target median latency ≤ 6 ms via TRT (so
  the Jetson inference node's wrapping infrastructure has budget
  to clear the ≤ 10 ms p95 end-to-end target in
  [`strafer-inference-package.md`](../active/trained-policy/inference-package.md)
  Phase 3).
- Surface a clear warning if the TRT EP is unavailable — that's
  the failure mode the inference node's launch-time benchmark
  catches.

## Acceptance criteria

### Build / structure

- [ ] `python Scripts/export_policy.py --help` documents the CLI:
      `--checkpoint`, `--output`, `--variant`.
- [ ] `python -m pytest source/strafer_lab/tests/test_export_policy.py`
      passes — exports a tiny dummy actor, round-trips through
      `load_policy()`, asserts dimensions + determinism + metadata.

### Determinism contract (load-bearing for [`strafer-inference-package.md`](../active/trained-policy/inference-package.md))

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
      ([`strafer-inference-package.md`](../active/trained-policy/inference-package.md))
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
- **Goal-position noise training.** That's
  [`policy-goal-noise-training.md`](../active/trained-policy/goal-noise-training.md).
  Orthogonal — happens *before* export, produces the checkpoint this
  brief consumes.
- **Subgoal-following training env.** That's
  [`strafer-lab-subgoal-env.md`](../active/trained-policy/subgoal-env.md).
  Produces the `NOCAM_SUBGOAL` checkpoint that this export tooling
  will then convert with `--variant NOCAM_SUBGOAL`.
- **Model registry / Databricks Model Serving.** Separate
  operator-side deployment work; not blocking for the LAN-HTTP /
  rsync path used by this brief.
- **Inference-side integration.** That's
  [`strafer-inference-package.md`](../active/trained-policy/inference-package.md).
  This brief's output is rsync'd onto the Jetson by the operator.
