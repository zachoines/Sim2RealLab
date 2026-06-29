# Extend `strafer_inference` hybrid mode to consume `DEPTH_SUBGOAL` checkpoints

**Type:** task / runtime extension
**Owner:** Jetson (extends `strafer_inference`)
**Priority:** P3 — completes the 2×2 deployment matrix on the runtime side; pairs with [`depth-subgoal-env`](depth-subgoal-env.md) which produces the trained checkpoint. Not blocking any current mission shape.
**Estimate:** S–M (1–2 days, the size depending on which implementation approach the picking-up agent chooses — see "Architecture choice" below)
**Branch:** task/depth-subgoal-hybrid-runtime

## Un-park trigger

This brief is parked until **all** of the following have shipped:

1. [`hybrid-mode`](../../active/trained-policy/hybrid-mode.md) shipped — provides the `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` dispatch, the Nav2 `/plan` subscription, the rolling-subgoal selection logic, and the `mode` runtime flag this brief's variant slots into.
2. [`depth-subgoal-env`](depth-subgoal-env.md) shipped — provides `PolicyVariant.DEPTH_SUBGOAL` and the converged training checkpoint this brief's runtime loads.
3. [`inference-package`](../../completed/inference-package.md) ✅ shipped (the DEPTH observation pipeline this brief reactivates under hybrid mode).

Un-park by `git mv parked/trained-policy/depth-subgoal-hybrid-runtime.md active/trained-policy/depth-subgoal-hybrid-runtime.md` in the PR that picks it up.

## Story

As a **mission operator who has a trained `DEPTH_SUBGOAL` checkpoint and wants to deploy it**, I want **the inference node to compose its depth pipeline with the hybrid backend's rolling-subgoal selection so the policy sees both Nav2's path AND the depth field**, so that **the costmap-trust caveat documented for NOCAM_SUBGOAL is lifted: the policy can detect late-arriving obstacles via depth and deviate from the path to avoid them**.

## Architecture choice (decide at PR-opening time)

The picking-up agent picks one of these two approaches. The choice determines the size of the PR.

### Option A — Variant-agnostic refactor (recommended)

Refactor `inference_node.py`'s observation-assembly path so the depth subscriber, the depth watchdog source, and the depth field in the raw obs dict are **conditional on `self._variant.fields` containing a `depth_image` field**, rather than hardcoded. Once that refactor lands, DEPTH_SUBGOAL deploys with no further code changes — load the checkpoint, point `model_path` at it, set `STRAFER_NAV_BACKEND=hybrid_nav2_strafer`. The runtime introspects the variant and composes pipelines accordingly.

The refactor is small (~50–80 lines across `inference_node.py`'s `_load_policy_from_param`, `_on_depth`, `_assemble_observation_or_none`, and the watchdog source list). The benefit is durable: any future `PolicyVariant` consuming a subset of the existing field types (e.g. a hypothetical `IMU_ONLY` debug variant) deploys without further runtime changes.

This is the recommended path because [`completed/inference-package.md`](../../completed/inference-package.md)'s implementation hardcoded DEPTH-specific assumptions in four places — that's an honest reflection of the Phase 3 scope at the time, not a design intent. The hardcoding becomes drag the moment a second non-DEPTH variant ships.

### Option B — Surgical extension

Add an explicit DEPTH_SUBGOAL branch alongside the existing DEPTH and NOCAM_SUBGOAL paths. Smaller diff at this brief's PR, larger diff at every future variant brief. Pick only if there's a reason variant-agnostic abstraction is harder than it looks (e.g. the watchdog source-list construction has hidden coupling the refactor would surface).

Recommendation: A. The picking-up PR description records *why* A was chosen (or the specific reason B was preferred).

## Context bundle

Read these before starting:

- [`depth-subgoal-env.md`](depth-subgoal-env.md) — the source of `PolicyVariant.DEPTH_SUBGOAL` + the trained checkpoint this brief loads.
- [`hybrid-mode.md`](../../active/trained-policy/hybrid-mode.md) — the consumer-side runtime brief. This brief extends its `mode: hybrid` plumbing to support a depth-aware variant.
- [`completed/inference-package.md`](../../completed/inference-package.md) — the DEPTH-direct runtime. Most of this brief's code is "make the DEPTH-direct obs pipeline composable with the hybrid-mode rolling-subgoal pipeline". If Option A is taken, this brief is the place where the DEPTH-direct path stops being DEPTH-specific.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md) — DEPTH_SUBGOAL inherits the contract from DEPTH; the reset-trigger set and thread-safety mutex from PR #55 apply unchanged.

## Approach

Three phases, sequenced. Phase 1 is the load-bearing refactor (Option A) or the surgical patch (Option B). Phases 2 and 3 are the same regardless of Option choice.

### Phase 1 — Variant-aware obs pipeline (Option A) or surgical DEPTH_SUBGOAL branch (Option B)

**Option A (recommended)** — in `inference_node.py`:

- `_load_policy_from_param`: gate the `Building TensorRT engine` log line on the loaded variant having any `depth_image` field; today it gates on `model_path.suffix == ".onnx"` only.
- `_assemble_observation_or_none`: branch on `self._variant.fields` containing each field type. Skip the depth subscriber wait for NOCAM variants; skip `build_raw_obs_dict`'s `depth_flat_normalized` arg for variants without a depth field.
- Watchdog: build the source list dynamically. Variants with a depth field include the depth source; variants without one don't. NOCAM_SUBGOAL today drops the depth source; DEPTH_SUBGOAL today restores it.
- `_on_depth` callback: skip caching when the loaded variant has no depth field (no behavioral difference today since the topic is still subscribed, but avoids decoding cost). Or simply skip subscribing in the first place when the variant has no depth field — small but cleaner.

**Option B** — explicit DEPTH_SUBGOAL handling alongside the existing variant branches. State the reason in the PR description; the burden of proof is on Option B since A is the recommended path.

### Phase 2 — Hybrid-mode watchdog source composition (½ day)

The hybrid-mode brief introduces a `/plan` freshness source (the 6th source overall). For DEPTH_SUBGOAL under hybrid mode, the source list is:

- goal — from the action server (replaces the `/strafer/goal` subscriber-driven source when hybrid mode is active per [`hybrid-mode`](../../active/trained-policy/hybrid-mode.md))
- IMU — `/d555/imu/filtered`
- joint_states — `/strafer/joint_states`
- odom — `/strafer/odom`
- depth — `/d555/depth/image_rect_raw` (restored vs NOCAM_SUBGOAL hybrid)
- TF — `map → base_link`
- plan — `/plan` (hybrid-mode-specific)

= 7 sources total. The composition is mechanically derivable from `(variant.fields, mode)`; if Option A is taken, this composition is automatic. If Option B is taken, hard-code it.

### Phase 3 — Validation (1 day)

Unit tests:

- DEPTH_SUBGOAL load: subscribes to `/d555/depth/image_rect_raw`, includes depth in the watchdog source list, includes `depth_image` in the raw obs dict.
- NOCAM_SUBGOAL load: does NOT subscribe to depth, does NOT include depth in the watchdog or obs dict. (Regression anchor — Option A's refactor must not break this.)
- DEPTH-direct load: unchanged behavior. (Regression anchor for the existing strafer_direct path.)
- Watchdog source list under all four (variant, mode) combinations enumerated in a parametrized test.

End-to-end (smoke): with a hand-built DEPTH_SUBGOAL-shaped ONNX stub and `STRAFER_NAV_BACKEND=hybrid_nav2_strafer`, the inference node:

1. Loads the artifact.
2. Subscribes to `/plan`, `/d555/depth/image_rect_raw`, plus the four common topics.
3. Once all 7 watchdog sources are fresh and a goal is staged, publishes non-zero `/cmd_vel`.
4. Stale `/plan` → zero twist.
5. Stale depth → zero twist.

Sim-validation of the trained checkpoint lives in `depth-subgoal-sim-validation.md` (to be filed when this brief picks up, mirroring the [`strafer-hybrid-sim-validation`](strafer-hybrid-sim-validation.md) precedent).

## Acceptance criteria

### Refactor (Option A) or branch (Option B)

- [ ] Implementation approach chosen and stated in the PR description.
- [ ] If Option A: the four hardcoded-DEPTH paths in [`completed/inference-package.md`](../../completed/inference-package.md)'s implementation (depth subscriber unconditional, watchdog depth source unconditional, depth required in `_assemble_observation_or_none`, depth required in `build_raw_obs_dict`) become variant-aware.
- [ ] If Option B: explicit DEPTH_SUBGOAL handling lands alongside the existing variant branches; the maintenance burden is acknowledged in the PR description.

### Variant + watchdog composition

- [ ] DEPTH_SUBGOAL + hybrid: 7-source watchdog (goal, IMU, joint_states, odom, depth, TF, plan); all sources zero-twist on independent staleness.
- [ ] NOCAM_SUBGOAL + hybrid: 6-source watchdog (no depth); regression-tested.
- [ ] DEPTH + direct: 6-source watchdog (no plan); regression-tested.
- [ ] NOCAM + direct: rejected at startup (the deployment lane is intentionally not supported per [`completed/inference-package.md`](../../completed/inference-package.md)'s Out of scope).

### Recurrent contract preservation

- [ ] DEPTH_SUBGOAL's reset triggers fire at the same call sites as DEPTH (action-server goal accept, mid-mission goal pose update with `is_mid_mission_reset=True`).
- [ ] Mutex-guarded `policy(obs)` and `policy.reset()` calls inherit unchanged.

### Integration

- [ ] `JetsonRosClient.navigate_to_pose` dispatch with `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` + a loaded DEPTH_SUBGOAL artifact routes correctly. No new dispatch logic needed — the dispatcher already targets the strafer_inference action server regardless of loaded variant.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context module — particularly [`completed/inference-package.md`](../../completed/inference-package.md)'s implementation notes about the hardcoded DEPTH paths — update those in the same commit.

## Investigation pointers

- [`source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py) — the file Option A refactors / Option B extends. The four hardcoded-DEPTH paths to surface and conditionalize.
- [`source/strafer_ros/strafer_inference/strafer_inference/watchdog.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/watchdog.py) — the watchdog `stale_sources` function; today it takes a fixed-shape `WatchdogTimeouts` dataclass. Variant-aware composition either reshapes that dataclass or filters its source list at call time.
- [`source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py) — the helpers are already variant-agnostic (`build_raw_obs_dict` takes every field; `assemble_observation` in `strafer_shared.policy_interface` does the field selection). Most of the refactor work is in the node, not the helpers.
- [`hybrid-mode.md`](../../active/trained-policy/hybrid-mode.md) — Phase 1's `mode` flag and `/plan` subscription. This brief composes with that work; pick up only after hybrid-mode has shipped.
- [`depth-subgoal-env.md`](depth-subgoal-env.md) — the source of the variant + checkpoint.

## Out of scope

- **The training environment + checkpoint.** That's [`depth-subgoal-env`](depth-subgoal-env.md). This brief is runtime-only; it loads what that brief produces.
- **Sim validation against the trained checkpoint.** File `depth-subgoal-sim-validation.md` as a follow-up at PR-opening time (same precedent as [`inference-package`](../../completed/inference-package.md) → [`strafer-direct-sim-validation`](../../active/trained-policy/strafer-direct-sim-validation.md) and [`hybrid-mode`](../../active/trained-policy/hybrid-mode.md) → [`strafer-hybrid-sim-validation`](strafer-hybrid-sim-validation.md)). Carries the DEPTH_SUBGOAL-specific parity bounds (19 NOCAM dims ≤ 1e-5 + 4800 depth dims ≤ 1e-3 + subgoal-pose pick ≤ MAP_RESOLUTION × 2), the 7-source watchdog acceptance, and the dynamic-obstacle test that's intentionally out of scope for NOCAM_SUBGOAL.
- **Real-robot DEPTH_SUBGOAL validation.** Files later, gated on the sim-validation follow-up passing.
- **Re-tuning the DEPTH_SUBGOAL checkpoint** if the runtime exposes a gap. Tuning responses route back to [`depth-subgoal-env`](depth-subgoal-env.md)'s follow-up queue.
- **NOCAM_DIRECT support.** Intentionally never per [`completed/inference-package.md`](../../completed/inference-package.md)'s Out of scope; the variant-aware refactor (Option A) does not change that — startup explicitly rejects NOCAM under `STRAFER_NAV_BACKEND` unset / `strafer_direct`.
