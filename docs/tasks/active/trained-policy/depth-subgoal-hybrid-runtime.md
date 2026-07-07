# Extend `strafer_inference` hybrid mode to consume `DEPTH_SUBGOAL` checkpoints

**Type:** task / runtime extension
**Owner:** Jetson (extends `strafer_inference`)
**Priority:** P3 — completes the 2×2 deployment matrix on the runtime side; pairs with [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md) which produces the trained checkpoint. Not blocking any current mission shape.
**Estimate:** S–M (1–2 days, the size depending on which implementation approach the picking-up agent chooses — see "Architecture choice" below)
**Branch:** task/depth-subgoal-hybrid-runtime
**Status:** In-flight (un-parked). Runtime is variant-agnostic (verified); this PR **consumes** `PolicyVariant.DEPTH_SUBGOAL` (shipped in #138) and adds the full combo test cell, the sim depth-timeout override, and the deploy depth-scale fix. Stays **active** for the live acceptance (load the converged `DEPTH_SUBGOAL` artifact + a hybrid sim mission), gated on the checkpoint (`depth-subgoal-env` Phase 5) + the rig.

## Un-park trigger

> **Resolved.** All three deps shipped. [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md) shipped `PolicyVariant.DEPTH_SUBGOAL` (its Phase 2) in **#138**; this brief **consumes** that variant — the runtime was already variant-agnostic, so there is nothing to add on the shared-contract side. The **live** acceptance still waits on `depth-subgoal-env`'s converged checkpoint (Phase 5, operator-gated).

This brief was parked until **all** of the following had shipped:

1. [`hybrid-mode`](../../completed/hybrid-mode.md) ✅ shipped (#119 / #122 / #123) — provides the `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` dispatch, the Nav2 replan/rolling-subgoal selection logic, and the runtime flags this brief's variant slots into.
2. [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md) ✅ shipped `PolicyVariant.DEPTH_SUBGOAL` in #138 (consumed here); its converged training checkpoint (Phase 5) is still pending.
3. [`inference-package`](../../completed/inference-package.md) ✅ shipped (the DEPTH observation pipeline this brief reactivates under hybrid mode).

Un-parked by `git mv parked/trained-policy/depth-subgoal-hybrid-runtime.md active/trained-policy/depth-subgoal-hybrid-runtime.md` in this PR.

## Story

As a **mission operator who has a trained `DEPTH_SUBGOAL` checkpoint and wants to deploy it**, I want **the inference node to compose its depth pipeline with the hybrid backend's rolling-subgoal selection so the policy sees both Nav2's path AND the depth field**, so that **the costmap-trust caveat documented for NOCAM_SUBGOAL is lifted: the policy can detect late-arriving obstacles via depth and deviate from the path to avoid them**.

## Architecture choice — RESOLVED: Option A, shipped in #122, verified here

> **Option A shipped in PR #122** (`strafer-inference-hybrid-mode-pr-b`). The observation pipeline is already variant-agnostic: `_has_depth` / `_uses_subgoal` are derived from `self._variant.fields` (`inference_node.py`), the depth subscriber and subgoal subscriber are gated on those flags, `_assemble_observation_or_none` composes referent + optional depth generically, the watchdog takes `depth_enabled` / `subgoal_enabled`, and the TRT cold-start warning gates on `_has_depth`. There was **no refactor left to do** — `DEPTH_SUBGOAL` (a depth field AND `subgoal_*` fields) lights up both paths with zero structural change. This PR's job was to (1) supply the `DEPTH_SUBGOAL` member the runtime introspects, (2) prove the composition with the combo test cell, and (3) close the one real sim-timing hole (depth staleness). The historical Option-A/Option-B decision below is retained for context; it is settled.

The original two options (kept for the record):

### Option A — Variant-agnostic refactor (recommended)

Refactor `inference_node.py`'s observation-assembly path so the depth subscriber, the depth watchdog source, and the depth field in the raw obs dict are **conditional on `self._variant.fields` containing a `depth_image` field**, rather than hardcoded. Once that refactor lands, DEPTH_SUBGOAL deploys with no further code changes — load the checkpoint, point `model_path` at it, set `STRAFER_NAV_BACKEND=hybrid_nav2_strafer`. The runtime introspects the variant and composes pipelines accordingly.

The refactor is small (~50–80 lines across `inference_node.py`'s `_load_policy_from_param`, `_on_depth`, `_assemble_observation_or_none`, and the watchdog source list). The benefit is durable: any future `PolicyVariant` consuming a subset of the existing field types (e.g. a hypothetical `IMU_ONLY` debug variant) deploys without further runtime changes.

This is the recommended path because [`completed/inference-package.md`](../../completed/inference-package.md)'s implementation hardcoded DEPTH-specific assumptions in four places — that's an honest reflection of the Phase 3 scope at the time, not a design intent. The hardcoding becomes drag the moment a second non-DEPTH variant ships.

### Option B — Surgical extension

Add an explicit DEPTH_SUBGOAL branch alongside the existing DEPTH and NOCAM_SUBGOAL paths. Smaller diff at this brief's PR, larger diff at every future variant brief. Pick only if there's a reason variant-agnostic abstraction is harder than it looks (e.g. the watchdog source-list construction has hidden coupling the refactor would surface).

Recommendation: A. The picking-up PR description records *why* A was chosen (or the specific reason B was preferred).

## Context bundle

Read these before starting:

- [`depth-subgoal-env.md`](../../active/trained-policy/depth-subgoal-env.md) — the source of `PolicyVariant.DEPTH_SUBGOAL` + the trained checkpoint this brief loads.
- [`hybrid-mode.md`](../../completed/hybrid-mode.md) — the consumer-side runtime brief. This brief extends its `mode: hybrid` plumbing to support a depth-aware variant.
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

- goal — presence-keyed on the executing `navigate_to_pose` action goal (there is no goal topic; see [`inference-goal-preemption`](../../completed/inference-goal-preemption.md))
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

Sim-validation of the trained checkpoint lives in `depth-subgoal-sim-validation.md` (to be filed when this brief picks up, mirroring the [`strafer-hybrid-sim-validation`](../../completed/trained-policy/strafer-hybrid-sim-validation.md) precedent).

## Acceptance criteria

### Refactor (Option A) or branch (Option B)

- [x] Implementation approach chosen and stated: **Option A, shipped in #122** — verified here, no new refactor (see the resolved Architecture-choice section).
- [x] Option A: the four formerly-hardcoded-DEPTH paths are variant-aware (landed #122; `DEPTH_SUBGOAL` exercises the depth-carrying half of every one). The `DEPTH_SUBGOAL` member is defined by #138 and consumed here.

### Variant + watchdog composition

> Terminology note: the parked brief wrote the 7th source as "plan". After #132 the subgoal generator owns `/plan`/replanning and the inference-node watchdog watches **`subgoal`** freshness (the inference-side half of the split stale-plan budget). The 7 sources are therefore: goal, IMU, joint_states, odom, depth, TF, **subgoal**.

- [x] DEPTH_SUBGOAL + hybrid: 7-source watchdog (goal, IMU, joint_states, odom, depth, TF, subgoal); depth and subgoal trip independently (`test_depth_subgoal_watchdog_has_both_depth_and_subgoal`), both-fresh + goal-active is clean.
- [x] NOCAM_SUBGOAL + hybrid: 6-source watchdog (no depth); regression cell unchanged (`test_nocam_subgoal_watchdog_swaps_depth_for_subgoal`).
- [x] DEPTH + direct: 6-source watchdog (no subgoal); regression cell unchanged (`test_depth_direct_watchdog_has_no_subgoal_source`).
- [x] NOCAM + direct: still rejected — Option A does not change that; `NOCAM` in `strafer_direct` remains unsupported per [`completed/inference-package.md`](../../completed/inference-package.md)'s Out of scope.

### Recurrent contract preservation

- [x] DEPTH_SUBGOAL's reset triggers fire at the same call sites as DEPTH — the trigger set is variant-independent (`_execute_callback` resets on every goal accept, incl. preempting goals) and `load_policy` keys recurrence off checkpoint structure, not variant (verified #136).
- [ ] Cross-format recurrent-contract parametrization over `DEPTH_SUBGOAL` — the e2e fixture (`test_recurrent_contract_e2e.py`, strafer_lab) does not parametrize over variant cheaply and is out of this brief's touch scope; filed as a follow-up (see Out of scope). Re-confirm with the converged `.pt`/`.onnx` artifact.

### Integration

- [x] `JetsonRosClient.navigate_to_pose` dispatch with `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` + a loaded DEPTH_SUBGOAL artifact routes correctly — no new dispatch logic; the bringup auto-launch gates on the backend, not the variant (`test_depth_subgoal_env_flows_to_launch_arg`), so DEPTH_SUBGOAL auto-launches inference + the subgoal generator exactly as NOCAM_SUBGOAL. Live artifact-load confirmation stays open (checkpoint pending).

### Deploy depth-scale correction (cross-lane flag from #138)

- [x] **Fixed a pre-existing double-scale in the deploy depth path** (whole DEPTH family, not DEPTH_SUBGOAL-specific). `downsample_depth` normalized by `1/max_depth` to [0,1] and `assemble_observation` then applied `DEPTH_SCALE` (=1/max_depth) again, feeding the network 6× too-small depth vs sim (a 3 m surface → 0.083 deployed vs 0.5 in sim). Sim scales exactly once (`mdp.depth_image` returns raw meters; the ObsTerm applies `DEPTH_SCALE`). Fix: `downsample_depth` now returns **raw meters**, mirroring `mdp.depth_image`, so `assemble_observation`'s single scale matches sim; the `build_raw_obs_dict` param is renamed `depth_flat_normalized`→`depth_flat_meters` to make the contract legible. Pinned end-to-end by `test_obs_pipeline.py::TestDepthSingleScaleParity` (3 m → 0.5, explicitly not 0.083) — the missing value assertion that let it hide. `inference-package.md`'s stale "mirror the 1/max_depth step" notes corrected. Latent until now, but #138's DeFM un-scale (anchored to absolute metric distance) makes the absolute-scale contract load-bearing, so it must land before any real-robot DEPTH / DEPTH_SUBGOAL run. **Changes the deployed DEPTH input distribution → wants a real-robot re-check (see Out of scope).**

### Maintenance

- [x] Facts updated in the same commit: operator env-file / launch-arg variant enumerations include DEPTH_SUBGOAL; `inference-package.md`'s depth-normalization implementation notes corrected for the raw-meters `downsample_depth` contract. (The `PolicyVariant.DEPTH_SUBGOAL` member, its docstring, and the strafer_lab variant-count guard are owned by #138.)

## Investigation pointers

- [`source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py) — the file Option A refactors / Option B extends. The four hardcoded-DEPTH paths to surface and conditionalize.
- [`source/strafer_ros/strafer_inference/strafer_inference/watchdog.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/watchdog.py) — the watchdog `stale_sources` function; today it takes a fixed-shape `WatchdogTimeouts` dataclass. Variant-aware composition either reshapes that dataclass or filters its source list at call time.
- [`source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py) — the helpers are already variant-agnostic (`build_raw_obs_dict` takes every field; `assemble_observation` in `strafer_shared.policy_interface` does the field selection). Most of the refactor work is in the node, not the helpers.
- [`hybrid-mode.md`](../../completed/hybrid-mode.md) — Phase 1's `mode` flag and `/plan` subscription. This brief composes with that work; pick up only after hybrid-mode has shipped.
- [`depth-subgoal-env.md`](../../active/trained-policy/depth-subgoal-env.md) — the source of the variant + checkpoint.

## Out of scope

- **The training environment + checkpoint.** That's [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md). This brief is runtime-only; it loads what that brief produces.
- **Sim validation against the trained checkpoint (the live acceptance, still OPEN on this brief).** Loading the converged `DEPTH_SUBGOAL` artifact + running a hybrid sim mission is goal-c's validation step; it waits on the checkpoint ([`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md) training) + the rig, so this brief stays **active** at merge. The dedicated `depth-subgoal-sim-validation.md` is filed once the checkpoint lands (same precedent as [`inference-package`](../../completed/inference-package.md) → [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md) and [`hybrid-mode`](../../completed/hybrid-mode.md) → [`strafer-hybrid-sim-validation`](../../completed/trained-policy/strafer-hybrid-sim-validation.md)). It carries the DEPTH_SUBGOAL-specific parity bounds (19 NOCAM dims ≤ 1e-5 + 4800 depth dims ≤ 1e-3 + subgoal-pose pick ≤ MAP_RESOLUTION × 2), the 7-source watchdog acceptance, and the dynamic-obstacle test that's intentionally out of scope for NOCAM_SUBGOAL.
- **Cross-format recurrent-contract parametrization over `DEPTH_SUBGOAL`.** Follow-up (one-liner): parametrize `test_recurrent_contract_e2e.py` (strafer_lab — out of this brief's touch scope) over `DEPTH_SUBGOAL` once the converged `.pt`/`.onnx` exports exist. Tracked on [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md)'s Phase 2 (which owns that test surface and produces the artifact); the reset-trigger contract itself is variant-independent and already covered here.
- **Real-robot re-check of the corrected deploy depth scale.** The double-scale fix changes the deployed DEPTH-family input distribution (now the correct sim-matching value). No real DEPTH checkpoint has been deployed to a robot yet, so nothing was locked into the wrong distribution — but the first real-robot DEPTH / DEPTH_SUBGOAL run must confirm depth reaches the network at the sim value (the sim-validation follow-up's 4800-dim ≤ 1e-3 parity bound covers this in sim).
- **Real-robot DEPTH_SUBGOAL validation.** Files later, gated on the sim-validation follow-up passing.
- **Re-tuning the DEPTH_SUBGOAL checkpoint** if the runtime exposes a gap. Tuning responses route back to [`depth-subgoal-env`](../../active/trained-policy/depth-subgoal-env.md)'s follow-up queue.
- **NOCAM_DIRECT support.** Intentionally never per [`completed/inference-package.md`](../../completed/inference-package.md)'s Out of scope; the variant-aware refactor (Option A) does not change that — startup explicitly rejects NOCAM under `STRAFER_NAV_BACKEND` unset / `strafer_direct`.
