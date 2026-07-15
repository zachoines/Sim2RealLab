**Status:** Shipped 2026-07-15 in `1bbbf08` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/152

# Bind inference.yaml to the launched node + audit the values it awakens

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_ros/strafer_inference/config/`)
**Priority:** P1
**Estimate:** S — one-line key change + a value audit + a regression pin. Runtime bind confirmed on the Jetson; the two-bringup TRT-persistence timing is operator-gated.
**Branch:** task/inference-config-bind

## Story

As a **mission operator launching a depth policy under bringup**, I want **inference.yaml's parameters to actually reach the inference node**, so that **the TRT engine persists across launches (no ~90 s cold rebuild every time) and the config I edit is the config that runs, not silently-ignored text**.

## The fault (measured)

`inference.launch.py` launches the node with `name="strafer_inference"` **and** `namespace="strafer_inference"` → FQN `/strafer_inference/strafer_inference`. `inference.yaml` keyed its params under the bare `strafer_inference:` top key, which matches only a node in the **global** namespace (`/strafer_inference`). The two never matched, so **every value in the file was silently ignored** and the node ran on code defaults. Visible consequence: `trt_engine_cache_enable`/`trt_engine_cache_path` fell back to the code defaults `False`/`""`, so TensorRT rebuilt the engine (~90 s for DEPTH_SUBGOAL on Orin Nano) on **every** launch instead of deserializing a cached one. `subgoal_generator.yaml` is **not** affected — its node runs unnamespaced, so its bare key binds (verified).

## The fix

`inference.yaml` top key `strafer_inference:` → `/**:` — the wildcard matches any fully-qualified name, so it binds the namespaced node and survives a future namespace change. One line; the rest of the work is the audit it forces, because dead config comes alive the moment the key binds.

## Overlay order (verified, load-bearing)

`inference.launch.py` sets `parameters=[<config_file: the yaml>, {model_path, policy_variant, use_sim_time}]` — the override dict is **last**. launch_ros writes that dict to a temp params-file keyed by the node's FQN and appends it after the yaml, so it wins over the `/**` wildcard by **both** command-line order **and** FQN-beats-wildcard specificity. So env keeps ownership of `model_path` / `policy_variant` / `use_sim_time` (the desired direction) — the stale `policy_variant: "DEPTH"` literal could never override the env-driven `DEPTH_SUBGOAL`. Confirmed by two independent adversarial verifiers **and** an in-process rcl reproduction (`/**` binds `trt_engine_cache_path`; the bare key leaves it empty), now pinned as a test.

## Value disposition — every key audited against the node contract

Deleted (3) — each was inert-or-duplicate and would have re-bound wrongly the moment the key matched:

| Key | Old value | Disposition | Why |
|---|---|---|---|
| `model_path` | `""` | **delete** | Launch/env-owned (`STRAFER_INFERENCE_MODEL_PATH`), applied last-wins; standalone `ros2 run` passes `-p model_path`. The YAML `""` only duplicated the node's "no model" sentinel and never won — one owner now. |
| `policy_variant` | `"DEPTH"` | **delete** | Launch/env-owned (`STRAFER_POLICY_VARIANT`), applied last-wins. The stale `DEPTH` literal predated the rig's `DEPTH_SUBGOAL` and was inert only because the override wins; deleting removes the hazard outright. |
| `infer_period_s` | `0.0333333` | **delete** | Still consumed (sets the tick timer; the depth-freshness gate is a separate per-frame skip), but a truncated hardcoded copy of the node's shared-constant default `POLICY_SIM_DT × POLICY_DECIMATION = 1/30`. Deleting lets the deploy rate default drift-proof from `strafer_shared`, exactly as `subgoal_generator.yaml` omits `update_period_s`. |

Kept (20): all match the node's declared default and pin a real deployment contract (topic/frame names, watchdog timeouts, the `/cmd_vel` remap contract, cross-node agreements with `subgoal_generator.yaml`, the ONNX EP order). The two that intentionally **differ** from the code default — `trt_engine_cache_enable: true` and `trt_engine_cache_path` (node defaults `False`/`""`) — are the whole point of the fix: the node delegates their real default to the YAML, so binding them is what makes the engine persist. `obs_dump_path` is now settable via the YAML (the fix's side benefit) but stays out of the normal-mission config (diagnostic only).

## Validation

- **Runtime bind pinned.** `test_inference_config.py::TestParamsFileBindsToLaunchedNode` exercises rcl's real params-file → FQN matching in-process: the shipped `/**` yaml binds `trt_engine_cache_path` to `/strafer_inference/strafer_inference`; a bare-key copy does **not** (negative control). This catches a future namespace change or a revert to a specific key — the static-parse tests could not.
- **Empirical, through the real launch.** `ros2 launch inference.launch.py` + `ros2 param get` on the live node: `trt_engine_cache_path` and `trt_engine_cache_enable=true` bound from the YAML (code defaults `""`/`False`); `infer_period_s` back at the shared-constant `0.03333333333333333` (not the deleted `0.0333333` literal); `policy_variant=DEPTH` from the launch default (env unset), not a YAML key.
- **Tests:** `colcon test --packages-select strafer_inference` → 337 passed, 9 skipped (pre-existing torch skips), 0 failures.
- **Operator-gated (rig):** two consecutive bringups with a real model — first persists the engine to `trt_engine_cache_path` (~90 s cold build), second deserializes it in seconds (the measured fix signature). That timing needs the rig + operator time.
