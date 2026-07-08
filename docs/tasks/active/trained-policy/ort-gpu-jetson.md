# Install + engage GPU execution providers for `strafer_inference`

**Type:** task / runtime enablement (Jetson host env + `strafer_inference` config/docs; minimal code)
**Owner:** Jetson
**Priority:** P2 — unblocks the real-robot latency budget. Satisfies the "TRT runtime on Jetson" prerequisite of [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md) (its Prerequisite 3 + Latency acceptance).
**Branch:** task/ort-gpu-jetson
**Status:** Shipped 2026-07-07 in this PR (Jetson). Host `onnxruntime-gpu` install + active-provider log + TRT engine cache + measured fallback matrix all landed and verified on-device. Stays **active** only for the cross-referenced **real-robot 33 ms verification**, which is deferred to the real-robot validation lane (see Out of scope).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/144

## Story

As a **mission operator deploying a trained policy on the Orin Nano**, I want **the inference node to actually run on the TensorRT / CUDA execution providers instead of silently falling back to CPU**, so that **DEPTH inference meets the 33 ms (30 Hz) control budget** — which it cannot on CPU (measured ~84 ms).

## Problem

The stock PyPI `onnxruntime` (1.23.2) was installed in the Jetson runtime env (`~/.local`, system python3.10). It is CPU-only: `get_available_providers()` returned `['AzureExecutionProvider', 'CPUExecutionProvider']` — no TRT, no CUDA. The `onnx_providers` preference list in `inference.yaml` already asked for TRT→CUDA→CPU, so ORT silently bound CPU and nothing surfaced the fallback. DEPTH_SUBGOAL on CPU is ~84 ms/inference — 2.5× over budget.

## What shipped

### 1. Host install — `onnxruntime-gpu` 1.23.0 (JetPack-matched)

- Machine: JetPack 6.x / L4T **r36.5.0**, CUDA **12.6**, cuDNN **9.3.0**, TensorRT **10.3.0.30**, Python **3.10.12**, aarch64.
- Wheel: `onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl` from the jetson-ai-lab index **`https://pypi.jetson-ai-lab.io/jp6/cu126`** (the `.dev` domain is dead — use `.io`; `jp6/cu126` = JetPack 6 / CUDA 12.6).
- Version choice: 1.23.0 over the index's 1.24.0 — the jp6/cu126 build links the JetPack **system** TRT 10.3 (ORT has supported the TRT 10.x API since ~1.18), it is the pairing NVIDIA's Orin-Nano/JP6.2 forum thread recommends, and it keeps the same 1.23 minor line as the pre-existing CPU `onnxruntime` 1.23.2 (minimal opset/kernel/API drift).
- Install order (**load-bearing**): CPU `onnxruntime` and `onnxruntime-gpu` share one `site-packages/onnxruntime/` dir and the last install wins **silently** (the CPU `pybind11_state.so` overwrites the GPU one, stripping the providers with no error). Since ORT ≥ 1.20 the GPU wheel no longer declares `Provides-Dist: onnxruntime`, so pip does not auto-replace. Uninstall first:
  ```bash
  python3 -m pip uninstall -y onnxruntime
  python3 -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.23.0
  ```
- Do **not** `pip install` the `nvidia-*-cu12` / `tensorrt` wheels — the Jetson build dlopens the JetPack system CUDA/cuDNN/TensorRT (`libnvinfer.so.10`, `libcudnn.so.9`, `libcudart.so.12`); pip copies conflict. No `LD_LIBRARY_PATH` tweak was needed here — the providers loaded cleanly.
- Verified: `python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"` →
  `1.23.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.

### 2. Prove the providers engage — active-provider startup log

The `LoadedPolicy` ONNX wrappers now record `active_providers = sess.get_providers()`; the node logs it once after load. Live standalone-node run (DEPTH_SUBGOAL, no mission):

```
[INFO] [strafer_inference]: Loaded policy from .../strafer_depth_subgoal_v0.onnx (recurrent=True)
[INFO] [strafer_inference]: ONNX Runtime active providers (priority order): ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
[INFO] [strafer_inference]: strafer_inference node up: variant=DEPTH_SUBGOAL tick=0.0333s ... policy_loaded=True
```

An operator can now read GPU-engaged vs CPU-fallback off the log. A NOCAM_SUBGOAL run with `-p onnx_providers:=[CPUExecutionProvider]` logs `active providers: ['CPUExecutionProvider']` and (correctly) omits the TRT cold-start line.

### 3. TRT engine cache

The TRT EP builds the engine on first inference — **~93 s cold for DEPTH_SUBGOAL** on the Orin Nano. New config (`inference.yaml`) + node params persist it:

```yaml
trt_engine_cache_enable: true
trt_engine_cache_path: "~/.cache/strafer_inference/trt_engines"
```

`inference_node._resolve_onnx_providers()` upgrades the `TensorrtExecutionProvider` entry from a plain string to a `(name, {trt_engine_cache_enable, trt_engine_cache_path})` tuple when the cache is configured; the plain-string list still works when it is not. `load_policy` forwards the mixed list to ORT verbatim (ORT natively accepts string / `(name, dict)` entries) — no change to `load_policy`'s provider/threading logic. The node `expanduser`s and `makedirs` the path; the defaults are OFF/empty **in code** (the recommended path lives in config, not the `.py`). Native Python `True` was verified to reach the EP as `'1'` via `get_provider_options()` (not the silent-no-op `'True'`), and a warm cache dropped the DEPTH session-create + first-inference from ~93 s to a few seconds. The existing cold-start log (updated to ~90 s) + the `ready`-param flip still read sensibly: operator sees the "building engine" line, a warm-up gap with `ready=False`, then `ready=True (first inference complete)`.

### 4. Fallback matrix (measured on-device, node-faithful `load_policy` path, `intra_op=1`)

TensorRT partitioning: `[TensorRT EP] Whole graph will run on TensorRT execution provider` — **no DeFM op fell back to CUDA**; the only build log is a benign `setComputePrecision ignored for strongly typed network` on a Normalization layer.

| Variant | Provider | median | p95 | p99 | proc CPU% | active EP |
|---|---|---|---|---|---|---|
| DEPTH_SUBGOAL (4819-dim, GRU) | **TRT** | **4.71 ms** | 6.70 | 7.43 | 74% | TRT |
| DEPTH_SUBGOAL | CUDA | 7.53 ms | 8.38 | 8.94 | 99% | CUDA |
| DEPTH_SUBGOAL | CPU | **83.6 ms** | 83.8 | 84.0 | 100% | CPU |
| NOCAM_SUBGOAL (19-dim MLP) | TRT | 0.228 ms | 0.238 | 0.247 | — | TRT |
| NOCAM_SUBGOAL | CUDA | 0.296 ms | 0.306 | 0.320 | — | CUDA |
| NOCAM_SUBGOAL | **CPU** | **0.056 ms** | 0.058 | 0.079 | — | CPU |

DEPTH_SUBGOAL: TRT is the pick — 4.71 ms median (p99 7.43 ms), comfortably under 33 ms, and lighter on the host CPU than CUDA (74% vs 99%). CUDA is the accepted fallback (also under budget). CPU is **unusable** for DEPTH (2.5× over budget).

### 5. Threading sanity

`onnx_intra_op_threads=1` governs only the CPU intra-op pool; it does not throttle the GPU EPs (their kernels run on-device with their own host threads). Measured: DEPTH TRT at `intra_op=1` → 4.71 ms / 74% CPU vs `intra_op=0` (ORT all-cores) → 4.72 ms / 73% CPU — identical. The `intra_op=1` + spin-off setting from the small-MLP work stays.

### 6. NOCAM regression / per-variant provider preference

The 19-dim MLP is **fastest on CPU** (0.056 ms) — GPU kernel-launch overhead (~0.2 ms) exceeds the math. All three EPs are trivially under budget, but CPU is fastest and skips the engine build / GPU context. Recommendation documented in `inference.yaml` (`onnx_providers` comment) and `env_autonomy.env`: **DEPTH/DEPTH_SUBGOAL → TRT-first (default); NOCAM/NOCAM_SUBGOAL → `onnx_providers=[CPUExecutionProvider]`**. No new mechanism — expressed via the existing config layer.

## Files touched

- `source/strafer_shared/strafer_shared/policy_interface.py` — `LoadedPolicy.active_providers` surface (read-only; ONNX wrappers set it from `sess.get_providers()`).
- `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py` — `trt_engine_cache_*` params, `_resolve_onnx_providers()`, active-provider log, cold-start message.
- `source/strafer_ros/strafer_inference/config/inference.yaml` — TRT cache params + per-variant provider guidance.
- `source/strafer_ros/strafer_bringup/config/env_autonomy.env`, `docs/example_commands_cheatsheet.md` — install + per-variant provider docs.

## Out of scope

- **Real-robot 33 ms verification.** The measured matrix is on synthetic obs through the node's `load_policy` path; the real-robot obs-receive→cmd_vel end-to-end latency on the actual rig **stays open** and joins the real-robot readiness deferrals (`depth-subgoal-hybrid-runtime` Out of scope / the not-yet-filed `strafer-inference-real-robot-validation`). This brief is the GPU-EP enabler that unblocks it.
- **Pre-built `.engine` sidecar shipping.** The engine cache is populated on first launch and reused; shipping a prebuilt engine (keyed to sm87 + TRT version) is a later optimization, not needed now.
