# Validate the `strafer_direct` backend end-to-end against the sim-in-the-loop rig

**Type:** task / validation
**Owner:** Either — recording the rosbag and emitting the gym-env
ground-truth obs requires the DGX-side sim bridge running; assembling
the inference-node obs and running the comparison script runs on the
Jetson side. The brief can be picked up by whichever lane the operator
is in at the time the rig is up.
**Priority:** P2 — gates the real-robot DEPTH validation follow-up but
not the merge of the inference-package PR (the plumbing has unit-test
coverage; this brief converts plumbing-correct into sim-deployable).
**Estimate:** M (1–2 days once the rig is up and a deployable
checkpoint exists; longer if either prerequisite needs work)
**Branch:** task/strafer-direct-sim-validation

## Story

As a **mission operator about to deploy `strafer_direct` against a real
chassis**, I want **the sim-in-the-loop run to demonstrate that the
inference node's observation vector matches the gym env's obs at the
same sim timestamp, that DEPTH inference holds latency targets on
Jetson Orin Nano via TRT EP, and that the trained policy actually
beats the MPPI plateau on a sustained translate-and-avoid mission**,
so that **I have evidence the sim-to-real contract is intact before
real-robot validation runs against sensor-noise distribution shift,
lighting variance, and dynamic obstacles**.

These five acceptance items lived in
[`inference-package.md`](inference-package.md) originally; they were
extracted into this brief so PR #55 could merge with all unit-testable
acceptance closed. Each item gates on operator-driven sim work that
has no unit-test analog.

## Context bundle

Read these before starting:

- [inference-package.md](inference-package.md) — the source of the
  five acceptance items; the implementation of every piece this brief
  validates ships in PR #55.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
  — point 5 (determinism) is exercised at the cross-format level in
  [`source/strafer_lab/tests/test_recurrent_contract_e2e.py`](../../../../source/strafer_lab/tests/test_recurrent_contract_e2e.py)
  and at the inference-node-fake-policy seam in
  [`source/strafer_ros/strafer_inference/test/test_inference_runtime.py`](../../../../source/strafer_ros/strafer_inference/test/test_inference_runtime.py);
  this brief's E2E mission is the third leg of that triangle.
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
  — "Camera resolutions (sim mirrors real)" pins the 640×360 vs 80×60
  split that makes the depth-parity acceptance non-trivial.

## What already exists

The Jetson-side plumbing is complete (PR #55):

- `strafer_inference` ROS package with the DEPTH observation pipeline,
  six-source watchdog, model loading with TRT-EP provider preference,
  recurrent reset triggers, L1 velocity clamp, and a `ready` parameter
  that flips on first successful inference.
- `JetsonRosClient.navigate_to_pose` backend dispatch honoring
  `STRAFER_NAV_BACKEND` with per-mission Nav2 fallback when the
  inference server is unavailable.
- 99 unit tests on the strafer_inference side, 10 new ros_client tests
  on the autonomy side, all green.

What this brief adds is the sim-driven validation that proves the
plumbing produces the right numbers under load.

## Prerequisites

1. **Sim-in-the-loop bringup**: `bringup_sim_in_the_loop.launch.py` up
   on the Jetson with the DGX bridge feeding `/d555/*`, `/strafer/*`,
   and `/clock`. Both required for the rosbag recording.
2. **Deployable DEPTH checkpoint** rsync'd to the Jetson under
   `~/strafer_ws/install/strafer_inference/share/strafer_inference/models/`
   (or wherever `inference.yaml`'s `model_path` is configured). Gates
   on the DGX-side training pipeline producing a converged checkpoint
   against `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` (or `-Robust-`)
   and the export tooling emitting a `.onnx` (plus optional `.engine`
   sidecar to skip the TRT cold-start cost).
3. **TRT runtime on Jetson**: ONNX Runtime built with the TensorRT
   execution provider visible to `load_policy()` — sanity-check via
   `python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"`.

If any prerequisite is missing, file the gap as its own follow-up
rather than working around it here. The point of this brief is the
end-to-end validation, not the rig.

## Acceptance criteria

### Contract parity (sim-to-real load-bearing; rig-only, no checkpoint required)

- [ ] **NOCAM-fields obs parity**: with a recorded sim-in-the-loop
      rosbag, the inference node's assembled NOCAM-portion (first
      19 dims) matches the gym-env obs at the same sim timestamp
      within float32 noise (≤ 1e-5 max abs delta). The
      `body_velocity_xy` parity bound is meaningful post-#56 because
      both sides derive it from the same encoder-FK signal chain.
- [ ] **DEPTH downsample parity**: the 4800-dim depth portion of
      the assembled obs, after the 640×360 → 80×60 resize +
      nearfield-fill + scale pipeline, matches the gym-env
      `depth_image` output for the same scene state within
      ≤ 1e-3 max abs delta. Higher tolerance than NOCAM because
      area-resampling vs. native-render isn't pixel-identical.

These two parity checks do NOT need a trained checkpoint — the
inference node assembles the obs vector on every tick regardless of
whether the policy fires. Set `model_path=""` (the empty sentinel)
and record the rosbag with the action server unadvertised.

### Latency (needs a real ONNX artifact + TRT runtime)

- [ ] **Latency p95 < 10 ms** (obs receive → cmd_vel publish) on
      Jetson Orin Nano via the TensorRT execution provider. Record
      in the merging PR via the
      [`tune_capture.py`](../../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
      harness extended to also capture inference-side timestamps, OR
      a dedicated `benchmark_inference_node.py`. CPU-only fallback
      latency surfaced separately for context but not gating.

A "real" ONNX artifact here means anything DEPTH-shaped that exports
through ONNX Runtime — does not require the final trained
checkpoint. A hand-built DEPTH-shaped stub (4819-dim input → 3-dim
output) is sufficient to measure the inference-path latency; the
trained checkpoint just produces useful actions on top.

### End-to-end (gates on the deployable checkpoint)

- [ ] On a `translate forward 3 m` sim mission with
      `STRAFER_NAV_BACKEND=strafer_direct` and the trained DEPTH
      checkpoint loaded, observed `/strafer/odom.linear.x` 1 s
      sustained median ≥ 1.0 m/s on the open segments. This is the
      architectural-win metric the MPPI critic-tuning brief
      plateaued under at 0.632 m/s.
- [ ] On a sim mission with one obstacle in the path, the robot
      reaches the goal *without colliding*. This is the safety win
      that NOCAM in `strafer_direct` cannot deliver and the reason
      DEPTH is the MVP variant.

## Approach

Three independent runs — each closes one acceptance group and each
can land as its own PR (or fold all three into one). Recommend
filing them in order; the dependency chain (rig → bag → trained
artifact) only flows one way.

### A. Parity runs (rig only, no checkpoint required)

1. Stand up `bringup_sim_in_the_loop.launch.py` on the Jetson with
   the DGX bridge running and producing `/clock`, `/d555/*`,
   `/strafer/*`. Confirm via `ros2 topic hz`.
2. Launch `strafer_inference` with `model_path=""` (action server
   stays unadvertised; the tick still publishes zero twist when the
   watchdog is clean but no inference fires).
3. On the DGX side, instrument the gym env to dump the assembled
   NOCAM obs vector (and the native-80×60 depth obs) per step
   alongside the sim timestamp. JSONL or numpy-npz works.
4. On the Jetson side, modify the inference node's debug log path
   (or add a temporary `--dump-obs-to-jsonl <path>` argument) to
   emit the assembled obs vector per tick alongside the sim
   timestamp.
5. Record a `bringup_sim_in_the_loop`-driven rosbag for ≥ 30 s with
   the robot moving (joystick or a scripted mission). Make sure the
   bag captures `/d555/depth/image_rect_raw`, `/d555/imu/filtered`,
   `/strafer/joint_states`, `/strafer/odom`, `/strafer/goal`, and
   `/clock`.
6. Replay the bag and run a comparison script that joins the two
   obs streams on sim timestamp and asserts the bounds. Land the
   comparison script under
   `source/strafer_ros/strafer_inference/scripts/` (or
   `Scripts/` if it ends up cross-lane).

### B. Latency benchmark (needs a real ONNX artifact)

1. Export a DEPTH-shaped ONNX stub via the export tooling (any
   network with the 4819-dim → 3-dim shape works for latency; the
   tape-out trained checkpoint plugs in identically once it lands).
2. Launch `strafer_inference` with `model_path` pointing at the
   stub; confirm the TRT engine cold-start log line fires and the
   `ready` parameter flips after ~30 s.
3. Run a benchmark harness for ≥ 60 s of sustained inference. Two
   acceptable shapes: extend
   [`tune_capture.py`](../../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
   to capture inference-side rx → publish timestamps via /tf or a
   sidecar topic, OR ship a new
   `source/strafer_ros/strafer_inference/scripts/benchmark_inference_node.py`
   that subscribes to the cmd_vel publish and pairs against the obs
   receive times. Report p50 / p95 / p99.
4. Re-run with `onnx_providers=["CPUExecutionProvider"]` for the
   non-gating CPU baseline; the gap to TRT is the operator-facing
   "did you boot with TRT" signal.

### C. End-to-end mission (gates on the trained checkpoint)

1. Operator exports a trained DEPTH checkpoint via
   `Scripts/export_policy.py --variant DEPTH --checkpoint <ckpt> --output models/strafer_depth_v0.onnx`
   (DGX-side). Rsync to the Jetson model path.
2. Re-confirm the deterministic-mean export with a same-obs +
   reset() round-trip on the real artifact (already covered by the
   in-tree `test_recurrent_contract_e2e.py` cross-format parity
   test; this is the operator-side re-confirmation against the
   actual export).
3. Run a `translate forward 3 m` sim mission with
   `STRAFER_NAV_BACKEND=strafer_direct` in a ProcRoom-like
   pre-warmed scene; record `/strafer/odom`. Verify the 1 s
   sustained median ≥ 1.0 m/s.
4. Run a mission with a single obstacle in the path; verify
   reach-without-collision via the bridge's collision sensor or by
   inspecting the trajectory in the rosbag.

## Investigation pointers

- [`inference-package.md`](inference-package.md) — the source brief.
  All implementation lives in PR #55; this brief validates that
  implementation in sim.
- [`source/strafer_ros/strafer_inference/`](../../../../source/strafer_ros/strafer_inference/)
  — the package this brief validates.
- [`source/strafer_lab/strafer_lab/sim_in_the_loop/`](../../../../source/strafer_lab/strafer_lab/sim_in_the_loop/)
  — DGX-side bridge / harness; the `bringup_sim_in_the_loop` launch
  consumes its published topics.
- [`source/strafer_ros/strafer_navigation/scripts/tune_capture.py`](../../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
  — example of an MPPI-side timing harness; the latency benchmark
  can extend it or mirror its structure.
- [`completed/mppi-critic-tuning-for-sim-envelope.md`](../../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the predecessor brief that established the 0.632 m/s plateau the
  architectural-win acceptance compares against.

## Out of scope

- **Real-robot DEPTH validation.** File as a separate brief
  (`strafer-inference-real-robot-validation.md`) once this brief's
  sim validation passes. Real-robot introduces sensor-noise
  distribution shift, lighting variance, and dynamic obstacles that
  warrant their own scope and acceptance criteria.
- **Re-tuning the policy if validation surfaces a gap.** This brief
  is pass/fail against the bounds; tuning is a separate DGX-lane
  training brief if the bounds don't hold.
- **Sim-in-the-loop rig setup.** Already in flight upstream; if the
  rig is broken when this brief is picked up, file the breakage as
  its own follow-up rather than fixing it inline here.
