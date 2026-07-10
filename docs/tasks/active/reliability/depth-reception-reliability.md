# Make the inference node's depth reception survive load

**Type:** task / bug (BLOCKER)
**Owner:** Jetson agent (`source/strafer_ros/strafer_inference/`, `source/strafer_ros/strafer_bringup/`)
**Priority:** P1
**Estimate:** M — code + unit shipped; the rig repro-then-fix and clean-mission re-run are operator-gated (see Acceptance)
**Branch:** task/depth-reception-reliability

## Story

As a **mission operator running a depth-variant policy under full bringup**, I want **the inference node to keep receiving depth frames while the CPU/memory is contended (bringup + TensorRT engine build)**, so that **the node stops tripping `stale=['depth']` → zero-twist → mission abort while depth is still flowing on the bus**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

**The fault (measured on the rig).** ~921 KB fragmented RELIABLE depth samples were dropped at the node's own DDS receive layer under CPU/memory contention — depth flowed on the bus (2.69 Hz) and reached other subscribers, while the node's `_on_depth` never fired → `stale=['depth']` every tick → zero-twist → mission abort. Load-correlated: total during full bringup + TRT build, intermittent when calm. Small topics in the same callback group stayed fresh. Reproduced at 4× fragment pressure (corrected-cadence ProcRoom run, `--camera-frame-skip 0`): the fault hit multiple subscribers at once — the inference node tripping `stale=['depth']` repeatedly, `point_cloud_xyz` seeing 0 Image vs 11 CameraInfo in 10 s, rtabmap 5 s no-data. The Image-lost / CameraInfo-received asymmetry is diagnostic: the DGX async publisher pushes Image+CameraInfo together, so the publish happened and the large Image was lost in transit/receive, not a publisher outage.

Lineage: [completed/bridge-procroom-scene-ab.md](../../completed/bridge-procroom-scene-ab.md) (the ProcRoom A/B rig this was measured on). The `PROCROOM_AB_FINDINGS.md` coordinator addendum and the depth-probe script are session artifacts, not checked in.

**The fix attacks three different links in the chain.**

### Layer 1 — QoS: depth subscription → SENSOR_DATA profile

Changed the depth subscription from default RELIABLE (`depth=10`) to **BEST_EFFORT, KEEP_LAST, depth=1** (`_DEPTH_QOS` in `inference_node.py`). Reliable retransmission is exactly wrong for this stream: the freshness gate only ever wants the *newest* frame, so a lost frame's correct disposition is "skip this tick" (designed-for behavior), not a retransmit burst that amplifies congestion at the worst moment.

**Real-driver QoS check (requested by the brief).** The real D555 publishes depth via `realsense2_camera` (`perception.launch.py` includes `rs_launch.py`); `config/d555_params.yaml` sets **no** `depth_qos` override. Honest result (verified against realsense-ros source — `constants.h` / `profile_manager.cpp` / `ros_utils.cpp`, identical at 4.51.1 and 4.x HEAD): the wrapper defaults its image/depth streams to **`SYSTEM_DEFAULT`** (`IMAGE_QOS = "SYSTEM_DEFAULT"` → `rmw_qos_profile_system_default` → **RELIABLE** on the standard Humble RMWs incl. Cyclone), so the prior RELIABLE sub **was** compatible with the real driver's default publisher. `SENSOR_DATA`/BEST_EFFORT is the wrapper default only for the **IMU/HID** streams (the likely source of the original "best-effort" assumption). BEST_EFFORT is still the right choice: it receives from the RELIABLE default (sim bridge `async_camera_publisher._IMAGE_QOS`, and the real driver) **and** stays compatible with a future `depth_qos:=SENSOR_DATA` (the wrapper's recommended sensor-profile override) — the congestion/no-retransmit rationale stands on its own.

**Real-driver landmine the QoS check surfaced (goal-b, NOT this PR).** realsense-ros publishes depth as **16UC1 millimeters** (`RS2_FORMAT_Z16 → TYPE_16UC1`), with no 32FC1 option in the 4.x line — and `_on_depth` hard-drops any non-32FC1 frame. On real hardware every driver depth frame would be rejected at the encoding gate (depth permanently stale) *before* QoS matters, and the deploy obs pipeline expects float meters. This belongs to the goal-b conformance list: convert downstream (`depth_image_proc/convert_metric`) or add a 16UC1→meters path in `_on_depth`, decided at goal-b bringup alongside the wheel-sign and madgwick-chain QoS audit. Do **not** fix here.

### Layer 2 — Concurrency: dedicated depth callback group + the thread-safety it forces

`_on_depth` moved to its own `MutuallyExclusiveCallbackGroup` and the `MultiThreadedExecutor` went 2→3 threads, so the heavy ~921 KB deserialize/take runs on a separate executor thread instead of contending for the tick's single serialized slot.

**The trap this opens, and how it's closed.** Today's code was race-free *because* every subscriber + the tick shared one mutex group. Moving depth out makes `_on_depth` (writes `_last_depth_meters`, `_last_depth_rx_t`, `_depth_seq`) concurrent with `_on_tick` (reads all three + the gate + obs assembly). The danger is tearing: a fresh seq paired with a stale array = inference on the wrong frame, silently — which re-opens the exact stale-frame hole the depth-freshness gate closed. Closed by:
- a lock (`_depth_lock`) around the `(array, rx_t, seq)` triple write in `_on_depth`;
- one locked snapshot at the top of each tick (`_tick_depth_meters` + `depth_rx_t` + `depth_seq` locals) that the watchdog, the gate, **and** obs assembly all read — so they see one coherent frame;
- the consume records the **snapshot** seq (the frame obs was built from), not the live counter a mid-tick `_on_depth` may have advanced.

The array is *replaced* (never mutated) by `_on_depth`, so the snapshotted ref stays valid without a copy. `_last_inferred_depth_seq` is written and read only by the tick, so it needs no lock. The threading contract is documented in the node's module docstring (updated where it documents the freshness gate's single-thread assumption).

### Layer 3 — Transport: Cyclone DDS receive tuning

`CYCLONEDDS_URI` was unset (kernel-default socket buffers). Added a checked-in `config/cyclonedds.xml` raising the receive socket buffer (`SocketReceiveBufferSize max="16MB"`) + fragment-reassembly headroom (`DefragUnreliableMaxSamples`/`DefragReliableMaxSamples` = 32), exported via both env files (`env_sim_in_the_loop.env` and `env_autonomy.env` — the real robot moves the same 921 KB frames). Each setting is commented with the mechanism it addresses. The config is self-locating (`CYCLONEDDS_URI` uses `BASH_SOURCE`), so it works from any checkout path, and takes effect only under `rmw_cyclonedds_cpp` (both lanes use it).

**Discovered while validating (material, encoded in the config + a checked-in sysctl):**
- `SocketReceiveBufferSize` is clamped by the OS to `net.core.rmem_max`, which is ~208 KB on the Jetson. Without raising it, Cyclone requests 16 MB but keeps the tiny default and this layer is **inert**. Shipped `config/99-cyclonedds-rmem.conf` (a `/etc/sysctl.d/` drop-in setting `rmem_max=16777216`) plus a one-line install step in the config/env comments.
- An unmet `min` on `SocketReceiveBufferSize` is **fatal** — Cyclone refuses to create the domain and every node fails to start. So the config uses `max` only (request-and-clamp), never a hard `min`, or a too-low kernel cap would brick an untuned host. Validated: the config loads cleanly on this Jetson (rmem_max=208 KB) with `max`-only; it aborted with `min="4MB"`.

### Also carried (docs only, no behavior change)

`env_autonomy.env` gained a warning **not** to set `STRAFER_DEPTH_TIMEOUT_S` on real hardware: in sim the bridge watchdog's 0.5 sim-s ≈ 5+ wall-s at the rig's low RTF, so the sim-only 2.0 s override is meaningless in wall time; on the real robot the node's depth watchdog and the downstream cmd-vel watchdog both sit at 0.5 wall-s, and overriding it opens a (0.5, 2.0] s window where the node drives on depth the downstream watchdog already treats as stale.

## Acceptance criteria

- [x] Depth subscription is BEST_EFFORT / KEEP_LAST / small depth, pinned by a test.
- [x] `_on_depth` on its own callback group; executor threads 2→3; depth triple lock-guarded; tick reads one coherent snapshot.
- [x] Thread-safety unit tests: a deterministic anti-tear test (a frame landing mid-tick must not tear the snapshot — mutation-verified it fails on the regressed consume) + a concurrent writer/reader harness.
- [x] `config/cyclonedds.xml` + `config/99-cyclonedds-rmem.conf` checked in and installed; `CYCLONEDDS_URI` exported from both env files; config validated to load under `rmw_cyclonedds_cpp`.
- [x] Real-driver depth QoS checked and recorded (above): realsense2_camera image/depth default is `SYSTEM_DEFAULT` → reliable (only IMU/HID defaults to best-effort), so the old RELIABLE sub was compatible; BEST_EFFORT is chosen for congestion behavior + forward-compat with a `depth_qos:=SENSOR_DATA` override. The real encoding landmine (16UC1 vs the node's 32FC1-only gate) is recorded for goal-b.
- [x] `env_autonomy.env` carries the `STRAFER_DEPTH_TIMEOUT_S` clock-domain warning.
- [x] `colcon`/pytest green for `strafer_inference` (counts in the PR body).
- [ ] **Rig repro-then-fix [operator-assisted].** On `main`, synthetic CPU load during bringup reproduces `stale=['depth']`; on this branch the same load profile shows depth received (count `_on_depth` firings vs bus frames via the depth-probe script). **Measure the bus first** (protocol below). Acceptance: **zero `stale=['depth']` across a full bringup + complete mission under load**, and the parity cadence histogram shows the regular depth spike.
- [ ] **Clean v1 mission re-run [operator-assisted].** The re-flown clean v1 mission closes the loop and is the drive-quality observation the training-lane routing is holding for. Stays open on this brief after merge.
- [ ] Raise `net.core.rmem_max` on both hosts (`config/99-cyclonedds-rmem.conf` → `/etc/sysctl.d/`) so Layer 3 is not inert.

## Bus-first measurement protocol (operator, before attributing loss to the receive layer)

`ros2 topic hz` the raw depth topic and compare against the bridge's sim-step rate:
- **publish-deficit** (the DGX `async_camera_publisher` drops frames when its single-slot worker is busy — at `--camera-frame-skip 0` it may self-throttle) = DGX capacity, **out of scope here — REPORT it back with numbers, do not fix in this PR**;
- **receive-drop** (depth on the bus at the expected rate but `_on_depth` misses it) = this PR's scope.

Secondary (not a gate): record rtabmap / `point_cloud_xyz` input freshness before/after — they should improve from the deployment-wide `cyclonedds.xml` layer.

## Out of scope

- The freshness gate's semantics (owned by the depth-fresh-tick-gate work), watchdog budgets, the parity tooling surfaces, anything DGX-side.
- The DGX `async_camera_publisher` publish-deficit — measure and report, file a DGX follow-up if the bus rate is short; do not fix here.

## Investigation pointers

- [`source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py) — `_DEPTH_QOS`, `_depth_cb_group`, `_depth_lock`, the `_on_tick` snapshot, `main()` thread count, the module docstring threading contract.
- [`source/strafer_ros/strafer_bringup/config/cyclonedds.xml`](../../../../source/strafer_ros/strafer_bringup/config/cyclonedds.xml) and [`config/99-cyclonedds-rmem.conf`](../../../../source/strafer_ros/strafer_bringup/config/99-cyclonedds-rmem.conf).
- [`source/strafer_ros/strafer_perception/config/d555_params.yaml`](../../../../source/strafer_ros/strafer_perception/config/d555_params.yaml) — no `depth_qos` override → real driver keeps the wrapper image default (`SYSTEM_DEFAULT` → reliable); see the Layer 1 real-driver check for the 16UC1 encoding landmine.
