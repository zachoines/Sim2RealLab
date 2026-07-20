# Bring up the Orin NX + containerize `strafer_ros` for skip-0 depth-policy deployment

**Type:** task / platform migration + deployment infra
**Owner:** Jetson (new Orin NX 16GB `strafer-nx`; replaces the Orin Nano)
**Priority:** P1 — unblocks the depth-policy A/B by giving the robot lane a device that runs the policy at **frame-skip 0**; the Nano's compute ceiling forced skip 3 and made the sim-bridge loop the bottleneck. Carries the skip-0 capacity gate.
**Estimate:** L (multi-session; hardware bringup + driver build + two-image containerization). Bulk shipped in this PR; the live acceptance is operator-gated.
**Branch:** `docker/strafer-ros-containerization` (deviates from the `task/<name>` convention — noted; not worth a rename mid-review).
**Status:** In-flight (PR #157). The containerization + NX bringup shipped here; the brief stays **active** for the open **skip-0 e2e** acceptance (operator-run, gated on the DGX sim host) and the deferred hardware lanes.

## Story

As **the robot lane, which needs to run the trained depth policy at the sim
control rate**, I want **the `strafer_ros` stack containerized and running on a
Jetson Orin NX that has the GPU/TRT headroom for frame-skip 0**, so that **the
depth-policy A/B (bare-metal Jetson vs DGX bridge) has a robot device that isn't
the bottleneck, and spinning up the next Jetson is a ~10-minute provision instead
of a bespoke bare-metal setup.**

## Context bundle

- [`ort-gpu-jetson`](../../completed/ort-gpu-jetson.md) — measured TRT inference (~4.7 ms in `load_policy`); the skip-0 budget headroom argument rests on it. This PR's `strafer-gpu` image reproduces that provider stack (`onnxruntime-gpu` jp6/cu126, TRT + CUDA EPs verified).
- [`depth-subgoal-hybrid-runtime`](depth-subgoal-hybrid-runtime.md) — the hybrid runtime this deployment must launch correctly (inference node **+** rolling-subgoal generator). The compose `policy` profile reproduces that coupling.
- `source/strafer_ros/deploy/` — the scaffolding this brief ships (two Dockerfiles, `docker-compose.{sim,,dev}.yml`, `host-setup/`, env mirrors + `tests/check_env_sync.py`).
- `docs/D555_IMU_KERNEL_FIX.md` — the D555 IMU hid-sensor-hub + IIO kernel-module rebuild (host-side; deferred hardware lane below).

## What shipped (this PR)

- **NX bringup:** JetPack 6.2 / L4T r36.4.3 confirmed, MAXN_SUPER; on-device Docker + nvidia default runtime + compose. WiFi via a from-source-built Netgear A8000 `mt7921u` driver (DKMS, reboot-validated) — recorded in the operator's device notes, out of this brief's touch scope.
- **Two images, decomposed over CycloneDDS** (kept — not FastDDS): `strafer-cpu` (base/perception/slam/navigation/autonomy) + `strafer-gpu` (inference, TRT/CUDA providers verified). Both built + validated on-device.
- **Compose lanes:** sim-in-the-loop (nav2 backend, no GPU/hardware — full stack launches in-container), full deploy (5 services + `policy`/`remote` profiles), dev bind-mount overlay. `build:` sections build both images.
- **Hybrid coupling reproduced** in `inference_policy.launch.py` (subgoal generator under `hybrid_nav2_strafer`) with a **fail-loud** guard (policy backend + empty/missing model = hard error, no silent nav2 fallback).
- **Canon-drift guards:** cyclonedds.xml + rmem.conf bind the canonical files (no byte-copies); the env mirrors are pinned by `tests/check_env_sync.py`.
- Six Jetson-Docker gotchas fixed in-repo (host-net build for the missing `iptable_raw`, setuptools≥64, cyclonedds RMW, `numpy<2`, `.dockerignore` cache, compose-v2).

## Acceptance criteria

- [x] Both images build on-device and the **sim-in-the-loop** stack launches in a container over CycloneDDS (nodes init; executor waits on VLM/planner as designed).
- [x] `strafer-gpu`: `onnxruntime.get_available_providers()` lists `TensorrtExecutionProvider` + `CUDAExecutionProvider` in the baked image; `strafer_inference` imports.
- [x] `docker compose config` valid for `docker-compose.yml` (+ dev overlay); `check_env_sync.py` green.
- [ ] **Containerized hybrid bringup smoke** — `--profile policy` with `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` + a DEPTH/DEPTH_SUBGOAL stub model: inference node **and** subgoal generator come up; the inference watchdog's `subgoal` source goes fresh (no zero-twist from a missing generator). Operator-run.
- [ ] **NX skip-0 e2e mission (the migration's acceptance).** With the DGX sim bridge up (`192.168.50.196`: bridge + VLM :8100 + planner :8200, `ROS_DOMAIN_ID=42`, wired-LAN CycloneDDS), run a hybrid depth-policy mission at **frame-skip 0** and confirm the obs→`cmd_vel` loop holds the 33 ms budget end-to-end. **Gated on the DGX being cleared.** This is also the **precondition for the relocated depth-policy A/B session.**

## Nano-parity gate — recommend **RETIRE**

The migration's original plan carried a "Nano-parity" gate (prove the NX
reproduces the Nano's behavior before trusting it). **Recommend retiring it**
rather than silently dropping it:

- On-device NX validation (sim-in-the-loop launch + the skip-0 e2e above)
  **supersedes** cross-device parity — it exercises the exact deployed images on
  the exact target, which parity-to-an-inferior-device does not.
- The Nano ran at **skip 3** (its ceiling); "parity" to a skip-3 baseline is the
  wrong bar for a device whose whole purpose is skip 0.
- **Rollback is intact and cheap:** the bare-metal Nano is untouched at skip 3, so
  a regression on the NX falls back to the known-good Nano with no un-migration
  work. That safety net is what a parity gate would otherwise provide.

Operator decision requested: confirm RETIRE (or override with the specific parity
signal wanted).

## Hardware lanes (deferred — open, host-side)

- [ ] **RoboClaw passthrough** — `base` service device-cgroup + `99-strafer.rules` (installed by `host-setup`) → `/dev/roboclaw{0,1}`; validate the driver auto-detects front/rear once the two controllers are connected.
- [ ] **RealSense D555 passthrough** — USB (`/dev` + cgroup 81/189) or Ethernet-DDS; decide transport (see Out of scope) and validate the `perception` service sees the camera.
- [ ] **D555 IMU kernel modules** — rebuild the hid-sensor-hub + IIO modules against the NX kernel (host-side, per `docs/D555_IMU_KERNEL_FIX.md`); without filtered IMU, DEPTH/NOCAM obs assembly returns `None`. Kernel modules can't live in a container.

## Out of scope

- **The depth-policy A/B session itself.** This brief delivers its robot-side precondition (a skip-0-capable containerized NX); the A/B is its own session, gated on the skip-0 e2e above + the DGX.
- **Ethernet-vs-USB D555 transport decision.** Filed with the hardware lane when the camera is connected; changes only the `perception` service block.
- **WiFi driver maintenance.** The A8000 `mt7921u` build + DKMS persistence is done and recorded in the operator's device notes; not a `strafer_ros` artifact.

## Follow-ups (file when picked up)

- **Cyclone interface pinning** (`General/Interfaces` in `cyclonedds.xml`): now that docker0-class interfaces exist on the host, Cyclone can pick the wrong NIC for cross-host discovery. Pre-existing latent risk, not a PR regression — pin the robot's real interface before the cross-host skip-0 e2e run.
- **Zenoh tag pin** is `${ZENOH_TAG:-1.5.1}`; confirm it matches the workstation bridge version when remote access is first used.
