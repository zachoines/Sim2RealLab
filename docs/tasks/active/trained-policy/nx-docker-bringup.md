# Bring up the Orin NX + containerize `strafer_ros` for skip-0 depth-policy deployment

**Type:** task / platform migration + deployment infra
**Owner:** Jetson (new Orin NX 16GB `strafer-nx`; replaces the Orin Nano)
**Priority:** P1 — unblocks the depth-policy A/B by giving the robot lane a device that runs the policy at **frame-skip 0**; the Nano's compute ceiling forced skip 3 and made the sim-bridge loop the bottleneck. Carries the skip-0 capacity gate.
**Estimate:** L (multi-session; hardware bringup + driver build + two-image containerization). Bulk shipped in this PR; the live acceptance is operator-gated.
**Branch:** `docker/strafer-ros-containerization` (deviates from the `task/<name>` convention — noted; not worth a rename mid-review).
**Status:** e2e gates **PASS** (PR #157; run 2026-07-23 on `strafer-nx`). Both acceptance gates below are green — the containerized hybrid smoke and the skip-0 ProcRoom mission. Making the containerized deploy self-sufficient against the DGX bridge required fixing **four in-branch gaps** the rev-3 runbook's Gate-2 topology had missed (see **What the e2e surfaced** below). Brief stays **active** only for the deferred hardware lanes + the goal-completion/park follow-up. **Pending human merge/review** — the Gate-2 topology changed materially.

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
- [x] **Containerized hybrid bringup smoke** — `--profile policy`, `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` + the v1 **DEPTH_SUBGOAL** model: **inference node AND rolling-subgoal generator both up**, fail-loud guard clean (real backend + model), overlay obs/depth timeouts applied. TRT engine builds at session init — **cold ~84 s / warm ~4 s** (cached to the `strafer_trt_engines` volume).
- [x] **NX skip-0 e2e mission (the migration's acceptance).** DGX bridge up (ProcRoom, `--decimation 4 --render-interval 4` → derived **skip 0 / 30 Hz sim**; VLM :8100 + planner :8200, domain 42, wired-LAN CycloneDDS). Hybrid DEPTH_SUBGOAL mission at **frame-skip 0**: **trip-free steady-state** (2 startup `subgoal` trips before the first plan, **0 mid-mission across 242 s**), obs→`cmd_vel` loop held, **warm first-inference 102 ms**. The **semantic VLM→policy** path was also exercised end-to-end (grounded "beige square" → `goal_projection` → policy drove to it). One open item: the policy **parks near the goal but doesn't trip `NavigateToPose`'s success radius**, so the action times out instead of formally completing (follow-up below). Clears the **precondition for the relocated depth-policy A/B session.**

## What the e2e surfaced — fixed in-branch

Running the gates end-to-end showed the containerized deploy lane was **not
self-sufficient against the DGX bridge** — it only appeared to work because a
stale sim-in-the-loop container was silently supplying the missing nodes. Four
gaps, all fixed on this branch:

1. **`use_sim_time` missing from the sim-bridge overlay** (`51e42cf`). The bridge
   publishes `/clock` + sim-stamped TF/sensors; the deploy lane defaulted
   `use_sim_time=false`, so rtabmap stamped `map→odom` in wall time while the
   bridge's `odom→base_link` was sim time → split TF tree, nav2 couldn't plan.
2. **Missing SLAM support nodes** (`f34e96d`). The Gate-2 topology
   (`inference slam navigation`) drops `base`+`perception` for their crash-looping
   hardware drivers, but each also carries a sim-needed support node —
   `robot_state_publisher` (`base_link→d555_link` TF) and `timestamp_fixer` (the
   `/d555/*_sync` topics rtabmap consumes, with sim depth remaps). Added back as an
   opt-in `sim-perception` compose service (`profile: sim-bridge`) +
   `sim_bridge_support.launch.py`, mirroring `bringup_sim_in_the_loop`.
3. **`make submit`/`submit-deploy` didn't source ROS** (`1889ce5`) — they ran the
   CLI via `docker compose exec`, which bypasses the entrypoint → `rclpy` import
   error. Now source the workspace in the exec.
4. **`strafer-cpu` lacked Pillow** (`1889ce5`) — the executor's VLM grounding
   encodes camera frames with PIL, but `pillow` wasn't a declared `strafer_autonomy`
   dependency (only the `[planner]` extra pulled heavy deps). Added to base deps.

**Follow-ups filed (not blocking this brief):** policy goal-completion — parks near
the goal without tripping `NavigateToPose`'s success radius; the mission-runner's
per-step nav timeout is measured on **wall-clock**, so a planner-emitted budget
(e.g. 7 s) starves the policy under the sim's sub-unity RTF (→ `task/hybrid-nav-sim-clock`);
DGX `/clock` hitches can crash rtabmap (`docker restart strafer_slam` recovers).

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
