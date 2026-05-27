# Windows workstation sim-bridge (deferred path; unblock when NVIDIA-Vulkan-on-WSL2 is fixed)

**Type:** investigation + task
**Owner:** DGX agent (lane: `source/strafer_lab/`, `Scripts/`, `docs/INTEGRATION_WINDOWS_WORKSTATION.md`)
**Priority:** P3 (lower than the data-collection path that ships in [`windows-workstation-bringup.md`](windows-workstation-bringup.md); upgrade to P2 once a viable Vulkan path is identified)
**Estimate:** M (~2-3 days; mostly waiting on NVIDIA driver fix, then perf capture + cross-host validation)
**Branch:** task/windows-workstation-bringup-sim-bridge

## Story

As a **DGX/sim developer with a Windows workstation already serving data collection via the native Path A install** (see [`docs/INTEGRATION_WINDOWS_WORKSTATION.md`](../../../INTEGRATION_WINDOWS_WORKSTATION.md)), I want **`make sim-bridge` to run end-to-end against the Jetson autonomy stack from the Windows host's WSL2 distro**, so that **bridge iteration moves off the DGX onto the gaming-class GPU and the DGX is freed up for training**.

## Context

The parent brief [`windows-workstation-bringup.md`](windows-workstation-bringup.md) established the WSL2 Ubuntu-22.04 install as the only architecturally viable home for the bridge on a Windows host (CycloneDDS is Linux-only per NVIDIA's docs; sim-bridge can't run native Windows). Phase 2 of that work proved the WSL2 stack installs cleanly end-to-end — CUDA passthrough, conda + isaacsim + isaaclab + rsl_rl, mirrored networking, env_setup.sh, all strafer_lab imports — **except for Kit's renderer**.

The renderer error is `vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.` Per [NVIDIA's developer forum](https://forums.developer.nvidia.com/t/critical-bug-rtx-5070-ti-mobile-missing-nvidia-icd-json-in-wsl2-causes-vulkan-initialization-failure-on-driver-591-86/359933) (March 2026) the Linux NVIDIA Vulkan ICD is not present in the CUDA-on-WSL2 shim at driver 591.86+. The bringup spike on the reference workstation confirmed this against driver 610.47: `/etc/vulkan/icd.d/` has no `nvidia_icd.json`; `/usr/lib/wsl/lib/` ships D3D12 + CUDA shim libs but no Vulkan producer; `vulkaninfo` only sees `llvmpipe`. Manual ICD JSON pointing at the new D3D12 lib (`libnvwgf2umx.so`) does not expose a Vulkan device; Mesa Dozen (Vulkan-over-D3D12) is not in Ubuntu 22.04's mesa-vulkan-drivers 23.2.

So this brief is **gated on external resolution** of one of:

1. NVIDIA ships a Windows driver build that includes the Linux Vulkan ICD in the WSL CUDA shim. (Most likely fix; watch driver-release notes.)
2. NVIDIA publishes `libnvidia-gl-<branch>` in their CUDA-on-WSL apt repo (this historically delivered the ICD JSON to the right path).
3. A community Vulkan-over-D3D12 path (Mesa Dozen on a newer Ubuntu mesa package) becomes mature enough to drive Isaac Sim.

When any of those land, this brief picks up.

## Approach

Sequence the work so the first two steps fail-fast if Vulkan is still broken:

### Phase 1 — Unblock verification (½ day)

- `wsl --shutdown` then re-launch Ubuntu-22.04. Check `/etc/vulkan/icd.d/` for `nvidia_icd.json`. Check `/usr/lib/wsl/lib/` for new Vulkan-related shim libs (`libnvidia-vulkan-producer*`, `libGLX_nvidia*`).
- `vulkaninfo --summary` should now list the RTX 4080 as a physical device, not `llvmpipe`.
- If still broken: stop, document the new driver version's gap, file as a follow-up. If unblocked: proceed.

### Phase 2 — Bridge smoke (½ day)

- From WSL2: `cd ~/Workspace/Sim2RealLab && source env_setup.sh && make sim-bridge`. Verify Kit boots through renderer init this time.
- From a second WSL2 terminal: `ros2 topic list` shows `/cmd_vel`, `/d555/*`, `/strafer/odom`, `/tf`, `/clock`.
- `ros2 topic hz /d555/color/image_raw` shows ~30 Hz.

### Phase 3 — Cross-host validation (½ day)

- Jetson up on LAN. From the Jetson, `ros2 topic list` should see the WSL2-published bridge topics (mirrored networking + Cyclone discovery).
- `make launch-sim` on the Jetson + manual mission via `strafer-autonomy-cli submit "translate forward 3 m"` → simulated robot advances in Kit, mission returns `final_state: succeeded`.

### Phase 4 — Perf capture (½ day)

- `$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py --mode bridge --headless --enable_cameras --profile --profile-interval 10 --profile-window 200`
- Capture per-phase p50/p99: `cmd_vel read`, `env.step (total)`, `publish_state`, `sim.render`, `simulation_app.update`, `camera :: GPU→CPU readback`, `camera :: rclpy publish`.
- Commit numbers to [`docs/tasks/context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md) under a new "WSL2 on RTX 4080" row alongside the DGX reference rows.

### Phase 5 — Runbook flip (1 day)

- [`docs/INTEGRATION_WINDOWS_WORKSTATION.md`](../../../INTEGRATION_WINDOWS_WORKSTATION.md): move the Vulkan gotcha out of Path B's "known gotchas" and into a "resolved (was: ...)" footnote. Update the use-case routing table to mark `make sim-bridge` as live.
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md): add a "DGX or Windows-via-WSL2" line to Stage 2's host topology.

## Acceptance criteria

- [ ] `make sim-bridge` runs end-to-end on the Windows workstation against the same Jetson stack used in the existing Linux flow.
- [ ] A `translate forward 3 m` mission completes from the Jetson side against the Windows bridge without code changes to the Jetson lane.
- [ ] Bridge perf numbers on Windows are committed to a new section of `bridge-runtime-invariants.md`.
- [ ] [`docs/INTEGRATION_WINDOWS_WORKSTATION.md`](../../../INTEGRATION_WINDOWS_WORKSTATION.md) Path B section flips the Vulkan gotcha to a "resolved on driver X.Y" footnote and the use-case-routing table marks `make sim-bridge` as live.
- [ ] The Linux DGX path continues to work byte-identically — verify by running the smoke missions on DGX before merging.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Out of scope

- **The data-collection / harness / teleop use case** — that's Path A in the runbook, shipped in [`windows-workstation-bringup.md`](windows-workstation-bringup.md). This brief is bridge-only.
- **Building a FastDDS↔CycloneDDS translator** as a Vulkan-unblock workaround. Considered and rejected during the parent brief's Phase 1 — adds latency that would invalidate perf reference numbers. Only revisit if NVIDIA never fixes WSL2-Vulkan and the bridge-on-Windows demand outweighs the perf hit.
- **Dual-booting the Windows workstation into native Ubuntu.** NVIDIA's forum recommends this for Vulkan reliability, but it costs the data-collection use case that Path A serves well, plus reboot friction.

## Investigation pointers

- NVIDIA CUDA-on-WSL user guide (driver-release-dependent): <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- NVIDIA developer forum bug thread (March 2026): <https://forums.developer.nvidia.com/t/critical-bug-rtx-5070-ti-mobile-missing-nvidia-icd-json-in-wsl2-causes-vulkan-initialization-failure-on-driver-591-86/359933>
- Mesa Dozen status / Ubuntu 22.04 mesa version vs Dozen support window
- Parent brief's decision log: [`windows-workstation-bringup.md`](windows-workstation-bringup.md) Phase 2 captured the exact failure mode and tried manual ICD + Dozen workarounds (both failed on driver 610.47 + mesa 23.2).
