# Windows workstation bringup for sim-bridge

**Type:** investigation + task
**Owner:** DGX agent (lane: `source/strafer_lab/`, `Scripts/`, `env_setup.sh`, top-level `Makefile`, `docs/INTEGRATION_*.md`)
**Priority:** P2
**Estimate:** L (~1 week; multi-day research + multi-host install scripting + cross-OS network testing)
**Branch:** task/windows-workstation-bringup

## Story

As a **DGX/sim developer with an i9-14900K + RTX 4080 Windows workstation**, I want **to run `make sim-bridge` and `make sim-bridge-gui` on Windows against the same Jetson autonomy stack the Linux DGX serves today**, so that **the bridge can run on a much faster gaming-class GPU for iteration (the DGX is more valuable for training large models with memory headroom) without forking the autonomy code or the sim assets**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md) — hosts, repo paths, the conda env contract.
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md) — bridge mainloop assumptions; some likely don't port cleanly to Windows.

## Context

The current stack assumes Ubuntu on both hosts (DGX Spark and Jetson Orin Nano). The DGX side runs Isaac Sim 6 + Isaac Lab 3 under conda env `env_isaaclab3` on Linux; `Scripts/`, `env_setup.sh`, and the `Makefile` all assume bash + a Linux conda layout.

Windows complicates this on several axes:

1. **Isaac Sim / Isaac Lab support.** Isaac Sim 6 has official Windows support; Isaac Lab 3's official support matrix lists Linux only, with "experimental" Windows usage reported by community. Need to investigate the actual install path (Omniverse Launcher install on Windows vs. extension-based install vs. WSL2).

2. **Toolchain.** `env_setup.sh` assumes bash + Linux conda. `Makefile` targets use POSIX features. Need either:
   - A PowerShell variant (`env_setup.ps1`, `Makefile.ps1` or `tasks.ps1`) that mirrors the Linux contract.
   - A WSL2-based bringup that runs the existing scripts inside Ubuntu while accessing Windows GPU via WSLg / CUDA on WSL2.

3. **ROS 2 cross-host.** rmw_cyclonedds_cpp + ROS 2 Humble on Windows is supported but less battle-tested than on Linux. Need to confirm cross-host discovery (Windows ↔ Jetson Linux on the same LAN, ROS_DOMAIN_ID=42).

4. **Bridge perf.** RTX 4080 has more raster + RT throughput than the DGX's iGPU, but Isaac Sim's perf on Windows-with-RTX has its own quirks. Bridge-runtime-invariants doc reference numbers were captured on Linux DGX — re-derive for Windows.

5. **Asset / repo sharing.** Both hosts share one git repo via separate clones; Windows clones need to honor LF-only line endings for the scripts and the `Scripts/` entry points. Isaac Sim USD asset paths are case-sensitive — need to verify generated `Assets/generated/scenes/` paths resolve on Windows.

## Approach

Phase the work so the early phases produce decision-grade evidence before committing to a full port:

### Phase 1 — Feasibility spike (1–2 days, investigation only)

- Stand up Isaac Sim 6 on the Windows workstation via the Omniverse Launcher.
- Install Isaac Lab 3 on Windows; document which install method works (extension vs. develop install). Capture the experimental-support caveats.
- Run a stock Isaac Lab example (e.g., the cartpole RL training quick-start) to confirm CUDA + Kit work.
- Stand up ROS 2 Humble on Windows. Verify `ros2 topic pub/echo` cross-host discovery with the Jetson on the same LAN.
- Report back: is this viable, what's broken, what's the minimal config that gets a single Isaac Lab scene running.

### Phase 2 — Bridge port (3–4 days)

- Replicate `source/strafer_lab/` setup on Windows: install Python deps in a Windows conda env mirroring `env_isaaclab3` (note: env name should differ to avoid confusion in dotfiles, e.g., `env_isaaclab3_win`).
- Adapt `env_setup.sh` to a PowerShell equivalent OR commit to WSL2 as the supported Windows path (recommend WSL2 if it works because it minimizes script duplication).
- Smoke-test `source/strafer_lab/scripts/run_sim_in_the_loop.py --mode bridge` against the Jetson stack.
- Capture bridge perf numbers (per the `--profile` mode) on Windows + RTX 4080. Compare to the DGX reference table in `bridge-runtime-invariants.md` so the operator knows what they're getting.

### Phase 3 — Docs (1 day)

- Write `docs/INTEGRATION_WINDOWS_WORKSTATION.md` mirroring the structure of the existing `INTEGRATION_*.md` runbooks: prerequisites, install steps, env file equivalents, smoke-test commands, troubleshooting.
- Update top-level `Readme.md`'s "Repository structure" / "Run" sections to mention the Windows path as an alternative DGX-side host.
- Update `docs/example_commands_cheatsheet.md` to note Windows-specific command variants (or PowerShell equivalents) where they differ.

## Acceptance criteria

- [ ] `make sim-bridge` (or its Windows equivalent — bare invocation OR `make` via WSL2) runs end-to-end on the Windows workstation against the same Jetson stack used in the existing Linux flow.
- [ ] A `translate forward 3 m` mission completes from the Jetson side against the Windows bridge without code changes to the Jetson lane.
- [ ] Bridge perf numbers on Windows are committed to a new section of `bridge-runtime-invariants.md` (or a sibling runbook) for future reference.
- [ ] `docs/INTEGRATION_WINDOWS_WORKSTATION.md` exists and walks a new developer from "fresh Windows workstation" to "running smoke missions" without verbal handoffs.
- [ ] The Linux DGX path continues to work byte-identically — verify by running the smoke missions on DGX before merging.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- Isaac Lab Windows support discussion (community threads, current status — verify against the version we're on, not stale 2024 threads).
- Isaac Sim 6 Windows docs (official): Omniverse Launcher install + Isaac Sim extension config.
- ROS 2 Humble on Windows: chocolatey-based install instructions; rmw_cyclonedds_cpp availability on Windows.
- WSL2 + CUDA: NVIDIA's CUDA-on-WSL2 docs; whether Isaac Sim runs inside WSL2 or needs native Windows.
- `env_setup.sh`, `.env.example`, `Makefile`, `Scripts/` — the surfaces that will need Windows equivalents or WSL2-compatible refactors.

## Out of scope

- **Real-robot bringup on Windows.** The Jetson stays on Jetson; this brief is only about the sim-side host.
- **Training large models on Windows.** Training stays on the Linux DGX where memory and CUDA driver maturity favor it.
- **Cross-OS shared-filesystem mounts.** Each host clones its own copy of the repo and rsyncs / git-pulls; we are NOT relying on a network filesystem shared between Windows and Linux.
- **Rewriting `Scripts/` for native Windows if WSL2 is sufficient.** Pick the path that minimizes duplication.

## Decision log

These facts were established during Phase 1 spike on this workstation
(RTX 4080, driver 595.97, Windows 11 build 26200, WSL 2.6.3) and pin the
architecture for the rest of the brief.

- **Architecture: full WSL2 Ubuntu-22.04**, not the originally-considered
  native-Windows or hybrid paths. Forcing finding from NVIDIA's docs:
  *"Isaac Sim supports Cyclone DDS middleware for Linux only. Windows is
  not supported at this time."* The project's `.env.example` pins
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` for documented reasons (FastDDS
  shared-memory transport doesn't traverse machines), so a Windows-native
  bridge could not speak Cyclone directly to the Jetson. The brief's own
  preferred path ("recommend WSL2 if it works because it minimizes script
  duplication") happens to also be the only RMW-compatible path.
- **Conda env name: `env_isaaclab3`**, not `env_isaaclab3_win`. The WSL2
  Linux is an Ubuntu Linux for `env_setup.sh` / `Makefile` purposes; the
  Windows-ness is below the filesystem layer. Identical env name avoids
  dotfile bifurcation.
- **Networking: WSL2 mirrored mode.** Default WSL2 NAT puts the distro
  on its own subnet (`172.x.x.x`), which breaks CycloneDDS multicast
  discovery to the Jetson. `mirrored` mode puts the WSL2 distro on the
  same LAN interface as the Windows host — in this bringup the WSL2
  guest gets `192.168.50.83` on the same `192.168.50.0/24` as the
  Jetson and DGX, and Cyclone discovery works without port forwards.
  Requires Windows 11 22H2+ (this host: build 26200).
- **JetPack kernel pin in `~/.wslconfig` is silently ignored** by WSL
  2.6.3 (path with spaces fails to parse). The new Ubuntu-22.04 distro
  boots on the standard MS WSL2 kernel (`6.6.87.2-microsoft-standard-WSL2+`)
  which carries the `dxgkrnl` passthrough Isaac Sim needs. The existing
  pin is left in place (it was installed by NVIDIA SDK Manager and may
  be needed by that workflow); the pre-bringup backup lives at
  `~/.wslconfig.pre-bringup.bak`.
- **Phase 1 acceptance — viable**: `nvidia-smi -L` inside the new
  distro reports the RTX 4080, confirming CUDA-on-WSL2 works.
- **Phase 2 deferral**: end-to-end mission validation against the
  Jetson is gated on the Jetson being powered up on the LAN. At spike
  time the Jetson was unreachable from both the Windows host and the
  WSL2 distro. Standalone install + bridge perf capture proceed
  unblocked; cross-host mission validation must run when the Jetson is
  back up.
