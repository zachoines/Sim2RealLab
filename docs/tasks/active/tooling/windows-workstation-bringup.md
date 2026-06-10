# Windows workstation bringup for sim-bridge

**Type:** investigation + task
**Owner:** DGX agent (lane: `source/strafer_lab/`, Windows launchers, `env_setup.sh`, top-level `Makefile`, `docs/INTEGRATION_*.md`, and the `### Windows` Install/Run subsections of the `strafer_lab` / `strafer_vlm` / `strafer_autonomy` (planner) READMEs)
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

The current stack assumes Ubuntu on both hosts (DGX Spark and Jetson Orin Nano). The DGX side runs Isaac Sim 6 + Isaac Lab 3 under conda env `env_isaaclab3` on Linux; `env_setup.sh` and the `Makefile` assume bash + a Linux conda layout, and the entry points under `source/<pkg>/scripts/` launch via `$ISAACLAB -p`. The legacy top-level `Scripts/*.ps1` Windows launchers were removed in the script-tooling layout consolidation — this brief authors the Windows launcher surface **fresh** against the current `source/<pkg>/scripts/` layout rather than porting the old ones.

Windows complicates this on several axes:

1. **Isaac Sim / Isaac Lab support.** Isaac Sim 6 has official Windows support; Isaac Lab 3's official support matrix lists Linux only, with "experimental" Windows usage reported by community. Need to investigate the actual install path (Omniverse Launcher install on Windows vs. extension-based install vs. WSL2).

2. **Toolchain.** `env_setup.sh` assumes bash + Linux conda. `Makefile` targets use POSIX features. Need either:
   - A PowerShell variant (`env_setup.ps1`, `Makefile.ps1` or `tasks.ps1`) that mirrors the Linux contract.
   - A WSL2-based bringup that runs the existing scripts inside Ubuntu while accessing Windows GPU via WSLg / CUDA on WSL2.

3. **ROS 2 cross-host.** rmw_cyclonedds_cpp + ROS 2 Humble on Windows is supported but less battle-tested than on Linux. Need to confirm cross-host discovery (Windows ↔ Jetson Linux on the same LAN, ROS_DOMAIN_ID=42).

4. **Bridge perf.** RTX 4080 has more raster + RT throughput than the DGX's iGPU, but Isaac Sim's perf on Windows-with-RTX has its own quirks. Bridge-runtime-invariants doc reference numbers were captured on Linux DGX — re-derive for Windows.

5. **Asset / repo sharing.** Both hosts share one git repo via separate clones; Windows clones need to honor LF-only line endings for the scripts and the `source/<pkg>/scripts/` entry points. Isaac Sim USD asset paths are case-sensitive — need to verify generated `Assets/generated/scenes/` paths resolve on Windows.

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
- Author the `### Windows (workstation)` Install/Run subsections in the package READMEs that run on Windows (`strafer_lab`, `strafer_vlm`, `strafer_autonomy` planner side), following the canonical shape and conventions [`install-docs-consolidation`](../../completed/install-docs-consolidation.md) established: the `## Install / ### Linux / ### Windows / ## Run` template, audited against a live Windows install, **no `_Last verified` footers**, and de-dup (link the shared bits to the Linux subsection / the new runbook rather than re-pasting). Replace the existing placeholder `venv_isaac` Windows stubs in those READMEs and in `Readme.md`'s Windows subsection. (This responsibility was sliced out of `install-docs-consolidation` so the live Windows port writes it first-hand.)
- Update top-level `Readme.md`'s "Repository structure" / "Run" sections to mention the Windows path as an alternative DGX-side host.
- Update `docs/example_commands_cheatsheet.md` to note Windows-specific command variants (or PowerShell equivalents) where they differ.

## Acceptance criteria

- [ ] `make sim-bridge` (or its Windows equivalent — bare invocation OR `make` via WSL2) runs end-to-end on the Windows workstation against the same Jetson stack used in the existing Linux flow.
- [ ] A `translate forward 3 m` mission completes from the Jetson side against the Windows bridge without code changes to the Jetson lane.
- [ ] Bridge perf numbers on Windows are committed to a new section of `bridge-runtime-invariants.md` (or a sibling runbook) for future reference.
- [ ] `docs/INTEGRATION_WINDOWS_WORKSTATION.md` exists and walks a new developer from "fresh Windows workstation" to "running smoke missions" without verbal handoffs.
- [ ] The `### Windows` Install/Run subsections in the `strafer_lab` / `strafer_vlm` / `strafer_autonomy` (planner) READMEs — and `Readme.md`'s Windows subsection — are authored against the live Windows install, replace the placeholder `venv_isaac` stubs, and match the shape/conventions `install-docs-consolidation` established (no `_Last verified` footers; de-dup via links).
- [ ] The Linux DGX path continues to work byte-identically — verify by running the smoke missions on DGX before merging.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- Isaac Lab Windows support discussion (community threads, current status — verify against the version we're on, not stale 2024 threads).
- Isaac Sim 6 Windows docs (official): Omniverse Launcher install + Isaac Sim extension config.
- ROS 2 Humble on Windows: chocolatey-based install instructions; rmw_cyclonedds_cpp availability on Windows.
- WSL2 + CUDA: NVIDIA's CUDA-on-WSL2 docs; whether Isaac Sim runs inside WSL2 or needs native Windows.
- `env_setup.sh`, `.env.example`, `Makefile` — the surfaces that will need Windows equivalents or WSL2-compatible refactors, plus fresh Windows launchers for the `source/<pkg>/scripts/` entry points.

## Out of scope

- **Real-robot bringup on Windows.** The Jetson stays on Jetson; this brief is only about the sim-side host.
- **Training large models on Windows.** Training stays on the Linux DGX where memory and CUDA driver maturity favor it.
- **Cross-OS shared-filesystem mounts.** Each host clones its own copy of the repo and rsyncs / git-pulls; we are NOT relying on a network filesystem shared between Windows and Linux.
- **Authoring native-Windows launchers if WSL2 is sufficient.** Pick the path that minimizes duplication.
