# Upgrade Isaac Lab from `ae41e2aca68` (VERSION 3.0.0 pre-release) to current develop

**Type:** investigation + task
**Owner:** DGX agent (lane: `source/strafer_lab/`)
**Priority:** P2 (raised from P3 by the Windows bringup spike — drift now blocks the Windows-native data-collection path without per-clone manual patches; user flagged catch-up as a priority once the in-flight harness epic on DGX stabilizes)
**Estimate:** M (~2-3 days; bisect + fix import drift, validate bridge + training smoke, re-publish pin)
**Branch:** task/isaaclab-develop-upgrade

## Story

As a **DGX/sim developer**, I want **strafer_lab to import cleanly against the current Isaac Lab `develop` tip (and ideally a tagged Isaac Lab 3.x release)**, so that **new clones on fresh hosts don't have to checkout a months-old commit hash and the Windows-bringup runbook can stop pinning by SHA**.

## Context

Today (May 2026) `env_isaaclab3` on the DGX is pinned to commit
`ae41e2aca68` (`VERSION = 3.0.0`, pre-release line of Isaac Lab 3.x).
This pin was the implicit baseline strafer_lab was developed against.

Since `ae41e2aca68`, the Isaac Lab develop branch has merged several
API-breaking refactors. The Windows-workstation bringup (commit
`fbd7c80`, brief `windows-workstation-bringup.md`) tried to clone
`develop` tip and surfaced the first of these breaks:

- **`isaaclab.utils.configclass` is now a lazy-loaded submodule, not
  a callable function.** PR #4741 "Lazifies IsaacLab export system"
  (commit `5001364ce`) replaced the eager re-export in
  `isaaclab/utils/__init__.py` with `lazy_loader.attach_stub(...)`.
  The stub does not promote the `configclass` *function* up to the
  `isaaclab.utils` namespace, so existing `from isaaclab.utils import
  configclass` resolves to the submodule object. Affected file:
  `source/strafer_lab/strafer_lab/tasks/navigation/agents/distributions.py:22`
  and any other `from isaaclab.utils import <name>` in `strafer_lab/`.

- **`rsl_rl-lib` version drift.** At commit `ae41e2aca68`,
  `./isaaclab.sh --install rl` pins `rsl-rl-lib==3.1.2`. strafer_lab's
  `distributions.py` (docstring: "Affine-Beta distribution for
  rsl_rl 5.0") imports `from rsl_rl.modules.distribution import
  Distribution` — that submodule first appears in `rsl-rl-lib` 4.x+
  (latest is 5.3.0). New installs must `pip install --upgrade
  rsl-rl-lib` after IsaacLab's installer or strafer_lab won't import.
  The Windows runbook documents this manual step; pulling rsl_rl
  into IsaacLab's own pin (or pinning it in `strafer_lab/pyproject.toml`)
  removes the foot-gun.

- **Windows incompatibility: unconditional `import fcntl` in
  `isaaclab.sim.spawners.from_files`.** At commit `ae41e2aca68`,
  `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py`
  line 8 imports the Unix-only `fcntl` module unconditionally, so any
  USD-spawning env crashes on Windows native at `gym.make()` time
  (every strafer_lab env spawns USDs). The runtime use is guarded by
  `if _world_size > 1` (multi-rank distributed training only), so the
  fix is a `sys.platform != "win32"` guard around the import. The
  Windows bringup ships `Scripts\Install-IsaacLab-WindowsPatches.ps1`
  to apply the patch per-clone; upstream resolution would let new
  operators skip that step.

- **`isaaclab.bat --install rl` is incomplete on Windows.** `.bat`
  installs only `isaaclab + isaaclab_rl`; the Linux `.sh` also
  installs `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_mimic`,
  `isaaclab_visualizers`. strafer_lab depends on `_assets` and
  `_tasks` (errors at import: `Could not find a version that
  satisfies the requirement isaaclab_assets`); the headed Kit
  visualizer triggered by `--video` and `play_strafer_navigation.py
  --viz kit` depends on `_visualizers` (errors at runtime:
  `Visualizer 'kit' skipped: isaaclab_visualizers is not installed.
  Install with: pip install isaaclab_visualizers[kit]`). The
  Windows-patches script does the `pip install -e
  IsaacLab/source/<submod> --no-deps` for all four. Upstream fix:
  align `.bat` install tokens with `.sh`.

- **`omni.kit.pip_archive` missing from `isaacsim[all,extscache]>=6.0.0`
  on Windows.** The Windows isaacsim pip wheel doesn't ship that
  extension, but Kit's solver expects it as a transitive dep from
  `omni.kit.telemetry` (which is loaded by `isaacsim.exp.full.kit`,
  the headed `--viz kit` experience). Without it, headed runs crash
  Kit's renderer init with `No versions of omni.kit.pip_archive that
  satisfies`, cascading into a follow-up
  `ModuleNotFoundError: omni.physxfabric` when fabric tries to
  enable transitively. Headless training is unaffected (a different
  Kit experience is used). The Windows-patches script drops a
  name-only stub extension at
  `venv_isaac/Lib/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.1-stub/`
  to satisfy the dep solver — runtime use of pip_archive is for
  pip-bundled boto3/etc which strafer training doesn't touch.
  Upstream fix: include `omni.kit.pip_archive` in the
  `isaacsim-extscache-kit` wheel for Windows, or remove the
  transitive dep from `omni.kit.telemetry`.

- **PhysX-on-WSL2 GPU integration is currently broken.** Spike via
  `test_strafer_env.py --env Isaac-Strafer-Nav-Real-NoCam-v0
  --num_envs 1 --duration 5` on the Windows-via-WSL2 bringup logs
  `omni.physx.plugin: No CUDA context manager available; forcing CPU
  simulation` then `PhysX warning: GPU Bp pipeline failed, switching
  to software`, followed by `Warp` kernel device mismatch
  (`cuda:0` vs `cpu`) when `articulation_data.body_com_pose_w` tries
  to read PhysX state. CUDA itself works (`nvidia-smi -L` and Warp
  cuda device detection are fine inside WSL2); the issue is
  specifically PhysX's CUDA-context handshake. Likely related to the
  Kit experience file or a missing env var on WSL2-x86_64 — needs
  investigation. This affects the Windows bringup's acceptance bullet
  "`make sim-bridge` runs end-to-end" until resolved.

The Windows bringup pinned IsaacLab to `ae41e2aca68` to ship; this
brief is the follow-up that actually catches up.

## Acceptance criteria

- [ ] strafer_lab imports cleanly against the current Isaac Lab
      `develop` tip with `./isaaclab.sh --install rl`. No
      `TypeError: 'module' object is not callable` or analogous
      `from isaaclab.X import Y` failures.
- [ ] `make sim-bridge` runs to the steady-state per-phase reference
      numbers in `docs/tasks/context/bridge-runtime-invariants.md` on
      the DGX with the new Isaac Lab.
- [ ] `make sim-harness` completes a multi-mission sweep against a
      stock Infinigen scene with no regressions in reachability
      labeling.
- [ ] `Scripts/train_strafer_navigation.py` reaches the first PPO
      iteration on `Isaac-Strafer-Nav-Real-NoCam-v0` with no
      Isaac-Lab-API errors (smoke test only — no convergence check).
- [ ] The runbook update is committed in the same PR:
      `docs/INTEGRATION_WINDOWS_WORKSTATION.md` Step 7 drops the
      SHA pin and points at the new known-good tag/branch, AND
      `docs/DGX_SPARK_SETUP.md` Step 5 (`Isaac Lab 3.0 (Develop
      Branch)` section) updates to the new install token / commit.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Approach

**Ordering: DGX-first, Windows-second.** The DGX is the canonical
training / bridge host and the operator's reference for "does this
upgrade hold." Land + validate the upgrade there on a stable
checkpoint, then port the resulting baseline to Windows (which has
its own additional drift items captured in the Decision log section
above). Don't try to fix both at once — Windows-specific gaps will
mask Linux-specific drift if mixed.

### Phase 1 — DGX: map the drift (½ day)

- `cd ~/Documents/repos/IsaacLab && git log ae41e2aca68..origin/develop
  --oneline -- source/isaaclab/isaaclab/utils/` — enumerate the
  utils-layer refactors landed since the pin.
- Grep strafer_lab for every `from isaaclab` / `from isaaclab_*`
  import; cross-reference with the diff above. Build a short list
  of the API surfaces affected.
- Decide pin target: latest stable Isaac Lab tag (if 3.x is out by
  then) OR a known-good commit on develop. Capture rationale in
  the PR.

### Phase 2 — DGX: fix the imports + validate Linux baseline (1-2 days)

- Mechanical changes: `from isaaclab.utils import X` →
  `from isaaclab.utils.X import X` for any X now lazy-loaded as a
  submodule. Probably `configclass`, possibly `string_utils`,
  `math_utils`, `dict_utils`.
- Run `python -c "import strafer_lab"` until it imports cleanly.
- `make sim-bridge` end-to-end (cross-host with the Jetson).
- `make sim-harness` for one scene.
- `Scripts/train_strafer_navigation.py --num_envs 16` for one iter.
- Re-derive bridge per-phase numbers, update
  `bridge-runtime-invariants.md` if anything drifted.

### Phase 3 — Windows: port baseline + retire `Install-IsaacLab-WindowsPatches.ps1` items that newer IsaacLab resolves (½-1 day)

- On the Windows workstation: pull the new IsaacLab pin (per the
  updated runbook), recreate `venv_isaac`, re-run
  `Scripts\Install-IsaacLab-WindowsPatches.ps1`, check which of the
  four patches the script reports as "already patched" / "already
  installed" — those are the ones newer IsaacLab fixed upstream.
- For each gap that the upgrade closed, drop that section from the
  Windows-patches script and update the runbook Step A5 to remove
  it from the documented list.
- If all four are closed: delete `Scripts\Install-IsaacLab-WindowsPatches.ps1`
  outright and collapse Step A5 to a note that it used to be needed.
- Re-run the Path A smoke (`launch_isaac_lab.ps1
  Scripts\test_strafer_env.py --env Isaac-Strafer-Nav-Real-NoCam-v0
  --num_envs 8 --duration 10`) and the headed `--video` training
  smoke captured under Step A7b/A7c. If either regresses, file a
  new follow-up brief rather than scoping into this one.

## Out of scope

- **Switching off rsl_rl.** If a newer Isaac Lab release renames its
  RL bundle (`rsl_rl` token already became `rl` between the pin and
  develop tip), update the install command — but don't migrate
  strafer_lab to a different RL framework.
- **Isaac Sim binary version upgrade beyond 6.0.x.** Isaac Sim 6.x
  is the validated runtime; this brief is about Isaac Lab only.
- **Re-architecting `env_isaaclab3` as a non-conda venv.** That's a
  separate "do we still need conda on Linux" discussion.

## Investigation pointers

- IsaacLab PR #4741 (commit `5001364ce`, "Lazifies IsaacLab export
  system") for the import-pattern change.
- The Windows-bringup brief's decision log (commit `fbd7c80`) names
  `ae41e2aca68` as the pre-upgrade pin.
- IsaacLab releases page — Isaac Lab 3.x tag if shipped by upgrade time.
