<#
.SYNOPSIS
Apply Windows-native install patches that bridge gaps between the
DGX `./isaaclab.sh --install rl` Linux flow and the Windows
`.\isaaclab.bat --install rl` flow.

.DESCRIPTION
The Windows pip-installable Isaac Sim 6 + IsaacLab develop @
ae41e2aca68 install has four documented gaps versus the DGX baseline.
This script applies all four in order so an operator following Path A
of `docs/INTEGRATION_WINDOWS_WORKSTATION.md` runs ONE command instead
of remembering five.

Gaps closed:

1. Patch IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/
   from_files.py: guard `import fcntl` (Unix-only) behind a
   sys.platform check. Otherwise every USD spawn crashes on Windows.

2. Editable-install the IsaacLab submodules that .bat skips but .sh
   includes: isaaclab_assets, isaaclab_tasks, isaaclab_mimic,
   isaaclab_visualizers. strafer_lab depends on the first two; the
   headed Kit visualizer needs the fourth.

3. Upgrade rsl-rl-lib to 5.x. `.\isaaclab.bat --install rl` at the
   pinned commit installs 3.1.2; strafer_lab's distribution module
   uses the 5.x API.

4. Create a name-only `omni.kit.pip_archive` stub extension under
   `venv_isaac/Lib/site-packages/isaacsim/extscache/`. The Windows
   isaacsim pip wheel doesn't ship that extension, but Kit's solver
   expects it as a transitive dep when the headed `exp.full.kit`
   experience loads `omni.kit.telemetry`. Without the stub,
   `--video` (headed env-centered video) crashes Kit's renderer
   init. The runtime use of pip_archive is for bundled boto3 / pip
   wheels; strafer training doesn't touch them, so a name-only stub
   satisfies the dep solver.

All four patches are per-clone (touch only files inside venv_isaac/
and IsaacLab/, both gitignored) and are idempotent. The script is
safe to re-run after `git pull` of IsaacLab or after re-creating
venv_isaac.

The upstream fix path is tracked in
`docs/tasks/active/tooling/isaaclab-develop-upgrade.md`. When that
brief ships, this script collapses to whatever bits remain unfixed.

.PARAMETER VenvPath
Path to the Windows venv (defaults to repo-relative `venv_isaac/`).

.PARAMETER IsaacLabPath
Path to the IsaacLab clone (defaults to repo-relative `IsaacLab/`).

.EXAMPLE
PS> .\Scripts\Install-IsaacLab-WindowsPatches.ps1
Apply all four patches against the default repo-local install paths.

.EXAMPLE
PS> .\Scripts\Install-IsaacLab-WindowsPatches.ps1 -VenvPath D:\envs\my_isaac
Apply against a custom venv location.
#>
Param(
    [string]$VenvPath = $null,
    [string]$IsaacLabPath = $null
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
if (-not $VenvPath) { $VenvPath = Join-Path $workspaceRoot "venv_isaac" }
if (-not $IsaacLabPath) { $IsaacLabPath = Join-Path $workspaceRoot "IsaacLab" }

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "venv_isaac python not found at $pythonExe. Run Step A2/A3 of the runbook first."
    exit 1
}
if (-not (Test-Path $IsaacLabPath)) {
    Write-Error "IsaacLab clone not found at $IsaacLabPath. Run Step A4 of the runbook first."
    exit 1
}

Write-Host "=== [1/4] Patching IsaacLab fcntl import (Unix-only) ===" -ForegroundColor Cyan
$spawnFile = Join-Path $IsaacLabPath "source\isaaclab\isaaclab\sim\spawners\from_files\from_files.py"
$content = Get-Content $spawnFile -Raw
if ($content -match '(?m)^import fcntl\s*$') {
    $patched = $content -replace '(?m)^import fcntl\s*$', @'
import sys
# fcntl is Unix-only. Single-process runs never enter the `_world_size > 1`
# branch below, so the import is conditional; multi-rank distributed training
# on Windows is not supported. Patch applied by Scripts\Install-IsaacLab-WindowsPatches.ps1.
if sys.platform != "win32":
    import fcntl
else:
    fcntl = None  # type: ignore[assignment]
'@
    Set-Content -Path $spawnFile -Value $patched -NoNewline
    Write-Host "    Patched $spawnFile"
} elseif ($content -match 'sys\.platform != "win32"') {
    Write-Host "    Already patched, skipping"
} else {
    Write-Warning "    Could not find 'import fcntl' line to patch. Inspect $spawnFile manually."
}

Write-Host "=== [2/4] Installing IsaacLab submodules .bat skips ===" -ForegroundColor Cyan
foreach ($submod in @("isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_visualizers")) {
    $submodDir = Join-Path $IsaacLabPath "source\$submod"
    if (-not (Test-Path $submodDir)) {
        Write-Warning "    $submod source dir not found at $submodDir, skipping"
        continue
    }
    Write-Host "    Installing $submod editable (--no-deps)..."
    & $pythonExe -m pip install -e $submodDir --no-deps --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "    pip install -e $submodDir returned exit $LASTEXITCODE"
    }
}

Write-Host "=== [3/4] Upgrading rsl-rl-lib to 5.x ===" -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade "rsl-rl-lib" --quiet
if ($LASTEXITCODE -ne 0) { Write-Warning "    rsl-rl-lib upgrade returned exit $LASTEXITCODE" }
$rslVersion = & $pythonExe -c "import rsl_rl; print(getattr(rsl_rl, '__version__', '?'))"
Write-Host "    rsl_rl: $rslVersion"

Write-Host "=== [4/4] Creating omni.kit.pip_archive stub ===" -ForegroundColor Cyan
$stubRoot = Join-Path $VenvPath "Lib\site-packages\isaacsim\extscache\omni.kit.pip_archive-0.0.1-stub"
if (Test-Path (Join-Path $stubRoot "config\extension.toml")) {
    Write-Host "    Stub already present, skipping"
} else {
    New-Item -ItemType Directory -Force -Path (Join-Path $stubRoot "config") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $stubRoot "omni\kit\pip_archive") | Out-Null
    $extToml = @'
[package]
authors = ["NVIDIA"]
category = "kit"
description = "Stub for omni.kit.pip_archive. The Windows isaacsim 6.0.0 pip distribution doesn't ship this extension, but Kit's dep solver expects it (loaded transitively from omni.kit.telemetry inside isaacsim.exp.base.kit). Without this stub, headed Kit experiences (--viz kit / --video) fail with 'No versions of omni.kit.pip_archive that satisfies'. The omni.services.pip_archive extension (which IS shipped) provides the actual pip-bundled packages; this stub is name-only. Created by Scripts\Install-IsaacLab-WindowsPatches.ps1."
title = "Kit Pip Archive (stub)"
version = "0.0.1-stub"
writeTarget.platform = true
writeTarget.python = true
writeTarget.kit = true

[core]
order = -1000
reloadable = false

[dependencies]

[[python.module]]
name = "omni.kit.pip_archive"
'@
    Set-Content -Path (Join-Path $stubRoot "config\extension.toml") -Value $extToml -NoNewline
    Set-Content -Path (Join-Path $stubRoot "omni\__init__.py") -Value ""
    Set-Content -Path (Join-Path $stubRoot "omni\kit\__init__.py") -Value ""
    Set-Content -Path (Join-Path $stubRoot "omni\kit\pip_archive\__init__.py") -Value "# Stub for Kit dep resolver - see config/extension.toml"
    Write-Host "    Stub created at $stubRoot"
}

Write-Host ""
Write-Host "All four patches applied. Verify with a headed smoke:" -ForegroundColor Green
Write-Host "  .\Scripts\launch_isaac_lab.ps1 Scripts\test_strafer_env.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 1 --duration 5"
