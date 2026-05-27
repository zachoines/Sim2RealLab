<#
.SYNOPSIS
Run an Isaac Lab Python script under the Windows-native venv_isaac
install — equivalent of `$ISAACLAB -p <script>` on Linux.

.DESCRIPTION
Native Windows path for data collection / harness / teleop. Resolves
the venv's python.exe, sets up Kit's extension search paths, EULA env
vars, and invokes the script with whatever args follow. Matches the
DGX-side `./isaaclab.sh -p script.py ...` workflow.

.PARAMETER ScriptPath
Path to the Python script to run (absolute or relative to repo root).

.PARAMETER ScriptArgs
All arguments after the script path are forwarded verbatim.

.EXAMPLE
PS> .\Scripts\launch_isaac_lab.ps1 Scripts\test_strafer_env.py `
        --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 8 --duration 10
Runs the NoCam smoke test for 10 seconds, headed.

.EXAMPLE
PS> .\Scripts\launch_isaac_lab.ps1 source\strafer_lab\scripts\collect_demos.py `
        --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 `
        --output demos/ --max_episodes 100 --viz kit
Drives the gamepad demo collector — the use case this Windows-native
path exists for.
#>
Param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

# NOTE: ErrorActionPreference stays "Continue" (the default). Kit emits
# informational lines to stderr, and "Stop" makes PowerShell abort on
# every stderr line as if it were a fatal error. We rely on $LASTEXITCODE
# at the end of the script for the real success/failure signal.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $workspaceRoot "venv_isaac"
$pythonExe = Join-Path $venvRoot "Scripts\python.exe"
$isaacPath = Join-Path $venvRoot "Lib\site-packages\isaacsim"

if (-not (Test-Path $pythonExe)) {
    Write-Error @"
Could not find python interpreter at $pythonExe.
Run the Windows-native install first; see
docs/INTEGRATION_WINDOWS_WORKSTATION.md (Path A) for the recipe.
"@
    exit 1
}

# Resolve the script path: absolute as-given, else relative to repo root.
if (-not [System.IO.Path]::IsPathRooted($ScriptPath)) {
    $ScriptPath = Join-Path $workspaceRoot $ScriptPath
}
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
}

# EULA + Kit environment (same trio as launch_isaac_sim.ps1).
$env:ACCEPT_EULA = "Y"
$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:PRIVACY_CONSENT = "Y"

$env:ISAAC_PATH = $isaacPath
$env:CARB_APP_PATH = Join-Path $isaacPath "kit"
$env:EXP_PATH = Join-Path $isaacPath "apps"
$env:PYTHONPATH = (
    (Join-Path $isaacPath "exts"),
    (Join-Path $isaacPath "extscache"),
    (Join-Path $isaacPath "extsDeprecated")
) -join ";"

Write-Host "Launching: $pythonExe $ScriptPath $($ScriptArgs -join ' ')"
& $pythonExe $ScriptPath @ScriptArgs
exit $LASTEXITCODE
