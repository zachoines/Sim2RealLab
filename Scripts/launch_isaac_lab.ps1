Param(
    [string[]]$ArgsToPass
)

$ErrorActionPreference = "Stop"

# Paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $workspaceRoot "venv_isaac"
$pythonExe = Join-Path $venvRoot "Scripts\python.exe"
$scriptPath = Join-Path $scriptDir "run_empty_lab.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Could not find python interpreter at $pythonExe. Ensure venv_isaac exists."
}

# Accept the EULA non-interactively and make sure kit finds assets.
$isaacPath = Join-Path $venvRoot "Lib\site-packages\isaacsim"
$env:ACCEPT_EULA = "Y"
$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:ISAAC_PATH = $isaacPath
$env:CARB_APP_PATH = Join-Path $isaacPath "kit"
$env:EXP_PATH = Join-Path $isaacPath "apps"
$env:PYTHONPATH = (
    (Join-Path $isaacPath "exts"),
    (Join-Path $isaacPath "extscache"),
    (Join-Path $isaacPath "extsDeprecated")
) -join ";"

# Make sure git is visible for any optional extension pulls.
$env:PATH = "C:\Program Files\Git\cmd;$env:PATH"

Write-Host "Starting Isaac Lab using $pythonExe ..."
& $pythonExe $scriptPath @ArgsToPass
