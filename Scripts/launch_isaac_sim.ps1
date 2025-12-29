Param(
    [switch]$Headless,
    [string]$AppConfig = "isaacsim.exp.base.kit",
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# Paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $workspaceRoot "venv_isaac"
$isaacPath = Join-Path $venvRoot "Lib\site-packages\isaacsim"
$isaacExe = Join-Path $venvRoot "Scripts\isaacsim.exe"

if (-not (Test-Path $isaacExe)) {
    Write-Error "Could not find isaacsim.exe at $isaacExe. Did the pip install finish correctly?"
}

# Accept the EULA non-interactively
$env:ACCEPT_EULA = "Y"
$env:OMNI_KIT_ACCEPT_EULA = "Y"

# Core Isaac Sim environment
$env:ISAAC_PATH = $isaacPath
$env:CARB_APP_PATH = Join-Path $isaacPath "kit"
$env:EXP_PATH = Join-Path $isaacPath "apps"
$env:PYTHONPATH = (
    (Join-Path $isaacPath "exts"),
    (Join-Path $isaacPath "extscache"),
    (Join-Path $isaacPath "extsDeprecated")
) -join ";"

# Build the argument list
$argsList = @($AppConfig)
if ($Headless) {
    # this hides the main window but still runs a minimal sim
    $argsList += "--no-window"
}
if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

Write-Host "Starting Isaac Sim from $isaacExe using $AppConfig ..."
& $isaacExe @argsList
