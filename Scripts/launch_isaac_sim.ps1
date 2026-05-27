<#
.SYNOPSIS
Launch Isaac Sim Kit standalone on the Windows-native venv_isaac install.

.DESCRIPTION
Native Windows path for data collection / harness / teleop. Use this
when you want Isaac Sim running on the host GPU directly with the
Kit viewport open. Cross-host DDS to the Jetson is NOT supported in
this configuration (CycloneDDS is Linux-only per NVIDIA's docs;
sim-bridge is tracked separately — see
docs/INTEGRATION_WINDOWS_WORKSTATION.md).

.PARAMETER Headless
Run without the Kit viewport. Default is headed.

.PARAMETER AppConfig
Kit app config (`.kit` file) to load. Defaults to the IsaacSim base
experience.

.PARAMETER ExtraArgs
Additional arguments passed verbatim to isaacsim.exe.

.EXAMPLE
PS> .\Scripts\launch_isaac_sim.ps1
Opens the Kit editor viewport with the default Isaac Sim experience.

.EXAMPLE
PS> .\Scripts\launch_isaac_sim.ps1 -Headless
Boots Isaac Sim minimally with no window (sanity check that Kit
starts).
#>
Param(
    [switch]$Headless,
    [string]$AppConfig = "isaacsim.exp.base.kit",
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $workspaceRoot "venv_isaac"
$isaacPath = Join-Path $venvRoot "Lib\site-packages\isaacsim"
$isaacExe = Join-Path $venvRoot "Scripts\isaacsim.exe"

if (-not (Test-Path $isaacExe)) {
    Write-Error @"
Could not find isaacsim.exe at $isaacExe.
Run the Windows-native install first; see
docs/INTEGRATION_WINDOWS_WORKSTATION.md (Path A) for the recipe.
"@
}

# Accept the EULA non-interactively. The trio is what Kit's
# bootstrap looks for; setting all three avoids a stdin prompt on
# fresh installs.
$env:ACCEPT_EULA = "Y"
$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:PRIVACY_CONSENT = "Y"

# Kit's extension search paths. exts / extscache / extsDeprecated all
# need to be on PYTHONPATH so the extension manager can resolve
# dependencies.
$env:ISAAC_PATH = $isaacPath
$env:CARB_APP_PATH = Join-Path $isaacPath "kit"
$env:EXP_PATH = Join-Path $isaacPath "apps"
$env:PYTHONPATH = (
    (Join-Path $isaacPath "exts"),
    (Join-Path $isaacPath "extscache"),
    (Join-Path $isaacPath "extsDeprecated")
) -join ";"

$argsList = @($AppConfig)
if ($Headless) {
    $argsList += "--no-window"
}
if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

Write-Host "Starting Isaac Sim from $isaacExe using $AppConfig ..."
& $isaacExe @argsList
