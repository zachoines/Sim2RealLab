<#
Headless STEP-to-USD conversion via the Omniverse CAD (HOOPS) converter.

Usage:
  .\convert_step_to_usd.ps1 -InputPath "C:\path\file.step" -OutputPath "C:\path\file.usd"

Optional:
  -ConfigPath points to a JSON options file; if omitted, an empty "{}" is used.
#>

Param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$venvRoot = Join-Path $workspaceRoot "venv_isaac"
$isaacExe = Join-Path $venvRoot "Scripts\isaacsim.exe"
$hoopsMain = Join-Path $venvRoot "Lib\site-packages\isaacsim\extscache\omni.services.convert.cad-507.0.2+107.3.1.u353\omni\services\convert\cad\services\process\hoops_main.py"

if (-not (Test-Path $isaacExe)) {
    Write-Error "isaacsim.exe not found at $isaacExe"
}
if (-not (Test-Path $hoopsMain)) {
    Write-Error "hoops_main.py not found at $hoopsMain"
}

if (-not $ConfigPath -or -not (Test-Path $ConfigPath)) {
    $ConfigPath = Join-Path $scriptDir "hoops_config.json"
    if (-not (Test-Path $ConfigPath)) {
        "{}" | Set-Content -Path $ConfigPath -NoNewline
    }
}

# Accept EULA non-interactively
$env:ACCEPT_EULA = "Y"
$env:OMNI_KIT_ACCEPT_EULA = "Y"

Write-Host "Converting STEP -> USD:"
Write-Host "  Input : $InputPath"
Write-Host "  Output: $OutputPath"
Write-Host "  Config: $ConfigPath"

& $isaacExe `
    --no-window `
    --/app/quitAfterExec=1 `
    --/app/fastShutdown=1 `
    --enable omni.kit.converter.hoops_core `
    --enable omni.kit.converter.cad `
    --enable omni.services.convert.cad `
    --exec "`"$hoopsMain`" --input-path `"$InputPath`" --output-path `"$OutputPath`" --config-path `"$ConfigPath`"`"
