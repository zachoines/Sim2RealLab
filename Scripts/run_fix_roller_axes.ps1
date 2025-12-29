Param(
    [string]$Stage = "C:/Worspace/mechanum_robot_v1.usd",
    [string]$Output = "C:/Worspace/mechanum_robot_v1_converted.usd"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$isaacExe = Join-Path $workspaceRoot "venv_isaac\\Scripts\\isaacsim.exe"
$fixScript = Join-Path $scriptDir "fix_roller_axes.py"

if (-not (Test-Path $isaacExe)) {
    Write-Error "isaacsim.exe not found at $isaacExe"
}
if (-not (Test-Path $fixScript)) {
    Write-Error "fix_roller_axes.py not found at $fixScript"
}

Write-Host "Running fix_roller_axes against $Stage ..."

$execArgs = if ([string]::IsNullOrWhiteSpace($Output)) {
    ("`"{0}`" --stage `"{1}`"" -f $fixScript, $Stage)
} else {
    ("`"{0}`" --stage `"{1}`" --out `"{2}`"" -f $fixScript, $Stage, $Output)
}

& $isaacExe `
    --no-window `
    --/app/renderer/mode=Tiny `
    --/app/quitAfterExec=1 `
    --/app/fastShutdown=1 `
    --exec $execArgs
