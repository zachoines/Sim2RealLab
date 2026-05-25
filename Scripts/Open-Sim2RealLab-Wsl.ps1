<#
.SYNOPSIS
Open a positioned WSL2 shell inside the Sim2RealLab repo with the conda
env activated and env_setup.sh sourced.

.DESCRIPTION
On the Windows workstation the bridge runs inside a WSL2 Ubuntu-22.04
distro (the only configuration in which CycloneDDS is available, per
NVIDIA's Isaac Sim 6 docs; see
`docs/INTEGRATION_WINDOWS_WORKSTATION.md`). The existing Linux scripts
(`env_setup.sh`, `Makefile`, `Scripts/`) are reused verbatim from inside
that distro.

This launcher is a single-purpose helper: it opens an interactive bash
shell, `cd`s into `~/Workspace/Sim2RealLab` (the WSL-side clone, NOT
the Windows `C:\Workspace\Sim2RealLab` checkout — see the runbook for
why two checkouts), activates `env_isaaclab3`, sources `env_setup.sh`,
and hands control to the operator. From there, `make sim-bridge`,
`make sim-bridge-gui`, `make sim-harness`, and the other DGX-side
targets behave identically to the Linux DGX.

.PARAMETER Distro
WSL2 distro name. Defaults to `Ubuntu-22.04`.

.PARAMETER RepoPath
WSL-side repo path. Defaults to `~/Workspace/Sim2RealLab`.

.PARAMETER CondaEnv
Conda env name. Defaults to `env_isaaclab3` (same as the Linux DGX).

.PARAMETER Command
Optional command to execute non-interactively. When supplied, the
shell runs the command and exits; without it, an interactive bash
shell is opened.

.EXAMPLE
PS> .\Scripts\Open-Sim2RealLab-Wsl.ps1
Opens an interactive WSL bash shell positioned at the repo, with the
conda env activated and env_setup.sh sourced.

.EXAMPLE
PS> .\Scripts\Open-Sim2RealLab-Wsl.ps1 -Command "make sim-bridge"
Runs `make sim-bridge` inside the configured shell and exits.

.EXAMPLE
PS> .\Scripts\Open-Sim2RealLab-Wsl.ps1 -Command "make sim-harness" `
        -EnvVars @{ SCENE_META='...'; SCENE_USD='...'; OUTPUT_DIR='...' }
Passes env-var overrides to a non-interactive command.
#>
Param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$RepoPath = "~/Workspace/Sim2RealLab",
    [string]$CondaEnv = "env_isaaclab3",
    [string]$Command = "",
    [hashtable]$EnvVars = @{}
)

$ErrorActionPreference = "Stop"

# Pre-flight: distro registered?
$distroList = (wsl --list --quiet) -join "`n"
if ($distroList -notmatch [regex]::Escape($Distro)) {
    Write-Error "WSL2 distro '$Distro' is not registered. See docs/INTEGRATION_WINDOWS_WORKSTATION.md for setup."
}

# Build the env-var prefix
$envPrefix = ""
if ($EnvVars.Count -gt 0) {
    $envPrefix = ($EnvVars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' '
    $envPrefix += ' '
}

# Compose the shell command: cd → conda activate → source env_setup.sh → exec
$inner = @"
cd $RepoPath || { echo 'repo not found at $RepoPath'; exit 1; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CondaEnv
source env_setup.sh
"@

if ($Command) {
    $inner += "`n${envPrefix}${Command}`n"
    wsl -d $Distro -- bash -lc $inner
} else {
    # Interactive: drop into bash with the env applied
    $inner += "`nexec bash"
    wsl -d $Distro -- bash -lc $inner
}
