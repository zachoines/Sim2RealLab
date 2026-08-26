# Provenance — isaac-lab-upgrade-pra-2026-08-26

## Hosts and trees

| | |
|---|---|
| Host | `gx10-d1d8` — DGX Spark, NVIDIA GB10 (Blackwell), driver 580.82.09 |
| OS | Ubuntu 24.04.3 LTS, kernel 6.11.0-1014-nvidia, aarch64 |
| CUDA | 13.0 (`/usr/local/cuda-13.0`) |
| Sim2RealLab | `main` @ `94d83c98074269a938de5aa37fe985bf49c4b69f`, plus the branch under measurement, `task/isaac-lab-upgrade-pra`, in a sibling worktree |

## The three Isaac Lab environments in play

| role | conda env | clone | Isaac Sim | torch |
|---|---|---|---|---|
| **currently selected** (control, read-only) | `env_isaaclab3` | `~/Documents/repos/IsaacLab` @ `ae41e2a` | 6.0.0.0 | 2.10.0+cu130 |
| **candidate** (read-only here) | `env_isaaclab3beta2` (Python 3.12.13) | `~/Documents/repos/IsaacLab-3beta2` @ `ffff603eafc6b74264a5261cc0183d6a65390d78`, tag `v3.0.0-beta2.patch1` | 6.0.1.0 | 2.11.0+cu130 |
| **throwaway, built for this record** | `env_isaaclab3verify` | `~/Documents/repos/IsaacLab-verify` @ the same `ffff603` | 6.0.1.0 | 2.11.0+cu130 |

Neither of the first two was written to. The candidate clone's working tree was
clean before and after (`git status --porcelain` empty). The earlier clone
carries its two long-standing local Kit edits, dated April 2026 and untouched
here. Both environments reported the same torch build before and after the
rebuild.

The throwaway environment and the third clone exist only for the rebuild. The
third clone additionally received the `omni.kit.telemetry` line deletion for the
second boot arm, which is why that arm's clone is not pristine.

## Interpreters

- earlier pair: `~/miniconda3/envs/env_isaaclab3/bin/python`, launcher
  `~/Documents/repos/IsaacLab/isaaclab.sh`
- candidate pair: `~/miniconda3/envs/env_isaaclab3beta2/bin/python`, launcher
  `~/Documents/repos/IsaacLab-3beta2/isaaclab.sh`
- rebuild: `~/miniconda3/envs/env_isaaclab3verify/bin/python`, launcher
  `~/Documents/repos/IsaacLab-verify/isaaclab.sh`

Candidate-pin runs entered the pair through the Stage 3 record's deposited
`newpin_env.sh`, which mirrors `env_setup.sh` against the candidate prefix
without editing `.env`.

## Inputs this repository does not carry

- **The original build's pip logs.** The recipe is transcribed from a numbered
  series of logs written when the candidate pair was built on 2026-08-14/15.
  They are machine-local, outside both repositories, and were never deposited.
  They are the reason the recipe could be reconstructed rather than guessed, and
  their absence from either repository is itself reported in the record.
- **The Stage 3 deposited freeze**, which the rebuild is accepted against:
  `isaac-lab-upgrade-stage3-2026-08-23/record-files/pip-freeze-env_isaaclab3beta2.txt`
  in the evidence repository.
- **The boot-hang forensics** the watchdog's detection rule is derived from:
  `isaac-lab-upgrade-stage3-followup-2026-08-23/r1/` in the evidence repository.

## Deposit

| | |
|---|---|
| Repository | `https://github.com/zachoines/Sim2RealLab-Artifacts` (private) |
| Directory | `isaac-lab-upgrade-pra-2026-08-26/` |
| Commit | `930434e68b733989696d0c7989ad7e334aa3633a` |
| Files | 75, sha256 of each listed in the record's evidence section |
