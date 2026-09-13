# Provenance — isaac-lab-3.0.0-golden-scouting-2026-09-12

## Host

DGX Spark `gx10-d1d8`. NVIDIA GB10, aarch64, Ubuntu 24.04. Measured after the 2026-09-12
maintenance window: kernel 7.0.0-1019-nvidia, driver 580.173.02, glibc 2.39-0ubuntu8.9.
Nothing here touches the GPU — the probe constructs configuration objects and never boots Kit,
so the host's graphics state is not an input.

## Trees

| role | tree | revision |
|---|---|---|
| reproducing baseline | `~/Documents/repos/IsaacLab` | the pair the tooling currently selects |
| candidate pin | `~/Documents/repos/IsaacLab-3beta2` | tag `v3.0.0-beta2.patch1` (`ffff603ea`), untouched |
| the arm under test | a detached worktree of `release/3.0.0` | `3cd1279e63effc1b82b51f70ec3250cec10d21f5` |
| our code | `~/Workspace/Sim2RealLab` | `main` at the merge of the boot watchdog |

The `release/3.0.0` revision was fetched into the disposable `IsaacLab-verify` clone and
checked out as a **detached worktree** in the session scratch directory. Neither pinned clone
was fetched into, checked out, or otherwise modified; `IsaacLab-3beta2` remained detached at
`ffff603ea` throughout.

`release/3.0.0` is a different branch line from the target tag. Their merge-base is
`4b587d17d32c` (2026-05-20), and the tag is contained in `origin/release/3.0.0-beta2` only.
There is no Isaac Lab release tag newer than the target tag, which is why the arm under test is
a branch tip and is recorded here by full hash.

## Interpreters

The baseline arm ran on `env_isaaclab3`'s interpreter; the candidate and `release/3.0.0` arms
ran on `env_isaaclab3beta2`'s. All three under `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`
with `PYTHONUNBUFFERED=1`, the established idiom for constructing env cfgs Kit-free. No conda
environment was created, modified, or installed into.

The `release/3.0.0` arm reached its Isaac Lab sources through `PYTHONPATH` at the worktree,
ahead of the editable installs. The record's Scope and limits section states which interpreter
that arm used and what it therefore does not establish.

## What produced the numbers

`probe/golden_probe.py` in the deposit, run once per arm. It imports `_canon`, `_hash`,
`_contract`, `_CONTRACT_FIELDS`, `_SIM_FIELDS`, `_CONTRACT_GOLDENS` and `_COMPOSED_RL` from
`source/strafer_lab/test_sim/env/test_composition_contract.py`, so the values it computes are
the values the contract gate computes, by construction rather than by agreement.

The field-level classification flattens each stored canonical preimage to path/scalar pairs and
compares the baseline against the arm under test, counting added paths, removed paths, and
paths whose value changed. Counts are over canonical paths, not over source lines.

The goldens' hermetic scene corpus fixture is autouse within the contract test module and is
documented there as inert for these variants; the baseline arm reproducing all 22 frozen
hashes is the check that it is inert here too.

## Inputs this repository does not carry

The three full probe runs, each with the complete canonical preimage of all 22 variants, and
the classification output, are in the deposit named in the record's Evidence section. The
`release/3.0.0` worktree itself is transient and is reconstructible from the hash above.
