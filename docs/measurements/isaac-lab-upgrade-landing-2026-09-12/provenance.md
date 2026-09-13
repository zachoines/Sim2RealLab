# Provenance — isaac-lab-upgrade-landing-2026-09-12

## Hosts and trees

| | |
|---|---|
| Host | `gx10-d1d8` — DGX Spark, NVIDIA GB10 (Blackwell), driver 580.173.02 |
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, aarch64 |
| CUDA | 13.0 |
| Sim2RealLab | `main` @ `66c01a1`, plus the branch under measurement, `task/isaac-lab-landing-flip` |

The driver and kernel both moved since the `-pra-2026-08-26` record (580.82.09 →
580.173.02, 6.11.0-1014 → 7.0.0-1019). Kernel 7.0 is the release that renamed the
`futex_wait_queue` wchan to `futex_do_wait`, which is why the boot-stall rate classifier
had to be re-derived; that work is recorded in `kit-boot-hang-2026-09-12`.

## The Isaac Lab environments in play

This is the record of a **rename**, so each environment appears under two names. The
"before" column is the name it carried up to 2026-09-12 20:41; the "after" column is what
it carries now.

| role | conda env before → after | clone before → after | Isaac Sim | torch |
|---|---|---|---|---|
| **canonical** (was: candidate) | `env_isaaclab3beta2` → **`env_isaaclab3`** | `~/Documents/repos/IsaacLab-3beta2` → **`~/Documents/repos/IsaacLab`** @ `ffff603eafc6b74264a5261cc0183d6a65390d78`, tag `v3.0.0-beta2.patch1` | 6.0.1.0 | 2.11.0+cu130 |
| **retired** (was: canonical) | `env_isaaclab3` → **`env_isaaclab3-retired`** | `~/Documents/repos/IsaacLab` → **`~/Documents/repos/IsaacLab-retired`** @ `ae41e2aca68` | 6.0.0.0 | 2.10.0+cu130 |

`.env` was not edited. `CONDA_ENV=env_isaaclab3`,
`ISAACLAB=~/Documents/repos/IsaacLab/isaaclab.sh` and
`STRAFER_ISAACLAB_PYTHON=~/miniconda3/envs/env_isaaclab3/bin/python` read the same
before and after the flip; they resolve to a different pair because the pair was renamed
underneath them. That is the whole mechanism, and it is why this change has no pointer
diff to show.

Two conda environments named in earlier records are **not** part of this flip and were
not touched: `env_isaaclab3probe-6001` and `env_isaaclab3probe-6100`, the 6.1.0.0
scouting probes.

`~/Documents/repos/IsaacLab-verify` and its `env_isaaclab3verify`, the throwaway rebuild
pair from the `-pra-2026-08-26` record, no longer exist on this host. The clone was
present at 2026-09-12 20:00 and gone by 20:09:58; this session did not remove it and
cannot attribute the removal. It is neither half of either pair and its absence does not
affect the flip or the rollback.

## Interpreters

- canonical pair: `~/miniconda3/envs/env_isaaclab3/bin/python`, launcher
  `~/Documents/repos/IsaacLab/isaaclab.sh`
- retired pair: `~/miniconda3/envs/env_isaaclab3-retired/bin/python`, launcher
  `~/Documents/repos/IsaacLab-retired/isaaclab.sh`

Retired-pair runs in this record are read-only executions. They enter the pair by
overriding `CONDA_ENV` / `ISAACLAB` / `STRAFER_ISAACLAB_PYTHON` on the command line
after `env_setup.sh`, and activating the retired prefix so `isaaclab.sh` selects it —
`.env` is never edited. Nothing was installed into, upgraded in, or written to the
retired environment or clone.

## Editable bindings

Both environments' `isaaclab_*` editables were re-installed in place after the moves,
with `pip install --no-deps --no-build-isolation -e <clone>/source/<pkg>`. The installer
(`isaaclab.sh --install`) was **not** used: it re-pins torch to 2.10 and drops
`torchaudio` on every invocation, per `source/strafer_lab/README.md` § Install.

After the re-link, every `__editable___isaaclab*_finder.py` mapping resolves inside its
own clone — 18 mappings under `IsaacLab` for the canonical env (15 packages, three of
which register sub-paths), 14 under `IsaacLab-retired` for the retired env.

## Inputs this repository does not carry

- `dump_golden_preimages.py` and `golden_compare.py`, the two tools that make the golden
  re-freeze auditable, live only in the private evidence repo
  (`isaac-lab-upgrade-baseline-2026-08-14/`). `dump_golden_preimages.py` resolves the
  repository as `parents[3]`, so it must be copied into a
  `docs/measurements/<record>/` directory before it will run. It was copied in for this
  measurement and removed afterwards; the record directory carries no tooling.
- The baseline preimages the re-freeze was attributed against
  (`isaac-lab-upgrade-baseline-2026-08-14/record-files/goldens/preimages`, 26 files).
- The Kit clone is not tracked by this repository, so the `omni.kit.telemetry` deletion
  it carries cannot appear as a diff in any pull request.

## Deposit

`isaac-lab-upgrade-landing-2026-09-12/` in the evidence repository. The README's
Evidence section carries its commit hash and the sha256 of every deposited file.
