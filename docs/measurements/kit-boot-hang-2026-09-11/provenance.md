# Provenance — kit-boot-hang-2026-09-11

## Host

DGX Spark `gx10-d1d8`. NVIDIA GB10, aarch64, Ubuntu 24.04, glibc 2.39, kernel
6.11.0-1014-nvidia. 121 GB RAM. `kernel.yama.ptrace_scope` = 1, unchanged by this
measurement. The GPU was declared before every boot and was idle throughout: the stall
never reaches GPU initialisation, and no compute application was resident at any check.

## Environments

All boots used Isaac Sim's own `SimulationApp` directly. Isaac Lab is not on this path
and none of its checkouts were read or run.

| conda environment | isaacsim | carb | role |
|---|---|---|---|
| `env_isaaclab3` | 6.0.0.0 | 210.0.1 | rollback artifact; source of the 0/30 figure carried forward from 2026-08-23 |
| `env_isaaclab3beta2` | 6.0.1.0 | 210.1.11 | candidate pin; the arm-1 and arm-2 subject and the bisect's control |
| `env_isaaclab3probe-6001` | 6.0.0.1 | 210.1.5 | built for this measurement |
| `env_isaaclab3probe-6100` | 6.1.0.0 | 210.3.2 | built for this measurement |

The first two were read-only for this measurement; nothing was installed into them and
no `.env` was touched. The two probe environments were built fresh from
`https://pypi.nvidia.com` as `isaacsim[all,extscache]==<version>`, Python 3.12.13,
each about 19 GB, each resolving `isaacsim.exp.base.python.kit` as its default
experience — the same experience the reference environments resolve. Both are retained
rather than deleted: 6.0.0.1 is the only environment on the host that still reproduces
the stall, and 6.1.0.0 is the arm whose result is unresolved. Deleting either would
discard the means to finish the ladder.

Two operational notes on the probe environments, neither of which affects the results
but both of which would otherwise masquerade as a failure rate. A freshly installed
Isaac Sim has no EULA marker: 6.0.0.1 required the marker file the reference
environments carry, and 6.1.0.0 required `OMNI_KIT_ACCEPT_EULA=YES` on every
invocation. Without them a boot exits in about a second rather than stalling.

## Interpreter and launch

Every boot ran the environment's own `python` directly, not through a launcher wrapper,
so that the process holding the stall is the Python process itself and can be traced.
`PYTHONUNBUFFERED=1` and `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` were set on
every arm, matching the conditions under which the 2026-08-23 baseline was taken.

## Tooling

`gdb` 15.0.50 and `elfutils`' `eu-stack` are the host's own. `py-spy` 0.4.2 was
installed into a throwaway virtual environment inside the session scratch directory so
that no measured environment was modified.

The resolver interposer deposited as `scripts/dnsshim.c` was compiled with the host
`gcc` 11.5.0 and applied only to the boot process under test, through `LD_PRELOAD`. It
changes no system state; `/etc/hosts` and `/etc/resolv.conf` were not touched.

Page-cache eviction used `posix_fadvise(POSIX_FADV_DONTNEED)` on the Isaac Sim tree.
That advises the kernel to drop clean cached pages and does not modify file contents.

Fisher exact p-values were computed with exact rational arithmetic over the
hypergeometric distribution rather than a normal approximation; the implementation is
deposited as `scripts/fisher.py`.

## Figures carried in from earlier measurements

Three numbers in the record were not produced by this session and are cited from
`docs/measurements/isaac-lab-upgrade-pra-2026-08-26/` and the deposit it references:
the 10/30 and 0/30 arms taken 2026-08-23, and the three dirty-lock non-correlations
across 80 launches. They are reproduced in the corrected upstream draft with the same
values.

## Inputs this repository does not carry

The four traced stalls, the per-boot rows for all three interleaved measurements, and
every script are in the deposit named in the record's Evidence section. The
disassembly offsets quoted in the record are file offsets into
`libomni.kit.app.plugin.so` and `libcarb.so` as shipped in the two reference
environments; they are reproducible from those files with `objdump -d` and are not
themselves deposited, the binaries being vendor artifacts.
