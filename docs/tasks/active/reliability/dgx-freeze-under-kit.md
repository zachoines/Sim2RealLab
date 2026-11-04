# Find why the sim host stops under Kit, and make the next stop leave its cause on record

**Type:** investigation (the sim host has stopped three times under Kit and the planner and VLM services, needing a
power cycle each time, with no logged cause)
**Owner:** DGX, with the maintainer for anything that needs root or the hardware
**Priority:** P1. Every lane campaign runs on this host, and each stop has cost a relaunch: two in the frame-health
session (2026-10-18, each relaunched within the hour) and one in the inside-goal session (2026-11-02, about 26 hours to
the power cycle).
**Estimate:** M
**Branch:** `task/dgx-freeze-under-kit`

## Story

As **the sim host that every lane campaign runs on**, I want **the cause of these stops known, or at least the next one
leaving its last seconds on record**, so that **a campaign stops for a reason someone can act on, not a power cycle.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [`lane-inside-goals-2026-11-02`](../../../measurements/lane-inside-goals-2026-11-02/README.md) §8, its provenance,
  and its deposit's `OPERATIONAL_NOTE_2026-11-03.md`, `campaign/telemetry.csv`, `campaign/meminfo.csv` and
  `campaign/evict.log`.
- [`lane-slam-frame-health-2026-10-17`](../../../measurements/lane-slam-frame-health-2026-10-17/README.md)'s deposit,
  `AMENDMENT_CRASH.md` (the two 2026-10-18 stops).

## Context

The host is `gx10-d1d8` (NVIDIA GB10, aarch64; kernel 7.0.0-1019-nvidia; driver 580.173.02). On the GB10 the CPU and
the GPU share one memory.

| stop | when (UTC) | into the bridge | what ran | logged before it |
|---|---|---|---|---|
| 1 | 2026-10-18, about 00:54:55 | 20 min (F2's third reach) | Kit, the planner and VLM services | no shutdown, thermal, power, Xid, OOM-kill or panic line |
| 2 | 2026-10-18, about 01:51:19 | 6 min (F2r) | the same | the same |
| 3 | 2026-11-02, about 14:34:43 | 20 min (IB1's slot 4) | the same | one `NVRM: … Out of memory [NV_ERR_NO_MEMORY]` line at 14:14:27, 12 s after the bridge launched |

- **Each needed a power cycle.** The first two read as power losses; the third left the host frozen until it was
  power-cycled at 2026-11-03 16:40 UTC. In each, the journal's last flushed lines come minutes before the stop (stop 3:
  14:31:55).
- **The 1 Hz telemetry** (`fh_telemetry.py`, fsync'd line by line, since stop 2) shows nothing out of course up to stop
  3's last line. Over that boot the GPU ran at 36–58 °C and 10.5–62.6 W, in short bursts about once per reach (the last
  at 14:34:28–29, 55–57 °C and 54–56 W) between quieter stretches near 50 °C and 20 W. No clock-event reason was set,
  and the 1-minute load was 1.8–2.8 in the last minute.
- **The lead,** the maintainer's: at stop 3's out-of-memory line about 80 GB of the host's memory sat in the page cache.
  It is not shown as the cause: the run went on for 20 minutes after the line.
- **After stop 3,** each bridge launch was preceded by dropping the page cache without root (`posix_fadvise`
  `POSIX_FADV_DONTNEED` over the home directory). A 1 Hz memory log ran, and three boots finished with no further
  driver line and no stop. That is three boots, not a test.

## Decisions (the maintainer's)

| # | Decision | Options |
|---|---|---|
| 1 | The page cache | keep the per-file eviction before every Kit launch as standing practice (and where it lives); a root-level `drop_caches` (a narrow sudoers rule for one script); a kernel or driver setting; neither |
| 2 | A persistent record of the last seconds | a short journald `SyncIntervalSec` (the journal already persists; its default 5 min fits the minutes of missing lines); `pstore`/`ramoops`; `kdump`; a serial or network console (all need root) |
| 3 | A reproduction | none; a dedicated session that loads Kit and the services under a filled page cache with every log on; when, and on what budget |
| 4 | Vendor status | read NVIDIA's notes for GB10 driver 580.173.02 and the kernel, and report a known issue if one matches |

## Acceptance criteria

- [x] **The memory log of the boots after stop 3, read** (`lane-inside-goals-2026-11-02`'s deposit,
      `campaign/meminfo.csv`, `campaign/journal_reads.txt`): from the first eviction to IB3's end MemAvailable stayed
      at 86.3 GiB or more and Cached at 6.9 GiB or less (the evictions: 17.17 → 1.87, 4.91 → 2.09 and 4.56 → 2.09
      GiB), with no out-of-memory or Xid kernel line and no stop.
- [ ] Decisions 1–4 recorded, each in this brief with its date.
- [ ] **The memory log beside the telemetry in every Kit session** until this brief closes, as a deposited tool or a
      tracked one, as decision 1 rules.
- [ ] If decision 2 enables a record, it is tested once by a deliberate, logged reboot, and the next stop's last seconds
      are read from it.
- [ ] If decision 3 runs a reproduction, it has its own pre-registration and record.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- `journalctl --list-boots`, `journalctl -b -<n> -k`, `last -x`: what each previous boot recorded before it ended.
- The deposit's `campaign/telemetry.csv` and `campaign/meminfo.csv` columns (each tool's docstring names them).
- `nvidia-smi` on the GB10 reports no total GPU memory; `/proc/meminfo` is the memory that matters there.

## Out of scope

- **The lane's reads.** Each stop was handled by its session's relaunch rule; this brief changes no record's reading.
- **Any change to the bridge or the services** before a cause is in view.
