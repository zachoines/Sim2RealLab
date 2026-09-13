# Kit boot hang — the stall located, and two candidate avoidances refused — 2026-09-11

The intermittent Kit boot stall on the candidate Isaac Sim pin has a measured locus.
A stalled process was traced live, and the trace replaces every inference that stood
in for a cause until now: two threads are parked in an unbounded futex wait inside the
carb plugin registry, one bringing up `carb.dictionary` on the main thread and the
other bringing up `carb.settings` on a `std::async` worker that Kit's app-settings
initialisation starts before it reads any setting. The two candidate avoidances that
were expected to remove the stall do not remove it, and the trace says why neither
could.

The stall is the gate on the Isaac Lab migration's landing sequence. The outcome of
this measurement is that no one-line avoidance exists, so the relaunch-on-stall
watchdog is the mitigation that lands.

One caveat governs how every rate below should be read, and it is not a small one.
The host's propensity to produce the stall varies by more than an order of magnitude
over hours, for reasons this measurement did not isolate. The same reproducer that
stalled 10 times in 30 boots in an earlier session stalled 9 times in 120 boots at the
start of this one and 0 times in the ~180 boots that followed, with nothing about the
installation changed. Page-cache residency and the check's own network outcome were
both tested directly and both excluded. A run of clean boots is therefore not evidence
that a change helped; only a run in which a known-affected control also stalls can
carry that weight.

## Verdicts

| item | result |
|---|---|
| Locus of the stall | Measured. Two threads in an unbounded private `FUTEX_WAIT_BITSET` inside the carb plugin registry, reached from `carbOnPluginPreStartup` of `carb.dictionary` on the main thread and `carbOnPluginStartup` of `carb.settings` on a `std::async` worker |
| Identity of the second thread | `std::async` worker running a pointer-to-member of `omni::kit::AppSettings`, launched unconditionally ~25 instructions into app-settings init, before any setting is read |
| Why the version gate exists | 6.0.1.0 moved a carb interface acquisition onto that worker; the corresponding 6.0.0.0 function reads an environment variable first and makes no carb call |
| `--/telemetry/disableInternalSessionCheck=true` | Refused. 5/40 vs 2/40 control, p = 0.43. The setting is applied and reads back true; the worker is created before any setting is read |
| `OMNI_DEVONLY_ASSUME_INTERNAL_SESSION_CONSENT=agreed` | Refused. 2/40 vs 2/40 control, p = 1.00. The variable is read after the acquisition that blocks |
| Version ladder | 6.0.0.1 reproduces the stall, 4/25, full signature. 6.1.0.0 read 0/25 but so did the known-affected 6.0.1.0 control in the same run, so 6.1.0.0 is inconclusive |
| Page-cache residency as the modulator | Excluded. 0/25 evicted vs 0/25 warm |
| The check's network outcome or latency as the trigger | Excluded. 0/24 across injected delays of 0 ms, 250 ms and 1000 ms with an internal address returned |
| Boot watchdog | Lands. No avoidance was found, and the mechanism it works around is now established rather than assumed |

## What a stalled boot is doing

Four stalled processes were traced, caught one at a time and killed after capture.
Every one carries the same shape: exactly two threads, both with `wchan` reading
`futex_wait_queue`, resident size 45–47 MB, no CUDA context, no NVIDIA file
descriptors open, and standard output stopping after the launcher's first line. The
main thread has accumulated 6–7 CPU ticks and stops advancing; the second thread has
accumulated **zero** — it was created and never ran user code to completion.

Both threads are in the same call:

```
syscall(98, uaddr, 0x89, 0, NULL, 0, 0xffffffff)
```

which is `futex`, `FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG`, a **NULL timeout**, and
`FUTEX_BITSET_MATCH_ANY`. That was read from `/proc/<tid>/syscall` on a live stall and
confirmed against the call site in the carb runtime, where the argument register is
loaded from the caller's own stack — so each thread waits on its own stack-local wait
word, and the two words differ. This is not one thread waiting on the other's
`std::future`; it is two threads parked in the same lock-acquisition path, each
waiting to be posted by whoever holds it.

The main thread's stack, outermost first:

```
SimulationApp.__init__ -> _start_app       (isaacsim.simulation_app)
omni.kit.app python binding
libomni.kit.app.plugin.so   (five frames)
libcarb.so                  (plugin framework)
carbOnPluginPreStartup      libcarb.dictionary.plugin.so
libcarb.so                  +0x2ff90, +0x47e50, +0x1a234, +0x1a700
syscall                     libc  <- unbounded futex wait
```

The worker's stack, outermost first. The three frames in the app plugin were each
verified by disassembling the instruction that the frame's return address points past,
so the call chain is read off the binary rather than guessed:

```
_Async_state_impl<_Invoker<tuple<bool (omni::kit::AppSettings::*)(), AppSettings*>>, bool>::_M_run
_State_baseV2::_M_do_set
_Task_setter<..., bool (omni::kit::AppSettings::*)(), ...>::_M_invoke
libomni.kit.app.plugin.so +0x1b9b90   return of `bl 0x1b97e0` at 0x1b9b8c
libomni.kit.app.plugin.so +0x1b9880   return of `bl 0x19a270` at 0x1b987c  (carb interface get)
libomni.kit.app.plugin.so +0x19a32c   return of `blr x2`     at 0x19a328  (interface vtable call)
libcarb.so                  (plugin framework)
carbOnPluginStartup         libcarb.settings.plugin.so
libcarb.so                  +0x2ff90, +0x47e50, +0x1a234, +0x1a700
syscall                     libc  <- unbounded futex wait
```

The four innermost carb frames are the same addresses in both threads. The worker's
task type is not inferred: it is the demangled `_Async_state_impl` symbol from the
shipped dynamic symbol table, which names a pointer-to-member of
`omni::kit::AppSettings` returning `bool`.

So one thread is driving startup of the settings plugin while the other is driving
startup of the dictionary plugin, both inside the plugin registry, and with only two
threads in the process neither can be posted.

## Why the two candidate avoidances cannot work

The app-settings initialisation function launches the worker about 25 instructions
from its entry, with no branch in between, and before it reads anything: it takes the
address of the check, passes `std::launch::async`, and calls the `std::async`
instantiation. Nothing that is expressed as a setting can gate a thread that is
created before settings are read. That is the whole account of the first switch, and
it matches the measurement — the setting is genuinely applied, reading back true from
a booted process, and the stall rate does not move.

The second switch fails for a different reason. In the candidate pin the check's first
act is to consult a new override helper, and that helper's first act is to acquire the
carb settings interface in order to read a settings key; only if that key is absent
does it fall back to the environment variable. The acquisition is the operation that
blocks. The environment variable is read strictly after it, so setting it cannot
prevent the re-entry — it can only suppress the network probe that would have followed.

This is also the version gate. In the rollback pin the corresponding function reads the
environment variable first and its call inventory contains no carb call at all, so its
worker never re-enters the plugin registry. The symbols behind the override helper and
the result-wait setting are absent from that binary entirely.

## The two switches, measured

Three arms interleaved round by round so that any drift in the host's sensitivity fell
on all three equally, with the arm order rotating so position could not bias the
result. 40 boots per arm, a 35-second classification timeout, on the candidate pin.

| arm | stalls | Fisher exact, two-sided, vs control |
|---|---|---|
| `--/telemetry/disableInternalSessionCheck=true` | 5/40 | p = 0.43 |
| `OMNI_DEVONLY_ASSUME_INTERNAL_SESSION_CONSENT=agreed` | 2/40 | p = 1.00 |
| control, neither set | 2/40 | — |

Neither switch drives the rate to zero, and each produced stalls of its own carrying
the full signature — so each is refused by direct counterexample, which does not
depend on the test having power. All nine stalls showed exactly two threads, both in
`futex_wait_queue`, at 46.2–46.4 MB.

The classification is unambiguous: successful boots took 7–8 seconds (n = 111) and
every stalled boot was still stalled at the timeout (n = 9), with nothing in between.

A separate 45-boot run with the first switch set produced no stall, but it ran after
the host had stopped producing stalls at all, so it carries no information either way.
It is recorded so that it is not later mistaken for evidence that the switch works.

`/app/waitForInternalSessionResult` reads back false by default, so the main thread
does not wait on the worker's result in the shipped configuration. It was not used as
an arm; setting it true adds a wait rather than removing one.

## The rate is unstable, and two explanations for that are excluded

Within the 120-boot A/B the stall rate fell as the run proceeded — 8 stalls in the
first 60 boots, 1 in the second 60, Fisher exact two-sided p = 0.032 — and across the
session as a whole it went from 10/30 in an earlier session to 9/120 and then to 0
across roughly 180 consecutive boots. Nothing about the installation changed.

Two candidate explanations were tested and both fail:

- **Page-cache residency.** Evicting the entire Isaac Sim tree — 121 254 files,
  15.6 GiB — with `posix_fadvise(POSIX_FADV_DONTNEED)` before each boot, interleaved
  against un-evicted boots, gave 0/25 against 0/25. The eviction demonstrably took
  effect: evicted boots averaged 10.1 s against 8.0 s warm.
- **The check's network outcome and duration.** The check resolves a hostname and
  treats one internal prefix as meaning "internal session"; on this host that name
  does not resolve and the probe fails in about 60 ms. Interposing the resolver for
  the boot process alone, so that it returned an internal address after a controlled
  delay, gave 0 stalls in 24 boots at delays of 0 ms, 250 ms and 1000 ms. Interception
  was confirmed by tracing the call from inside the Kit process, so the negative is
  real and not a failed injection.

What does modulate the rate is not established. The practical consequence is stated in
the lead: only a run in which a known-affected control also stalls can support a claim
that some change helped.

## Version ladder

Two throwaway environments were built to fill in the ladder, then measured against the
candidate pin in one interleaved run, 25 boots per arm, with the arm order rotating.

| isaacsim | carb | stalls | reading |
|---|---|---|---|
| 6.0.0.0 | 210.0.1 | 0/30 | earlier session; the rollback artifact |
| 6.0.0.1 | 210.1.5 | 4/25 | reproduces, full signature on all four |
| 6.0.1.0 | 210.1.11 | 0/25 this run; 10/30 earlier | no sensitivity in this run |
| 6.1.0.0 | 210.3.2 | 0/25 | inconclusive, see below |

The positive result stands on its own: **6.0.0.1 reproduces the stall**, four times,
each with exactly two threads in `futex_wait_queue` at about 45.5 MB and no boot
completion. That places the change no later than 6.0.0.1, between carb 210.0.1 and
210.1.5, and it means the midpoint is not a safe landing place.

The two zeroes are not evidence of a fix. The known-affected candidate pin also read
0/25 in the same interleaved run, so the run demonstrated no sensitivity for anything
except the 6.0.0.1 environment. Comparing the two newly built environments against
each other — the best-controlled pair available, both freshly installed and measured
in the same run — gives 4/25 against 0/25, Fisher exact two-sided p = 0.11. That is
directional and not significant. Whether 6.1.0.0 fixes the defect is open, and settling
it needs a host on which the 6.0.1.0 control reproduces.

One boot in the control arm failed differently: a SIGSEGV inside the carb tasking
plugin, exit 139, one second into startup. That is a separate failure mode, already
noted in the earlier record as something the watchdog deliberately does not retry, and
it is recorded here only so it is not counted as a stall.

## Tracing a stalled process on this host

`kernel.yama.ptrace_scope` is 1, so a debugger started beside a stalled process is a
sibling and is refused — `gdb` reports `ptrace: Inappropriate ioctl for device` and
`eu-stack` reports `Operation not permitted`. Both refusals were reproduced against a
live stall before any workaround was applied.

No system setting had to change. The reproducer nominates its own tracer on its first
line, via `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`, after which any process may
attach. A stall is detected as no growth in either captured output or accumulated CPU
ticks over a fixed window, which distinguishes a stall from a slow boot without
depending on wall-clock alone — a distinction that earned its keep when the freshly
built environments took 66 s and 21 s for their first boots.

## Correction and follow-up — 2026-09-12

Three things are now known that this record either got wrong or could not say. The measurements
above stand; what follows sharpens one attribution, adds the fact that decides what to do next,
and records a hazard the instrument turned out to have.

### The change is new in 6.0.0.1, not in the candidate pin

The sections above contrast "the candidate pin" with "the rollback pin", which reads as though
the carb interface acquisition inside the internal-session check arrived in 6.0.1.0. It arrived
one release earlier. Read from the shipped `libomni.kit.app.plugin.so` of all four installed
pins:

| isaacsim | `getInternalSessionOverride` | `/app/waitForInternalSessionResult` |
|---|---|---|
| 6.0.0.0 | absent | absent |
| 6.0.0.1 | **present** | **present** |
| 6.0.1.0 | present | present |
| 6.1.0.0 | present | present |

The first build carrying the new override helper is the first build that stalls, so the marker
and the onset coincide exactly. The Version ladder section's conclusion — that the change lands
no later than 6.0.0.1 — was already correct; this makes it exact rather than bounded.

### Isaac Sim 6.1.0.0 relocates the spawn, and the race cannot form there

The open arm in the Version ladder section is closed, and not by the rate measurement that
section said would be needed.

`AppSettings::init` still starts the same worker with `std::launch::async` in 6.1.0.0, and
`_checkInternalSession` itself is unchanged — 3 of 259 instructions differ, one of them a
`__LINE__` constant. What moved is where the worker is started. In 6.0.0.1 and 6.0.1.0 the
20-instruction spawn sequence sits in the entry block at `init+0x64`, with no intervening
branch. In 6.1.0.0 it is deleted from there and reinserted at `init+0x1268`, **after**
`loadCorePlugin()` of `carb.dictionary` and `carb.settings`, with everything between shifted by
exactly 0x70 bytes — one statement relocated in the source.

That is observable at runtime on every boot, without waiting for a stall. Interposing
`pthread_create` and reading the process's own mappings at its first thread creation asks
directly whether the two plugins the deadlock is between were already loaded:

| isaacsim | core plugins loaded at first spawn | boots | verdict |
|---|---|---|---|
| 6.0.0.0 | no | 10/10 | exposed ordering, harmless — its worker calls `getenv` and never touches carb |
| 6.0.0.1 | no | 10/10 | **exposed** |
| 6.0.1.0 | no | 10/10 | **exposed** |
| 6.1.0.0 | **yes** | 10/10 | **not exposed** |

Unanimous on every pin. On 6.1.0.0 the deadlock described above cannot form, because both
plugins are loaded before the second thread exists.

The limit matters and is not a formality: this establishes that **the documented race is
structurally impossible** at 6.1.0.0. It does not establish that 6.1.0.0 never stalls for some
other reason. A rate measurement now supports the same direction — in one pass 6.0.0.1 stalled
7/25 and 6.0.1.0 4/25 while 6.1.0.0 stalled 0/25, Fisher exact two-sided p = 0.0125 against the
two affected pins pooled — but one pass is one pass.

NVIDIA's Carbonite changelog carries a single candidate acknowledgement,
`OMPE-98376: omni.kit.app : Fixed a possible rare hang on startup`, the only startup-hang entry
across 209.x-212.x and in exactly the module involved. Whether the relocation is that fix
cannot be settled from outside.

### The pre-touch candidate is not implementable, and is retired

The avoidance that follows most directly from the mechanism — acquire the carb settings
interface on the main thread before `SimulationApp`, leaving the worker no plugin startup to
drive — cannot be built from Python. After `import isaacsim` the carb framework object exists
with six plugins loaded, but `carb::settings::ISettings` is not among them, and every
acquisition entry point raises
`Failed to acquire interface: carb::settings::ISettings (pluginName: nullptr)`. Kit registers
that interface during `app.startup()`, the same call that starts the worker, so there is no seam
between them. Three attempts, three identical failures; the boot never reaches the spawn, so
there is no ordering to measure. The avoidance can only be made upstream.

### A host upgrade silently broke the rate instrument, and would have read as good news

The host was rebooted on 2026-09-12 with a new kernel, driver and libc, and with
`fs.inotify.max_user_watches` raised from 65536 — where it had been 99.86% exhausted, so every
Kit boot failed to create its file-change watches — to 1048576.

The kernel renamed the `wchan` symbol for a futex wait from `futex_wait_queue` (6.11) to
`futex_do_wait` (7.0). The rate harness matched the old name exactly, so the first post-upgrade
pass labelled a genuine stall — two threads, 45812 kB, no boot completion — as
`slow_or_other`. Nothing errored. A harness that stops recognising stalls reports a clean run,
which is indistinguishable from the defect being fixed.

That pass was discarded rather than patched after the fact, and is deposited with the raw rows
that show the stall it mislabelled. The classifier now matches the futex wait by prefix and
reports loudly when a two-thread stall carries an unrecognised `wchan` set. Two things were
checked and are **not** exposed to this: the boot watchdog detects a stalled boot by absence of
CPU time and output and never reads `wchan`, and the ordering check above reads mappings.

The stall rate also rose sharply after the upgrade — 1/25 on 6.0.0.1 before, 7/25 after. Whether
that is the maintenance window or the same unexplained drift the record documents above is not
established, and the boundary is recorded in the deposit so later comparisons can account for it.

## Evidence

Two deposits, in the private repository
`https://github.com/zachoines/Sim2RealLab-Artifacts`.

### `kit-boot-hang-2026-09-11/` — commit `abe9bd6442e0cee66639e70aba3692ffb91244e2`

It holds the four traced stalls with their raw `/proc` state and both threads'
backtraces, one row per boot for each of the three interleaved measurements, the
first boots of the two environments built for the ladder, the separate SIGSEGV, and
every script the session ran. The corrected upstream issue draft sits under
`upstream/`; it supersedes the draft deposited on 2026-08-26, which is immutable
because a merged record cites it. The deposit's own `DEPOSIT.md` says which directory
holds what. This record was README-only from the start and its evidence never lived in
the record directory, so there is no `record-files/` tree.

sha256 of every file, paths relative to that deposit directory:

```
8a391ac275c77402ee342950e2231ac4155f92846338b1f3ffec11489ba08287  r1-backtraces/backtrace.txt
812da349a22767b658f0e615788d12b92333ccdc52c8ce13693611076544cc4b  r1-backtraces/catch_trace.console.log
3401213fbcbaf37ffe78f08dcbdc7baf7840e1ae8946e8a99b9787440979683a  r1-backtraces/specimens.txt
03d05803db3a8e7345b81c918224613b14b25b0f57a5156451b2ac17d3066501  r2-switch-ab/ab.console.log
cb5991a116cfd2cd76f2748a8fd49e0ce2aaa467c6e5ad2fc0f213ce8ff59a7c  r2-switch-ab/ab.csv
4d64bba36ec21999117edb81d4b02ca5c5c57fcc6e5908f7017b84703ea89189  r3-version-bisect/bisect-coldenv.console.log
02e30696b71724d623c6708d689193736f80990a67c86806be2f95033e4f55f8  r3-version-bisect/bisect-coldenv-rounds.csv
0aebcbb0757f69ce1995a77d771143dd47dd1b2dde6abbb35ea1768e01b21699  r3-version-bisect/bisect.console.log
1d9e92c15cf9fbbda555b33bce00d6b2b6eb307799c8050fd7436579e683fd18  r3-version-bisect/bisect.csv
c5e46ce9ca267f2da0787ff8569a4250a0b766bf453ecd517658ff001aab59cd  r3-version-bisect/ctrl601-6-crash139.log
0c13de9b723a659429d8c7d98db5d15bb7af744d598e3d81b7d0ed556f16bb19  r3-version-bisect/warm6001.log
e428f9963c0b9f53efed611e621622a4b6eca2100c682a2c624639bbfddd563b  r3-version-bisect/warm6100.log
01d67a43e1c57331037a81e84e99c8111ea97c02253b557b248e1bf61adc4042  r4-page-cache/evict_ab.console.log
a51c31b4b39ad664b392755bbb0de2c84bee93ea71d76fe7e4f1694314632918  r4-page-cache/evict_ab.csv
6853d8b4d1e3ab88ced509f2a3993c8bf7e77a22172e34ab803749c7ba93fd9b  scripts/ab.sh
685e75ef14054034a9f4dea7a6428f205ab4cc22b66d685b42df2e1bfab4e9de  scripts/bisect.sh
8b88571a5a9ff7f4fb07a53ca69675b1d3d35c90299952b3cef0d4ecc5283f6d  scripts/boot_probe.py
3790ded99cb0d77657513406fdb3666f54a46d8f48ce31e9ebc5bdc4918503d4  scripts/catch_trace.sh
8b41c793d999636f2210b147a8dbf2608451b05fc9f435dc9321386b94d26907  scripts/dnsshim.c
94893f016d1b2b829d2eb11eb832899fb222308f5d1bf21ded37dee54cc231f4  scripts/evict_ab.sh
2a9ad7f267838a9d58a8fed2275bc1ca1c6ca6f0852b9a07d566c5d691dab4dd  scripts/evict.py
42ee1aca95c0c059763e16af9037a7cac985810b452c1efde0f2d1a39e89c4eb  scripts/fisher.py
bae3989bef51341b0ebe8cb2359fecfd516a72926a461071fc18aa935719933a  scripts/isaacsim_only.py
ee1814a6fe7c298b5809a38bfb440a5dc66dade5e82f7d923a36674914400395  scripts/probe_rate.sh
f15ebc5d10d291aca2ee1572b5bc631c592118d7605dd686f600823da1fec9be  scripts/resolve.py
e09e34446a052d05dc925c146536bd02319b89ea86ef75a59a24ff6cce116042  scripts/semprobe.py
b30ba0139f954a9f6ecc48a0d01540944cef11e6a2537ee20ca1e2120ed8b030  scripts/specimen_s1.sh
6c4993f09747a54d833badb246c04e22df6f15b317eecd58d7be9dc9d0cd08e9  scripts/specimens.sh
aada077031a32915eca6039609392a9f846ef437fc678608d0bfbf9576de5e2a  scripts/symframes.py
794ecd654d4b85d17b46e3632f9f87e62519e3669f0c41ffa99aa76923cad00d  scripts/trace_boot.py
3a5916ba3205dae06ae570710b8c5db542c9054720d211159d81bdd2afb4bbcf  scripts/trace_boot_s1.py
c164b1f142b6b4ec06039213bfbcfbe0e9e65248c4bb782ce599382cb31c2c4d  upstream/UPSTREAM-ISSUE-DRAFT.md
```

### `kit-boot-hang-2026-09-12/` — commit `04ce2fa1ca2e0e79335ccf2a1bdfaace37f3f23d`

Backs the Correction and follow-up section. It holds the spawn-ordering check with its
interposer, its per-boot rows for all four pins and its verdict at 10 boots each; the
disassembly excerpts for the spawn at `init+0x64` and its relocation to `init+0x1268`; the
evidence that the pre-touch candidate cannot be built from Python; and the rate ladder either
side of the host maintenance window, including the discarded pass whose labels the kernel's
`wchan` rename invalidated. The superseding upstream draft sits under its `upstream/`. The
deposit's own `DEPOSIT.md` says which is which. Like the record above, it has no
`record-files/` tree.

sha256 of every file, paths relative to that deposit directory:

```
ac1827583b67493d4d058d71b870536d02b94f4beba1f0706b9ca77caf45dffa  canary/aborted/boots-20260912T174552-misclassified.csv
4e00ab3f93be85ceb769935a4301bca1cf42c367fa7f748b959c085df91f0802  canary/aborted/pass2-posthygiene.console.log
7b06cb201fe7d4c7601b93330c789babed744eaa1aa4851f76ddaa99b83678c1  canary/boots.csv
63ac03d52dada892ae5ea64c626c6ba9e6898a06ef1a378751e5519f06b42175  canary/canary_pass.sh
f5c3ea32363f2e859baa10647337ba01bdd575ec7ebfa7151142fc1134746f41  canary/canary_status.py
d7da49f7f5933df6c7e33fd6d4d333c9628c86fdc53e875df5854604561b6ce7  canary/HYGIENE.md
10e8131cc8ef0eacd2290694a4b15f32c7006e8ba9140c24e3ce4cb29a141487  canary/passes.csv
a7e28ec6d7bf6e3ae527d8ee005a1938be18a46330ba11e7605742657f3900a8  disassembly/disasm-6.0.1.0-spawn-at-entry.txt
d344631f6023dace4b61ca4fe855d404398f6ec298e16a79f9361b054337d4ab  disassembly/disasm-6.1.0.0-spawn-after-plugin-loads.txt
3cc858b37163b89bdc7fb28ff41ba915eff8a4ed8a9285addb8b338a1a5f90cf  ordering/arm4-not-implementable.txt
135e287ccae01d90045f183b1248fcbdd5d7217c4a5e72460cb5ff06fda82584  ordering/boot_arm4.py
6af8fda3986f3cb326c078113c3e931ccb073ab7ee91ee363420913c6d3d241b  ordering/boot_plain.py
70797dc38eab585ccc84a4b25692ee65f71bb90825359e9209d5ad3142c64785  ordering/check_pin_ordering.sh
0f444ce4a485bdf3f63a78d626bf55a7a1c5173d4dc56feb6fcc0ad932a883cc  ordering/ordering-all-pins-10boots.txt
0e9c10da0655bc75c64796c6dc5004b79df3baa5ffcd792a9236a32ee78bcdfe  ordering/perboot-6.0.0.0.csv
6bf95393594adb0db4c3ccc01541761227ab9456d436c5579201c1cb03d75ac7  ordering/perboot-6.0.0.1.csv
396f67f71be44dbc13de9596301c5b6f3aa621ed67b8f09aa56cef1405373e65  ordering/perboot-6.0.1.0.csv
6929dd6b0bad70886212144c4a769babe7e3e74671207589ae0425515486d322  ordering/perboot-6.1.0.0.csv
f3168cf30088f895faab5adf275326233770544a4c626b7a664cf39448763af9  ordering/spawnprobe.c
7fc3666dfd75a6899eb1f4b9fbe66ecbd0566c58dc6d6df6c8e17dea5ace8a11  upstream/UPSTREAM-ISSUE-DRAFT.md
```
