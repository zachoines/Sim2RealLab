# Isaac Lab upgrade — Stage 3 follow-up: R1 diagnosis, PR-B, G8 — 2026-08-23

Outputs of the Stage 3 follow-up dispatch: the bounded R1 diagnosis, the compatibility
work that unblocked gate **G8**, and G8 itself. The R3 disposition is filed separately as
[`docs/tasks/active/reliability/render-photometric-shift-isaacsim6.md`](../../tasks/active/reliability/render-photometric-shift-isaacsim6.md).

Under the record policy in force from this dispatch, this record is a README only. Every
script, log, CSV, JSON and capture is in the companion evidence repository under
`isaac-lab-upgrade-stage3-followup-2026-08-23/`, deposited **before** this record at
commit **`11cab6f8755d7c57aba6778b0443b6538e058f9a`** (117 files, each with a sha256 in
that deposit's `DEPOSIT.md`).

`.env` still names the old pair. The old pair was executed read-only as a control and is
unchanged; the new clone is still pristine. Nothing here touches `noise_models.py` or any
depth-noise configuration, and no renderer setting was altered.

---

## Verdicts

| item | result |
|---|---|
| **R1** — where the hang sits | **strictly pre-Kit-boot**, uniform signature over 16+ instances |
| **R1** — the 2×2 matrix | **both hypotheses refuted**; no launch discipline changes the rate |
| **R1** — mechanism | **dirty-lock (OMPE-97109) refuted by direct measurement**; four mechanisms excluded |
| **R1** — isolation | **isaacsim 6.0.1.0 10/30 vs 6.0.0.0 0/30, p = 0.0008** — Isaac Lab exonerated |
| **PR-B** | compat module lands the 8 sites; both-pin green; Kit smoke `all_ok` on both pins |
| **G8** — cadence | **PASS — 1012/1012 deltas at 33.3333 ms, sd = 0.000000** |
| **G8** — obs dump | written, parses under `parity.py`, documented NaN mask intact |
| **G8** — `bridge_harness_smoke.py` | **PASS** |
| **G8** — cross-lane parity | **not runnable** — no gym-side dumper exists in the tree |

---

## R1 — the diagnosis

### (1) Where the hang sits: strictly pre-Kit-boot

Every hung run from the Stage 3 session left a log of **exactly 249 B**, byte-identical
across all five, stopping at the launcher's deprecation warning — which is printed
*before* Kit initialises. Zero CUDA markers, zero Kit-boot markers. Successful runs of the
same commands are 19–24 KB and carry both. **No mid-run instance exists.**

A live specimen taken on the minimal surface (`hang-forensics.txt`, `e0e76105…`):

| observation | value |
|---|---|
| `VmRSS` | 47 228 kB — a booted Kit is gigabytes |
| threads | **2**, both `state=S wchan=futex_wait_queue`, 6 and 0 CPU ticks |
| mapped libraries | only the earliest carb plugins: dictionary serializers, variant, events, eventdispatcher |
| `/dev/nvidia*` fds | **0** — no CUDA context was ever created |
| stdout | 85 B |

Across the 16 matrix hangs the signature is uniform: threads always 2, wchan always
`futex_wait_queue`, RSS 46 976–47 232 kB (0.5 % spread). **Successful boots all completed
in exactly 3 s**, against a 60 s cap — a 20× margin, which matters for the mitigation.

### (2) The controlled matrix — both hypotheses refuted

80 launches, 20 per cell, on the bare `AppLauncher` boot. That surface was adopted after a
pilot showed it reproduces at the session rate (3/14) with the identical signature, and it
completes in 3 s where the terminations suite takes ~30 s. Every cell starts from a swept
rest state, so what differs between cells is the within-cell discipline. Raw data:
`matrix.csv` (`ddff67d7…`), worked statistics `analysis.txt` (`8b008af8…`).

| cell | hangs |
|---|---|
| nosweep-overlap | 3/20 (15 %) |
| swept-overlap | 3/20 (15 %) |
| nosweep-serial | 5/20 (25 %) |
| swept-serial | 5/20 (25 %) |

| comparison | result | Fisher exact (two-sided) |
|---|---|---|
| swept vs not swept | 8/40 vs 8/40 | **p = 1.000** |
| serialized vs overlapped | 10/40 vs 6/40 | p = 0.402 |
| P(hang \| previous hung) vs P(hang \| previous ok) | 2/14 (14 %) vs 14/62 (23 %) | p = 0.721 |

- **Sweeping does nothing.** Identical rates, p = 1.000. There is also no dose–response on
  the number of stale pairs present at launch: 0 pairs 9/41 hung, 8 pairs 1/5 hung.
- **Serializing does not help**, and the point estimate runs the *wrong way* — the
  teardown-window hypothesis predicted serialization would reduce the rate; it did not.
- **Hangs do not cluster.** A hang does not poison the next launch.

### (3) Upstream trackers

Inherited from the coordinator's addendum and not repeated.

### (4) Forensics — the documented mechanism is refuted by direct measurement

OMPE-97109 describes `kill -9` leaving *a global lock in a dirty state*. That lock is a
POSIX named semaphore, whose value is readable without ptrace: 1 when free, 0 when held by
a process that died holding it. Read during a live hang:

```
idle baseline        /carbonite-sharedmemory = 1
DURING the hang      /carbonite-sharedmemory = 1      <- free, not held
                     /carb-RStringInternals-<pid> = 2
after SIGKILL        /carbonite-sharedmemory = 1      <- the hang left nothing dirty
```

**The hung process is not blocked on the global carbonite lock, and killing it does not
dirty that lock.** Taken with the matrix, this removes the dirty-lock class as the
explanation for what is observed here — while leaving the changelog entries as evidence
that a related defect class exists in this kernel.

### The isolating experiment

Two further mechanisms were tested beyond the dispatch, because the matrix had exhausted
the ones it named.

| experiment | result | p |
|---|---|---|
| `omni.kit.telemetry` excluded vs enabled (the one line the old clone deletes) | 10/40 (25 %) vs 9/40 (22 %) | **1.000** |
| **`SimulationApp({"headless": True})` — isaacsim only, no Isaac Lab** — 6.0.1.0 vs 6.0.0.0 | **10/30 (33 %) vs 0/30 (0 %)** | **0.0008** |

The second is the result that matters. It boots isaacsim's own `SimulationApp` directly —
no Isaac Lab, no `AppLauncher`, no kit app of ours, no strafer code — and the hang
reproduces at 33 % on the new env and **never** on the old one across 30 launches each.

> **The defect is in isaacsim 6.0.1.0 / Kit 110.1.2 / carb 210.1.11 itself. Isaac Lab, this
> repository, the local kit apps, the telemetry extension, the stale shared memory and the
> launch cadence are all exonerated.** The reproducer is six lines of Python against a
> pip-installed isaacsim, which is what an upstream report needs.

### What this means for the mitigation

No discipline drives the rate toward zero, so a boot watchdog is the only mitigation
available. Its precondition — *"ONLY IF the hang is proven boot-only"* — is met by (1) and
(4): every instance is pre-CUDA, with a uniform signature. The threshold is trivially
separable: successes take 3 s, hangs never complete. **Adopting the watchdog across every
launch path (Make targets, rig prompts, bridge) is PR-A material and is flagged here, not
silently added.**

---

## PR-B — the compat module (gate G8's blocker)

`isaacsim.core.utils` is deprecated and now lives in `extsDeprecated/`, which Isaac Lab's
kit apps do not put on the extension search path. One compat module now owns the three
symbols the repo uses, and all eight call sites route through it.

Three things the documented replacements did not give for free:

- **`set_camera_view` changed shape** — free function `(eye, target, camera_prim_path)` →
  `ViewportManager.set_camera_view(camera, *, eye, target)`.
- **`isaacsim.core.rendering_manager` ships under `exts/` but is not enabled**, so the
  import fails though the files are present; the module enables it on demand. A
  disk-level check passes here while the call still fails at runtime — which is the same
  class of mistake that produced the original breakage, and it was caught only by running
  the code.
- **`add_labels` differs in behaviour**: the deprecated helper defaults to *replacing* a
  taxonomy's labels, the replacement always *appends*. The module preserves the
  replace-by-default contract the metadata writer relies on.

Kit smoke on both pins (`smoke-oldpin.json` `60f51e43…`, `smoke-newpin.json` `c1aab417…`)
returns `all_ok: true`: `enable_extension` true, the camera lands on its requested
translation `[3.0, 4.0, 5.0]`, and labels go `["alpha"]` → `["beta"]` (overwrite) →
`["beta","gamma"]` (append). The new-pin run also records
`deprecated_surface_importable: false`, so the fallback is not quietly carrying the calls.

CPU suites: old pin 1252 passed / 1 skipped, contract 148, orphaned 29; new pin 1252 / 1,
contract 126 / 22 (the moved goldens, unchanged from Stage 3), orphaned 29.

---

## G8 — the gate

Run on the PR-B branch, since the fix must be in the tree for the bridge to start.

**The bridge starts.** Zero `ModuleNotFoundError`; the cadence contract prints
`publish 30.00 Hz sim (policy period 30.00 Hz) | frame_skip=3 (derived, derived 3) |
bridge tick 120.00 Hz | renders/tick 1.00`; both async publishers come up; no stall.

### Cadence — PASS

1013 camera ticks timed off the live ROS 2 stream (`cam_ticks.json`, `12bc1d42…`), from
`/d555/depth/image_rect_raw` header stamps — sim time straight off `/clock`, which is what
the 2026-07-31 baseline's "1302/1302 at 33.33 ms" measured.

| quantity | value |
|---|---|
| camera ticks | **1013** (gate wants ≥ 1000) |
| deltas | 1012 |
| min / max / mean | 33.3333 / 33.3333 / 33.33333 ms |
| sd | **0.000000 ms** — one distinct value across all 1012 |
| at 33.33 ms ± 0.05 | **1012/1012 = 100.0000 %** (gate wants ≥ 99.9 %) |
| non-monotonic jumps | 0 |

Two measurement paths were tried and rejected before this one, and both are recorded
because each would have produced a wrong answer:

- The **obs dump** is written once per *bridge step* (120 Hz), so its deltas are a uniform
  8.3333 ms. It cannot express the camera cadence at all.
- **Detecting camera frames by watching the depth block change** is unsound while the
  robot is stationary — with no `cmd_vel` the scene is static, consecutive frames are
  bit-identical, and duplicates read as skipped ticks. That method reported a spurious
  89.15 %.

### Obs dump — written and parity-clean

`parity.load_obs_jsonl` parses the dump to 4129 rows × 3619 dims, variant
`DEPTH_SUBGOAL`, with exactly **7 NaN-masked dims** — `subgoal_relative[0..1]`,
`subgoal_distance`, `subgoal_heading_to_subgoal`, `last_action[0..2]` — which is the
documented convention. A self-parity pass joins 4129/4129 ticks with worst |dt| 0.000 ms
and both bounds PASS (`obs-parity-report.txt`, `dfd96880…`).

**The cross-lane comparison the gate names is not runnable.** `--obs-dump-path` is
declared only by `run_sim_in_the_loop.py`; there is no gym-side obs dumper in the tree, so
the reference stream has no producer. Building one is new feature work rather than an
accommodation, so it is named here rather than improvised.

### `bridge_harness_smoke.py` — PASS

34 s, no retries, on the PR-B branch: writer lifecycle, detections annotator (10-label
vocab), depth PNG sidecars, the discard path, and `reloaded: episodes=1 frames=32` — the
LeRobot read-back working under torchcodec on a booted Kit.

---

## GPU

Declared at session start; `nvidia-smi --query-compute-apps` verified empty before every
Kit launch; no rig session displaced. The R1 matrix deliberately launched ~240 Kit
processes, all on the new pair except the 30-launch old-pin control.
