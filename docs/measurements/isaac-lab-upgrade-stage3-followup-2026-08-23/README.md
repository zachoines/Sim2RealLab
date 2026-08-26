# Isaac Lab upgrade — Kit start-up hang, compatibility layer, bridge gate — 2026-08-23

Diagnosis of the intermittent Kit start-up hang on the candidate stack, the compatibility
work that unblocked gate **G8**, and G8 itself. The renderer photometric shift is filed
separately as
[`docs/tasks/active/reliability/render-photometric-shift-isaacsim6.md`](../../tasks/active/reliability/render-photometric-shift-isaacsim6.md).

Under the measurements evidence policy this record is a README only; the evidence lives in
the companion repository and is cited by digest at the end of this file. This record was
README-only from the start and its evidence never lived in the record directory, so there
is no `record-files/` tree to migrate.

`.env` still names the old pair. The old pair was executed read-only as a control and is
unchanged; the new clone is still pristine. Nothing here touches `noise_models.py` or any
depth-noise configuration, and no renderer setting was altered.

---

## Verdicts

| item | result |
|---|---|
| **start-up hang** — where it sits | **strictly pre-Kit-boot**, uniform signature over 16+ instances |
| **start-up hang** — the 2×2 matrix | **both candidate causes refuted**; no launch discipline changes the rate |
| **start-up hang** — mechanism | **dirty-lock (OMPE-97109) refuted by direct measurement**; four mechanisms excluded |
| **start-up hang** — isolation | **isaacsim 6.0.1.0 10/30 vs 6.0.0.0 0/30, p = 0.0008** — Isaac Lab exonerated |
| **compatibility layer** | one importer for the 8 sites; green on both stacks; Kit smoke `all_ok` on both |
| **G8** — cadence | **PASS — 1012/1012 deltas at 33.3333 ms, sd = 0.000000** |
| **G8** — obs dump | written, parses under `parity.py`, documented NaN mask intact |
| **G8** — `bridge_harness_smoke.py` | **PASS** |
| **G8** — cross-lane parity | **not runnable on this host** — needs the inference node's dump (`--node-dump`) from a Jetson-in-the-loop session; the bridge dump is the gym-side half |

---

## The Kit start-up hang — the diagnosis

### (1) Where the hang sits: strictly pre-Kit-boot

Every hung run in the pre-bump validation session left a log of **exactly 249 B**, byte-identical
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

Already surveyed against the Carbonite changelog and the Isaac Sim issue trackers; not
repeated here.

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

Two further mechanisms were tested once the matrix had exhausted the candidates it was
built to separate.

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
launch path (Make targets, rig prompts, bridge) belongs with the change that lands the
bump recipe, and is flagged here rather than silently added.**

---

## The compatibility layer — what unblocked G8

`isaacsim.core.utils` is deprecated and now lives in `extsDeprecated/`, which Isaac Lab's
kit apps at `3.0.0-beta2` no longer put on the extension search path. One compat module
now owns the three symbols the repo uses, and all eight call sites route through it.

Four things the documented replacements did not give for free:

- **`enable_extension` returns `False` rather than raising** when Kit declines, and every
  call site discarded that return — so a missing extension proceeded silently. The module
  raises instead, which closes every site at once.
- **`set_camera_view` changed shape** — free function `(eye, target, camera_prim_path)` →
  `ViewportManager.set_camera_view(camera, *, eye, target)`.
- **`isaacsim.core.rendering_manager` ships under `exts/` but is not loaded everywhere.**
  Five of Isaac Lab's six kit apps pull it in; `isaaclab.python.headless.kit` — the one
  `AppLauncher` selects for headless without cameras — does not, so the import fails there
  though the files are present. The module enables it on demand. A disk-level check passes
  while the call still fails at runtime — the same class of mistake that produced the
  original breakage, and caught only by running the code.
- **`add_labels` differs in behaviour**: the deprecated helper defaults to *replacing* a
  taxonomy's labels, the replacement always *appends*. The module preserves the
  replace-by-default contract the metadata writer relies on.

### The camera path changes on the currently pinned stack too

The replacement API is importable on **both** stacks, so both take the replacement branch
and the deprecated helper runs on neither — the camera behaviour changes now, not at the
bump. A translation-only check cannot see that. Measured where both implementations are
reachable, at `eye = (2, -3, 12)`:

| straight down, `target = (2, -3, 0)` | translation | quaternion | forward | up |
|---|---|---|---|---|
| deprecated | `(2, -3, 12)` | `[1, 0, 0, 0]` | `[0, 0, -1]` | `[0, 1, 0]` |
| replacement, before the fix | `(2, -3, 12)` | `[0.707107, 0, -0, -0.707107]` | `[1e-06, 0, -1]` | `[1, 0, 1e-06]` |
| **max abs delta** | 0.0 | **0.707107** | 1e-06 | **1.0** |

Both look straight down, but the up vector differed by a **quarter turn** — and that is
the pose `coverage_capture` and `teleop_capture` request every step for the overhead
follow, so it would have rotated recorded overhead video by 90°. The module now offsets a
degenerate target along **+Y** rather than letting the replacement break collinearity
along +X, which reproduces the deprecated orientation through the replacement's own
look-at:

| case | translation | quaternion | forward | up |
|---|---|---|---|---|
| straight down, after | 0.0 | **4e-06** | **8e-06** | **8e-06** |
| oblique control, `target = (5, 1, 0)` | 0.0 | **0.0** | **0.0** | **0.0** |

One difference is documented rather than reproduced: the deprecated helper authors
`omni:kit:centerOfInterest` and the replacement leaves it absent. Nothing in this
repository reads that attribute.

Kit smoke on both stacks (`prb/smoke-oldpin.json`, `prb/smoke-newpin.json`; orientation in
`prb/ORIENTATION-CHECK.md`) returns `all_ok: true`: `enable_extension` true, the camera lands on its requested
translation `[3.0, 4.0, 5.0]`, and labels go `["alpha"]` → `["beta"]` (overwrite) →
`["beta","gamma"]` (append). The new-pin run also records
`deprecated_surface_importable: false`, so the fallback is not quietly carrying the calls.

CPU suites, re-run after the orientation fix: currently pinned 1252 passed / 1 skipped,
contract 148, compat guard 3; candidate 1252 / 1, contract 126 / 22 (the moved
composition goldens, unchanged), compat guard 3.

---

## G8 — the gate

Run with the compatibility layer applied, since the bridge cannot start without it.

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

**The cross-lane comparison the gate names is not runnable on this host.** The bridge's
`--obs-dump-path` output *is* the gym-side stream (`strafer_lab/bridge/obs_dump.py`
evaluates the same observation terms training assembles, and `obs_parity.py` consumes it
as `--gym-dump`). Its counterpart is the **inference node's** dump (`obs_dump_path`
parameter in `strafer_inference/inference_node.py`, consumed as `--node-dump`), which
only exists when the Jetson node runs against the same bridge session. Both producers
and the comparator exist in the tree; what is missing is a Jetson-in-the-loop session,
so the comparison is named here for that lane rather than improvised.

### `bridge_harness_smoke.py` — PASS

34 s, no retries, with the compatibility layer applied: writer lifecycle, detections annotator (10-label
vocab), depth PNG sidecars, the discard path, and `reloaded: episodes=1 frames=32` — the
LeRobot read-back working under torchcodec on a booted Kit.

---

## GPU

Declared at session start; `nvidia-smi --query-compute-apps` verified empty before every
Kit launch; no rig session displaced. The start-up-hang matrix deliberately launched ~240 Kit
processes, all on the new pair except the 30-launch old-pin control.

---

## Evidence

Every script, log, CSV, JSON and capture behind the numbers above is deposited in the
companion evidence repository, pushed **before** this record so the reference is
content-stable.

| field | value |
|---|---|
| repository | `https://github.com/zachoines/Sim2RealLab-Artifacts` |
| directory | `isaac-lab-upgrade-stage3-followup-2026-08-23/` |
| commit | **`9ed9487a830496558f7c42d1a5ff03d89452b03d`** |
| files | 124 |

Verify by cloning the evidence repository alongside this checkout, checking out that
commit, and running `sha256sum -c` against the list below.

```
6668bb44eb7b6aad7fbc95b67302263af6d2414d4c72066622f682d91719b42b  ./g8/bridge.log
074736b5ffa86a6f55a921a364aec9f42e688c6f344ac1b55211e748824bf882  ./g8/bridge_obs_dump-head200.jsonl.gz
5a6366c37a42e917d4e4327ed12f4b8c624fd0a16773fc2e8199d5c4584f2b5f  ./g8/bridge_obs_dump-NOT-DEPOSITED.md
c34830de8f56e8b1d1f25940f6503bd1e374a409c6ad37d196d2e823cab3247d  ./g8/cadence-final.txt
b62282c69f86e230e7df5f4eb85e8b6a4303578beaeed01fc0f7c7e6e4e4fe25  ./g8/cadence.txt
845794b660272022b511f5aa8337e4d8a6867869fc5b681c76fa6c62f550fd68  ./g8/camera-cadence.txt
12bc1d420bf714e2eb7db5fe574cfb36dc2a625f9dcaeb099ce2f3565a62c675  ./g8/cam_ticks.json
dfd9688061eea42b739d886a9ddbdb522d2cd112a29d4620afadb5d33ee941a0  ./g8/obs-parity-report.txt
ab8acce2f571ccd650a6a771fb95c70e30ae6599fd9ab843737d1a3efc33f28c  ./g8/smoke.log
6af11f9e1c72c76d07f633b956927b8dbdc6df552f5b08ed89004336752c891f  ./prb/orientation-AFTER-fix-newpin.json
fb4073cf92038a6a8f93f6df1d42c076f4a6f5ff98566a70ec1f656d99806422  ./prb/orientation-AFTER-fix-newpin.log
0e4488301ad0b92a88812ab54cedf9e68bde40b423cff16da99d5f678537f5b7  ./prb/orientation-AFTER-fix-oldpin.json
48cafd50cc06294b8b083b91a3697d944ebae8330763bafdcc49bb9ffe6e5a5d  ./prb/orientation-AFTER-fix-oldpin.log
46105c2e91dd59b0cea46ac3bb7e058af0fc52dfa71b7278ba782d08232b7f02  ./prb/orientation-BEFORE-fix-oldpin.json
4737f35c52e34828171e6a99f965797a24815bd1b3dece4a78c911c32f878885  ./prb/orientation-BEFORE-fix-oldpin.log
c08e260ac59e1a4e803ad9fedba5beb0f3e3080f60e6f50399cb0c781061869d  ./prb/ORIENTATION-CHECK.md
c1aab41705311acab528a8baee992ce0bd345db0b6984988b88671c7d23ddb44  ./prb/smoke-newpin.json
54ba5b8957aec06f541354110886844c54c55e112dcc42f8198d0b82fae5161f  ./prb/smoke-newpin.log
60f51e4346ab161b13112eb70e850c3f5c04d817383bad8d07ab3360fd7fde89  ./prb/smoke-oldpin.json
00486164601f34ef64b04a44230f4cc8430bdc386bd9fe3fe19ecb82c822b995  ./prb/smoke-oldpin.log
8b008af8a1e8915418500dc3e979a6c12feacdd8e42b1195ae8e1abb8adc6af2  ./r1/analysis.txt
eff6acbaf93168be6b760cd155d9194b5ca48ccfdc1c3308bb2731ea92c1237e  ./r1/boot-logs/nosweep-overlap-10.log
f01da3e947df6b4b1b4533d3d6eb098f8450858dcff4edc57e171c42f19ecb0a  ./r1/boot-logs/nosweep-overlap-11.log
0de7da0370eb4a3d337e5ad62b4c1728c8c908a6a42807d44c849637ad3c9c62  ./r1/boot-logs/nosweep-overlap-12.log
04f14e345121262c83e705946f9fb9db5d9d360b729179927e6967f1f3775d69  ./r1/boot-logs/nosweep-overlap-13.log
e5225c9fdda6ac22019f36b7814e1c0b1c6cdb46b08ce62a1b6b55f53e22a8d0  ./r1/boot-logs/nosweep-overlap-14.log
d8aef6f9efe2c02657c21e1474cf560a5d4014d5dad66750b91ee1c40fe29ffd  ./r1/boot-logs/nosweep-overlap-15.log
366f495cb1e9005392b2fc1bf884ea3179ebd8ccdf665d10fcb25d9a9ce374ec  ./r1/boot-logs/nosweep-overlap-16.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-overlap-17.log
31ee2f3ce476896834f40ca7fc176a8240ce097001220d9a2a021ea0624e8bc6  ./r1/boot-logs/nosweep-overlap-18.log
98edf1ccfe5f67666dc61a8fba13b80a92a7531124bcdc382919f606ebae27c8  ./r1/boot-logs/nosweep-overlap-19.log
0f6e22d7dbcfd3e77a4a56d7300b5a2645627ba55da5033ceaad766b16b1104a  ./r1/boot-logs/nosweep-overlap-1.log
b92f090abe863a884f0dcdc89135b16b644b5b358f7932c5d7b357e2c875d11f  ./r1/boot-logs/nosweep-overlap-20.log
5f755872f6642ce5e8ad8798c52f6d6ee1e4f3bc9cb45d6e57423703018ac126  ./r1/boot-logs/nosweep-overlap-2.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-overlap-3.log
ca5a98855c774ae06477a13d41b3f9e2c060710ea1eda4385d633d95c3599017  ./r1/boot-logs/nosweep-overlap-4.log
0b00185fb8cdbe10c5911868634c5f448323f3754b1e2466c22c5ff8cc38fd70  ./r1/boot-logs/nosweep-overlap-5.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-overlap-6.log
eda1335c43641c443b5aee531b69cbcd0c4bc59f41ee0dede202ffa093231bc6  ./r1/boot-logs/nosweep-overlap-7.log
30f52c5cd04a1d9cdb8a39b33160499bd98311c2166c94c341c265e2f652a3b5  ./r1/boot-logs/nosweep-overlap-8.log
165065904b35a79e2a0ce18c860e44f4c080f8ed468c57b559b9b5c58d144011  ./r1/boot-logs/nosweep-overlap-9.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-serial-10.log
acb3f28d7be8e0fb6ac3a8515619000aafb813307a9a273f101a40512b50037b  ./r1/boot-logs/nosweep-serial-11.log
6d103e8feab8488f79742e25405bbdd79f0cc03f1b28b96e6e26369953aa87f6  ./r1/boot-logs/nosweep-serial-12.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-serial-13.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-serial-14.log
ed847eb552d23d15ac64c3ed80ecfb596ffd4c3b6bd1ce723bf880aef1ff7123  ./r1/boot-logs/nosweep-serial-15.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-serial-16.log
9bc825d7fa53e8545360bea3f1a5ea9d594bf7778cb2d2d320ef67d0e5e5c068  ./r1/boot-logs/nosweep-serial-17.log
590596a2f3d07328abf5852408969f3fca027eeed6c537fb9ed428c43f655ef4  ./r1/boot-logs/nosweep-serial-18.log
1f034dfbfa24e70bad602df122e2c7a5c21946cc4e0b2f2993dfe43224f5b6a7  ./r1/boot-logs/nosweep-serial-19.log
12ff33e5dfee0327e1a50bd5dc36885c6319b66a08698d03feb98bd0fd1e1615  ./r1/boot-logs/nosweep-serial-1.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/nosweep-serial-20.log
b44568af554f847bef4574f2deecff065d30ac75452502235bfc70bec75a05c9  ./r1/boot-logs/nosweep-serial-2.log
462364e4ca57a75fcb8dffefe68480a63f1f2f722d5a40168084a9111abee7a2  ./r1/boot-logs/nosweep-serial-3.log
869e49989b50d3e61f18cf2a96df978398f6f9cd74b896ea24cb11f0f5c00c56  ./r1/boot-logs/nosweep-serial-4.log
5e5fd74a3b92cda8030db2a5c6de427c86804306f9b42be4cd309653b36f1226  ./r1/boot-logs/nosweep-serial-5.log
99c614f682f7c16426cb9a5a790ed1ddcfc6dbe1b9ec765ddb9dbe834e0bbc89  ./r1/boot-logs/nosweep-serial-6.log
3f7fd7358c3b667c86dcca361dbe34d121f89754f0ea3d9bf20f416fcd317c74  ./r1/boot-logs/nosweep-serial-7.log
39a600a838761e65dbc48b22240bab4903559fd3a6b2e18505ddb6c994f88512  ./r1/boot-logs/nosweep-serial-8.log
eace64e543f49d21ab4d2876328b2373bfc706b09365d40557fbee4690e4e3f6  ./r1/boot-logs/nosweep-serial-9.log
9d4372ad9519102a793a7bc71ed04fa0228284d989cba699cff819a63500b943  ./r1/boot-logs/swept-overlap-10.log
31aafdbc654ce796f70ebb9ab72003f69792e5b73848bd4c7f31ef0d8a00ae7e  ./r1/boot-logs/swept-overlap-11.log
6ada6f19efdd6c28d840044a2e32bc60871c3853e4dac00cda06c5d562262595  ./r1/boot-logs/swept-overlap-12.log
484c9ae92996e1d6959e742f83a4d216238fe685df24760ec733281e5f9a6705  ./r1/boot-logs/swept-overlap-13.log
d788c13b8f70f60049fa9e4ae967687980dc3539ab433c0ed40da03421e66ea5  ./r1/boot-logs/swept-overlap-14.log
37d7ea068b4f046d251bac4e7013159b5b395df1bb8c02c3d35ca5b8add08062  ./r1/boot-logs/swept-overlap-15.log
9bd9b160ba3c95f83ec81712934f06cfd25e1677fb56e48c3e2c8d170f2da0d8  ./r1/boot-logs/swept-overlap-16.log
96ecab0582f7e9fc9b2b5ca9f439c7bc9b8d30c1852c4e5a01482be8f7812662  ./r1/boot-logs/swept-overlap-17.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-overlap-18.log
41d8c7e97b252eae0162aca86dba6fb2669aa22e694768f15396bb0337c525b2  ./r1/boot-logs/swept-overlap-19.log
9aaca6dbfab08db304bc88b6de7b29612905a844c5ca2b6605c76260115d398a  ./r1/boot-logs/swept-overlap-1.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-overlap-20.log
ac0242f0528ea9caa0abf56728b7b26ea3a68484fe1c083cf7f5d4cc1d89cf16  ./r1/boot-logs/swept-overlap-2.log
4811067f3b574bc5842f1ceb4ad81eac3e39aec95633bda17e805a64b4c0ac74  ./r1/boot-logs/swept-overlap-3.log
ef2db850256b7c2e8fb050ba4b0883e8300bbeb4768be38ed0868c69b3422e76  ./r1/boot-logs/swept-overlap-4.log
f9e20fcc6d306f53180889745bb0b61e9609f25ba77cd2d8e92e45664fcf310f  ./r1/boot-logs/swept-overlap-5.log
5e7b0d52dd9a235a83693e38fbb68a196b331885d2b0ab80bc66b9da209e2d0b  ./r1/boot-logs/swept-overlap-6.log
9ac7635240d53f7d36a672e19dc3e7a0b6b169beb5a6aa19a8efa60ff7563c64  ./r1/boot-logs/swept-overlap-7.log
88c0a490362a8c35b9aeb72dfb76595da1e3715922bcb273d50f2eed422f0235  ./r1/boot-logs/swept-overlap-8.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-overlap-9.log
8aa6febbd83e5cbc3ac5db0b80a4dddd4f5fec562a9cb09803624461e882eefe  ./r1/boot-logs/swept-serial-10.log
9c0cb76f25b317d5f61ddd0c71d5ec6b7558ae50332b78433e27446bcf5a362f  ./r1/boot-logs/swept-serial-11.log
1e2cf78e7b0145e5592bda7ad9dd77c9287b2baa7c94cdd9d719ccacbfa51437  ./r1/boot-logs/swept-serial-12.log
ae0bb6605d5b88af186ef0dc8e1f954798c5293c85957664b38ef280b411b7ed  ./r1/boot-logs/swept-serial-13.log
888bdf19727402dc5abd637773e314440297ee89b32ea619c4446d467680d5a5  ./r1/boot-logs/swept-serial-14.log
643b19dcafa4ce29e13e8312c40dea569fa14878c0f2303adac476c49e7133d6  ./r1/boot-logs/swept-serial-15.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-serial-16.log
54b53970fbf2a80084b33585151a0e2190f778f460c8291ea2806cacac21aa03  ./r1/boot-logs/swept-serial-17.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-serial-18.log
70fdcb6bebeb4beb55d8f1082a17a74c49f100d5e04bc74446b5fda46efab432  ./r1/boot-logs/swept-serial-19.log
42af20d737635816193c7ede0e1a69b566de6d3899be8ce9f3c5987ff4dda447  ./r1/boot-logs/swept-serial-1.log
09e9adb1de796c41705cb4b9258a1fc726efa4a63d086be37efe1e13b01ba2e5  ./r1/boot-logs/swept-serial-20.log
100fb0c2bf13bf9245b42624f243e1dd3ce8509e527894356103a9532128c22b  ./r1/boot-logs/swept-serial-2.log
9d1435176dcd43315754df954341f3d54d3c62fa6f20dc7be904202d1054ec79  ./r1/boot-logs/swept-serial-3.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-serial-4.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-serial-5.log
4f971962466be92aba141d2a13f0e643bf4e2b997bf315fc01e5b72db8290ec3  ./r1/boot-logs/swept-serial-6.log
7be676241ef420f51146d410bf569ad8f5aee125101dab5ba0df278d0204283a  ./r1/boot-logs/swept-serial-7.log
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  ./r1/boot-logs/swept-serial-8.log
a418dd60dd7f489cb3c9e5487603874713681177d44a39d342b548ab115dbb2d  ./r1/boot-logs/swept-serial-9.log
debbe676e91440cff91a725f9de9b3b47b37e1db0c7f9ef6c8b597da467075f7  ./r1/driver.log
e0e76105f70dd3426a78c08b8675ef28ca67955daeb9bbf73a1a39395f67b4b4  ./r1/hang-forensics.txt
d8578d8a71b6c1bec8475a97c41ae6350307ab964c86d091ccd179d3a6b35368  ./r1/isaacsim_only.csv
494daa3918f943083ffdde615dbe3da87bbc1a1d670dff2c9267a25aed9a14d5  ./r1/isaacsim_only.log
ddff67d782a545f26ada41666fb1d5697e0a0c32b79e1978361e0f122965157e  ./r1/matrix.csv
f8ce68c5d011496c2869e225690b6301a69596c161a0cb7fb7933b0573326228  ./r1/pilot.csv
4a320d1a52655059280699a4e0d859f1a687c7c8b750cddcc538c7376e813c44  ./r1/telemetry.csv
49b3f01ff5ccf9e4743f01a1f055dca16b8c9e4b9724c73e8d433eacb8a5b6d1  ./r1/telemetry.log
775e3a2efc20633785e90a2db69fba36901af0650e7eb35b7dbcc148db3e72cf  ./r3/render-photometric-shift-isaacsim6.md
505f66f62735325883e28c6c3332116f68de553be84d6e9a72696ab020d170e8  ./scripts/bareboot_notel.py
d8f89bb37cfca46b647e53ddb2a87b2eea311cc5b02b29bf08e2d2edfe24925e  ./scripts/bareboot.py
b97338d382c4ddc5b2ccdfab3f5da998a89510ce2dcfc376db03d4de45164775  ./scripts/bareboot_tel.py
fa784cbc4b88bcd32a84d8dc5afdbec3b6bca0f01a547f1ca9b39c661bf78b2b  ./scripts/cam_sub.py
1ec4ec51e728309b2330bab92decc92dbee1ef46a91f7e541116fb35d0812f3a  ./scripts/catch.sh
cba50914a6a0145cf48881f1dfb8948f1653a64d8e463d6374c8aea55621e7b5  ./scripts/compat_smoke.py
d1828068b8f3d65a699a6e8afefe17a3d8bed67cb6cef84eff61bdbd0993d687  ./scripts/driver.sh
570b4458018859144e0c74c27d377920c1650b9b7acc1ffadb59d5ee70ba577c  ./scripts/isaacsim_cell.sh
bae3989bef51341b0ebe8cb2359fecfd516a72926a461071fc18aa935719933a  ./scripts/isaacsim_only.py
f2ee1d5d0dd8996de37a534d41bc08a7d598bb2220b96d2bb1e7de6d07b732ae  ./scripts/kit_retry.sh
37cd30fef454d44161842a2fe345560f448ba1246e6c6f08cde4703ab4d40d55  ./scripts/matrix.sh
8814c4eaeb5f1916cb1ab26dff1ed7f8762d2da52c5e8eea1463550d4e9cacae  ./scripts/run_g8.sh
e09e34446a052d05dc925c146536bd02319b89ea86ef75a59a24ff6cce116042  ./scripts/semprobe.py
5acda9334072b69ad0384ae55b38d842ca006d42c2a7f2dd5dc0e170ad1c1189  ./scripts/telcell.sh
```
