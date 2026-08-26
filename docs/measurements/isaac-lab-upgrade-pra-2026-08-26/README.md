# Isaac Lab upgrade — environment recipe verified by rebuild, and launch reliability — 2026-08-26

Two things measured together, because the second is what makes the first
repeatable: the environment-creation recipe for the candidate Isaac Lab pair was
reconstructed and then **proved by building it again from scratch**, and a Kit
boot that deadlocks is now detected and relaunched instead of hanging a gate.

Under the measurements evidence policy this record is a README only; the
evidence lives in the companion repository and is cited by digest at the end.
This record was README-only from the start and its evidence never lived in the
record directory, so there is no `record-files/` tree.

`.env` and the Makefile defaults still name the earlier pair. The earlier pair
and the candidate clone were both executed read-only and are unchanged; the
rebuild went into a throwaway environment and a third clone. Nothing here
touches `noise_models.py` or any depth-noise configuration.

---

## Verdicts

| item | result |
|---|---|
| **recipe** — was one ever written down | **no** — the pair was built before any recipe existed; the only surviving description was a `pip freeze`, which cannot recover ordering |
| **recipe** — reconstruction source | the build's own numbered pip logs, machine-local and never deposited |
| **recipe** — rebuild acceptance | **288 of 300 distributions byte-identical**; every pinned package, all 25 `isaacsim` wheels and all 15 `isaaclab_*` editables match |
| **recipe** — what does not reproduce | 10 unpinned transitives drifted; `usd-exchange` crossed a **major** version (2.3.0 → 3.0.0) |
| **watchdog** — detection rule | resident-size thresholds are unusable inside pytest; **no CPU and no output** is the signature that holds everywhere |
| **watchdog** — candidate pin | 16 boots, **4 relaunches**, all 16 eventually clean; one boot needed two relaunches |
| **watchdog** — earlier pin | **0 relaunches across all 14 Kit suites** — the wrapper costs nothing where the defect is absent |
| **new failure mode** — SIGSEGV in `libomni.kit.telemetry.plugin.so` | **2 / 16 boots** on the pristine candidate clone; **0 / 16** with the extension removed |
| **gates** — earlier pin, on this branch | Kit **487 / 487**, pure **1252 passed / 1 skipped**, contract **148** |

---

## The recipe was never written down

The candidate pair — conda `env_isaaclab3beta2` and an Isaac Lab clone at tag
`v3.0.0-beta2.patch1`, commit `ffff603` — was built on 2026-08-14/15. Nothing in
this repository or in the evidence repository recorded how. A `pip freeze` was
deposited with the Stage 3 record, and a freeze describes an end state but not
an order, which matters here because the order is the whole difficulty (see
*The torch pin has to be last*, below).

The build **was** logged, into a numbered series of pip logs that stayed
machine-local and was never deposited. Those logs are what this recipe is
transcribed from. Their existence is the finding: a recipe that reads as
unrecoverable can be recoverable and simply undeposited, and the way to tell is
to look before concluding.

## Proving it: the rebuild

The reconstructed sequence was executed end to end into a throwaway conda
environment against a third clone of Isaac Lab at the same commit. The two
existing pairs were never activated; the driver aborts if the active environment
is not the throwaway one, because `isaaclab.sh` picks its interpreter from
`CONDA_PREFIX` and would otherwise install into whichever pair happened to be
active — which, for the candidate pair, would have destroyed it.

All 17 steps exited zero. The resulting environment holds **300 distributions**,
the same count as the deposited freeze, and differs from it in **12 lines**:

| lines | what | why |
|---|---|---|
| 2 | the `strafer_shared` / `strafer_lab` editables | they carry the Sim2RealLab working-tree revision, not an environment fact |
| 10 | `usd-exchange` 2.3.0 → 3.0.0, `viser` 1.0.30 → 1.1.0, `rerun-sdk` 0.36.0 → 0.36.3, `protobuf` 7.35.1 → 7.36.0, and `GitPython` / `Pygments` / `python-dotenv` / `xxhash` / `jupyterlab_widgets` / `widgetsnbextension` at patch level | unpinned transitives, resolved eleven days later |

Everything a command names is identical: `torch` 2.11.0+cu130, `torchvision`
0.26.0+cu130, `torchaudio` 2.11.0+cu130, `torchcodec` 0.16.0, all 25 `isaacsim`
wheels at 6.0.1.0, all 15 `isaaclab_*` editables at `ffff603`, `rsl-rl-lib`
5.4.2, `onnxscript` 0.7.1, `onnxruntime` 1.25.1, `packaging` 26.0,
`nvidia-cudnn-cu13` 9.19.0.56, `lerobot` 0.5.1, `numpy` 2.3.1, `moviepy` 1.0.3.

So the recipe is correct and complete, and the reproducibility limit is named
rather than assumed: only about 15 of the 300 distributions are version-
controlled by the commands. A rebuild that must match exactly needs a
constraints file, which does not exist. `usd-exchange` crossing a major version
is the first thing to check if a future rebuild misbehaves.

The rebuilt environment carries **no `EULA_ACCEPTED` marker**, which confirms
the recipe's own claim that the marker comes from an interactive first boot and
not from anything the install sequence does.

## The torch pin has to be last

`isaaclab.sh --install` compares the installed torch against `2.10.0+cu130` and,
on any mismatch, uninstalls `torch`, `torchvision` **and** `torchaudio` and
reinstalls only the first two. It does this twice per invocation, so the second
pass reports `PyTorch 2.10.0+cu130 already installed.` and the log reads clean.

The consequence is stronger than an ordering rule: the environment is **not
idempotent under re-install**. Any later `--install`, including one run only to
add an extra, silently downgrades torch and leaves `torchaudio` missing. The
recipe says so, and says which two steps repair it.

Two related traps the recipe now names:

- `--install rsl_rl` no longer exists. An unknown token only **warns** and is
  skipped, so the stale command appears to succeed while installing no rsl-rl at
  all. The replacement extra pins `rsl-rl-lib==5.0.1`, which is not the 5.4.2
  this stack needs, so the correct move is to skip the extra and install the
  version directly.
- Isaac Sim is installed by hand as `isaacsim[all,extscache]==6.0.1.0`, not
  through the `isaacsim` install token, which asks for `isaacsim[all]>=6.0.0` —
  pinning neither the version nor the `extscache` extra that decides the Kit
  extension payload.

## Detecting a deadlocked Kit boot

The defect: on Isaac Sim 6.0.1.0 a Kit process intermittently deadlocks inside
carb initialisation — two threads in `futex_wait_queue`, resident size near
47 MB, no CUDA context, output stopped after the launcher's first line, no
recovery. Isaac Sim 6.0.0.0 does not do this.

A resident-size threshold was the obvious detector and was measured before being
adopted, which is what ruled it out. Sampling one healthy boot every 250 ms:

| t | process-group RSS | log bytes |
|---|---|---|
| 2 ms | 11.9 MB | 0 |
| 277 ms | **458 MB** | 85 |
| 827 ms | 1032 MB | 1348 |
| 2178 ms | 1349 MB | 1348 |

A healthy boot is an order of magnitude above the deadlocked 47 MB within
277 ms, so the separation is real for a bare launch. It is useless one level up:
inside a pytest subprocess the imports alone exceed any such threshold long
before Kit starts, so the check would pass before the boot it is meant to watch.

The rule that holds at both levels is **no CPU time and no output**. A
deadlocked tree moves neither counter — the live specimen sat at 6 CPU ticks and
did not advance — while every healthy phase of a run, import, plugin load,
collection, stepping, moves both continuously. Watching stops once the boot
window has elapsed, so a capture that runs for hours is never a candidate for
being killed, and only a launch that never started can be relaunched.

Measured on the candidate pin, 16 consecutive boots:

| | |
|---|---|
| boots | 16 |
| relaunched | 4 attempts across 3 boots (one needed two) |
| eventually clean | 16 / 16 |
| healthy boot | 3 s |
| detection latency | 61 s (a 60 s no-progress window plus one poll) |

On the earlier pin the same wrapper ran every one of the 14 Kit suites and
reported **zero** relaunches. That zero is the evidence that the wrapper is
inert where the defect is absent, which is what lets it land before the pair is
switched.

## A second failure mode, and what it says about the telemetry extension

Two of those 16 boots did not deadlock — they **crashed**, exit 139, about one
second in. The captured stack names `libomni.kit.telemetry.plugin.so` in two
consecutive frames.

The same 16-boot arm was then run against a clone with the `omni.kit.telemetry`
entry removed from both Kit app files, everything else held constant:

| arm | boots | SIGSEGV | deadlock relaunches |
|---|---|---|---|
| pristine clone | 16 | **2** | 4 |
| `omni.kit.telemetry` removed | 16 | **0** | 6 |

The deadlock rate is unchanged, which agrees with the earlier finding that
removing the extension is inert for *that* defect. The crash is a different
matter: it disappeared, and the stack points at the extension that was removed.

Two-by-two on 2/16 against 0/16 gives Fisher exact p = 0.48, so this is
**suggestive and not established** — the sample is far too small to carry on its
own. It is recorded because the mechanism and the observation agree, and because
it changes what the removal is for: it was adopted as a privacy step, and it may
also be a stability one. The claim to test, if it matters later, is a larger arm.

The wrapper does not retry a crash. A deadlock is silent and produces nothing; a
crash is loud and may be a real regression, and relaunching it would hide that.

## Where a failing suite's results used to go

Suites write JUnit XML to a fixed per-suite path. A run that died before pytest
wrote results would parse **the previous run's file** and report its counts as
the current run's — a crashed suite reading green. The file is now removed
before each spawn, so that case falls through to the existing "XML not
generated" error instead.

This is a different defect from the one that timed-out suites were thought to
have. On a timeout the code never reaches the parse at all, so the stale file
never misled the in-process report; it misled a later tally of the directory,
which is how the earlier baseline capture was affected. Both paths are now
covered, and a failing suite's XML is additionally copied aside under a
timestamped name so the next run cannot erase the evidence of why it failed.

A timed-out suite is also now terminated with SIGTERM, a bounded grace period,
and only then SIGKILL. The deadlocked process catches neither signal and blocks
nothing, so SIGTERM alone ends it; the escalation earns its keep on a healthy
run interrupted mid-write, which is what left carb shared memory and named
semaphores behind.

## Gate results on the earlier pin

Run on this branch, against the pair the tooling currently selects:

| gate | result |
|---|---|
| pure suite, unmodified `main` (control) | 1252 passed, 1 skipped |
| pure suite, this branch | 1252 passed, 1 skipped |
| Kit suites, this branch, through the watchdog | 487 tests, **487 passed**, **0 relaunches** |
| contract, two files | 148 passed |

An earlier run of the same gates, mid-work, came back 486 of 487: it caught
`test_collision_imu_mean_differs_from_free`, the flake tracked in
[`collision-imu-signal-flaky`](../../tasks/active/investigations/collision-imu-signal-flaky.md),
which then passed on the run above. Both runs are deposited. The failing one is
worth keeping for a second reason: its XML was preserved under a `-FAILRUN-`
name by the change described above, which is the only recording of that
mechanism firing on a real failure rather than a synthetic one.

## Evidence

| | |
|---|---|
| Repository | `https://github.com/zachoines/Sim2RealLab-Artifacts` (private) |
| Directory | `isaac-lab-upgrade-pra-2026-08-26/` |
| Commit | `930434e68b733989696d0c7989ad7e334aa3633a` |

The groups are `recipe-verify/` (the rebuild: its driver, the per-step pip logs,
the resulting freeze and the acceptance diff), `gates/` (the suite runs, the RSS
trajectory, and the two 16-boot arms with every per-attempt log), and
`upstream/` (an unposted draft issue for the carb deadlock). The deposit's own
`DEPOSIT.md` says which is which.

sha256 of every file in the deposit, paths relative to the deposit directory:

```
bf71f9db9e257af3c4c0afb98fe1493287390c0cc6f9cdc907d615ccba065bb6  gates/contract-oldpin.log
75b791f03d264b63b751d76b7bc9d54b165a578adfac84cd7a01976777a09769  gates/healthy-boot-rss-trajectory.csv
6a4159e894707d10463aa2aefc9cd9414a35c83cd6fa6184f862510fee5c4b3d  gates/pure-oldpin-main-CONTROL.log
0ba8b8180f5b402bdf92d3d933685220ff1c9ca2cab1dce7350f3f871565ae14  gates/telemetry-arm/boot_probe.py
d94adc24a9d2dcdbedda4ceb85b8ecae5100fd2ebe274f5a465218b02dc7abeb  gates/telemetry-arm/run.sh
9b32e5ca0791e3308d7b99fa2ac14e805f8429791c22a6f039f0fd3a0b411bbc  gates/telemetry-arm/tel1.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel10.log.attempt1
1974cb052b8cad53c4c3a6bb941e4fd586af47ee208026b791bc193ba01f01b0  gates/telemetry-arm/tel10.log.attempt2
041dc39a908f1468888d040706686a92d6c6984a45ddc882fef769f2991cd866  gates/telemetry-arm/tel11.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel12.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel12.log.attempt2
766f734d2092566073c7eb0b3a67528ec7c1082d73012067e92544cf8087b0e4  gates/telemetry-arm/tel12.log.attempt3
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel13.log.attempt1
db84b4762f87f9cb0bd16bea40af1725fb767b3847e668421f4bdfdcce0059e1  gates/telemetry-arm/tel13.log.attempt2
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel14.log.attempt1
ba0eb8bbc639c0322d4e55c7c6c9ec8d42ddd7fd043bfba3060c49c8acd56a55  gates/telemetry-arm/tel14.log.attempt2
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/telemetry-arm/tel15.log.attempt1
ea55077cb1c9a157cc0dce283658565084121740a828a130fed96df3687b3c9d  gates/telemetry-arm/tel15.log.attempt2
09ac8e30f4363682e93de33a7d563ae52ddfd80359f17e8546019716e2e3187e  gates/telemetry-arm/tel16.log.attempt1
059ef65b2a382b2ccb8799545313cefa15601739dfe4869eca4617f339d66d33  gates/telemetry-arm/tel2.log.attempt1
97fc6a44bd01764268833bdf36cdaca452ca148741081aa841829529f88d7982  gates/telemetry-arm/tel3.log.attempt1
18014d15a479cb09d73000e8cc168f477e36203191e8258b08372b17e14f61f3  gates/telemetry-arm/tel4.log.attempt1
10d8c60907bc325ee29edfd7beb81c40138a6fbf41442dcbf8418c1621de2f3d  gates/telemetry-arm/tel5.log.attempt1
0fcb2552d0705bb74914b512ab5fe463879a379c2ea917c180f30f88faec2fef  gates/telemetry-arm/tel6.log.attempt1
4cc3a5aaa29f659c74d93bfad438e24dbbf2ed75a8295dd70796745d494478f2  gates/telemetry-arm/tel7.log.attempt1
2c0f73d86d19db70e8f6d9f9c8c33fa728a1281549a1372461f4ea674221093a  gates/telemetry-arm/tel8.log.attempt1
36f802b4b19b93d993b86ee7a9849afee9f99cd8c5aed1cdeee0a1f50cf09612  gates/telemetry-arm/tel9.log.attempt1
8757b8c15d1747d1d4677b9e476ed282a58007635f11641560b3ed7d52c253dd  gates/telemetry-arm/telemetry-removed-16-boots.log
3df7c8a6242a7f4644105b717217802eeaa3bcf41d4ba00c7b8ada85d6f0115c  gates/test-lab-oldpin-FINAL.log
0faeed652f90452fa0ed53f1962f32db8688692552415a7ae81eaeeaf8939b59  gates/test-lab-oldpin-INTERIM.log
c9f013452b110769315b8891c38cb22b291599a47ad3caaef981a5e0f89ec0d6  gates/watchdog-validation/boot1.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/watchdog-validation/boot10.log.attempt1
1eacd4fdc0109615fcbedf89c4cca6df5280b316bb0cc4fc82d42a377008fd18  gates/watchdog-validation/boot10.log.attempt2
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/watchdog-validation/boot11.log.attempt1
9db78a1a2be070813be239dfc3cd5fbea33f1cb6a1a30922aeb6ce39f3bf8077  gates/watchdog-validation/boot11.log.attempt2
dcc4b8c2ff8eb352fcb401b577e2f5e7a4182bc6ed5da11de92b30cdf9e69fc0  gates/watchdog-validation/boot12.log.attempt1
d4916403e718c2afd3a38754d596191b36591d7e150a5830db6f47136e1d147c  gates/watchdog-validation/boot13.log.attempt1
4470cb0075261014cce13ed49b8297cfa56ea300767e6ce9e59d4867ca62a341  gates/watchdog-validation/boot14.log.attempt1
adf9aefa0f23bcb5b5e5287b10217d46550d61977ed5f9098df74673964b4404  gates/watchdog-validation/boot15.log.attempt1
caf2322d0ebcf629b560b44183b6e4e0add81b8354ab3c46d4265cf02a466f6b  gates/watchdog-validation/boot16.log.attempt1
900790ec814b921674bae121a1f587d178da829162b9080af844b83eeefb2f45  gates/watchdog-validation/boot2.log.attempt1
63961b70600e5d474037bcff2f6d64aea6d4132f20df639513cea5c89b05b0b8  gates/watchdog-validation/boot3.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/watchdog-validation/boot4.log.attempt1
827d47e74cca32589ed0f4f8c74e94f48c45496d23c21aadf755c2378e8420e7  gates/watchdog-validation/boot4.log.attempt2
df05a10840f25df25be442d8833b642d53c6b29338ae8ee3f617c5f5ccc3750d  gates/watchdog-validation/boot4.log.attempt3
8211cd670be90f55cd20fb74320801adc13b7be12d43caf6793c2956a7c3a5d2  gates/watchdog-validation/boot5.log.attempt1
ebaabfd7332967775ef6f3f8e91abaf3d3e16c495a8b65ba0fb2e52ec7454ba0  gates/watchdog-validation/boot6.log.attempt1
f1428062f7e8316647d664bcfc6af8acc0b92a6769f444906f03bec6f0c444d6  gates/watchdog-validation/boot7.log.attempt1
c175485a9667f46a84c82a83c0a63248231bd6837ec5bb9edf26063fcc81c899  gates/watchdog-validation/boot8.log.attempt1
378934e6d009a5632afb6f979d97f6bd670d77a37994b6b0f6708903a7584d31  gates/watchdog-validation/boot9.log.attempt1
0ba8b8180f5b402bdf92d3d933685220ff1c9ca2cab1dce7350f3f871565ae14  gates/watchdog-validation/boot_probe.py
9a09cd5ab46b03702bcebffbf90a9816d92b0fa5f3250756a8e27849f5f58927  gates/watchdog-validation/run.sh
e70c6e903ec125f7c45dbaed13761aaa24e3a9b930394d062d207601d007dd97  gates/watchdog-validation/watchdog-16-boot-validation.log
d9acf03a7005f10cfcb1993067e19b018c339cc54c31befc03ae75ac7b753d9f  recipe-verify/02-conda-create.log
ee229d7f91c8c763010814acaa303eeab2445b49b33596082c143dbef7bb4379  recipe-verify/03-torch.log
c8513f0aff2fab2a2c0475b1a58e5264d787a6231a7ed73962621742aed9061e  recipe-verify/04-isaacsim.log
0dfa56fc268ed7fdda1195fdde1f200322643d20293fddefda82b0507bbbee1b  recipe-verify/05-torch-force.log
41f264ba60f38e8cd5b71b36a4cdadfdc2cd279396cb3ab27d36ce41cbe88943  recipe-verify/06-isaaclab-editables.log
49fb91536d61b58171833b20f38818b6cafc9de7ed532e51cc5cd06db55bf7d8  recipe-verify/07-rslrl.log
ad06f724bd1ac198d6f90d021f4c0b0f679db258958adb6f7825183c33162070  recipe-verify/08-onnxscript.log
7deb218e542338247d2baf2fa55e83d204e667efa60786209eca78db1e04886c  recipe-verify/09-strafer-shared.log
1b972489379adbaa8c449121bba19f4d15861426254983e10874f1dbc50934a8  recipe-verify/10-strafer-lab.log
dde3dd991acdd998f4229ad0a0148a1f4144a16ad17c5c76c5b0af64e43f53b5  recipe-verify/11-lerobot.log
60b974c2e5cb024c03c21ca36c59f3039038ef1cad7cabfc78f955de15050dda  recipe-verify/12-lerobot-deps.log
19a9d060d11a8cca3953186f4183f9f822f9ce69b4960e498373e2335683ab04  recipe-verify/13-torch-final.log
caef26a20df20c3b55a11baf4ba4cac536c6189d5d39e54dcc8989ccd625bb3d  recipe-verify/14-cudnn-repair.log
40dbabfc9e490b1db5f692b25234c7dc09169edfe9b2fd314e6e463ffb666e18  recipe-verify/21-onnxruntime.log
b86e51d113c4e0c6ce1ac56552a160d3baa03c7d96539656a02ec054cfbe42a4  recipe-verify/24-packaging.log
66a08bf84c08e3ead5a29a5a8ad1f41880e70df5b4e95d0200e07f7d42b31108  recipe-verify/31-torchcodec.log
1e7c7604dae119e93f7861f817dbe92f3dfff1801638b2e966def707118cae62  recipe-verify/40-pyspy.log
c4378cc1694568b9433521931a5c8eacebbec205b53c63c36e3dca0136cdf000  recipe-verify/ACCEPTANCE.diff
30c19b4612c2bf30ca6c4a82b27e1a39dd82c7f0a7fc6a04e1e85dae53e55b81  recipe-verify/DRIVER.log
9f06d5ff0a4b7f5f65afa656016c7e028a9f75e78cb4fda31e8a71eab01c8836  recipe-verify/freeze-verify.txt
a5544d33e5af758addc205cb9fa8af602937a3834e3a96a2df4708a5f0abb383  recipe-verify/verify_build.sh
0f9b24d24f3b8c8338880e5306977ffd159f7f4dc727272bca5c46eeb1e5f3f4  upstream/UPSTREAM-ISSUE-DRAFT.md
```
