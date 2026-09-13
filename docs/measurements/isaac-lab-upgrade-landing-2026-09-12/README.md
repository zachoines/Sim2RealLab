# Isaac Lab upgrade — the pair flip, and what the rename does to an environment — 2026-09-12

The tagged pair (Isaac Sim 6.0.1.0 + Isaac Lab `v3.0.0-beta2.patch1`) was promoted to the
canonical names and the previous pair retired beside it. This record is the flip's ledger:
what was renamed, what the rename silently broke, what the gates read afterwards, and
whether the camera anchor the Stage 3 render comparison was confounded by now takes.

The flip produces almost no tracked diff by design. `.env`, `.env.example`, the `Makefile`
defaults and `repo-topology.md` name the pair by role, so they read the same before and
after and resolve to a different stack. The evidence is therefore this record, not a
diff.

## Verdicts

| item | result |
|---|---|
| **rename** — canonical pair after the flip | `env_isaaclab3` + `~/Documents/repos/IsaacLab` → Isaac Sim **6.0.1.0**, torch **2.11.0+cu130**, clone at tag `v3.0.0-beta2.patch1` |
| **rename** — retired pair | `env_isaaclab3-retired` + `~/Documents/repos/IsaacLab-retired` → Isaac Sim **6.0.0.0**, torch **2.10.0+cu130**, intact and importable |
| **rename** — editable bindings | **both** environments needed re-linking; the retired one fails *silently* if it is skipped (below) |
| **rename** — pip in the promoted env | **broken by `conda rename`** and repaired; two pip versions were layered by the clone step (below) |
| **rename** — console-script shebangs | **no repair needed** — conda 26.1.1 rewrote all 129 correctly |
| **Kit app** — telemetry deletion | does **not** ride the `mv`; re-applied to the promoted clone, 2 lines, verified 0 occurrences |
| **goldens** | **22 of 26 moved, all `contract`**; the 4 layout/depth-obs/palette goldens byte-identical; movement reproduced exactly by dropping `{cmd_kind, element_names}`; **purely additive** |
| **gate** — contract | **148 / 148** |
| **gate** — pure | **1252 passed, 1 skipped** (the expected `strafer_inference` skip) |
| **gate** — Kit suite | **487 / 487, 0 failed, 0 errors — ALL PASSED**, twice; **6 watchdog relaunches** on the first pass and **0** on the second |
| **gate** — harness smoke | **PASS**, `reloaded: episodes=1 frames=32`, 0 relaunches |
| **gate** — bridge cadence | reaches `publish 30.00 Hz sim`, no `WARNING` lines |
| **prerequisite the Kit gate exposed** | Stage 3 FINDING 1 (the zero-match contact filter) was still unfixed on `main` and had to be removed to reach 487 |
| **render anchor** — both pins | **takes on both**, proven by stage readback: retired 0.0 m, canonical 1.8e-15 m |
| **render shift** — matched poses | **CONFIRMED, and larger than the confounded estimate**: luma 87.6 → 62.6, crush 0.0005 → 0.1713; rescale factor **0.715**, not 0.792 |

## The rename silently repoints the retired environment

This is the finding that changed how the flip was run, and it is the one worth carrying
forward.

Every `isaaclab_*` editable is bound by a setuptools finder shim holding an **absolute**
path:

```python
MAPPING: dict[str, str] = {'isaaclab': '/home/zachoines/Documents/repos/IsaacLab/source/isaaclab/isaaclab'}
```

Neither `conda rename` nor `mv` rewrites it. The dispatched sequence renames the old env,
promotes the candidate, then moves `IsaacLab` → `IsaacLab-retired` and
`IsaacLab-3beta2` → `IsaacLab`. After those two moves:

- the **promoted** environment's 15 finders name `IsaacLab-3beta2`, which no longer
  exists, so `import isaaclab` raises. Loud, and repaired by the re-link the flip
  already planned;
- the **retired** environment's 14 finders name `.../repos/IsaacLab`, which still
  exists — and now holds the *candidate* clone.

Both were captured live, between the moves and the re-link:

```
promoted env  : find_spec('isaaclab') -> NOT FOUND
retired  env  : find_spec('isaaclab') -> .../repos/IsaacLab/source/isaaclab/isaaclab/__init__.py
```

The second is Isaac Sim 6.0.0.0 site-packages resolving Isaac Lab v3.0.0-beta2 sources:
a combination that is neither pair, with nothing raised. The upgrade brief calls the old
pair "the only way to recompute pre-bump config hashes"; left unrepaired, the flip would
have quietly destroyed exactly that. **Both** environments were therefore re-linked, the
retired one against `IsaacLab-retired`.

`isaaclab.sh` cannot detect this. Line 29 exports
`PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"` with `ISAACLAB_PATH` derived
from the script's own location, so `isaaclab.sh -p -c "import isaaclab"` self-corrects
after any move regardless of the editable state — it was observed importing `isaaclab`
successfully under **base** Python, in an environment with no Isaac Sim at all. Only a
bare-interpreter `find_spec` sees the truth. That asymmetry is why the new `make
test-lab` guard asserts the finder mapping rather than the launcher.

## `conda rename` broke pip in the promoted environment

Unforeseen, and worth knowing before the next rename.

```
ImportError: cannot import name 'get_runnable_pip' from 'pip._internal.utils.misc'
```

`conda rename` is clone-then-remove, and the clone re-materialises conda-tracked files.
The environment carried conda-tracked pip **26.1.2** underneath a pip-installed
**26.2.1**. The clone restored 26.1.2's `__init__.py` and `misc.py` over 26.2.1's but
left 26.2.1's `pip/_internal/build_env/` **package** directory beside 26.1.2's
`build_env.py` **module**. Python prefers the package, so 26.2.1's `build_env` loaded
against 26.1.2's `misc` and the import failed. Both `bin/pip` and `python -m pip` were
dead — and the editable re-link needs pip.

Repaired by removing the mixed `pip/` tree and both dist-infos, bootstrapping with
`ensurepip` (25.0.1), then restoring 26.2.1. `setuptools 78.1.0` and `wheel 0.47.0` were
verified unchanged afterwards; pip is not pinned in `constraints-isaac-lab.txt`. The
retired environment was checked and is unaffected — it carries a single coherent pip
26.0.1.

The predicted failure in the same area did **not** occur: 129 console scripts in the
candidate environment's `bin/` did carry a `env_isaaclab3beta2` shebang, and conda
rewrote every one. Post-rename count of stale shebangs is 0 in both environments.

**The rule to carry:** `conda rename` is safe for conda-tracked content and for shebangs,
but an environment holding a pip-upgraded copy of a conda-tracked package can come out of
the clone with two versions layered. Check `python -m pip --version` after any
`conda rename` of a pip-heavy environment.

## The Kit-app telemetry deletion does not ride the move

The retiring clone carried the deletion as an uncommitted working-tree edit; the
candidate clone was pristine. Moving the candidate into the canonical path therefore
re-enables the extension unless the deletion is re-applied — which also matters for boot
stability, since the `-pra-2026-08-26` record measured a SIGSEGV in
`libomni.kit.telemetry.plugin.so` on 2 of 16 boots of the pristine clone against 0 of 16
with it removed.

Re-applied with the recipe's own command
(`source/strafer_lab/README.md` § Install), which names **two** files, not one:

```
apps/isaaclab.python.kit           line 75
apps/isaaclab.python.headless.kit  line 30
```

Both now read 0 occurrences of `"omni.kit.telemetry" = {}`, and those were the only two
in the clone. The four other `.kit` apps never carried the key. The obsolete
`omni.kit.pip_archive` shim was confirmed absent and deliberately not carried over.

Because the clone is untracked, none of this can appear as a diff.

## The goldens moved exactly as pre-registered

22 of the 26 hashes moved, all of kind `contract`. The other four —
`depthobs-RLDepth_Real`, `layout-depth`, `layout-nocam`, `palette-pre_enrichment` —
have byte-identical preimages, which is the half a deployed checkpoint depends on.

Attribution, from `golden_compare.py` with the pre-registered drop set:

```
drop set                    : ['cmd_kind', 'element_names']
hashes unmoved              : 4/26
hashes moved                : 22
all moved reproduced by drop: True
unmoved preimages identical : True
union added lines           : ['"cmd_kind",', '"element_names",', '[', '],', 'null']
union removed lines         : []
```

Every moved hash reproduces its stored value once the two dropped keys are removed, and
no line is *removed* from any preimage — the movement is purely additive, which is what
a field rename plus two new keys should look like and is the licence for the re-freeze.
The contract test forbids editing goldens to make a test pass; this attribution is what
distinguishes a re-freeze from that.

The 22 literals were rewritten programmatically from `comparison.json`, each guarded by
an assertion that the value in the file matched the tool's `stored_golden` before
substitution. The diff is 22 changed lines, all inside `_CONTRACT_GOLDENS`
(`test_composition_contract.py:110-136`); the four unmoved goldens at `:143`, `:150` and
`:159-162` are untouched.

One correction to the method, recorded because it nearly read as a result:
`golden_compare.py` appends `preimages/` to its `--baseline` argument itself. Invoked as
`--baseline <dir>/preimages` it reads nothing, and then reports
`unmoved preimages identical: False` with empty added/removed unions — a red flag and two
vacuous greens. The numbers above are from the corrected invocation.

## Gates on the canonical pair

| gate | result |
|---|---|
| contract (`test_composition_contract.py` + `test_obs_contract.py`) | **148 passed** |
| orphaned pair (`test_d555_perception_cfg.py` + `depth_noise/test_scene_cfg.py`) | **29 passed** |
| pure (`source/strafer_lab/tests/`) | **1252 passed, 1 skipped** |
| Kit (`run_tests.py all`) | **487 / 487, 0 failed, 0 errors — ALL PASSED** |
| harness smoke (`make harness-smoke`) | **PASS**, `reloaded: episodes=1 frames=32` |
| bridge (`make sim-bridge`) | cadence print reached, no warnings |

The one pure-suite skip is the expected one:
`tests/contracts/test_action_clamp.py:135 — could not import 'strafer_inference.obs_pipeline'`.

The bridge line, verbatim:

```
[sim_in_the_loop] camera cadence: publish 30.00 Hz sim (policy period 30.00 Hz) | frame_skip=3 (derived, derived 3) | bridge tick 120.00 Hz | renders/tick 1.00
```

### Boot-stall telemetry — 6 relaunches across 14 suites

The canonical pin is the pin with the boot defect, so the watchdog stopped being a no-op
the moment the flip landed. Every stall showed the same signature — no CPU and no output
for 60 s during boot — and every one recovered:

| suite | relaunches |
|---|---|
| curriculums | 1 |
| sensors | 1 |
| noise_models | 1 |
| imu | 3 (`test_imu` 1, `test_imu_collision` 2) |
| the other 10 suites | 0 |

`imu:test_imu_collision` needed two relaunches before its third attempt exited clean.
No suite exhausted its attempts.

**A second full pass over the same tree, ninety minutes later, relaunched 0 times** — same
487/487, every suite first-attempt clean. Two passes of the same 14 suites on the same
host and the same pin therefore produced 6 relaunches and 0. That is the behaviour
`kit-boot-hang-2026-09-11` describes ("the host's propensity to produce the stall varies by
more than an order of magnitude over hours"), reproduced here within a single session, and
it is why neither number is a rate. A clean pass is not evidence the defect is gone; the
same-session control below is what carries that weight.

The tracked `collision-imu-signal-flaky` flake did not appear: imu read 4/4.

## The Kit gate could not reach 487 without an unfixed Stage 3 finding

Stage 3 FINDING 1 — `filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"]` at
`test_sim/sensors/depth_noise/scene_cfg.py:120`, a filter matching zero prims because the
scene defines no `Obstacle_*` prim — was assigned to the compatibility PR and did not
land with it. On the canonical pin's `omni.physics.tensors`, a zero-match filter leaves
the sensor's filter buffers unbuilt and `ContactSensor._create_buffers` raises
`AttributeError: 'NoneType' object has no attribute 'filter_count'`.

The filter was removed here because the 487 gate cannot pass with it. The change is safe
on both pins: it matched zero prims on the retired pin too, `Obstacle_` appears nowhere
else in the repository, and no test reads `force_matrix_w` from this scene —
`test_collision_rewards.py` builds its own unfiltered sensor and asserts precisely that
`force_matrix_w` is `None` without a filter. `depth_noise` reads 6/6.

## The render shift is real, and larger than the confounded estimate

Stage 3 measured a photometric gap between the pins and then retracted it in place: the
two stacks had been filmed from different camera poses, because the anchor wrote
`camera_position`/`camera_target`, which v3.0.0-beta2 renamed to `eye`/`lookat`. The dual
write landed in `main` (#214), but a `hasattr` probe proves only that a field *exists* —
not that the recorder read it. Nothing in the repository could tell the difference, and
the reassuring `Recording camera anchored on env_0` line prints the pose the script
*intended* whether or not the write took, and even when no capture object was found.

So this leg adds a readback. After the write, exactly one render is forced — that call is
where `IsaacsimKitPerspectiveVideo.render_rgb_array` first builds its annotator and poses
the camera prim from its own config, so anything read earlier is not the pose the clip was
filmed from — and then the camera prim's world transform is read off the stage and checked
two ways: the eye must land on the request, and the view ray must pass through the target.
The result is written to `camera-anchor.json` beside the MP4.

Both pins were filmed with the same command, environment, seed, env count and iteration
count, and both anchors are now **proven**, not asserted:

| pin | field pair written | eye error | target-ray error |
|---|---|---|---|
| retired (Isaac Lab 4.6.12) | `camera_position/camera_target` | **0.0 m** | **0.0 m** |
| canonical (Isaac Lab 6.1.14) | `eye/lookat` | **1.8e-15 m** | **1.0e-05 m** |

That is also the first direct evidence that #214's dual write does what it was written to
do: each pin picked the field pair its own Isaac Lab exposes, and each landed on the mark.

With the framing controlled, the readings:

| clip | mean luma | crush |
|---|---|---|
| retired pair, this session | **87.6** | **0.0005** |
| canonical pair, this session | **62.6** | **0.1713** |
| Stage 3's deposited old-pin control, re-measured today | 86.6 | 0.0056 |

The third row is the instrument check: the deposited nine-day-old clip reproduces its
recorded 86.6 / 0.0056 to every digit through today's tool, so the measurement path is
the same one Stage 3 used.

**This confirms the brief; it does not close it.** The matched-pose deltas are
**25.0 luma** and **0.171 crush**, against the thresholds that would have exonerated the
renderer (Δ ≤ 10 luma, Δ ≤ 0.01 crush). The gap is not pose.

The correction runs the *opposite* way from what the confound suggested. Stage 3's
unanchored new-pin clips read 68.6 and 68.3 luma; anchored on env 0 the new pin reads
**62.6**. The recorder's default pose was flattering the new stack, so removing the
confound made the shift bigger, not smaller. The disposition's rescale therefore fires,
but the factor is **0.715** (62.6 / 87.6), not the 0.792 the brief carries — and crush,
not luma, is the larger violation: 0.171 is 17× the retired pair's and past the tool's own
0.10 bound, while both pins already fail its absolute luma floor of 90.0, so the tool's
pass/fail verdict cannot separate the stacks and only the numbers can.

Two same-session controls make the comparison sturdier than Stage 3's: the retired-pair
clip was recorded today rather than borrowed from the earlier deposit, and it reads 87.6
against that deposit's 86.6 — about 1 luma of session-to-session spread on one pin, which
is 4% of the 25.0 delta being attributed.

### Boot-stall control, same session, same workload

The retired pair recorded its clip in **one attempt, 0 relaunches**; the canonical pair
needed **1 relaunch** on the identical workload. That is the known-affected control the
boot-hang record asks for whenever a clean-boot claim is made, and it reproduces the
pin asymmetry on a non-pytest launch path.

## Scope and limits

- **6 relaunches is one sample.** It is reported because it is the number this run
  produced, not as a rate. The same caveat the boot-stall record states applies.
- **The retired pair was executed, never written to.** No install, upgrade or write
  touched it; the only change to it is the editable re-link that keeps it pointing at its
  own clone, which is what preserves it as a rollback artifact.
- **Rollback was not rehearsed.** It is written down and its mechanism is now proven in
  one direction (the re-link repairs a moved clone), but no end-to-end rename-back was
  performed, so the upgrade brief's acceptance item asking for a demonstrated rollback is
  not met by this record.
- **The golden re-freeze is auditable only with the evidence repo.** Neither tool ships
  in this repository, so a reviewer with only `Sim2RealLab` checked out cannot
  independently reproduce the 22 hashes.
- **`IsaacLab-verify` disappeared mid-session** and this session cannot say how; see
  `provenance.md`.

## Evidence

### `isaac-lab-upgrade-landing-2026-09-12/` — commit `fe54878c9fe8727e68fbf37694ecadd2908316db`

Deposited in the evidence repository; `DEPOSIT.md` there describes each file. The
flip transcript carries the live capture of both predicted binding failures, the
two gate logs are the 6-relaunch and 0-relaunch passes over the same tree, and the
two `camera-anchor.json` readbacks are what make the render comparison valid.

```
72180de1f8b3fd2833a2918f7ddbf3ea7074c42a784908b83ee35c33761935f0  DEPOSIT.md
4bf83bb7dc33c8dd0ef233a393019d947ccfba617ba666742bec691447f1333c  flip/flip-transcript.log
c64488615124e49152f59ad5589b67d0d1b3f744628683a57056f2cab3c4a545  flip/relink_editables.sh
25837c23d9b0dc4d95e385cb959341a5856518ed28e0d7a40e709881c8c6fc26  gates/kit-suite.log
a2f69e5609a011f77ccb2cffc35d1867d8ffea6ee4d3c63cba9fc5f9f89aceb3  gates/make-test-lab.log
d617d4a40a7dac12a0c0c09c976f306cfaa416f2a23cdf88972c6235233ea5a0  gates/sim-bridge-cadence.log
bcb02cfc6c42fba9e81c784c9a4f26e33d9408cd041c43277b707f9a11013d31  goldens/comparison.json
98c120453f7eedcec678a1cbd2d287949a20e517170254a56d659ecf1cb117bd  goldens/golden-hashes-newpair.json
6cde838a0ff64ea9c8e34c525f01ff9bce9484433bad5e6507e1c981cccfd048  render/camera-anchor-newpin.json
fc5ac9ed8038cdb3b4ef1f6ce834a0edbaeb8bd987b2ef841847ff7a286a9e95  render/camera-anchor-retired.json
ffe40811b6c32f18a7ce385e17a6ec2b4861b69a8513747e31992e040e22b4fa  render/exposure-readings.txt
6952719752ba9b1f27b435c86f38c4a75c9b3d9de41eb732e1cd7462cc2b5647  render/g5-train-clip-newpin.mp4
a06468ebb22904ad05ef0ae079eb305be176a93d53f8be4d09aa28ccbca80351  render/g5-train-clip-retired.mp4
36df0a159a71283cf693506ebd53006c7965b8c85173a010cae13c8507a9364d  render/watchdog-newpin.txt
c07a6c7c754fe016d36a6861ecaa65fbbb3296f30593646e2028041068af1579  render/watchdog-retired.txt
```
