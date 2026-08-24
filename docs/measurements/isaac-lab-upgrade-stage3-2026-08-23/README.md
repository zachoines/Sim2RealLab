# Isaac Lab / Isaac Sim upgrade — Stage 3 validation of the new pair — 2026-08-23

Validation of the post-bump pair (`env_isaaclab3beta2` / IsaacLab `v3.0.0-beta2.patch1`) against
the pre-bump baseline recorded in
[`../isaac-lab-upgrade-baseline-2026-08-14/`](../isaac-lab-upgrade-baseline-2026-08-14/).

**Nothing was flipped.** `.env` still names the old pair, and no repo configuration, Makefile
target, recipe document or golden was changed. The old pair was executed read-only, as a
same-session control, and is byte-for-byte as it was found.

Numbers come from committed scripts and saved files. Where a gate's verdict rests on a
comparison, the comparison is against a **same-session control**, not against the nine-day-old
baseline alone — three times in this session that distinction changed the answer.

Provenance, stack versions and binding proof: [`provenance.md`](provenance.md).

---

## Verdicts

| gate | anchor | measured | verdict |
|---|---|---|---|
| **G0** Kit-mod experiment | first boot of a pristine clone | 13/13, no mods needed, clone still unmodified | **STRICT PASS** |
| **G1** suites | 486/14 + 1230/1 + 147 + 29 | 487 collected: 459 pass, 22 expected golden failures, **6 broken**; pure 1249/1; contract 126/22; orphaned 29 | **PASS with 2 findings** |
| **G2** roller physics | PGS md5 `d1ef40cd…`, p2p ≤3 mm, ride 48±1 | md5 moved; run1≡run2; p2p max 2.151 mm; ride **47.99 mm** | **ATTRIBUTED PASS** |
| **G3** pose trace | 3 hashes | `action_hash` identical; DR draw bit-identical; `inertias` 1.16e−10; trace diverges chaotically | **ATTRIBUTED PASS** |
| **G4** depth obs | array hashes, moments, profiles | hashes moved; mean +0.02 %, std +0.001 %, profiles r = 1.000000, same argmax | **ATTRIBUTED PASS** |
| **G5** render / video | luma 86.6, crush 0.0056 | **luma 68.6/68.3, crush 0.122/0.128**; old-pin control reproduces 86.6/0.0056 exactly | **FAIL — recorded; disposition in the upgrade brief** |
| **G6** export A/B/C | 0.0 same-format; 2.9e−6 / 1.7e−5 cross | A: 8.345e−07 / 4.247e−06 (below investigate floor); **B: export axis exactly 0.0**; C: 85 pass / 1 skip | **ATTRIBUTED PASS** |
| **G7** deterministic eval | REF 0.8119 | same-session REF **0.8575 ± 0.0435** (n=4); new pin **0.8650 ± 0.0289** (n=4); **+0.29 σ** | **PASS** |
| **G8** bridge cadence + parity | 1302/1302 at 33.33 ms | **bridge does not start**: `ModuleNotFoundError: isaacsim.core.utils`; `bridge_harness_smoke.py` PASSes | **BLOCKED — stop-condition #8** |
| **G9** training curve | mean 53.4 steps/s (45.4–61.4) | **53.86** steps/s, no NaN, curves r = 0.86 / 0.89 | **PASS** |

**Two results block the flip pending disposition in the upgrade brief: G5 and G8.** A third, FINDING 4, is not
a gate but is the most consequential thing measured today.

---

## Findings

| # | finding | class |
|---|---|---|
| **1** | `depth_noise` (6 tests): a zero-match contact filter is fatal at patch1 | test-side, for the compatibility PR (the both-pin code-accommodation PR the upgrade brief names) |
| **2** | Kit-booting processes intermittently hang ~24 % of launches — **new pair only** | **blocking; disposition in the upgrade brief** |
| **3** | A timed-out suite silently inherits the previous session's junit XML | instrument, later PR |
| **4** | `Articulation.data` now returns `ProxyArray`; production unaffected, probes not | measurement-side |
| **5** | The eval gate is **not** rerun-deterministic; its band was binomial-only | method |
| **6** | `isaacsim.core.utils` moved to `extsDeprecated/` — 8 sites, 7 scripts | **blocking; disposition in the upgrade brief** |
| **7** | RGB render is ~21 % darker with 22× crushed pixels | **blocking; disposition in the upgrade brief** |
| 8 | `env_setup.sh` does not activate conda — a mixed pair is one inherited var away | operational |
| 9 | Both Kit mods obsolete, but the telemetry deletion was never a correctness fix | recipe |

---

## Binding — and the mixed-pair run produced on purpose

`env_setup.sh` sources `.env` but **does not `conda activate`**; `isaaclab.sh` resolves its
interpreter from `CONDA_PREFIX` (`isaaclab.sh:17-18`). So `source env_setup.sh && $ISAACLAB -p …`
run from a shell where another env is active executes the **old clone's Isaac Lab source on the
other env's interpreter** — silently. That was demonstrated during setup: old `$ISAACLAB` with
`CONDA_PREFIX` pointing at the new env produced `isaaclab.__file__` in the old clone and
`sys.executable` in the new env. `make test-lab` does not activate either (`Makefile:263`); it
inherits whatever environment the invoking shell has active.

Every old-pin leg in this record therefore used
`source env_setup.sh && conda activate env_isaaclab3 && $ISAACLAB -p …`, and every run logs
`sys.executable` + `isaaclab.__file__` before doing anything else.

New-pin shell: [`newpin_env.sh`](newpin_env.sh) — mirrors what `env_setup.sh` derives, against
the new prefix, without sourcing `.env`.

---

---

## Gate detail

### G0 — first boot / Kit-mod verdicts — STRICT PASS
cmd: `$NEWLAB -p source/strafer_lab/run_tests.py terminations`
- 13/13 passed on the **pristine** new clone. No telemetry deletion, no `omni.kit.pip_archive` shim.
- **Verdict: both Kit mods are obsolete at patch1.** Neither is needed in the recipe PR (the landing PR that rewrites the install recipe).
- Clone still shows zero modifications after the boot (`git status --porcelain` empty) — the strong form.
- No EULA prompt; the interactively accepted `EULA_ACCEPTED` marker under the new env's
  `isaacsim/kit/` is sufficient. `OMNI_KIT_ACCEPT_EULA` was never set.
- Boot-to-first-test ~86 s cold.

### G0 addendum — what the telemetry deletion actually bought
Booting the pristine clone leaves an `omni.telemetry.transmitter` process resident after the
Kit run (observed as pid 889884, parented to init, started at the G0 boot; extension
`omni.kit.telemetry-0.5.2+f9bf0dda`, endpoint `opentelemetry.analytics.nvidiagrid.net`,
`enableAnonymousData=true`). It is **not** a GPU compute process — it never appears in
`nvidia-smi --query-compute-apps` — so it does not violate the resident-process rule, but it
outlives the run and one is spawned per Kit boot.

So the two Kit mods split, and the recipe PR should say so:
- `omni.kit.pip_archive` shim — **obsolete**, nothing to decide.
- `omni.kit.telemetry` deletion — **not required for function** (the boot proves that), but it
  was never a correctness fix; removing it re-enables an outbound telemetry daemon. Keeping or
  dropping it is a policy choice, and the recipe should present it as one rather than
  as a build step that patch1 made unnecessary.

### G1 — Kit suite `all`, new pin — FINDINGS (not the pre-registered result)
cmd: `$NEWLAB -p source/strafer_lab/run_tests.py all` (`run_tests.py` verified unchanged since
`e7ea7bd`; SUITES map still 14 entries, orphaned pair still excluded).
Result: **TOTAL 216 tests, 212 pass, 0 fail, 6 errors** — against a baseline of 486/14, 485 pass.
Twelve of fourteen suites are green and match baseline exactly:

| suite | baseline | new pin |
|---|---|---|
| terminations | 13 | 13 pass |
| events | 14 | 14 pass |
| commands | 8 | 8 pass |
| observations | 13 | 13 pass |
| curriculums | 4 | 4 pass |
| rewards | 46 | 46 pass |
| sensors | 1 | 1 pass |
| actions | 47 | 47 pass |
| **env** | **268** | **0 collected — ERROR TIMEOUT after 300 s** |
| noise_models | 55 | 55 pass |
| **depth_noise** | **6** | **0/4 passed, 5 errors** |
| imu | 4 | 4 pass (no flake this run) |
| obs_dump | 4 | 4 pass |
| camera_jitter | 3 | 3 pass |

The imu flake did **not** fire this run (4/4). Baseline recorded it as a coin flip; one green
run is consistent with that and is not evidence it is fixed.
Failing XMLs preserved with a `-FAILRUN` suffix before anything was re-run
(`suites/kit-junit-xml-FAILRUN/`, 18 files) together with the full-run log.

#### FINDING 1 — `depth_noise` (6 tests): a zero-match contact filter is now fatal
Signature, identical on all four collected tests (`test_gaussian` ×2, `test_frame_drops` ×2;
`test_holes` hit the 300 s per-file cap before reporting):

```
test_sim/sensors/depth_noise/utils.py:824: in create_depth_test_env
    env = ManagerBasedRLEnv(cfg)
  ... isaaclab/envs/manager_based_env.py:219: self.sim.reset()
  ... isaaclab_physx/sensors/contact_sensor/contact_sensor.py:357: in _create_buffers
    self._num_filter_shapes = self.contact_view.filter_count if self.cfg.filter_prim_paths_expr else 0
  ... isaacsim/extscache/omni.physics.tensors-110.1.13.../omni/physics/tensors/api.py:4879: in filter_count
    return self._backend.filter_count
E   AttributeError: 'NoneType' object has no attribute 'filter_count'
```

Cause, and it is not the bump inventing a problem: the depth-noise **test** scene declares
`filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"]`
(`test_sim/sensors/depth_noise/scene_cfg.py:120`) while **defining no `Obstacle_*` prim at
all** — that string occurs exactly once in the file, in the filter itself. The scene's only
obstacle is `TestWall`. So the filter has always matched zero prims; the old stack returned a
usable view anyway, and `omni.physics.tensors 110.1.13` now hands back a view whose
`_backend` is `None`, which upstream's `_create_buffers` dereferences without a guard.

**Blast radius is provably test-only.** `filter_prim_paths_expr` is set in exactly one place in
the whole repo — that test scene. Every production `ContactSensorCfg`
(`strafer_env_cfg.py:214, 258, 334, 412, 1506, 1571, 1617`) leaves it unset and reads
`net_forces_w`, which `test_collision_rewards.py:61` documents as the deliberate choice. No
shipped env, no deployed policy path, and no training env constructs a filtered contact view.

Class: same family as the §12.5 `set_debug_vis` stub-env break — a **test-side** accommodation,
both-pin compatible (deleting a filter that matches nothing changes nothing at the old pin).
It belongs in the compatibility PR, not in the measurement record, and it is **not** a STOP: the pre-registered
STOP clause covers runtime failures in the **env** suite via manager/warp/`.torch` call sites;
this is the depth_noise suite via a filtered contact view.

#### FINDING 2 — `env` suite: 300 s TIMEOUT under `run_tests.py`, 19 s outside it
The env suite's *tests* are fine on the new pin. Run outside the harness the suite is
**269 collected / 22 failed / 0 errors in 18.98 s**, and the 22 are **exactly**
`test_composed_rl_variant_matches_frozen_contract` and nothing else — the pre-registered
expected result, with **no** manager/warp/`.torch` call-site failure. (269 vs the baseline's
268 is the same +1 that moved the contract base 147 -> 148 at #206.)

```
$NEWLAB -p -m pytest source/strafer_lab/test_sim/env --tb=short -q --junit-xml=...
  -> 269 tests, 22 failures, 0 errors, 18.98 s      [suites/env-uncapped-newpin.xml]
$NEWPY  -m pytest source/strafer_lab/test_sim/env --tb=short -q --junit-xml=...
  -> exit 0 in 25 s wall (bare interpreter, run_tests.py's exact argv)
$NEWLAB -p source/strafer_lab/run_tests.py env      (nothing else running)
  -> ERROR TIMEOUT after 300 s, 0 collected          [reproduced twice]
```

Excluded as causes:
- **Contention.** The reproduction ran with an empty `nvidia-smi --query-compute-apps` and no
  other job; the Gate A export leg finished at 10:38:47, before the first env pytest started
  at 10:39.
- **Pipe-buffer deadlock.** `run_tests.py::_run_subprocess` already redirects the child to
  `tempfile.TemporaryFile` rather than `PIPE`, specifically to avoid that.
- **The invocation.** Bare `python -m pytest` with run_tests.py's exact argv completes in 25 s.

So `make test-lab` does not produce a usable Kit total on the new pin, while the tests it runs
are green. Forensics in progress; see FINDING 2b.

#### FINDING 2b — the hang is intermittent and roves between suites
A second full run (RUN2, nothing else running, 1656 s wall) put the timeout somewhere else
entirely. Union of the two full runs plus the standalone repetitions:

| suite | RUN1 | RUN2 | standalone | best observed |
|---|---|---|---|---|
| terminations | 13 pass | 13 pass | 13 pass (G0) | **13/13** |
| events | 14 pass | 14 pass | — | **14/14** |
| commands | 8 pass | 8 pass | — | **8/8** |
| observations | 13 pass | 13 pass | — | **13/13** |
| curriculums | 4 pass | 4 pass | — | **4/4** |
| rewards | 46 pass | **TIMEOUT** (test_rewards) | — | **46/46** |
| sensors | 1 pass | 1 pass | — | **1/1** |
| actions | 47 pass | 47 pass | — | **47/47** |
| env | **TIMEOUT** | 247 pass / 22 fail | 269 in 19 s; 269 in 25 s; **TIMEOUT** | **269 (247+22)** |
| noise_models | 55 pass | 55 pass | — | **55/55** |
| depth_noise | 0/4 + TIMEOUT | 0/4 + TIMEOUT | — | **0/6 — real break, FINDING 1** |
| imu | 4 pass | **TIMEOUT** ×2 | — | **4/4** |
| obs_dump | 4 pass | 4 pass | — | **4/4** |
| camera_jitter | 3 pass | **TIMEOUT** | — | **3/3** |
| **total** | 216 | 439 | — | **487 collected** |

Every suite except `depth_noise` has been observed running to completion and green, and the 22
env failures are the pre-registered moved goldens. So the new-pin Kit result, assembled from
runs that each completed the suite in question, is:

> **487 collected** (= baseline 486 + the one env test #206 added) — **459 pass**, **22 fail**
> (all `test_composed_rl_variant_matches_frozen_contract`), **6 broken** (all `depth_noise`,
> FINDING 1). The `collision-imu` flake did not fire in the one run where imu completed (4/4).

**But no single run produces that total**, because between one and four Kit-booting pytest
subprocesses hang past the 300 s cap on every full run, in a different set of suites each time
(RUN1: env, holes. RUN2: test_rewards, holes, imu ×2, camera_jitter). The hung suite completes
in ~20 s when run again. This is a **new-pin behaviour with no baseline counterpart** — the
08-14 baseline ran 486/14 with only the imu flake — and it makes `make test-lab` non-reproducible
on the new pair. An old-pin control run of the identical command is in progress to establish
whether the host, rather than the bump, is responsible.

`test_holes` is the one consistent timeout (both runs). It shares `create_depth_test_env(...,
use_test_scene=True)` with the two files that error (`test_holes.py:92`), so it hits the same
broken contact filter; the difference is that the exception is raised from inside a physics
callback during `sim.reset()`, and there the process appears to hang on teardown rather than
report. FINDING 1 covers both.

#### FINDING 3 — a timeout silently inherits the previous session's junit XML
`run_tests.py` writes each suite's junit to a fixed path and does not unlink it first; on
timeout the child is killed before pytest writes results, so **the file left at that path from
the previous run survives and reads as this run's result**. Concretely, when the `-FAILRUN` set
was preserved, `kit-suite-env.xml` and `kit-suite-depth_noise_test_holes.xml` carried internal
timestamps of `2026-08-14T15:48` and `2026-08-14T15:52` — the **pre-bump baseline's** files,
which would have contributed 270 passing tests to a run that collected neither suite. The same
thing recurred for RUN2 (5 of its fixed-path files were not written by RUN2).
Both quarantined under `suites/stale-not-this-session/` with a README; each preserved set now
sums only to its own run (`-FAILRUN` 216/0/4, `RUN2` 439/22/4 — the console error counts are
those plus the synthesised TIMEOUT pseudo-errors, which have no XML).
This is the mirror of the §11.8 rerun hazard and wants the same durable fix: unlink the XML
before launching a suite, so an absent result can never read as a stale pass. Instrument change,
so it belongs to a later PR — named here, not made here.

#### FINDING 2c — the old-pin control settles the attribution
Identical command, same session, same host, nothing else running:
`source env_setup.sh && conda activate env_isaaclab3 && $ISAACLAB -p source/strafer_lab/run_tests.py all`
(binding proof logged: `env_isaaclab3` interpreter, old-clone `isaaclab.__file__`, torch 2.10.0+cu130).

> **OLD pin: 487 collected, 487 passed, 0 failed, 0 errors, ALL PASSED, 666 s wall. No timeouts.**
> XML set: 18 files summing to exactly 487/0/0, every file written by that run.

Four things follow, and they are the reason the control was worth the GPU time:
1. **487 is the right collected total at `ab2daed`.** The baseline's 486 was taken at `e7ea7bd`;
   #206 added one env test. Both pins now collect 487, so the comparison is like-for-like.
2. **The roving 300 s hang is a property of the new pair**, not of this host, this checkout, or
   this session. Old pin: 0 timeouts in 666 s. New pin: 2 timeouts in RUN1, 4 in RUN2 (1656 s).
3. **The `depth_noise` break is new-pin only** — 6/6 pass on the old pin. The zero-match contact
   filter at `scene_cfg.py:120` has always been dead config; only the new
   `omni.physics.tensors 110.1.13` turns it into an `AttributeError`.
4. **The 22 contract failures are new-pin only** — the old pin runs 269/269 in `env`, which is
   also the independent confirmation that #203 is inert on the pin `main` ships.

Wall-clock is **not** a clean new-vs-old comparison and is not claimed as one: RUN2's 1656 s
contains 1200 s of pure timeout waiting and omits the work of the four suites it never finished.

---

### G1 (Kit-free half) — pure suite + contract — PASS, matches pre-registration exactly
- pure suite `$NEWPY -m pytest -q source/strafer_lab/tests/`: **1249 passed, 1 skipped**, 111.7 s
  — exactly the pre-registered 1230 + 19 (#206). **Skip count is exactly 1**, and its identity is
  the standing one: `tests/contracts/test_action_clamp.py:135 — could not import
  'strafer_inference.obs_pipeline'`. The export gate is therefore **running**, not silently
  skipped: all four `test_recurrent_contract_e2e` tests report **PASSED**, including
  `test_pt_and_onnx_produce_numerically_close_actions_across_sequence`.
- contract two-file: **126 passed / 22 failed**, all 22 the moved goldens, nothing else.
- orphaned pair standalone: **29 passed** — identical to baseline.

### G2 — roller / physics — ATTRIBUTED PASS
cmd (each leg via `kit_retry.sh`, new pin, `--solver-type 0` explicit on every PGS leg):
`$NEWLAB -p source/strafer_lab/scripts/roller_bounce_probe.py --headless --solver-type 0
 --omega-fracs 0.25,0.5,0.75,1.0 --duration 5.0 --csv <dir>/roller_z_pgs_run{1,2}.csv`
Retries: PGS run1 0, PGS run2 0, TGS 0, `--inspect` **1 hang then success** (post-sweep).

**Strict gate does not hold.** Baseline PGS run1 md5 `d1ef40cdd2b04e4058e3d4369670079d` (matches
the pre-registered anchor exactly, so the right file is being compared); new pin
`dc0570e27ec934aaa71517209b5602d8`. Physics moved across the bump.

**But D0 still holds on the new pin**, so the gate stays strict-grade rather than degrading to
bands: `roller_z_pgs_run1.csv` and `run2.csv` are **BIT-IDENTICAL** (one md5 across two separate
Kit boots), exactly as the baseline was bit-identical across four. Every delta below is therefore
real signal, not spread.

| frac | late p2p mm (base -> new) | growth (base -> new) | roller rad/s (base -> new) |
|---|---|---|---|
| 0.25 | 0.266 -> **0.249** | 0.53 -> 0.558 | 66.4 -> 64.6 |
| 0.50 | 0.945 -> **1.001** | 0.72 -> 0.913 | 118.5 -> 118.0 |
| 0.75 | 1.598 -> **1.297** | 0.81 -> 0.643 | 154.8 -> 151.2 |
| 1.00 | 2.108 -> **2.151** | 0.48 -> 0.734 | 185.4 -> 183.8 |

Diagnostics, all inside their pre-registered bands:
- **late p2p ≤ 3 mm at every frac** — max **2.151 mm** (baseline max 2.108; +2 %). PASS
- **growth ≤ 1.3** — max **0.913**. PASS
- **ride height 48 ± 1 mm** — rest **47.99 mm**, *identical to the baseline's 47.99*;
  spin-mean 47.56 (baseline 47.56), min 47.26 (baseline 47.31). PASS, and it is the strongest
  single statement in this gate: contact/penetration behaviour did not move.
- roller speed at frac 1.0 = **183.8 rad/s** — band retired per §11.4, reported only.

**TGS positive control still has its eyes** (`--solver-type 1`, not the shipped config):
late p2p 0.370 / 4.884 / **14.556** / 14.040 mm, peak chassis **69.4 mm** against PGS's 48.7.
Separation from PGS is **11.2×** at frac 0.75 and **6.5×** at frac 1.0, with a **+20.7 mm**
peak-chassis lift (baseline: 11.9× and +22.5 mm). The fault class the probe exists to detect is
still detected. The TGS numbers themselves moved more than the PGS ones — expected, and
immaterial: TGS is not what the nav envs ship.

**No STOP condition fired**: no TGS-class signature on PGS (late p2p 2.151 mm vs the ≳16 mm
stop line; growth 0.913 vs ≳2×), ride height inside 48 ± 1, control reproduces.
Verdict: **ATTRIBUTED PASS** — a small, self-consistent, bit-reproducible PhysX 6.0.0 -> 6.0.1
change, with the shipped solver's discriminating quantities (late p2p, ride height) unmoved.

---

### G3 — seeded pose trace — ATTRIBUTED PASS (+ a gate-design note)
The committed baseline probe **cannot run on the new pin**: every `Articulation.data` field is
now `isaaclab.utils.warp.proxy_array.ProxyArray` — neither `wp.array` nor `torch.Tensor` — so
`_np()` fell through to `np.asarray`, whose forwarded `__array__` raises
`TypeError: can't convert cuda:0 device type tensor to numpy`. This is the §2.10/§12.5
warp->`.torch` churn firing as a real call-site break rather than as deprecation noise.
(`root_physx_view.get_masses()` still returns a genuine `warp._src.types.array`, so the probe
now meets three shapes, not one.)

**Production is not exposed.** `filter`-free grep over the shipped tree: 73 `wp.to_torch` call
sites (all covered by patch1's shim, and exercised green by the 459 passing Kit tests) and
**zero** bare `np.asarray`/`.cpu()`/`.numpy()` applied to a `.data` field. The break is specific
to code that converts `.data` without going through warp — which here is the measurement probe.

A Stage 3 copy of the probe (`pose_trace_probe.py`, committed here) adds **one guarded branch**
— unwrap `.torch` when the object is neither a `wp.array` nor a `torch.Tensor` — and nothing
else. **The edit is proven inert, not assumed inert**: run on the OLD pin it reproduces all
three baseline hashes exactly (`action` `8dce355f…`, `dr` `194fe656…`, `trace` `c6632a4e…`),
so every movement below is the stack's, not the edit's.

Retries under the FINDING 4 hang: run1 1 hang then success, run2 0.

| hash | baseline | new run1 | new run2 | verdict |
|---|---|---|---|---|
| `action_hash` | `8dce355f…` | `8dce355f…` | same | **IDENTICAL** |
| `dr_hash` | `194fe656…` | `76829356…` | same | MOVED |
| `trace_hash` | `c6632a4e…` | `1f66ade4…` | same | MOVED |

**Step 1 of the tree passes outright**: `action_hash` is identical, so torch 2.11 did **not**
change the seeded CUDA uniform stream and the pre-registered `--action-device cpu` fallback is
not needed. The baseline trace stays comparable.

**Step 2 needs the arrays, not the hashes, and they say something different from "RNG order
changed".** Per-array deltas, baseline vs new run1:

| array | max abs delta | reading |
|---|---|---|
| `actions`, `joint_pos`, `joint_vel`, `masses`, `material_properties`, `root_state_w` | **exactly 0** | the DR draw and reset state are **bit-identical** — the RNG stream did not move |
| `inertias` | **1.16e-10** | a float-level difference in a derived tensor — and the sole reason `dr_hash` moved |
| `positions` | 3.55 | the rollout |
| `quats` | 1.88 | the rollout |

So `dr_hash` did not move because randomisation changed; it moved because `inertias` sits inside
the hash and PhysX 6.0.1 computes it 1e-10 differently. **New pin is self-deterministic**:
run1 == run2 to 0.0 on every array, as at baseline.

The trajectory difference is that 1e-10 amplifying, and it is measured rather than asserted:

```
step   0: 1.49e-08 m   step   5: 1.24e-02   step  40: 1.60e-01   step 200: 3.22e+00
step   1: 4.47e-08     step  10: 4.90e-02   step 100: 1.77e+00   step 299: 2.56e+00
step   2: 1.69e-04     step  20: 8.34e-02   step 150: 1.84e+00
fit over steps 0-119: ~exp(0.0465 * step) — e-folding every 21.5 steps
saturates after step 150 in [1.74, 3.45] m — the scale of the paths themselves (1.87-10.36 m)
```

Seed at float32 resolution, positive Lyapunov growth, saturation at the attractor scale by
~step 100. **No discontinuity, no step change** — the signature of chaotic amplification in a
contact-rich rollout under random actions, not of a behavioural change.

Verdict: **ATTRIBUTED PASS.** Per the pre-registered decision tree (`dr` moved), this is not a physics
verdict and physics rests on **G2 + G7** — which is the right place for it, and both are
answered independently.

**Gate-design note for the coordinator.** The ≤0.01 m per-step / ≤0.10 m terminal diagnostics
are not applicable to a 300-step rollout and should not be read as failed: once *any* float
moves, chaotic growth exceeds them by construction, and no build differing in the last bit could
ever satisfy them. The probe remains an excellent **determinism** instrument — it is what proved
D0 on both pins — but it is not a cross-build **equality** instrument at 300 steps. If a
cross-build form is wanted, the deviation is still 1.69e-04 m at step 2 and only crosses 0.01 m
between steps 2 and 5, so a ≤2-step comparison would carry the intended meaning.

---

### G4 — depth-observation statistics — ATTRIBUTED PASS
cmd (both tiers, committed `depth_obs_stats.py` unmodified, 30 frames × 8 envs, seed 42):
`$NEWLAB -p <baseline>/depth_obs_stats.py --headless --env <tier> --num_envs 8 --seed 42
 --frames 30 --out <dir>/render/depth_obs_<tier>.json`
Retries: enriched-robust 1 hang then success; subgoal-real 0.

**Strict gate does not hold** — array hashes moved on both tiers
(`f4c3375a…` -> `85b75459…` robust, `a6529378…` -> `66943f9d…` real; float32 hashes likewise).
Everything the strict gate exists to protect is nonetheless intact:

| quantity | bound | Enriched-Robust | Subgoal-Real |
|---|---|---|---|
| mean | ≤ 2 % | **+0.0217 %** (0.351818 -> 0.351895) | **+0.0269 %** (0.283900 -> 0.283976) |
| std | ≤ 5 % | **+0.0009 %** | **+0.0086 %** |
| p50 | — | +0.0001 % | +0.0235 % |
| min / max / p01 / p99 | — | **delta exactly 0** | **delta exactly 0** |
| fraction at min | — | **delta exactly 0** | **delta exactly 0** |
| fraction at max | — | −9.26e−06 (0.0094 %) | **delta exactly 0** |
| row-band profile (45) | same argmax, r ≥ 0.99 | argmax **4 -> 4**, **r = 1.000000**, max abs d 5.81e−04 | argmax **0 -> 0**, **r = 1.000000**, max abs d 5.22e−04 |
| col-band profile (80) | same argmax, r ≥ 0.99 | argmax **20 -> 20**, r = 0.999999, max abs d 3.86e−04 | argmax **12 -> 12**, **r = 1.000000**, max abs d 7.82e−05 |
| per-frame mean, 30 frames | — | max abs d 1.13e−03 | max abs d 1.15e−03 |

The mean moves by **1/92nd** of its allowed budget and the std by **1/580th**; the row profile —
the vertical-FOV fingerprint, and the quantity §11 nominates as "the render fingerprint with
shape in it" — is unchanged to six decimal places of correlation with an identical peak row.
The baseline's documented degeneracy reproduces exactly: frame 0 of the robust tier is still
all zeros, the real tier's is 0.1686 -> 0.1692.

Verdict: **ATTRIBUTED PASS** — a float-level RTX change. Specifically **not** the class the gate
is aimed at: a VFOV, denoiser or AA change would move the profile shape or the peak row, and
neither moved at all. The policy's input distribution is unchanged for practical purposes.

---

### G5 — render / video path — **FAIL (band exceeded), attributed to the new stack**

cmd (identical on both pins except the launcher):
`<launcher> -p source/strafer_lab/scripts/train_strafer_navigation.py --env
Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0 --num_envs 64 --max_iterations 6 --seed 42
--headless --video --video_length 200 --video_interval 2000 --log_dir <scratch>`
then `measure_perception_exposure.py <clip>.mp4` — **numbers compared, not the tool's verdict**.
New-pin retries under FINDING 4: **1 hang (1200 s) then success (458 s)**.

| metric | 2026-08-14 baseline | **OLD pin, same session** | **NEW pin** | bound | result |
|---|---|---|---|---|---|
| frames | 201 | 201 | 201 | — | ok |
| resolution | 720×1280 | 720×1280 | 720×1280 | — | ok |
| **mean_luma** | **86.6** | **86.6** | **68.6** | Δ ≤ 10 | **Δ = 18.0 — FAIL** |
| clip_frac | 0.0 | 0.0 | 0.0 | Δ ≤ 0.005 | Δ = 0 — pass |
| **crush_frac** | **0.0056** | **0.0056** | **0.1218** | Δ ≤ 0.01 | **Δ = 0.1162 — FAIL** |
| white_frames | 0 | 0 | 0 | — | ok |

**The old-pin same-session control reproduces the baseline exactly** — 86.6 / 0.0 / 0.0056, to
every recorded digit. So this is not run-to-run spread, not scene randomisation, and not drift
in the measuring tool: the RGB render is genuinely **~21 % darker on the new stack, with 22×
more crushed-black pixels** (0.56 % → 12.2 %).

**Video-path integrity is intact** — the failure is photometric, not structural: stdout shows
`Source prim name : /World/envs/env_0` and `Recording camera anchored on env_0 at world
(35.0, -35.0, 12.0) -> (35.0, -35.0, 0.0)`, so the `_capture` patch found its target and did
**not** take the silent world-frame fallback. The MP4 is non-degenerate (201 frames, 240 134 B,
sha256 `2aa0e4ec41170d82…`).

**Why this is not contradicted by G4.** G4 found the *depth* observation essentially unchanged
(mean +0.02 %, profiles r = 1.000000). Depth is geometric; this is photometric. A tonemapping /
auto-exposure re-default in RTX moves the RGB image and leaves `distance_to_image_plane`
untouched, which is exactly the pattern observed.

**Why it matters beyond a training video.** The clip is cosmetic, but the quantity is not: the
same RGB path feeds the capture and perception lanes (D555 RGB, mission capture, the VLM
grounding corpus). A 21 % luma drop with 12 % crushed pixels is a change to the image
distribution those lanes are calibrated against — and the repo already carries an
exposure-calibration history (`measure_perception_exposure`'s `[90,150]` window, the RTX
histogram auto-exposure work). **Recorded as failed rather than worked around**, per the
standing rule: no exposure or renderer setting was touched to make this number move.

**G5 confirmation — both pins are internally stable, so the gap is real.** A second clip on each
pin was taken (new-pin run2 required 1 retry after a 1200 s hang, then 434 s):

| clip | mean_luma | clip_frac | crush_frac |
|---|---|---|---|
| 2026-08-14 baseline | 86.6 | 0.0 | 0.0056 |
| **OLD pin, this session** | **86.6** | **0.0** | **0.0056** |
| NEW pin run 1 | 68.6 | 0.0 | 0.1218 |
| NEW pin run 2 | 68.3 | 0.0 | 0.1279 |

The old pin reproduces the nine-day-old baseline to every digit; the new pin reproduces its own
value to 0.3 luma. **Δmean_luma ≈ −18.2, Δcrush_frac ≈ +0.117** — 1.8× and 12× their bounds.
Verdict: **FAIL, attributed to the new stack**, recorded as failed; disposition in the upgrade brief.

---

### G6 — export gates A / B / C — ATTRIBUTED PASS (and the two axes separate cleanly)

**Gate A (runtime axis)** — preserved artifact *bytes* under torch 2.11 / ORT 1.25.1, frozen
inputs replayed. Replay integrity verified first: every `|obs` array delta **0.0**.

| path | max action delta | max hidden delta |
|---|---|---|
| ONNX, all three artifacts | **0.0** | **0.0** |
| TorchScript `strafer_nocam_subgoal_v0`, `..._gru_smoke` | **0.0** | **0.0** |
| TorchScript `strafer_depth_subgoal_v2_998` | **8.345e−07** | **4.247e−06** |

Both below the investigate floor (1e−6 / 1e−5); nothing near STOP (1e−5 / 1e−4).
Attribution: a torch 2.10 -> 2.11 TorchScript kernel change confined to the one 3619-dim
recurrent artifact. ONNX is bit-identical because ORT did **not** move (1.25.1 on both pins),
and the two 19-dim artifacts are bit-identical, which bounds the change to the large model's
kernels rather than to the format or the harness.

**Gate B (export axis)** — re-exported from the preserved checkpoint with the repo's own
`export_policy.py` on the new stack, `dynamo=False`, into
`~/Documents/upgrade_stage3_artifacts/` (**live `models/` untouched**, mtime still 2026-08-08).
Retries: 0.

The artifacts are genuinely new bytes — `.pt` sha `03871cb5…` -> `03654da6…` (13 424 358 ->
13 425 690 B), `.onnx` sha `855e1df7…` -> `7dd5c504…` — and yet the trajectory dump they produce
is **byte-identical to Gate A's** (both npz sha256 `619f551bb1fb003d…`). So:

> **The export axis contributes exactly 0.0.** Every delta observed in G6 is the runtime axis
> already measured in Gate A. A re-export on the new stack is numerically indistinguishable from
> the artifact shipped before the bump.

Cross-format agreement of the NEW pair, against bounds action ≤ 1e−5 / hidden ≤ 1e−4:

| sequence | baseline | new | bound |
|---|---|---|---|
| `v2_998` normal | 2.921e−06 / 1.492e−05 | **2.980e−06 / 1.670e−05** | 1e−5 / 1e−4 |
| `v2_998` indist | 2.623e−06 / 1.676e−05 | **2.027e−06 / 1.740e−05** | 1e−5 / 1e−4 |
| `nocam_v0` (untouched control) | 2.861e−06 / 2.384e−06 | **identical to baseline** | — |

The untouched control artifact reproducing its baseline cross-format numbers *exactly* is the
proof that the harness is measuring the re-export rather than drifting. Margin to the known
wrongness class (0.064 action / 0.193 hidden) is **~4.3 decades**. Recorded stack:
torch 2.10.0->2.11.0, onnxscript 0.6.2->0.7.1, onnxruntime 1.25.1 (unchanged).

**Sidecar diff — exactly the pre-registered set, and nothing else.** Changed: `git_commit`
(`69014c6f` -> `ab2daed`), `export_timestamp`, `source_checkpoint`. **Appeared:
`trained_period_s = 0.03333333333333333`** (= 1/30 exactly). Unchanged: `action_dim`, `env_id`,
`formats`, `is_recurrent`, `obs_dim` (3619), `onnx_opset` (18), `policy_variant`,
`training_preset`.

> **Jetson-lane note, verbatim as instructed:** their rebuilt images now read the field, so a
> shipped re-export switches cadence from configured to artifact-driven; the value equals the
> configured 30 Hz, so behaviour is unchanged in value but the code path changes.

Minor note for the recipe PR: `source_checkpoint` records whatever path was passed, so this run's sidecar
carries an absolute out-of-repo path. A shipping re-export should be invoked with the in-repo
relative checkpoint path so the field stays meaningful.

**Gate C** — `tests/contracts/`: **85 passed, 1 skipped** (the standing `strafer_inference`
skip), green **and demonstrably run**.

**Deferred to the Jetson lane, named here:** ORT 1.23.0 load + numeric compare of the
re-exported `.onnx`. Not runnable on this host — the DGX carries ORT 1.25.1 in both envs and no
1.23.0 runtime exists here. The re-exported artifacts are in the evidence deposit for that lane.

---

### G7 — deterministic eval — PASS
cmd (identical on both pins except the launcher; preserved `model_998.pt`, sha-verified):
`<launcher> -p source/strafer_lab/scripts/eval_cadence_emulation.py --env
Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 --checkpoint <preserved model_998.pt>
--profile clean --num_envs 16 --episodes 100 --seed 42 --headless --out-dir <dir>`
Four samples per pin, taken in one session, alternating nothing else on the GPU.
New-pin retries under FINDING 4: 0 across all four runs (~128 s each).

| metric | OLD pin (REF), mean ± sd, n=4 | NEW pin, mean ± sd, n=4 | new − old | 2026-08-14 |
|---|---|---|---|---|
| **completion rate** | **0.8575 ± 0.0435** | **0.8650 ± 0.0289** | **+0.0075** | 0.8119 |
| `path_complete` | 0.8575 ± 0.0435 | 0.8650 ± 0.0289 | +0.0075 | 0.8119 |
| `off_path_divergence` | 0.0075 ± 0.0096 | 0.0125 ± 0.0050 | +0.0050 | 0.0297 |
| `sustained_collision` | 0.1350 ± 0.0465 | 0.1225 ± 0.0250 | −0.0125 | 0.1584 |
| direction offset median | 4.5725° ± 0.685 | 4.5325° ± 0.407 | −0.0400° | 4.510° |
| fraction left | 0.6248 ± 0.0249 | 0.6180 ± 0.0157 | −0.0068 | 0.6216 |
| near-arrival rate | 0.6125 ± 0.0275 | 0.6500 ± 0.0440 | +0.0375 | 0.5743 |
| progress fraction mean | 0.8925 ± 0.0191 | 0.8985 ± 0.0187 | +0.0060 | 0.8680 |

Individual completions — OLD 0.880 / 0.850 / 0.900 / 0.800 · NEW 0.860 / 0.870 / 0.900 / 0.830.

**Verdict PASS, on both readings of the gate:**
- *As specified.* Same-session REF = **0.8575**; binomial ±2·SE band = **[0.7876, 0.9274]**.
  Every one of the four new-pin samples, and their mean, falls inside it.
- *Properly powered.* Difference of means **+0.0075** against an SE-of-difference of 0.0261 —
  **0.29 σ**. There is no detectable behavioural change.

**The 0.8119 → 0.8575 "shift" in the REF is the instrument, not the tree.** 0.8119 lies inside
the old pin's own observed range [0.800, 0.900] measured today, and the run-to-run sd on a fixed
pin (0.0435) is **1.24×** the binomial SE (0.0350) the band was built from. The attribution work
the gate specification asked for was done first (FINDING 5): no env/harness commit, no checkpoint drift,
no DeFM drift, no cadence variation explains it — it is the eval's own spread.

**Cause buckets named, per the gate specification:** no bucket shifts between pins beyond noise.
`off_path_divergence` sits near zero on both (0.0075 vs 0.0125) — both far below the baseline's
0.0297, consistent across pins, so that is a same-session property rather than a bump effect.
`sustained_collision` is flat (0.135 vs 0.1225). Direction-offset median and fraction-left are
the most stable quantities measured all session: both pins reproduce the 2026-08-14 values
(4.510°, 0.6216) to within 0.06° and 0.004 respectively — which is independent evidence that the
policy's *steering behaviour* is unchanged across the bump, and that completion is simply the
noisy readout.

---

### G8 — bridge cadence + obs parity — **BLOCKED (stop-condition #8: import-surface removal)**

The bridge entry point **does not start on the new pin.** Kit boots, then:

```
File "source/strafer_lab/scripts/run_sim_in_the_loop.py", line 762, in main
    from isaacsim.core.utils.extensions import enable_extension
ModuleNotFoundError: No module named 'isaacsim.core.utils'
```

**Cause, located exactly.** isaacsim 6.0.1.0 relocated a large block of extensions from
`isaacsim/exts/` to `isaacsim/extsDeprecated/`, which is not on the default enabled-extension
path. The old pin ships **1** extension in `extsDeprecated/`; the new pin ships **24**, among
them `isaacsim.core.utils`, `isaacsim.core.api`, `isaacsim.core.prims`, the whole
`isaacsim.sensors.*` family and `isaacsim.robot.wheeled_robots`.
Old pin: `…/isaacsim/exts/isaacsim.core.utils` (auto-enabled).
New pin: `…/isaacsim/extsDeprecated/isaacsim.core.utils` (not auto-enabled).

**Blast radius is bounded and small** — of the 24 relocated extensions strafer imports exactly
**one**, at **8 sites in 7 scripts**, using **3 symbols**:

| symbol | sites |
|---|---|
| `isaacsim.core.utils.viewports.set_camera_view` | `collect_demos.py:303`, `coverage_capture.py:746`, `teleop_capture.py:406,438`, `test_strafer_env.py:162` |
| `isaacsim.core.utils.extensions.enable_extension` | **`run_sim_in_the_loop.py:762`**, `validate_scene_connectivity.py:840` |
| `isaacsim.core.utils.semantics.add_labels` | `extract_scene_metadata.py:640` |

A replacement surface exists and is *not* deprecated: `isaacsim.core.experimental.utils` sits
under `exts/`, and `enable_extension` is at
`isaacsim/exts/isaacsim.core.experimental.utils/isaacsim/core/experimental/utils/impl/app.py`.
The deprecated tree is also still physically present, so re-enabling the extension explicitly is
a second possible route. **Neither was applied**: this is §6 stop-condition **#8** — "an
import-surface removal at the target that forces strafer code migration beyond mechanical
accommodation … whether the accommodation rides the compatibility PR or re-targets the bump" — and the standing rule
is to record it for disposition in the upgrade brief, not to work around it. Seven Tier-1/Tier-2 entry points is past
"mechanical". Note the scoping report's §2.10 candidate list did **not** anticipate this; it
named `TiledCameraCfg` (cleared) and the warp churn. This is a new, unlisted blocker.

**What could still be measured, and passes:**
- `bridge_harness_smoke.py` — **PASS** on the new pin (34 s, 0 retries). It exercises the
  capture path end to end: env reset/step on normalized `/cmd_vel`, the
  `create() → add_frame() → save_episode() → finalize()` writer lifecycle, the `bbox_2d_tight`
  detections annotator (10-label vocab), 16UC1 depth PNG sidecars, the discard path (the
  cancelled mission's 9 frames correctly did **not** reach disk), and — importantly —
  **`reloaded: episodes=1 frames=32`**, i.e. the LeRobot dataset re-opens and decodes. That is
  the §13.1 torchcodec closure confirmed at runtime on a booted Kit, not just in unit tests.

**Not measured, and why:** the ≥1000-tick `t_sim` cadence capture (baseline 1302/1302 at
33.33 ms) and the `parity.py` obs comparison both require the bridge loop to run, so both are
blocked behind the same import. No stall forensics were needed — the process exits immediately
with a traceback rather than stalling, so the ptrace-free stall protocol did not apply.


---

**Per-site severity — the 8 sites do not fail the same way**, which matters for the disposition:

| site | guard | behaviour on the new pin |
|---|---|---|
| `run_sim_in_the_loop.py:762` | **none** | **hard crash — the Tier-1 bridge does not start** (observed) |
| `collect_demos.py:303` | none | hard crash when the camera re-pose path runs |
| `coverage_capture.py:746` | none | hard crash when the overhead camera re-poses each step |
| `test_strafer_env.py:162` | none, but behind `if args.video` | hard crash only under `--video` |
| `validate_scene_connectivity.py:840` | `except Exception` -> `logger.warning` | **silently degrades** — `isaacsim.asset.gen.omap` is never enabled and the tool continues, so occupancy generation proceeds without the extension it asked for |
| `extract_scene_metadata.py:640` | `except ImportError` -> `RuntimeError` | fails loudly and intentionally, with a message naming the runtime |
| `teleop_capture.py:406` | `except ImportError` -> `return False` | graceful fallback |
| `teleop_capture.py:438` | `except ImportError` -> `return` | graceful fallback |

The `validate_scene_connectivity.py` case is the one worth flagging beyond the bridge: it is the
only site that neither crashes nor falls back, so on the new pin it would keep running with the
occupancy extension absent and report success.

---

### G9 — training-curve overlay — PASS

cmd (new pin, same fixed-seed short run as the baseline; **no `--lr_schedule`, no recipe flags**):
`$NEWLAB -p source/strafer_lab/scripts/train_strafer_navigation.py --env
Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0 --num_envs 64 --max_iterations 100 --seed 42
--headless --log_dir <scratch>` — 1 h 32 m wall, exit 0, **0 retries**.

`--max_iterations 100` runs iterations **0–99**; there is no row 100. Iteration 0 carries
first-iteration warm-up and is excluded from throughput, per §11.6.

| quantity | baseline | new pin | delta |
|---|---|---|---|
| **steps/s, excluding iteration 0** | **53.49** (band 45.5–61.5) | **53.86** | **+0.36 (+0.7 %) — INSIDE** |
| mean reward, iters 1–99 | −4.65 | −4.69 | −0.04 |
| mean episode length, iters 1–99 | 287.73 | 283.92 | −3.81 (−1.3 %) |
| NaN / inf rows | none | **none** | — |

Sampled rows, against the baseline's own sampled table:

| iter | steps/s base → new | reward base → new | ep-len base → new |
|---|---|---|---|
| 0 | 39 → 36 | −0.28 → −0.31 | 35.20 → 28.25 |
| 10 | 52 → 57 | −3.82 → −3.74 | 179.29 → 210.82 |
| 25 | 53 → 57 | −4.81 → −5.01 | 278.72 → 280.43 |
| 50 | 53 → 43 | −5.06 → −5.16 | 319.09 → 309.24 |
| 75 | 54 → 55 | −4.60 → −4.64 | 334.22 → 309.02 |
| 99 | 55 → 53 | −5.09 → −5.36 | 385.86 → 371.39 |

"Overlays qualitatively" is stated as a number rather than eyeballed: across the 99 compared
iterations the curves correlate at **r = 0.8641** (mean reward) and **r = 0.8861** (mean episode
length). The characteristic early shape is reproduced — episode length climbs steadily
(28 → 371 against the baseline's 35 → 386) while reward falls and partly recovers — which is the
expected trajectory when a fresh policy survives longer and accumulates more per-episode penalty
before it learns to complete. Per-iteration steps/s correlates weakly (r = 0.40), which is
timing jitter rather than curve shape; the means agree to 0.7 %.

This is deliberately **not** an equality gate — rsl-rl 5.4.2's `compute_returns` fix changes
GRU-critic dynamics by design. What it establishes is that the training path runs end to end on
the new stack at baseline throughput, exercising the 5.4.2 runner and `tensordict 0.14.0`
together. It did that, and it was the one leg of the session that needed no retry.

**Provenance note.** The curve comes from the run's own TensorBoard event file, not from stdout:
the launcher's final step copied an earlier partial `/proc` recovery of the log over the complete
one, so the stdout log in the deposit holds only the first three iterations. The event file was
untouched and carries all 100 points per tag, and `model_99.pt` in the same directory
independently confirms the run reached iteration 99.
[`extract_training_curve.py`](extract_training_curve.py) reads the event file; the recovered
curve is [`training/training-curve.csv`](training/training-curve.csv) and the comparison is
[`training/curve-comparison.txt`](training/curve-comparison.txt).


## The two cross-cutting findings

## FINDING 4 (the session's headline) — the new pair intermittently deadlocks during Kit init

FINDING 2's "test-harness timeout" is not a test-harness problem. The roller probe — no pytest,
no `run_tests.py` — hung the same way on its second PGS leg, and that gave a live specimen.

**Specimen** (evidence deposit, `logs/G2-hang-specimen-forensics.log`; captured non-ptrace per the standing rule):

| observation | value | reading |
|---|---|---|
| elapsed | 20 min 43 s, still hung | does not self-recover |
| `VmRSS` | **48 296 kB** | a booted Kit is gigabytes — this never got near booting |
| threads | **2** | a booted Kit runs dozens |
| both threads | `state=S wchan=futex_wait_queue`, 6 CPU ticks total | blocked on a lock, doing no work |
| open fds | `/dev/null` ×2, `/dev/shm/carb-RStringInternals-<pid>` ×3 | **no `/dev/nvidia*` fd** |
| GPU | absent from `--query-compute-apps`, 0 % util | no CUDA context was ever created |

So the process deadlocks inside **carb initialisation**, after mapping the carbonite
shared-string segment and before opening the GPU or loading a single extension.

**Leading hypothesis, with its evidence and its limit.** Every SIGKILLed Kit process leaves a
`/dev/shm/carb-RStringInternals-<pid>` segment *and* its named POSIX semaphore
`sem.carb-RStringInternals-<pid>` behind; POSIX named semaphores have no owner-death recovery,
so a process killed while holding one leaves it locked permanently. 17 such stale pairs had
accumulated (dating from 2026-08-04 through this session), and the today-dated ones are exactly
the pids `run_tests.py` killed on timeout — 893571, 903006, 903091, 903463. A shared
`sem.carbonite-sharedmemory` also persists, unowned, from 2026-08-02.
**The limit: this does not reproduce on demand.** Three consecutive `AppLauncher` boots with the
stale set still in place succeeded in 2–3 s each. So the mechanism is a **race**, not a
permanently poisoned lock, and the shm/semaphore leak is a contributing condition rather than a
proven single cause. Stated as a hypothesis, not a conclusion.


**Rate, and how firmly it is attributed to the pair.** Counting every Kit launch of the session:

| pin | launches | hangs | rate |
|---|---|---|---|
| **NEW** | 55 (RUN1 14, RUN2 14, env standalone 3, roller 1, 21 under `kit_retry`, 2 misc) | **13** | **23.6 %** |
| **OLD** (same session, same host) | 20 (14 suite subprocesses, 4 evals, 1 `--video` train, 1 pose-trace equivalence) | **0** | 0 % |

If the old pair carried the new pair's rate, the chance of seeing **zero** hangs in 20 launches
is **0.45 %**. That is the attribution: the pair, not the host, the checkout, or the session.

**Frequency by leg.** 
RUN1 2 of 14 suites · RUN2 4 of 14 suites · roller 1 of 4 legs · standalone env 1 of 3.
The **same-session old-pin control run of the identical command hung 0 of 14** and finished
487/487 in 666 s. That is the attribution: **new pair, not this host**.

**Why it matters beyond this session.** It is not a test-only nuisance. It hit an ordinary
measurement script, so it will hit `train_strafer_navigation.py`, `eval_cadence_emulation.py`,
and `run_sim_in_the_loop.py` — including a long unattended training run or a rig session, where
a silent 20-minute stall is expensive and looks like something else entirely. **This is the one
result that should gate the flip**, and it is recorded rather than worked around.

**Mitigation used for the rest of this session, so results stay honest.** Every remaining Kit
leg runs through `kit_retry.sh` (committed here): bounded timeout, retry on hang, dead-pid
shm/semaphore sweep between attempts, and one accounting line per attempt so the retry count is
part of the record instead of being lost. Retries are reported per gate.
17 stale pairs were swept at 2026-08-23T12:12:28 before G2 resumed; hangs after that timestamp
are post-sweep data.

---

## FINDING 5 — the primary behavioural gate is not rerun-deterministic (method, not stack)

G7 step 1 asks for a same-session REF re-measure. Two runs of the **identical** command on the
**same** (old) pin, same seed, same session, same checkpoint bytes:

| old-pin run | episodes | completion | path_complete | off_path_div | sustained_coll | offset med | left |
|---|---|---|---|---|---|---|---|
| 1 | 100 | **0.880** | 0.880 | 0.000 | 0.120 | +3.91° | 0.595 |
| 2 | 100 | **0.850** | 0.850 | 0.020 | 0.130 | +5.53° | 0.655 |

**The instrument moves by 0.030 between two runs of the same build.** §11.2's D0 result
established bit-determinism for the roller probe, the pose trace and the export dump — but
**never for the eval**, and the eval is the behavioural gate the whole migration turns on. The
±2·SE band in §11.3/§11.6 is a *binomial sampling* band (SE ≈ 0.039 at n=101); it does not
include this run-to-run term, so a one-run-vs-one-run comparison against that band is
under-powered and could read a stack change into what the instrument does on its own.

Excluded as causes, checked rather than assumed:
- **Tree movement.** The only commits touching the env/agents between `e7ea7bd` and `ab2daed`
  are the PR-0 deletions (eval-neutral by construction, §11.3), the 11× `configclass` import
  rewrite, and comment-only edits. Diffs of `depth_rnn_model.py` and `rsl_rl_ppo_cfg.py` — the
  two files in the policy-reconstruction path — are **comments plus that one import line**.
  `eval_cadence_emulation.py` itself is untouched since the baseline.
- **Checkpoint drift.** The preserved `model_998.pt` verifies against `SHA256SUMS.txt`.
- **DeFM encoder drift.** The `torch.hub` clone, the `efficientnet_b0` checkpoint and the
  HuggingFace `models--leggedrobotics--defm` cache are all dated **2026-04-09**, months before
  the baseline. The "Downloading…" line is a cache load, not a fetch.
- **Cadence emulation.** Both runs report `hold 0.000 dup 0.000` at 30.00 Hz under `--profile
  clean`, so the tick-profile injector is not the variable.

Mechanism, consistent with everything else measured today: **nothing in the eval or policy path
pins GPU kernel determinism** — no `torch.use_deterministic_algorithms`, no
`cudnn.benchmark = False`, no `CUBLAS_WORKSPACE_CONFIG`; the script seeds only its numpy RNGs
(`eval_cadence_emulation.py:1621, :1634`). Autotuned/atomic kernels in the DeFM depth encoder
therefore produce float-level action differences, and a **closed-loop** rollout amplifies them
exactly the way G3 measured open-loop (e-folding every ~21.5 steps). Bit-deterministic physics
under replayed actions and a non-deterministic closed loop are not in conflict — G2/G3 replay a
fixed action sequence, G7 does not.

**Consequence for this gate, adopted here:** REF is treated as a *distribution*, not a number.
Four samples are taken on each pin and the pins are compared on their means with the observed
spread, instead of one run against a binomial-only band.

**Recommendation (not actioned — instrument change, and out of this session's scope):** either
pin determinism in the eval path, or state the gate in terms of a multi-run mean with a measured
run-to-run term. As written, a ±0.078 band around a single run cannot distinguish a real
regression from this instrument's own spread, and the baseline's 0.8119 is a single sample of it.

---


---

## Evidence

In-tree here: this README and [`provenance.md`](provenance.md) only, per the
measurements policy. Everything else this record cites — the reproducing scripts
(`newpin_env.sh`, `kit_retry.sh`, `pose_trace_probe.py`, `obs_dtype_probe.py`,
`compare_export_npz.py`, `extract_training_curve.py`), the interpreter state, the junit
XML sets with their provenance-suffixed directories, the roller CSVs and physics
metrics, the depth and exposure results, the export comparisons, the eval JSONLs, the
training-curve extraction, and the record's original `MANIFEST.sha256` — lives in the
companion evidence repository, alongside the bulk payloads (run logs including the live
hang specimen, the `.npz` trajectory and observation stacks, the four MP4 clips, and the
Gate B re-exported artifacts) deposited earlier in the same session.

| field | value |
|---|---|
| repository | <https://github.com/zachoines/Sim2RealLab-Artifacts> (private) |
| directory | `isaac-lab-upgrade-stage3-2026-08-23/` |
| deposit commit | `03995f3e3c3fca7c8f81fefa86248f4952cd94f52e110de82272b2b230985c8c02d8495c4` |
| conventions | the repository's `README.md`; per-file digests in the deposit's `DEPOSIT.md` |

sha256 of every file this record moved there, laid out under `record-files/` in the
record's own layout so `cp -a record-files/. docs/measurements/<record>/` restores it
(first-tranche bulk files are siblings, listed in `DEPOSIT.md`):

```
ba76f7522107dc9fb3d367f1b4f0c02fe41b3a4b4dd8312c749e8306e94f39ca  record-files/suites/stale-not-this-session/README.md
7e62e1cc7babf62af83738e30e0b0fa98c384e6b7a9674a65e9b27b11cbdf181  record-files/compare_export_npz.py
310746bde4baa0b231b3ed277b2a6b9a125cff6a5a1a2e12628fe16824d79d95  record-files/eval/g7-recomputed-from-jsonl.txt
970bc30dfe4c1c9100904958111326475aab2b64e68388755c11e369fc1add87  record-files/eval/g7-summary.txt
530a1fbb5663fb7b080ea9b3e9b96350b71e93ae8858468b1851054885790527  record-files/eval/newpin_run1/cadence_20260823_131757.jsonl
1f3af85f6a4ccaa09c404f2566e9b51c1edacaabf5557a4f058c928500c7a0c1  record-files/eval/newpin_run2/cadence_20260823_132005.jsonl
d17d628174d9084c22c4c484b89446adbbf4960e3e8e96f17d3bba27636785fe  record-files/eval/newpin_run3/cadence_20260823_132216.jsonl
2a3ac7a8940b9501be76b89d73765ac92b118bac41f2cd16e5f614b5d93673e2  record-files/eval/newpin_run4/cadence_20260823_132424.jsonl
78d8335f52346b59a683669569e57ca80f0a368a6104a7ebf80ccbc811c8098f  record-files/eval/oldpin/cadence_20260823_130714.jsonl
b7a8695bfff7e83de6031bba4e7db7af0a5c4a71bf6f7aebad6c94b1dd5d44ea  record-files/eval/oldpin_run2/cadence_20260823_131103.jsonl
258b41fd9c99295173ab6b769ca753ce1ed724816b5d9ddcea198a5694c0bdcf  record-files/eval/oldpin_run3/cadence_20260823_131348.jsonl
d34c71900f448c432c1b0e77d64b090943e26726ec8991c277f81c609e94e01b  record-files/eval/oldpin_run4/cadence_20260823_131553.jsonl
de2448026582a0d334ca1ab03f127e46d370293da9c00f916e8c979010d11633  record-files/export/gateA-comparison.json
ad834d19262e514d15c2c64b6266dae92b38c58cdb3670de0be9a63c784139be  record-files/export/gateA/export-trajectories-manifest.json
4e06da1a4af8e288d5d58b9714180ea03323cedc957ddccf491f358df3bf30ed  record-files/export/gateA/export-trajectories.npz.sha256
dbb464b9a44dcfd3348045e5b070d77944fe28bbbedac5e1d886c093ab321036  record-files/export/gateB-comparison.json
ac815f1dcfa5f9c35d4d57d8cd10c1761baacbe4ff7e55e74810c7aa76928111  record-files/export/gateB/export-trajectories-manifest.json
4e06da1a4af8e288d5d58b9714180ea03323cedc957ddccf491f358df3bf30ed  record-files/export/gateB/export-trajectories.npz.sha256
8ef7cc0e65e4bd54e47c06ebd3b7dd9e6ff9f7f0f9946f2ac41a24736ca44cde  record-files/extract_training_curve.py
f392731ec59047cc6411a3f9ebf1a982aa9551ce03d67f1f75512aa91ddacc38  record-files/kit_retry.sh
20123d322769267b98e36e7a8caa8a6870d16e98e97c84548f535489cc54f723  record-files/MANIFEST.sha256
ff313f05a0e8c49c46a6f76802635ef237e93290ab0e4203afb2162d6ec6f1bf  record-files/newpin_env.sh
8b2ab15e680354d866fd3014261311e6a0067b150a449997e85512f6feda200b  record-files/obs_dtype_probe.py
2d3265f59ab14a1740a2b95d3ffec3f89eccb5d7fb051bdec586cc148a72f8b1  record-files/physics/obs-dtypes.json
e046f152df857c73119862a55ccdc1697e98666026cd70f1c3745e9cec23e442  record-files/physics/pose_trace_OLDPIN_equivalence.json
2fabfc62e937192d968e47eb281c648963e397501b29fc3eb4dbad91dbdffe0c  record-files/physics/pose_trace_run1.json
3cd79443512257fd0e79c3ff077e10fa8cee68d31cf8b5b1b448549071afe8f3  record-files/physics/pose_trace_run2.json
3cf7643e5e4afe5feb17f9660ae9ad3f4dd3ec6e63b9740e642f5c335cd0e4f9  record-files/physics/ride-height.txt
ba5376a24b6cb01dc2a4fc6cd323621b7de76096bca85489c214fe637e3cad18  record-files/physics/roller-analysis.txt
e62f7ad1249d61788a374cd45e6472e34942fea421b44711fdd23e8448ac91d1  record-files/physics/roller-metrics.json
d1285d3c616c9ffd3ab539ed238e18abd10e4a42a2d91c087254d3898cd528fa  record-files/physics/roller_z_pgs_run1.csv
d1285d3c616c9ffd3ab539ed238e18abd10e4a42a2d91c087254d3898cd528fa  record-files/physics/roller_z_pgs_run2.csv
6d4110dd6a27b5189e19cdce59d035374d3dcc4dd5e7f378d7186e882e411065  record-files/physics/roller_z_tgs_control.csv
8f3b352bb884396823ae273fa52a699cf34a078536ef7220c24a0220ef55f7e5  record-files/pip-freeze-env_isaaclab3beta2.txt
8337afcd563aa5aa2b238f1e151fcd008d3027efb64729b9055bad47762585db  record-files/pose_trace_probe.py
5d344299804c1284b635582062bdcf6db1fa0dd82e89c64ecbc4c363e0decd3f  record-files/render/depth_obs_enriched_robust_play.json
f8666067d9a36f97e479f5d6ce077529fd834db7c96bdd7de7d3a12af27e8bfe  record-files/render/depth_obs_subgoal_real_play.json
2cab38b3abea0ba440d4b8ac76e66f7fe846c78f04b5864b5ab0bc9dd5a19af1  record-files/render/exposure-train-video-newpin-run2.txt
4a22059068ddae12fdcfaa80f3d0ecf9cc2ff10f96b4ad1fe07a53aa2c2f2ce3  record-files/render/exposure-train-video-OLDPIN.txt
d6dbc781295f1a2abe163fc504ecbe842646d82a3c858be8475678ea9cb7f249  record-files/render/exposure-train-video.txt
f74feda9d3b37b8962f0834f5dcfd222d30b5ce072af20ac1d6bf54d6b45ac06  record-files/suites/contract-two-file-newpin.txt
6db539354f027e67a4bd5bf78e6560ad117b9524ba8815e3c5e19e745e893084  record-files/suites/env-uncapped-newpin.xml
e9d75fd1b630c5a43ce1e3af92351002e92de26f8bb9fcf7b64669b8bf6746f0  record-files/suites/kit-junit-xml-envrerun/kit-suite-env-RERUN-PASSING.xml
b03e70752cfd2e7ee0f98563114b6a076309d2d57a154334be5a0a50bbbcb6a0  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-actions.xml
bf499b599d1f56648055b689d9cc67bd95e641ab310f17a926fb056aba4d3a1d  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-camera_jitter.xml
c88e63ac5642fdd7235d99f57ac7ba67f4c8bd46e5ffa571d516c8f6d5f77bc4  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-commands.xml
dcb5824deaa04d05b08973ae0acf35ed6a1c575b13463d3cbd00f71372b4f364  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-curriculums.xml
a456cdc01d1ba85bd20a9072e7519d6daa53c3447cc33bfe50bbbfb3c9d9512a  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-depth_noise_test_frame_drops.xml
d05f56577c1cebdec54786b4740ba50f5d0d46563d38244a914fec3d0efd3f66  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-depth_noise_test_gaussian.xml
ec20d16a7163a3bfd9605683b419e92d5391db1d9ca08e6aac08b872d2dd791c  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-events.xml
d25eca4177acffd281bbccf4723621c968bc6a45113831fd20ccf563f50b128e  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-imu_test_imu_collision.xml
40e32ce8156561a48f54030324af8604536fc918fdec99861cf778800b2365c8  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-imu_test_imu.xml
ddceba392bc12d684191f4bd61fbe72f26eb31d8faa0a4820945f3551ed12556  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-noise_models.xml
dfab705687d4c5aae6599b3b359ad09e0167ce1dd44e02516776b2a80f812c9d  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-obs_dump.xml
41c20209b12e01734cfcdd29f6a2672936204070a13ebbe7f3e2172c7cf10c3f  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-observations.xml
4c17e606bab5ec27022e28f305f1761c85ccfaedcdfab40e427dee8764ce0178  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-rewards_test_collision_rewards.xml
f417048bc4b601c740d9f57a416a8daf5f082c745189c42cb93ae3888744fae0  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-rewards_test_rewards.xml
f1a687faafd3386c5f520207801ed2ff0c53f70e6ede08b6f4e5f761137f42ce  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-sensors.xml
8135071b1052739f9dd53405f30e45e7a9ade997e6dff92cb6ebebfe1d3c51e1  record-files/suites/kit-junit-xml-FAILRUN/kit-suite-terminations.xml
1bab129d2cf7746a3ee121ca83605739362b7d84bebb8c2eb3dd1086dba3e5cb  record-files/suites/kit-junit-xml/G0-boot-terminations.xml
e2810aec2e01e68c09df2580ee53882c9509d09b5deb8289ab41dfb728cf38b6  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-actions.xml
920252547f957ae2abb815b639b6ed7f3b2280e3a366f1ddd3bc6be862a81b4f  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-camera_jitter.xml
c5a22c097d8790542c372e09abe268f3f7f57d9e08cf298b0ca926191db3a85e  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-commands.xml
aa26d675691a8ba945e3f68944cb7c580e854ac5ac2c6d06e1a80f2ab6e37506  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-curriculums.xml
aa8e4eaaecf859b3517f44d3bd5d27e9ac2852d011f9d3a7e798301625072da2  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-depth_noise_test_frame_drops.xml
1107687abdcf23108bd3a0310d8be14451f75935022317c644bd05cabd343a5b  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-depth_noise_test_gaussian.xml
fa6898c7a4c648487e3dceb067b3adbb2038a6f306f2f26e9172ebb02eeb5495  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-depth_noise_test_holes.xml
8f48d1a28b0556988e90d513d361dceea53fe5e7ee177940f02deae4522282b3  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-env.xml
2bac56d629fbbf9f3f05a0eb9b4f2bb37e8f9148c25229eb9d0a79c9dfda9f12  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-events.xml
035de7c67345ffc96444b6fe8befb596a0fd5c62d580bb68ec9497fa84d19f52  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-imu_test_imu_collision.xml
59dc997f7e569b46e15440722e8507fabd008bcef81bf91f9299eb56dbee7e66  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-imu_test_imu.xml
d44f5880d94d0b40a037b56e15b832f4adbf85b78a60a4261fe824d2a975608c  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-noise_models.xml
d053c689a72cb64dab5771dce1aebb333aa296068039bdf3b28d5fc4f3df9b50  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-obs_dump.xml
d4e4fa9cb7537e3cc81218509a6f7cf9ceb62f13dfe8416902245a622b8108db  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-observations.xml
435bd81eb60d13445cc8611a8a27e1dd6bb0306ccf7ade023acf7e37295244f6  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-rewards_test_collision_rewards.xml
d190821ab92300f6ecab28b1866c92b6d250be5554385db93c3b3dfaccc5544d  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-rewards_test_rewards.xml
6d8db09e50b167aa93c4a83cb819bf344b84105cbc88368b8f314a1d07633fe4  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-sensors.xml
3f4e0f7dbe49f3364dbb1551e051d5b15a69d18dd03e67b27f7a736d9b644d77  record-files/suites/kit-junit-xml-OLDPIN-control/kit-suite-terminations.xml
a0b8208897db65fa3f3c7745c25a5cb16fbcd6e062761151a9415918db05a55a  record-files/suites/kit-junit-xml-RUN2/kit-suite-actions.xml
8c86d3752fc3e9558039d8b33168c6122c1ff6a46d4d54c3ea288ca32d1615ca  record-files/suites/kit-junit-xml-RUN2/kit-suite-commands.xml
2c682e61d25592b882e198d27c438f76f2088fbcef1f4715e0b7825e5e3671e1  record-files/suites/kit-junit-xml-RUN2/kit-suite-curriculums.xml
130a389722a2b4db6617020b00eb3768218e6e30881622ea6bd45f472aa58a4c  record-files/suites/kit-junit-xml-RUN2/kit-suite-depth_noise_test_frame_drops.xml
fbaaac86984891df4564d3127d4953bdebac225dd0f63cefe10e2c3696480a22  record-files/suites/kit-junit-xml-RUN2/kit-suite-depth_noise_test_gaussian.xml
4580818d370b7a7cdf56184395d3a4e56cec21ea3fd19f737ca1e6679337f0eb  record-files/suites/kit-junit-xml-RUN2/kit-suite-env.xml
25fd37281e70ec952c76aa6627a534554301c0ebf86e9e6dec1539e9b57724a0  record-files/suites/kit-junit-xml-RUN2/kit-suite-events.xml
930b41bc94c6dcb6422232309d33a10eff9e441270646fea1c75db88eef32a81  record-files/suites/kit-junit-xml-RUN2/kit-suite-noise_models.xml
0835675ca4f2be544421f09d196b735f71df2910c7d2e5e34092ae7c2f56f43c  record-files/suites/kit-junit-xml-RUN2/kit-suite-obs_dump.xml
53fd2ff597491a56e0a0689db211921b7a4a0a27f7527d9e1f6c5162a0ba0665  record-files/suites/kit-junit-xml-RUN2/kit-suite-observations.xml
8ae6a161651464cfd793a47f8acc2518c12b6cdf075f6e26279ae1e74a9bd427  record-files/suites/kit-junit-xml-RUN2/kit-suite-rewards_test_collision_rewards.xml
2f30ef228c50bc9e342f08dc520dff6e35e9505eb730dbd6e770384a691e7c5c  record-files/suites/kit-junit-xml-RUN2/kit-suite-sensors.xml
0f6444bcb66e9e7d50656561d371525c28823b2657178bb5c07c8fb1783960bf  record-files/suites/kit-junit-xml-RUN2/kit-suite-terminations.xml
a2771ff7a442cde356cd043489e0b312f0624e0c5336ef39664d0af69a060df5  record-files/suites/orphaned-standalone-newpin.txt
3a80f62a3821f7e4661c0c5c1e408304eb5caceaeba07f82f61a9e024d828e7f  record-files/suites/pure-suite-newpin.txt
ecfcff929d5f7aad5d0fd2f0b7d4801ddd60878c0aec5463393bf3505186111f  record-files/suites/stale-not-this-session/kit-suite-depth_noise_test_holes--STALE-2026-08-14-baseline.xml
dd21b5a402e3a3731c7b4d749b87a6387afedd904b15610fc1ee83d64638844e  record-files/suites/stale-not-this-session/kit-suite-env--STALE-2026-08-14-baseline.xml
6c3541c02d7fdd7a009ff0c47c040ddbaabbe22915c3c60cc6ebb48ce1277d9e  record-files/training/curve-comparison.txt
17797ecdbf901282406db6a7dde71af7bca18179d318039b0c38fb15f1eb4dd9  record-files/training/training-curve.csv
```
