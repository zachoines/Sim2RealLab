# Reconciling the near-field depth convention — 2026-09-13

The 2026-08-17 mission gate's 0 of 6 was attributed to a convention mismatch in
the near field: training's realism noise wrote the far clamp at pixels the
deploy pipeline writes the near fill at, on 18.89 % of every frame
([`goal-a-attribution-2026-08-22`](../goal-a-attribution-2026-08-22/README.md)).
This record measures the reconciliation. The convention divergence is **gone** —
the far-clamp share of the near-field class falls from 0.4990 to **0.0000**, and
the noise-bearing depth's distance from the clean observation term collapses from
p95 0.96667 to **0.00480** scaled.

It also records a result the dispatch pre-registered the other way. Feeding the
reconciled depth to the deployed v2 artifact does **not** make it emit the rig
class; it still commands toward its referent. The reason is measurable here and
is the attribution record's own open item: v2's command is sensitive to the
*presence* of realism noise, not to the convention. The stereo Gaussian alone —
0.029 m, no class structure — already moves the command 52° off the clean
answer. The convention fix removes the corruption it was adjudicated to remove;
it does not by itself put the artifact on the robot's side of the divide, and
the retrain is what has to.

Setup, digests, interpreter and the machine-local inputs: [`provenance.md`](provenance.md).

---

## 1. What changed in the source

Two writes in the depth realism noise model, and two config fields that keep the
retired behaviour selectable.

`too_close` pixels now take `min_range`, which is the number the observation term
has already written at every pixel it filled. The comparison therefore cannot
reclassify the class the term created: the rewritten value is not itself below
`min_range`, so the write is idempotent, and the threshold coincidence that made
the rule a coin flip becomes a no-op by construction rather than by retuning.

Holes now take the median of their valid 3×3 neighbours, falling back to
`min_range` where every neighbour is invalid too. The property this buys is the
one the deploy reduction has and a per-pixel constant does not: an isolated
stereo failure is outvoted by the surface around it instead of presenting as
open space. It is **not** the same operation as the deploy rescue, and the
record does not claim it is — §2.1 states where the two diverge.

The median is written only at the invalid pixels, so its cost tracks the hole
rate rather than the frame size: measured 16.8 ms against 58.2 ms per frame at
1024 envs, bit-identical to the dense formulation. The frame's edge reads the
neighbours it actually has; padding the window to keep it square must not let a
replicated cell vote, or two far neighbours and one near one tie and resolve
near, moving an edge pixel off its surface.

`too_close_fill` (`"near"` | `"max"`) and `hole_fill` (`"median"` | `"near"` |
`"max"`) carry the new values as defaults, and an unrecognised value raises
rather than falling through — a misspelling would otherwise reinstate the far
clamp silently. Setting both to `"max"` reproduces the pre-fix tree
**bit-exactly**, at both realism tiers, RNG draw order included (`torch.equal`
true) — the median path consumes no randomness, so the arms are directly
comparable.

`min_range` stays 0.2, and it is now the shared constant the observation term
fills with rather than a second literal that happens to match: `depth_image`'s
`nearfield_fill` and `DepthCameraNoiseCfg.min_range_m` both read
`strafer_shared.constants.DEPTH_NEARFIELD_FILL`. Before this they were four
independent `0.2` literals, so the collision could have returned by editing any
one of them with every test still green.

### 2.1 Where the two paths still differ, deliberately

| case | deployment | training, after this change |
|---|---|---|
| one unresolvable pixel inside a resolvable neighbourhood | outvoted; reads the surface | outvoted; reads the surface |
| a whole neighbourhood unresolvable | far clamp | near fill |
| the reduction | median of the pixel's own 8×8 raw footprint, invalids counted as the far clamp | median of the valid neighbours among adjacent policy pixels |
| an even count | `np.median` averages the two middle values | the lower of the two, biasing nearer |

The first row is the property under test and both paths have it. The rest are
recorded rather than reconciled: the near fill is the conservative reading for a
policy that has to avoid what it cannot resolve, and the saturated case needs
every pixel of a neighbourhood to fail at once (0.03⁹ per pixel at the robust
tier). Closing the remaining gap is fix direction A, which is not taken here.

### 2.2 A consequence beyond the near field

The hole channel is now numerically much weaker, because a hole on a smooth
surface is invisible where it used to be a far-clamp spike. With the stereo term
disabled so only holes move, mean |Δ| from the clean image falls from 0.006854 to
0.000009 scaled at the realistic tier and from 0.020892 to 0.000049 at the
robust tier — roughly 760× and 430×. It still moves 0.65 % and 1.92 % of pixels,
at depth discontinuities where the neighbourhood disagrees with itself, and the
largest single excursion is 0.056 and 0.654 scaled.

`hole_probability` is therefore a much weaker randomization knob than it was.
That follows from the ruling rather than from this implementation, but it is a
material change to the training distribution outside the class the brief names,
and the retrain should not assume the knob still carries its old strength.

`hole_cluster_size` on the tier config has no reader and never had one, so holes
are independent per pixel. It matters more now than it did: a clustered failure
is the only kind that reaches the near-fill fallback at any realistic rate, and
also the only kind the deploy reduction cannot outvote. It is left dead rather
than wired, and named here so it stops reading as coverage that exists.

### 2.3 Two things still write the far clamp

Camera failure does, to every pixel of a failed frame. It is the only
`max_range`-means-invalid write left in the model, the ruling did not name it,
and it is untouched here. It is also why the robust tier's far-clamp share of the
near-field class is not identically zero over a long sample: at
`failure_probability` 0.001 a whole-frame event lands about once per thousand
env-steps, measured at 0.000625 over 3 200 env-steps against 0.000000 with
failures disabled. The 0.0000 in §2's table is the 30-tick sample the A/B ran,
where no failure event occurred. The convention itself reconciles exactly;
whether "camera dead" should read as open space is the adjacent question
[`d555-depth-decode-validity`](../../tasks/active/trained-policy/d555-depth-decode-validity.md)
owns. Deployment's counterpart is a zero Twist from the watchdog, not a cleared
frame.

The observation latency buffer does the other, in the opposite direction: it is
zero-filled, so the first one to three frames of every episode are 0.0 m
everywhere, below `min_range` and below anything `downsample_depth` can emit
(its floor is the near fill). Measured on an already-filled frame: the realistic
tier reads 21 600 of 28 800 pixels below `min_range` at step 0 and is clean from
step 2; the robust tier is entirely zero at step 0 and clean from step 3. This
predates the change and is outside the ruling, but it bounds the near-field
invariant — the invariant holds of the convention, not of the model's output at
episode start.

### A stale field the median exposed

`DepthNoiseModelCfg.height` / `width` were 60×80. The policy camera is 80×45, so
the declared shape described 4 800 pixels of a 3 600-pixel frame. Nothing read
the fields — `get_depth_noise` never forwarded them and no test pinned them — so
the error was inert. A neighbourhood median cannot be inert about it: it has to
unflatten. The defaults now come from `strafer_shared.constants`, which is where
every other consumer of the policy depth resolution already reads them, and a
shape mismatch raises rather than silently reshaping rows into each other.

The declared shape is now checked against every frame the model receives, not
only when the median runs, so a shape that disagrees with reality cannot sit
inert until a tier with holes reaches it. That check found the same stale 60×80
in two test fixtures, which are corrected here.

This is why the golden movement is three keys and not the two the dispatch
pre-registered; §3 attributes it.

---

## 2. The convention A/B, on one fixed scene

The Kit probe that produced the attribution cannot be re-run comparably on the
flipped pair (§5), so the measurement holds the scene fixed and varies only the
convention. The input is the attribution record's own **clean** observation-term
output at the defect pose — 30 ticks whose near-field class is 0.3786 of the
frame, the class the attribution measured. The production noise model is
imported, never reimplemented. `probes/convention_ab.py`.

Each row feeds its depth into the robot's own tick-0 observation and runs
v2@998 on CPU, which is the attribution record's patch-replay design.

| depth fed to the policy | far clamp, of frame | far clamp, of the near-field class | p95 \|Δ\| vs clean | mean off-goal | toward goal | in the rig class |
|---|---:|---:|---:|---:|---:|---:|
| clean observation term | 0.0000 | 0.0000 | 0.00000 | −78.83° | 0/30 | **29/30** |
| silent, retired convention | 0.0000 | 0.0000 | 0.00000 | −78.83° | 0/30 | 29/30 |
| silent, reconciled | 0.0000 | 0.0000 | 0.00000 | −78.83° | 0/30 | 29/30 |
| stereo only, retired | 0.1892 | 0.4998 | 0.96667 | +67.82° | 0/30 | 0/30 |
| stereo only, reconciled | 0.0000 | 0.0000 | 0.00479 | −26.64° | 30/30 | 0/30 |
| realistic tier, retired | 0.1976 | 0.5042 | 0.96667 | +15.20° | 30/30 | 0/30 |
| realistic tier, reconciled | **0.0000** | **0.0000** | **0.00480** | −26.75° | 30/30 | 0/30 |
| robust tier, retired | 0.2136 | 0.5146 | 0.96667 | +7.34° | 30/30 | 0/30 |
| robust tier, reconciled | **0.0000** | **0.0000** | **0.00959** | −23.40° | 30/30 | 0/30 |
| the published 2026-08-22 rows | 0.1957 | 0.4990 | 0.96667 | — | — | — |

The retired arm reproduces the record: 0.5042 against the published 0.4990, and
p95 = 0.96667 scaled, which is 5.79999 m, the figure the attribution published as
its paired maximum. That agreement is what licenses reading the reconciled rows.

**The reconciled convention is exactly a no-op on an already-filled image.** The
two silent rows — noise disabled, one convention each — are bit-identical to
clean and produce the identical command. Nothing about the fix perturbs a frame
the term has already filled; it only stops the model from inverting it.

**The divergence is gone at both tiers.** Far-clamp share of the near-field
class 0.0000, and p95 |Δ| from the clean term 0.00480 / 0.00959 scaled. The
robust tier's larger residual is its doubled disparity noise, not the convention:
its maximum, 0.39943, is a hole whose neighbourhood median legitimately came from
a nearer surface.

### What the dispatch pre-registered, and what happened

The dispatch's second acceptance clause asked that node-vs-noise-bearing depth
reach p95 |Δ| ≤ 0.01 scaled. **A correct fix cannot meet that as written.** The
node's own tick-0 depth differs from the *clean* sim referent by p95 0.02004
scaled — twice the band — and the attribution record already publishes that
residual as node-vs-sim reconstruction geometry, causally inert but not small.
The convention cannot touch it. Measured: node-vs-noise-bearing p95 goes
0.96667 → 0.02095, i.e. it converges on the reconstruction floor of 0.02004,
which is the most a convention fix can do. The p50 does sit inside the band, at
0.005014 → 0.002819 against a clean-referent floor of 0.000587.

The dispatch's third clause asked that v2 now emit the rig class (−79° to −83°)
where it previously drove. **It does not.** Reconciled depth gives 30/30 toward
the referent and 0/30 in the rig class. The decomposition says why: the stereo
Gaussian alone, with holes and the slam both disabled, already carries the
command from −26.64° instead of clean's −78.83°, and the slam then adds the rest.
Class membership tracks whether realism noise is present at all, not which
convention it writes. This is the attribution record's open item 1 — "the
slammed pixel class is not isolated" — measured directly, and it sharpens it: the
minimal sufficient perturbation is 0.029 m of plain Gaussian.

None of that weakens the fix. The divergence was real, source-level, and 18.89 %
of every frame; it is removed; clean depth reproduces the rig class 29/30, so the
reconciled data now sits within 0.0048 of a referent that is on the robot's side.
What it bounds is the claim: this PR fixes the data, and only a retrain can
produce an artifact that does not key on noise signature.

---

## 3. The golden movement, attributed key by key

The composition contract hash walks the observation cfgs, so the depth noise
model's own fields are inside it at one path,
`observations.policy.depth_image.noise`. `probes/golden_attribution.py` renders
the preimages with the test module's own `_canon` / `_contract` / `_hash` —
serialization is never reimplemented — once per tree state, and diffs them by
field name rather than by position, so an inserted field reads as one addition
instead of shifting every field after it.

All 25 goldens reproduce byte-for-byte on the pre-fix tree, which is the
precondition. **17 then move: 16 of the 22 contract goldens, plus the depth-obs
golden.** The complete pooled delta across every preimage is three keys:

```
added    observations.policy.depth_image.noise.too_close_fill  'near'      x16 (+1)
added    observations.policy.depth_image.noise.hole_fill       'median'    x16 (+1)
altered  observations.policy.depth_image.noise.height          60 -> 45    x16 (+1)
```

Nothing is removed and nothing else is altered; `width` was already 80. The
counts are the 16 depth-bearing contract variants, with the depth-obs golden
carrying the same three at its own shorter path.

**What did not move is the acceptance criterion.** Both
`_POLICY_OBS_LAYOUT_GOLDENS` hold, and they are the half a deployed checkpoint
depends on — the test module's own comment says a randomization change is
expected to move every contract hash and that the layout golden exists precisely
to prove the observation layout did not. The six NoCam contract goldens hold too,
having no depth term, and the palette golden holds. The 17 literals were
rewritten programmatically, each guarded by an assertion that the value in the
file matched the pre-fix hash the tool read before substitution.

---

## 4. The depth fingerprint

The fingerprint moves by design; these are the post-fix reference numbers, not a
gate. Both tiers, 30 frames × 8 envs, seed 42, captured on the canonical pair
once per tree state so the delta is the fix alone rather than the fix plus the
pair flip.

| | Enriched-Robust-Play | | Subgoal-Real-Play | |
|---|---:|---:|---:|---:|
| | pre-fix | post-fix | pre-fix | post-fix |
| mean | 0.3518946758 | 0.3272571001 | 0.2839761985 | 0.2621622123 |
| std | 0.3152281887 | 0.2954513944 | 0.3039665079 | 0.2843566985 |
| p50 | 0.2103566229 | 0.1945360973 | 0.1600223035 | 0.1491441429 |
| frac at max | 0.0986631944 | **0.0655590278** | 0.0805462963 | **0.0559710648** |
| frac at min | 0.0583333333 | 0.0583333333 | 0.0416666667 | 0.0416666667 |
| row band, bottom 3 | 0.1366 0.1326 0.1267 | 0.1039 0.0995 0.0956 | 0.1202 0.1166 0.1122 | 0.0963 0.0923 0.0887 |

The far-clamp share falls 3.31 pp and 2.46 pp, the floor band falls furthest, and
the zero-depth share is unchanged to the digit. That is the expected signature:
the affected class is the floor band the camera looks down onto, the fix stops
writing 6.0 m there, and it touches nothing at the other end of the range.

The pre-fix numbers on this pair differ from the 2026-08-14 baseline only in the
fourth decimal (mean 0.3518947 against 0.3518183), so the pair flip contributed
almost none of the movement above.

---

## 5. The anchor probe does not reproduce its pose on the flipped pair

The attribution's same-pose probe was re-run first, and the run is deposited, but
it cannot carry the acceptance measurement. It launched and completed on the
canonical pair in one attempt — every API it touches survives the flip, including
the `ProxyArray` pose reads its `wp.to_torch` helper rides — and its goal bearing
reproduces to 0.03° (−8.13° against the record's −8.1°). But it anchors on
whatever pose the env spawns the robot at and treats the capture's map
coordinates as an offset from it, and seed-42 room generation moved across the
flip. The robot therefore landed at (−2.149, −1.001) instead of (−0.147, −2.172),
facing open space: its clean near-field share is **0.0000** where the record's is
0.3786, and its clean far-clamp share 0.3889 where the record's is 0.0000. The
metric the acceptance names — the far-clamp share *of the near-field class* — has
an empty denominator there.

That is a property of the probe's anchoring, not of the fix, and it is why §2
holds the scene fixed instead. Reproducing that room would need the retired pin
or a captured scene USD; neither is in this PR's scope, and it is worth a brief
for whoever next needs a pose-stable sim probe.

---

## 6. Gates

| gate | result |
|---|---|
| depth noise, pure | 6 passed, including a near-field survival test that fails on the pre-fix tree at 0.5003 |
| near-field parity contract | 4 passed, running here rather than skipping |
| composition + obs contracts | 148 passed after the re-freeze |
| temporal texture | 40 passed |

The full `strafer_lab` gate, both halves, on the canonical pair: **Kit 499 of
499 across 14 suites, pure 1 259 passed and 1 skipped.** The boot watchdog
relaunched once, in the `rewards` suite — the upstream 6.0.1.0 boot stall, which
is telemetry rather than a failure.

`test_collision_imu_mean_differs_from_free` failed in an earlier full gate and
does not belong to this change. Its environment is `NoCam_Ideal`, which composes
no depth term and no depth noise model at all, so nothing in this change can
reach it; and it fails intermittently either side of the change (one failure and
one pass after, one pass before). The skipped pure test is a deploy-side alias
check that needs a sourced ROS 2, which this host does not have.

The near-field survival test was run against the pre-fix noise model to confirm
it fails there: 0.5003 of an already-filled image reached the far clamp, against
the brief's predicted ~50 %.

---

## 7. What this record does not establish

- **That the fix makes v2 deployable.** It does not; §2 measures the opposite,
  and the retrain owns it.
- **That the slammed class is the mechanism.** The decomposition shows realism
  noise in general moves the command, which leaves the attribution's open item 1
  open and narrows what can be claimed about *why* v2 depended on the corruption.
- **The affected share over a representative rollout.** The brief asks for it on
  the training distribution; §4's fingerprint is 30 frames × 8 envs at one seed
  on two Play tiers, which is a distribution sample, not a rollout.
- **Anything about v1.** It trained under the same rule at a smaller share and is
  untouched here.
- **The 8×8 block median.** Training still renders one ray per policy pixel and
  performs no block reduction; only the hole rescue was brought to parity. Fix
  direction A remains the stronger answer to drift and is not taken.

## Evidence deposit

None of the paths this record names are in this directory. They are in the
companion evidence repository `https://github.com/zachoines/Sim2RealLab-Artifacts`,
a private repository holding the evidence behind these records, under
`depth-convention-fix-2026-09-13/`, deposited at commit
`a6011e51cbb83be050f515a4bfee6b9eee571a46`. Two directories there:

- `record-files/` mirrors this one, so a path the prose above names is the same
  path in the deposit. Restoring it into this directory is what makes the
  commands run:

  ```
  cp -a <clone>/depth-convention-fix-2026-09-13/record-files/. \
      docs/measurements/depth-convention-fix-2026-09-13/
  ```

- `postflip-anchor-probe/` holds the same-pose probe's own outputs from its
  re-run on the canonical pair (§5). It never lived in this directory, so it is
  a sibling rather than part of `record-files/`.

Each carries its own `DEPOSIT.md`, which comes across with the files and is not
part of the record. Neither it nor the restored files belong in a commit.

The two probe scripts resolve their inputs relative to the repository root and
read the 2026-08-22 record as well, so that record has to be restored from its
own deposit for the A/B to re-run — `provenance.md` says which files and where.

sha256 of every deposited file, paths relative to this directory:

```
db52dde2db5b5ea0e5f181efa40169d1863edd1c4c545985bf60332131eac635  ab/convention_ab.json
073cbb8593609f870eba0cc2f5e183e62f147f701850ef4f49553c4c6a987c7b  ab/node_obs_rec0.json
8f344d1f662045056379354d415ae01ef991de0c48ab90b5c6ec981a608aabdb  gates/imu_post_fix_run1.log
8dc5e8c369f76927a0d9852768d2c7a261d91458d3c763a111477189afa7f381  gates/imu_post_fix_run2.log
cb020b25b09c7da180e20f3dd980c277621d88f123f566bd43f5437824305067  gates/imu_pre_fix.log
65e6275f9e7ee8e12bb53a8d2f1509923552285a050f4bc3ff280e2c2d2ca712  gates/test_lab.log
fea6b3f97a092ea0f4556b321550f7ef6f84d9939475efd8384adf1f33f51c51  gates/test_lab_first_pass.log
c5cb7c7f3750ffb8a26be44d703c442d808e71e05d56ce62933936111bfadb86  goldens/after/hashes.json
0f1ec0458956793bd69028a9cf74bad77a806bc0585add2998b9d8d0a4aa3e86  goldens/after/preimages/contract-RLDepthEnriched_Real.json
6615b0a3021cce935cb6199be9a6b9d404001138ff0cfbff4504810f73b472ae  goldens/after/preimages/contract-RLDepthEnriched_Real_PLAY.json
d9109417089ccad39ae0f099d3de54d18d973e0f4068fe2196894cf9a01ec6bf  goldens/after/preimages/contract-RLDepthEnriched_Robust.json
2d1325f67d737398d4056729f215c48df87c1f804c8bfeec20cce8bf1befa0ed  goldens/after/preimages/contract-RLDepthEnriched_Robust_PLAY.json
56e53425b0e922b95b304ef6b5aaec80a9de3c067824ca48b26a52642fb226e4  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real.json
2a3085ba188c7e55dd54a541f47cecb75f0e36e541b1e1421919070f1b139a5e  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
5fbbc180ea7d47527582f306cb1e1de163ca2a2656e79df871b3ac136eab87ab  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust.json
9bb384aa0ce9248045600f573e6ac985bd0bea70f6dfa11d72121f40ca8761db  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
d19a5d600bd759aa9cacdaaf9876b74e91c3be729cebda6fe6b16c013b4bfc3a  goldens/after/preimages/contract-RLDepthSubgoal_Real.json
8ad659d04bc6a120e0af87d38cb42c05b96b48e16887416d99d4d07cecbc342e  goldens/after/preimages/contract-RLDepthSubgoal_Real_PLAY.json
b0708d162ffa3d7d690f264a1fb0f2ffb3c66de32b3e30e5eda38bbff48adc2b  goldens/after/preimages/contract-RLDepthSubgoal_Robust.json
c18bcd56b998bb0396b615a9bbfadbf1cb0d4df84d5d97312d5c0fe764bc5e8c  goldens/after/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
59cfc1fce4a93095ff99f004836272bcb6b43225524181a97dd3db1c5b822246  goldens/after/preimages/contract-RLDepth_Real.json
2d141166c220bbf91876e2ce3a6e530290754749bad04ffe28c6c439a42c12c8  goldens/after/preimages/contract-RLDepth_Real_PLAY.json
dbebf9616149094c3112bb4b74ed7d54035375a643e9f2baae81156ebec210ad  goldens/after/preimages/contract-RLDepth_Robust.json
ecd30523975af7fd291acc407d29c3bcf4d28bfc7d37d88e48aed35fa685a3db  goldens/after/preimages/contract-RLDepth_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/after/preimages/contract-RLNoCam.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/after/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/after/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/after/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/after/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/after/preimages/contract-RLNoCam_PLAY.json
b605f785f5a4ec301034ce73e5279e274f3b86faf05952c8005f2b5103097e65  goldens/after/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/after/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/after/preimages/layout-nocam.json
d0d5283561a7d09eb061c6b5f86fa3349bcbad8541332928491ad2265b6f9418  goldens/before/hashes.json
60c68ccceaca85cc8e11233528c4f0ecdc4a59ffb0254529b3bd13aaa8073cbb  goldens/before/preimages/contract-RLDepthEnriched_Real.json
74857628ed2087e05c68ad7d91e502ca42515c235167f128fb87d1c15f3a6f07  goldens/before/preimages/contract-RLDepthEnriched_Real_PLAY.json
0c928dec66da63b42f2c083645ed1a9aea8015a94f2a99d7c74e396bffb1da5e  goldens/before/preimages/contract-RLDepthEnriched_Robust.json
1b853bf9db728a2444169c536b94f81da6ba576972bb32f1e0bdb56adea96d51  goldens/before/preimages/contract-RLDepthEnriched_Robust_PLAY.json
8c4cc2c62795d1a38dcfb15c9d94dfc0eb0627c4c8b851d888dd786d0158581b  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real.json
8065f92d02b8553690562cb480de51f89c54661f231ebefe88a41337a8c01cc2  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
3a0089f08aaeac9ce2e9755d84ed2470e95d48dafc744f264f2be6410f4c2625  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust.json
143de630af652019fa99fe9f4d5d314fa84bf9b201efc2935f98bc8a8a5ef691  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
27b609dd24e52264b209e8d246d0cef880bf3ee1f7de82cb18bcd320667bf4f9  goldens/before/preimages/contract-RLDepthSubgoal_Real.json
34d7eb3131f91ecaca5befea59086f445da32ba60f3ac849df73c653a325f33e  goldens/before/preimages/contract-RLDepthSubgoal_Real_PLAY.json
e575d202d524a69df2335fbf3bcbde2f5b092287d021c1fbbc5af5f7861f6536  goldens/before/preimages/contract-RLDepthSubgoal_Robust.json
ced2baefaea957c29fc8129797e5723ad38d978886cdc147e15b9a5409399a52  goldens/before/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
42919518766f653a80b2b4661b05cb61e4e5a598eb0b9ca933f49d90bcd57c4f  goldens/before/preimages/contract-RLDepth_Real.json
c032bf44e578d20afa56fff7e7e5867946dbed38450852cc2e1360f4f210d89f  goldens/before/preimages/contract-RLDepth_Real_PLAY.json
05a3f5290486d88e3cc1c52d5122171897be11b29f294cfe5e89854b1cf14356  goldens/before/preimages/contract-RLDepth_Robust.json
394c1fd2496076c1ebf64394655a5fadf5ebda324a2f7707ac08721cf444b748  goldens/before/preimages/contract-RLDepth_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/before/preimages/contract-RLNoCam.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/before/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/before/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/before/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/before/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/before/preimages/contract-RLNoCam_PLAY.json
eaa1e30057b24234d3f9077a26a836ed42e2b48204ffaad6f0817e6ffa71e4dc  goldens/before/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/before/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/before/preimages/layout-nocam.json
e3103106608a6e4bf2c7cb4c3d4a5ed96228ab2925b031ced8221fd9a5fe9c4d  probes/convention_ab.py
69ae2c4dbf711446a77bfb2fc14d939eed5ece410424f87cb6dde3179b9dcf1a  probes/golden_attribution.py
d8a0baa2d0c0e8f0eefaebb9cc963348541e81662850b3ba65095a3f7042185a  render/postfix_depth_obs_enriched_robust_play.json
9b560d5f4df72525152f35aebe215955fd4ac0f252b6b99eecf8c2f0b24316d9  render/postfix_depth_obs_enriched_robust_play.npz
9733b6820cca6ec6f8e61d8ead11b7098ad995c0c3477fc4f691766be5a9880a  render/postfix_depth_obs_subgoal_real_play.json
9d6df72983a3a27bc52e8bb1a2e68ea3cc40151baae05e2d29c68437a9089a44  render/postfix_depth_obs_subgoal_real_play.npz
5d344299804c1284b635582062bdcf6db1fa0dd82e89c64ecbc4c363e0decd3f  render/prefix_depth_obs_enriched_robust_play.json
db517372877e30326b7d91c78fe30c17fa6183478f6060f2f6d343722eb7bf38  render/prefix_depth_obs_enriched_robust_play.npz
f8666067d9a36f97e479f5d6ce077529fd834db7c96bdd7de7d3a12af27e8bfe  render/prefix_depth_obs_subgoal_real_play.json
33d4bf8536cd6dda6777f203b2c964709a3f0a9a9013656c6d70ffa363c7e8cc  render/prefix_depth_obs_subgoal_real_play.npz
```

sha256 of every file in `postflip-anchor-probe/`:

```
10540e1a80efcb831229359962885147d02686ef532384868a28f09fba8f0a99  closed_loop_summary.json
6ff15f028300305d2ec9e8fdbae130fa2335b945ce15e9ad96623917dc35025f  env_obsbuf.jsonl
93f07003d19394626b3d763f5a74b505248d53da5c9a5f52cbed4021e8c33fa4  gym_obs.jsonl
c02d0ae59a13905597b06bfab5a8489a22ca853cbd0dab5b0e9f58f48e56c222  meta.json
d849e28655f93426e1f81cfcf10652b5492d6f6b4703eca8f3a4cf2925384db8  probe_stdout_postflip.log
```

