# Re-validate subgoal tracking on the rig with a real v1 control

**Status:** Shipped 2026-08-01 in `4880064` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/177

**Type:** investigation (behavioural acceptance)
**Owner:** Jetson (rig session; DGX supplies the bridge)
**Priority:** P0 — it is the acceptance for three shipped deploy-side fixes and
the gate on whether the v2 bias is the only remaining defect.
**Estimate:** M (one rig session)
**Branch:** `task/subgoal-anchoring-rig-revalidation`

## Story

As the **operator deciding whether the deploy lane is now clean**, I want **one
behavioural session that A/Bs the shipped anchoring semantics and puts a real v1
control beside v2 at comparable speed**, so that **any residual tracking failure
is attributable to the policy rather than to the deploy stack.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

The 2026-07-31 obs-parity session ruled the observation channel clean and routed
the v2 signed leftward bias to the training lane. It also found three deploy-side
defects that stop *either* policy from tracking, all now shipped:

- [`subgoal-mission-anchoring`](subgoal-mission-anchoring.md) —
  deploy paths were robot-rooted, so cross-track error could never develop.
- [`inference-cadence-shortfall`](inference-cadence-shortfall.md) —
  the node ran 23.5 Hz sim against a 30 Hz training cadence.
- [`depth-camera-vfov-parity`](depth-camera-vfov-parity.md) —
  the deploy depth downsample's vertical geometry.

None of the three can produce a signed left/right asymmetry, so **do not expect
this session to move the bias.** What it must establish is whether tracking
itself is fixed.

> **Superseded on 2026-08-01 by the session's own measurements.** Two of the
> three framing claims above did not survive contact with the rig.
>
> **The bias line is superseded.** On the current build the historical lateral
> skew is absent under *both* anchoring modes — commands sit 50.1% left-of-goal
> under `rolling` (median offset +0.6°) against 76–85% in the historical
> captures. But the session cannot attribute that to these three fixes: build,
> procedural scene instance and goal set all changed together. What it can say
> is that the skew is gone and start heading does not explain it. Separately,
> the historical "95.6% left on ahead/right goals" figure is **withdrawn** — it
> rested on one mission labelled from a single instant, which held the goal to
> its left for 95.5% of the samples counted.
>
> **"A config flip, not a rebuild" is false.** `strafer_inference/config/` is
> `COPY`'d into `strafer-gpu` at build time (`docker/Dockerfile.gpu:54`) and
> colcon-installs to the package share dir; no shipped compose file bind-mounts
> it, and `inference_policy.launch.py:99-102` hard-codes the path when it
> includes `subgoal_generator.launch.py`, so that launch file's `config_file`
> argument is unreachable from the container's command. Editing the YAML in the
> worktree forces a rebuild between arms **and** stamps the image `-dirty`,
> destroying the revision-label currency check this same brief requires. The
> session resolved it with a read-only bind-mount over the installed YAML.
>
> **"v1 is not established as bias-free, only as barely moving" is also
> superseded.** v1 is not slow — it is duty-cycled. At a 0.01 m/s cut its median
> speed reads 0.044 m/s; at 0.10 m/s only 25.5% of its commands survive but
> those read median 0.416 m/s and 90.2% forward.

Two facts from that session shape the protocol:

- **The existing v1 data is not a control.** `v1_mA` shows 65.3% leftward, but at
  a median speed of 0.059 m/s against v2's 0.156–0.371 m/s — v1's "moving"
  samples are near-stationary, so its sign split is not meaningful. v1 is *not*
  established as bias-free.
- **The A/B is a config flip, not a rebuild.** `subgoal_anchoring: mission` vs
  `rolling` on the shipped generator, so the two arms differ in exactly one
  parameter.

## Acceptance criteria

- [x] Same missions, same scene token, fresh SLAM db, run four arms:
      v2×`mission`, v2×`rolling`, v1×`mission`, v1×`rolling`.
      **20 missions, all on-reference.** One continuous sim run, scene token
      `reval_reval1`, db fresh at 0 MB. Every mission additionally starts from a
      fixed *heading* (60°), restored by Nav2 — without that the goal set is
      bearing-skewed and the signed split is unreadable.
- [x] **v1 runs at a speed comparable to v2** (median moving speed within a
      stated factor), or the session records explicitly that a real v1 control
      could not be obtained and why.
      **The premise needs restating: v1 is duty-cycled, not slow.** At a 0.01 m/s
      cut its median moving speed is 0.035–0.044 m/s against v2's 0.081–0.137 —
      nominally 2–4× slower. At a 0.10 m/s cut only 13–26% of its commands
      survive but they read median **0.397–0.416 m/s, 71–90% forward**, i.e.
      *faster* than v2 at the same cut. A single threshold cannot characterise
      it; the per-arm threshold sweep is reported instead.
- [x] Subgoal-tracking heading error reported per arm. The `mission` arms must
      show cross-track error that *grows* — the `anchor status:` log line's
      `cross_track` field is the direct read — where the `rolling` arms pin it at
      ~0.03 m (one costmap cell), reproducing the measurement this A/B exists to
      test.
      **Confirmed within both models:** v2 0.028 → 0.308 m, v1 0.047 → 0.129 m,
      with `held` going 0 → 3440 and 0 → 3025. Caveat: under `rolling` the value
      scales with how far the robot travels between re-rootings, so it must be
      read within a model, never across.
- [x] The inference node's cadence counters reported per arm: achieved sim Hz,
      `depth_frames_received` vs `inferences`, and the skip histogram. Expect
      ~30 Hz sim; a shortfall with `depth_frames_received` also short indicts the
      BEST_EFFORT/depth=1 depth QoS rather than the node.
      **23.7 / 25.1 / 28.0 / 27.5 Hz sim at 97.9–98.4% consumption.** Note the
      counter is `depth_rx`, there is no skip *histogram* (a five-way
      skip-by-cause breakdown ships instead), and the counters are cumulative
      with no reset path — each arm needs a force-recreated container.
- [x] Signed strafe split per arm, on moving samples only, with the speed
      distribution alongside it so the v1 control is judged on its own terms.
      Reported with the speed distribution, a threshold sweep, and the
      **per-sample** goal bearing, because a split bucketed from a single
      instant manufactures a bias that is not there.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      Swept: no doc states the anchoring A/B is a config flip, names
      `depth_frames_received`, or claims a signed cross-track. The three
      corrections live in this brief's Context block.
- [x] No regression in the workflows the touched code supports — this is a
      measurement session; any code change it motivates is a separate brief.
      **Zero source changes.** The anchoring lever was a read-only bind-mount
      over the installed YAML, so all four arms ran the audited
      `25305a4c4103` image. The one code change this motivates is filed as
      [`depth-qos-reliable-flip`](../active/reliability/depth-qos-reliable-flip.md).

## Findings (2026-08-01)

Full report and preserved artifacts: `~/strafer_v2_validation/`
(`REVALIDATION_FINDINGS.md`, per-arm mission JSON under `arms/`, inference logs
under `logs/`, tooling under `tools/`). Build `25305a4c4103`, verified by
`docker inspect` on both images. Vanilla `ProcRoom-v0`, one continuous sim run,
fresh SLAM db `reval_reval1`.

**The deploy-side anchoring defect is fixed. The v2 advance failure is not, and
it is isolated to the v2 policy *under the tested conditions* — not, on this
session's evidence, to the weights themselves.**

> **Attribution caveat (amended 2026-08-02 on coordinator ruling).** Two
> confounds bound what the four arms can conclude, and a third cell is unfilled.
>
> 1. **Scene class.** The session ran the **vanilla**
>    `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`. That was a deliberate
>    session-side choice — it makes arm 1 directly comparable to the 2026-07-31
>    parity control.
>
>    **v1 trained on vanilla, as a matter of record**: the whole enrichment
>    feature (`enrich_depth`, the `…Enriched…` cfg classes and task ids) first
>    landed 2026-07-19/20, and v1's checkpoint dir `run_20260708_005923` predates
>    it by eleven days — no enriched variant existed for it to train on.
>
>    **v2's training scene is UNKNOWN.** It is *not* recorded anywhere: all three
>    artifact sidecars carry the same `env_id`
>    (`Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0` — the *vanilla* Play env),
>    because that field is the export-time default rather than the training task;
>    v0 carries it too despite predating the env rework. `train_strafer_navigation.py`
>    persists no task id. All that is established is that enriched training was
>    *possible* — `69014c6` (2026-07-26) contains the enriched classes and
>    `run_20260727_171735` postdates it. A competing hypothesis is live: v2 may be
>    the **vFOV retrain on the vanilla generator** (`depth-camera-vfov-parity`
>    calls a "v2 retrain" closed while `procroom-depth-enrichment` still treats
>    the enrichment retrain as an open operator decision).
>
>    **An unknown match disqualifies the cross-model contrast exactly as a known
>    mismatch would.** v1's distribution is on the record and v2's is not, so
>    **"v1 10/10, v2 4/10, in vanilla" cannot separate policy-broken from
>    scene-out-of-distribution** either way.
>
>    *Mechanism, stated correctly:* vanilla is **open-top** (`wall_height=1.0`,
>    `p_ceil=0.0`, no ceiling entity) where enriched raises walls to 2.7 m and
>    adds a ceiling at p=0.7 — measured in-repo as top-11-row depth pinned at the
>    6 m clamp **58.9% vanilla vs 7.3% enriched**. It is **not** a furniture
>    difference: vanilla *pins* difficulty at level 7/7
>    (`[2 internal walls, 8 furniture, 16 clutter]`, the generator's maximum)
>    while enrichment *un-pins* to U[4,7], so enriched rooms average **less**
>    furniture and clutter, and are **farther** on average, not nearer.
> 2. **The offline replay is circular for this attribution.** Its inputs are the
>    2026-07-31 node observations — **vanilla scene** (the 17:00 obs-dump
>    capture leg ran `ProcRoom-v0`; confirmed from the DGX Kit log), recorded
>    under `rolling` anchoring *from a loop in which v2 was already failing*.
>    All three defects therefore stack in the same direction: wrong scene class
>    for v2, legacy anchoring, and inputs that embody the failure. That makes the
>    replay **consistent with the scene axis, not evidence against it**.
>    Observations
>    recorded from a failing loop embody the failure (frozen dead-ahead
>    subgoals, viewpoints wherever the robot hovered). The replay shows v2
>    retreats **on those inputs**, which the frozen-regime account already
>    predicted, and shows v1 is robust off-distribution. It does **not** show v2
>    is broken on-distribution.
> 3. **The decisive cell has never been run:** enriched scene × `mission`
>    anchoring. Every historical enriched session used `rolling`; this session's
>    `mission` arms were vanilla. Owned by
>    [`enriched-scene-anchoring-addendum`](enriched-scene-anchoring-addendum.md).
>
> **What survives unamended:** the anchoring semantics verification (measured in
> both models), the cadence numbers with their load-dependent attribution, the
> QoS discriminator, P2, P3, and the withdrawal of the historical left-bias
> statistic. Those are within-session or within-model measurements that the
> scene class does not confound.

Every mission starts from a fixed pose *and heading* (60°), restored by Nav2
rather than the policy, giving body-frame goal bearings of −60/−33/0/+28/+60° —
bearing-balanced, because `+vy` serves a left-lying goal and an unbalanced set
makes the signed split unreadable.

| arm | advance (`v_par` median / closing) | outcomes | anchor `cross_track` median |
|---|---|---|---|
| v2 × `rolling` | −0.067 m/s / 31.7% | 2 adv, 3 ret | 0.028 m |
| v2 × `mission` | −0.006 m/s / 45.9% | 2 adv, 3 ret | **0.308 m** |
| v1 × `mission` | **+0.013 m/s / 72.2%** | **5 adv, 0 ret** | 0.129 m, growing |

- **Anchoring works.** Under `mission`, cross-track develops to ~0.31 m median
  before the 0.5 m admission bound re-anchors (`held=3440 / admitted=13`,
  `cross_track_exceeded=7`); under `rolling` it never exceeds ~0.05 m at p75
  with `held=0` and 100% `rolling_mode` admissions.
- **v2 does not advance under either mode**, and the dead-ahead goal — the
  frozen-subgoal regime — retreated in both arms without ever closing on its
  start distance.
- **Advance is measured as `v_par = v · ĝ`**, not `vx`. The base is holonomic
  mecanum, so `vx>0` scores a correct sideways approach as a failure on any goal
  not dead ahead.
- **Offline replay, read with its limits.** Replaying byte-identical recorded
  observations through both ONNX artifacts with the GRU state threaded as the
  deploy loader threads it: **v1 commands vx median +1.067 m/s (100% forward),
  v2 commands −0.244 m/s (0% forward)** — raw interpreted velocity, *pre-clamp*
  (see the units note below). The separation is real and rules out the
  observation vector, the replay harness, the recurrent threading and the action
  interpretation as sources. It does **not** establish that the v2 weights are
  broken: the inputs were recorded from an already-failing v2 loop, so they
  embody the failure. See the attribution caveat above.
- **The export is exonerated — do not chase it.** (Coordinator, 2026-08-02.) The
  v2 ONNX was checked against `model_998.pt` at the constants level:
  `obs_normalizer._mean` byte-identical, the std divisor identical up to a
  uniform +0.01 epsilon, the output head byte-identical; the apparent GRU weight
  difference is the standard PyTorch→ONNX gate reorder. `export_policy.py` also
  runs a torch-vs-ONNX round-trip probe at export time that raises on mismatch.
  **The artifact faithfully reproduces the trained actor.**
- **v2's failure surface is wider than the frozen-straight-ahead story.**
  (Coordinator, 2026-08-02.) At a cold start (h=0, last_action=0, real
  enriched-scene depth and proprio, subgoal pasted at 1 m), v2 commands negative
  vx at **every** bearing from −60° to +60°, on 20/20 sampled frames — while the
  same weights complete ~88% of closed-loop sim episodes and its lateral output
  tracks bearing correctly throughout. This also **refutes** the hypothesis that
  tick-1 bearing response explains this session's bearing↔outcome pattern.
  **Narrowed 2026-08-02:** those probe frames were **vanilla-scene**, not
  enriched as first reported. The wider "brittle to the joint (depth, subgoal,
  state-history) distribution" synthesis is **retracted**. What the probes
  actually demonstrate is narrower and cleaner: **v2 refuses to advance on
  vanilla depth at any bearing, cold-start** — a blunt scene-OOD response, on
  the same scene axis as the rig confound. v2's enriched closed-loop record is
  healthy (0.883 completion).

### Units and clipping convention

Offline-replay velocities are **raw interpreted, pre-clamp**: the network's
`action[0:2]` multiplied by `MAX_LINEAR_VEL` (≈1.568 m/s). Rig-measured
velocities are read from the recorded `/cmd_vel` topic and are therefore
**post-clamp**, bounded by `NAV_LINEAR_VEL` (0.784 m/s = `MAX_LINEAR_VEL` ×
`NAV_VEL_SCALE` 0.5). That is why v1's replay median of +1.067 m/s legitimately
exceeds the 0.784 m/s cap quoted for rig figures — the two are not the same
quantity. The deploy clamp is an L1 **proportional scale of the whole velocity
vector**, not a per-component clip, so it preserves commanded direction: a
pre-clamp median remains a valid *direction* statistic even where its magnitude
is not achievable. Do not compare a pre-clamp magnitude against a post-clamp one
without converting.
- **Two start-condition hypotheses tested and closed offline.** A carried-over
  `last_action` is refuted (the sign is inverted, and a first-step-only
  perturbation moves the median by 0.0005 m/s); body-velocity conditioning is
  real and monotonic but far too small (v2 stays 0% forward even when told it is
  moving at +0.6 m/s).
- **Cadence: 23.7 / 25.1 / 28.0 Hz sim** across arms against a 30 Hz target,
  consumption 97.9–98.4%. A **concurrent** independent subscriber on the node's
  own QoS saw 29.84 Hz while the node saw 22–25 Hz, which indicts the
  BEST_EFFORT/depth=1 receive path rather than the bridge. Filed as
  [`depth-qos-reliable-flip`](../active/reliability/depth-qos-reliable-flip.md).

### Standoff and corner ride-alongs

The standoff read **could not be scored for v2**: it never approached, so the
"hesitation ≈0.35 m out" signature had no opportunity to appear. v1 closed to
0.435 m on the same goal and was still closing at budget expiry.

## Investigation pointers

- `deploy/.env` is the artifact lever (`STRAFER_INFERENCE_MODEL_PATH`) and the
  scene-token lever (`STRAFER_SLAM_SCENE_TOKEN` — **bump per sim run**). The
  local `docker-compose.override.autonomy-local.yml` pins that used to shadow
  both were commented out on 2026-07-31.
- The generator's A/B knob is `subgoal_anchoring` in
  `source/strafer_ros/strafer_inference/config/subgoal_generator.yaml`.
- Prior session findings and preserved artifacts: `~/strafer_v2_validation/`
  (`OBS_PARITY_FINDINGS.md`, the parity bag, both obs dumps).

## Out of scope

- The v2 signed bias — training lane owns it; this session only measures it.
- Any deploy-side code change. If the session finds one, file it.
