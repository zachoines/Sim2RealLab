# Re-validate subgoal tracking on the rig with a real v1 control

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

**The deploy-side anchoring defect is fixed; the v2 advance failure is not, and
it is now isolated to the policy weights.**

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
- **The decisive isolation is offline.** Replaying byte-identical recorded
  observations through both ONNX artifacts with the GRU state threaded as the
  deploy loader threads it: **v1 commands vx median +1.067 m/s (100% forward),
  v2 commands −0.244 m/s (0% forward).** No anchoring, no deploy stack, no start
  condition — the separation is in the weights.
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
