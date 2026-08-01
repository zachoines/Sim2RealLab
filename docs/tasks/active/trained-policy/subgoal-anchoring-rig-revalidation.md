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

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The 2026-07-31 obs-parity session ruled the observation channel clean and routed
the v2 signed leftward bias to the training lane. It also found three deploy-side
defects that stop *either* policy from tracking, all now shipped:

- [`subgoal-mission-anchoring`](../../completed/subgoal-mission-anchoring.md) —
  deploy paths were robot-rooted, so cross-track error could never develop.
- [`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md) —
  the node ran 23.5 Hz sim against a 30 Hz training cadence.
- [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md) —
  the deploy depth downsample's vertical geometry.

None of the three can produce a signed left/right asymmetry, so **do not expect
this session to move the bias.** What it must establish is whether tracking
itself is fixed.

Two facts from that session shape the protocol:

- **The existing v1 data is not a control.** `v1_mA` shows 65.3% leftward, but at
  a median speed of 0.059 m/s against v2's 0.156–0.371 m/s — v1's "moving"
  samples are near-stationary, so its sign split is not meaningful. v1 is *not*
  established as bias-free.
- **The A/B is a config flip, not a rebuild.** `subgoal_anchoring: mission` vs
  `rolling` on the shipped generator, so the two arms differ in exactly one
  parameter.

## Acceptance criteria

- [ ] Same missions, same scene token, fresh SLAM db, run four arms:
      v2×`mission`, v2×`rolling`, v1×`mission`, v1×`rolling`.
- [ ] **v1 runs at a speed comparable to v2** (median moving speed within a
      stated factor), or the session records explicitly that a real v1 control
      could not be obtained and why.
- [ ] Subgoal-tracking heading error reported per arm. The `mission` arms must
      show cross-track error that *grows* — the `anchor status:` log line's
      `cross_track` field is the direct read — where the `rolling` arms pin it at
      ~0.03 m (one costmap cell), reproducing the measurement this A/B exists to
      test.
- [ ] The inference node's cadence counters reported per arm: achieved sim Hz,
      `depth_frames_received` vs `inferences`, and the skip histogram. Expect
      ~30 Hz sim; a shortfall with `depth_frames_received` also short indicts the
      BEST_EFFORT/depth=1 depth QoS rather than the node.
- [ ] Signed strafe split per arm, on moving samples only, with the speed
      distribution alongside it so the v1 control is judged on its own terms.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — this is a
      measurement session; any code change it motivates is a separate brief.

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
