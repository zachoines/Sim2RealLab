# Fill the enriched × `mission` cell the four-arm session could not

**Type:** investigation (behavioural acceptance — addendum)
**Owner:** Jetson (rig session; DGX supplies the enriched bridge)
**Priority:** P0 — it is the only cell that can attribute the v2 advance failure,
and two shipped decisions currently rest on an attribution the evidence does not
support.
**Estimate:** S (one short rig session; two arms, protocol already built)
**Branch:** `task/enriched-scene-anchoring-addendum`

## Story

As the **coordinator deciding whether the v2 advance failure is policy-owned or
scene-out-of-distribution**, I want **the one arm combination that has never been
run — enriched scene under `mission` anchoring** — so that **the attribution rests
on a closed loop generating its own in-support observations rather than on a
vanilla-scene contrast or a replay of already-failing inputs.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The [2026-08-01 four-arm session](../../completed/subgoal-anchoring-rig-revalidation.md)
verified the deploy stack and concluded the residual defect was policy-owned.
**That attribution is bounded and has been amended.** Three facts:

1. **The session ran the vanilla scene.**
   `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`, a deliberate session-side
   choice to keep arm 1 comparable to the 2026-07-31 parity control. But **v2
   trained on the enriched distribution** (`StraferNavCfg_RLDepthSubgoalEnriched_Robust`)
   and **v1 on vanilla**. Vanilla rooms are open-top and furniture-free —
   far-clip-heavy in depth, the exact axis the enrichment program moved v2's
   training away from. Every arm was a home game for v1 and an away game for v2.
   "v1 10/10, v2 4/10, in vanilla" cannot separate policy-broken from scene-OOD.
2. **The offline replay is circular for the attribution.** Its inputs were
   recorded from a loop in which v2 was already failing, so they embody the
   failure. It shows v2 retreats on those inputs — which the frozen-regime
   account already predicted — and that v1 is robust off-distribution. It does
   not show v2 is broken on-distribution.
3. **Enriched × `mission` has never been run.** Every historical enriched
   session used `rolling`; the 2026-08-01 `mission` arms were vanilla.

Two coordinator findings (2026-08-02) shape what this session should expect:

- **The export is exonerated.** The v2 ONNX matches `model_998.pt` at the
  constants level (normalizer mean byte-identical, std divisor identical up to a
  uniform +0.01 epsilon, output head byte-identical; the apparent GRU difference
  is the standard PyTorch→ONNX gate reorder), and `export_policy.py` runs a
  torch-vs-ONNX round-trip probe at export. **Do not chase the artifact.**
- **v2's failure surface is wider than "frozen straight-ahead".** At a cold start
  on real enriched-scene inputs it commands negative vx at **every** bearing from
  −60° to +60° (20/20 frames), while the same weights complete ~88% of
  closed-loop sim episodes and track bearing correctly in lateral output. The
  synthesis is **joint-distribution brittleness**: v2's advance decision falls
  over when the (depth, subgoal, state-history) joint leaves its training
  support, and both synthetic and deploy-manufactured joints do. This is exactly
  why only a closed loop can settle it.

## Acceptance criteria

- [ ] Two arms on `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`:
      **v2×`mission`** and **v2×`rolling`**. v1×`mission` optional if the window
      allows; v1 needs no further characterisation otherwise.
- [ ] Protocol identical to the four-arm session: fixed start pose **and
      heading**, fresh SLAM db, **new** scene token, per-arm force-recreated
      `inference` container on the audited image, and the same instrumentation
      (`anchor status:` cross_track, cadence counters, `v_par`, per-sample goal
      bearing, threshold sweep).
- [ ] **Goal set crosses bearing sign with map region** — at least one
      left-bearing and one right-bearing goal in EACH of two regions — plus one
      dead-ahead goal and one near-furniture goal. The 2026-08-01 set confounded
      bearing with region and could not separate them; the near-furniture goal
      makes the standoff read scorable if v2 approaches at all.
- [ ] Report the same per-arm deliverables: achieved sim Hz, `depth_rx` vs
      `inferences`, skip-by-cause breakdown, `anchor status:` cross_track,
      `v_par` with the speed distribution and threshold sweep, mission outcomes.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — measurement
      session; any code change it motivates is a separate brief.

## Pre-registered decision rule

Committed before the session runs, so the outcome cannot be read backwards.

- **v2 advances under enriched×`mission`** → the deploy stack is validated for
  goal-directed motion in the deployment-relevant scene class (real deployment
  rooms are furnished, i.e. enriched-like). The vanilla failure re-files as a
  **scene-robustness training-lane note, not a deploy defect**, and the
  2026-08-01 attribution is retired.
- **v2 still fails under enriched×`mission`** → the failure is policy-owned with
  clean attribution and the training lever fires. Its shape is informed by the
  joint-brittleness finding: **robustness augmentation — off-joint subgoal
  exposure and scene-mix — not merely frozen-subgoal replay.**
- **Either way**, v2×`rolling` on enriched is the within-session regression
  control; it is expected to reproduce the historical enriched-session collapse,
  and if it does not, that is itself a finding about what changed on this build.

## Investigation pointers

- The enriched bridge task **is registered**:
  `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`
  (`strafer_lab/tasks/navigation/__init__.py`). The coordinator supplies the DGX
  launch.
- The anchoring lever is a **read-only bind-mount** over the installed YAML at
  `/ws/install/strafer_inference/share/strafer_inference/config/subgoal_generator.yaml`
  — the config is baked into `strafer-gpu` at build time, so editing the
  worktree copy would force a rebuild and stamp the image `-dirty`. Overlay and
  per-arm configs: `~/strafer_v2_validation/arm_configs/`.
- v1 is `/models/policy.onnx`; there is no `/models/strafer_depth_subgoal_v1.onnx`.
- Session tooling is reusable as-is: `~/strafer_v2_validation/tools/`
  (`run_arm.sh`, `reposition.py`, `arm_summary.py`, `scrape_logs.py`,
  `bridge_probe.py`, `costmap_probe.py`, `offline_probe.py`).
- Cadence counters are **cumulative with no reset path**, and container logs are
  destroyed by the force-recreate — capture them before switching arms.

## Out of scope

- Any code change.
- The depth QoS flip — [`depth-qos-reliable-flip`](../reliability/depth-qos-reliable-flip.md)
  owns it and may proceed in parallel or first. **If it lands before this
  session, note the changed build label** in the findings.
- The v2 ONNX export, which is exonerated above.
