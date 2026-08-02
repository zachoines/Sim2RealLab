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

1. **The session ran the vanilla scene**
   (`Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`), a deliberate session-side
   choice to keep arm 1 comparable to the 2026-07-31 parity control.

   **v1 trained on vanilla as a matter of record** — enrichment first landed
   2026-07-19/20 and v1's `run_20260708_005923` predates it by eleven days.
   **v2's training scene is not recorded anywhere in this repo** (see the
   precondition below). Since v1's distribution is known and v2's is not, **an
   unknown match disqualifies the cross-model contrast exactly as a known
   mismatch would**: "v1 10/10, v2 4/10, in vanilla" cannot separate
   policy-broken from scene-out-of-distribution either way.

   *The scene axis, stated correctly:* vanilla is **open-top** (`wall_height=1.0`,
   `p_ceil=0.0`) where enriched raises walls to 2.7 m and adds a ceiling at
   p=0.7 — measured in-repo as top-11-row depth pinned at the 6 m clamp **58.9%
   vanilla vs 7.3% enriched**. It is **not** a furniture difference: vanilla
   *pins* difficulty at level 7/7 (the generator's maximum: 8 furniture, 16
   clutter) while enrichment *un-pins* to U[4,7], so enriched averages **less**
   furniture and clutter and is **farther** on average, not nearer. Do not
   restate this as "vanilla is furniture-free" — that is false.
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

## Precondition — establish v2's training scene BEFORE booking rig time

**This brief's premise is that v2's training distribution differs from vanilla.
That is currently an assumption, not a fact, and it is not establishable from
this repo.**

- All three artifact sidecars carry
  `"env_id": "Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0"` — the **vanilla**
  Play env — including v0, whose `obs_dim` 4819 predates the env rework. That
  field is `export_policy.py`'s export-time `--env` default
  (`_DEFAULT_ENV_BY_VARIANT`), not the training task, though its docstring
  claims otherwise.
- `train_strafer_navigation.py` prints `env_name` to stdout and persists no task
  id in the run directory.
- v1's recorded `git_commit` is **absent from this clone**, so even the tree it
  trained against cannot be inspected here.
- A competing hypothesis is live: v2 may be the **vFOV retrain on the vanilla
  generator** — `depth-camera-vfov-parity` calls a "v2 retrain" closed, while
  `procroom-depth-enrichment` still treats the enrichment retrain as an open
  operator decision. Both are referred to as "the v2 retrain".

**Required before running:** the coordinator confirms, from DGX-side training
records (run logs, launch command, or the run directory for
`run_20260727_171735`), which task id produced `model_998.pt`, and that answer is
recorded here.

- If **v2 trained enriched** → this brief proceeds as written.
- If **v2 trained vanilla** → the scene-class confound evaporates, the
  2026-08-01 attribution is reinstated as-is, and this brief closes without a rig
  session. The four arms would then already be the clean test.
- If it **cannot be recovered** → run the session anyway (an enriched arm is
  informative regardless), but the decision rule below must be read as
  "v2 in enriched" rather than "v2 on-distribution".

**Root cause worth fixing separately (source change, not this brief):** the
export sidecar's `env_id` cannot identify a training run, and the training script
writes no manifest. A `train_manifest.json` carrying task id, git SHA, seed and
`num_envs` would have made this precondition a lookup instead of an
investigation.

## Acceptance criteria

- [ ] **Precondition discharged:** v2's training task id recorded above, or
      explicitly marked unrecoverable with the decision rule reinterpreted.
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
