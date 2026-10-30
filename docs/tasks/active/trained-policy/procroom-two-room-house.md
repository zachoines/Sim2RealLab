# Add a two-room ProcRoom house on new IDs, so training paths cross an interior doorway

**Type:** task (a ProcRoom variant on new IDs, behind CPU and Kit gates; training on it is a separate decision)
**Owner:** DGX (`strafer_lab`: the ProcRoom generator, the composition, the subgoal command, the instruments)
**Priority:** P2. Proposed by the maintainer on 2026-10-23: no training path leaves its room, and the deployment's
houses are made of doorways. Independent of v4: by default the house trains after v4, as a warm-start leg from it (v4h).
**Estimate:** L (the corridor re-measure, CPU gates, the house IDs and their Kit gates, a bridge variant; no training)
**Branch:** `task/procroom-two-room-house`

**Filing this brief starts no training and opens no Kit window.** Every Kit window below is scheduled by the operator,
never runs during a lane session, and runs from its own worktree, never from the lane's checkout.

## Story

As **the depth-subgoal policy's training distribution**, I want **a ProcRoom variant in which a second, furnished room
joins the first through an interior doorway, with spawns, goals and planned paths drawn over both rooms, on new IDs**,
so that **cross-room paths enter training and no v3 ID or golden moves.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md): a new variant is a new ID.
- [`procroom-placement-architecture`](procroom-placement-architecture.md): F2, the July ruling, F-3, the consult list.
- [`procroom-depth-enrichment`](procroom-depth-enrichment.md): the descriptors D1–D5 and the enriched parameters.
- [`depth-subgoal-v4-scoping`](depth-subgoal-v4-scoping.md): v4's one list of changes, which does not carry the house.
- [`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md):
  v3's completion-time bound, which the horizon rule uses.
- **The scoping analysis**, deposited as `procroom-two-room-scoping-2026-10-23/record-files/` in the evidence
  repository ([Sim2RealLab-Artifacts](https://github.com/zachoines/Sim2RealLab-Artifacts), at `c0dd04b`): its
  `analysis/procroom-two-room-house_analysis.md`, cited below by section name, beside the replica, corpus replay and
  models it cites. Its figures are post hoc; its replica is a numpy stand-in, not the generator.

## Context

Training's ProcRoom is one room per env, and free space marks every cell outside its bounding box occupied, so no
spawn, goal or planned path leaves the room. The Infinigen heavy houses' planned paths cross a room boundary (a door
or an open-plan edge) at the shares below, on PR-3's endpoint-pair protocol (post hoc, read after the house was
proposed: the analysis, "What the Infinigen houses look like"). The July ruling that parked F2 reads: "Re-trigger for
any F2 revival: a descriptor case (D2 mid-row discontinuities) with its own gate — the path-topology case is closed"
(`procroom-placement-architecture`, "Findings for the operator — the PR-3 outcome, ruled 2026-07-25"). It closed the
curvature case for F2's single-room partition and does not bar a house-scale variant (decision 1). So this brief
claims no curvature target, and its acceptance rests on crossing support, descriptors and cost.

| straight-line band | Infinigen heavy corpus: paths crossing a room boundary | ProcRoom today |
|---|---|---|
| 1.5–4 m | 31.9 % | 0 % (by construction) |
| 4–8 m | 73.2 % | 0 % (by construction) |
| 8–15 m | 97.5 % | no paths |

## Decisions

| # | Decision | Options and status | Who, when |
|---|---|---|---|
| 1 | Scope of "the path-topology case is closed" | **Ruled 2026-10-26:** it closed the curvature case for F2's single-room partition and does not bar a house-scale variant. This brief claims no curvature target; its acceptance stays on crossing support, descriptors and cost | the coordinator; ruled |
| 2 | Interior door width | (i) [1.2, 2.0] m, ruling F-3's floor applied by analogy; (ii) today's perimeter law, [0.8, 2.0] m; (iii) Infinigen's cutter band, 0.77–0.96 m, which needs an aperture-aware off-path tolerance (out of scope). The analysis leans to (i), which meets F-3's letter for a centred robot but may not meet its intent at off-centre crossings, and leaves every trained door wider than every corpus interior door measured ("The horizon, and the doorway's binding limit") | the maintainer, after the corridor re-measure and before the CPU gates |
| 3 | Episode horizon on the house IDs | a longer episode, E from Gate C's rule (the replica predicts 25–30 s, against today's 20 s); a path-length cap, which defeats the long band; a per-episode time-out scaled to the planned length, a new termination used only by house IDs | the maintainer, with Gate C's value, before the house IDs fix their episode length |
| 4 | v4 row or v4h | **v4h, the default:** v4 launches without the house, and a warm-start leg from v4 on the house IDs follows on the maintainer's word. A v4 row needs, before v4's go, the house IDs merged with their gates passed, the v3 house baseline read, and a short warm-start leg from v3 on the house IDs passing its one-room non-regression rule; it also confounds every v4-against-v3 read of v4's other rows ("Relation to v4") | the maintainer, before v4's go |
| 5 | `p_house`, the per-episode share of house episodes | 0.5, provisional, after the column-knob precedent; another value, chosen after a leg has measured 0.5, re-runs that leg or is labelled post hoc | the maintainer, with decision 4 |

Fixed in the CPU gates' pre-registration, before any measurement, with no ruling needed now: the room-2 size rule's
selection, the far-corner gap, an exterior door on room 2, and the D2 mid-row measure ("Decisions this brief asks for").

## Acceptance criteria

Thresholds are fixed here, before any build; one changed after its first measurement is labelled post hoc, and no
gate rests on a modelled number. The design is in the analysis ("The design", "Keeping v3's contract byte-identical").

**Before generator code**
- [x] **The corridor re-measure, first** (done 2026-10-29, the deposit's `corridor_remeasure/`): on the generator's own
      functions a gap connects at every phase from 0.80 m (one corridor cell; 0.77–0.79 m at 70–90 % of phases; 1.20 m,
      4–10 cells), and a drawn door from 0.81 m. The replica agreed; the in-repository probe read one cell wide. Decision 2
      follows.
- [ ] Decisions 2, 3 and 5 recorded (or labelled provisional), and the coordinator's consult on the gated structural
      extensions: a second primitive collection, the larger house grid, a new `SceneSourceCfg` field and new gym IDs.

**The variant, on new IDs only**
- [ ] New IDs (for example `Isaac-Strafer-Nav-RLDepth-Subgoal-House-{Real,Robust}-v0` and their `-Play-` pairs) with
      new goldens, composed with `enrich_depth=True` under `_COMPOSED_RL` keys containing "Enriched"; their policy
      layout equals the depth layout golden (`obs_dim` 3619). Per env they export a room-label grid and the interior
      door segment, which a draw-sequence test shows draw no randomness.
- [ ] **No frozen literal moves:** the 22 contract goldens, `_PALETTE_GOLDEN`, `_DEPTH_OBS_GOLDEN`, both
      `_POLICY_OBS_LAYOUT_GOLDENS`, the 201-case vanilla sweep, the 340-op draw sequence, `_SIGNATURE_GOLDEN`,
      `_PALETTE_ORDER_GOLDEN` and the slot-range pins.
- [ ] **Additions only.** No edit to an existing function body in `commands.py`, `rewards.py` or the shared generator
      code; in `composed_env_cfg.py`, only branches taken when the new house field is set. No change to `NUM_OBJECTS`,
      `GRID_SIZE` or `OBJECT_SIZES`; no `PlacementCfg` field; no new field on `SubgoalCommandCfg` or on a reward cfg
      that existing IDs use; no `p_house` draw outside the house path.
- [ ] **The routing edits move nothing:** every existing ID's composed cfg, rendered whole on worktrees of `main` and
      of the branch (the new field dropped, one root-path token, one stand-in scene corpus), gives identical sha256
      lists, both in the record; a new test pins each existing variant's scene class.

**CPU gates on the real generator** (Kit-free, from a worktree of the branch, at `p_house` = 1, at least 1 536
episodes per size-rule arm, one per house, on training's goal draw; a failure stops at the report)
- [ ] **Gate A, crossing support.** The 4–8 m straight-line crossing share is at least 67.9 %, the lower 95 % bound of
      the corpus's 73.2 %. The 1.5–4 m and 8–15 m shares are reported against the corpus, ungated.
- [ ] **Gate D, the doorway.** Every drawn door width lies in the law decision 2 rules, and every open width is at
      least its minimum; room 2 lies in room 1's free-space component with at least 100 reachable cells in at least
      99 % of houses; the door's passage after inflation is at least 3 cells wide in every house; `bfs_failed` is 0,
      `spawn_count_min` stays above 100, and the search iterations stay below their caps.
- [ ] **Gate C, the horizon rule.** E, for decision 3, is the least of {20, 25, 30, 35, 40} s at which the house share
      of planned paths over (E − 11.87) / 1.53 m is at most the one-room share at 20 s; if none is, the gate fails.
- [ ] **Reported, ungated:** turning, bending, turn-density p90 and PS2 split by crossing (no curvature target,
      decision 1); per-room fragmentation and park-ladder rates against the shipped arm.

**Kit windows** (each scheduled by the operator, never during a lane session, from its own worktree)
- [ ] **Gate E, descriptors**, on house episodes against the shipped arm re-measured in the same session. D1, D3
      top-11 and D4 each pass if the house arm's 95 % interval overlaps the shipped arm's, or lies between the shipped
      point estimate and the Infinigen mean (D1 1.19 m, D3 8.3 %, D4 49.9 %). D5 is at most 2.21 % and D3 top-11 at
      most 17.3 %, the Infinigen maxima; D2 top-row and D2 mid-row are reported. A failure stops at the report.
- [ ] **Gate F, cost.** 20 iterations at 96 envs on the house ID and on v3's ID. Stop and consult if peak `sys_used`
      leaves under 10 GiB of the host's 124 543 MiB free, or if collection time exceeds twice v3's.
- [ ] **The lane room is unchanged:** after the merge, the seed-42 grid re-dumped on
      `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0` has sha256 `36929c68…`; a mismatch is a stop.
- [ ] **The v3 house baseline** (an eval, ungated): v3 on the house at `p_house` = 1, cross-room against same-room
      completion and door failures by length bin. It reads whether doorway competence is missing before any training.
- [ ] **A house bridge variant:** a capture ID beside the ProcRoom-Enriched bridge, its seed chosen by pre-registered
      criteria read from a Kit boot's grid dump, covered by `test_the_capture_variants_record_the_true_referent`.

**Docs and regression**
- [ ] Dated corrections: `procroom-depth-enrichment.md` T3 (Infinigen's two-room opening is 0.80 m, not 2 m), and
      `procroom-placement-architecture.md`'s corridor probe (one cell wide; the raster floor is 0.80 m, not 0.70 m).
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-lab-pure` passes from a worktree of the branch; Kit gates are read from the JUnit
      XML, never from a raw pytest exit code.

## Investigation pointers

- The analysis: "What training sees today", "The design" (the keep-out and house park ladder that connectivity
  needed, the grid window, the spacing), "Keeping v3's contract byte-identical", "The horizon, and the doorway's
  binding limit", "Cost", "Relation to v4", "Relation to the lane".
- Source: `mdp/proc_room.py` (`generate_proc_room`, `_gpu_bfs`, `_free_space_components`,
  `build_proc_room_collection_cfg`); `composed_env_cfg.py` (`SceneSourceCfg`, `_select_scene`,
  `_ComposedStraferNavEnvCfg.__post_init__`, `_prune_scene_cameras`); `mdp/commands.py`
  (`SubgoalCommand._resample_command`); `mdp/rewards.py` (`procroom_obstacle_proximity_penalty`).
- Pins: `test_sim/env/test_composition_contract.py` (`_canon`, the goldens, `_OPEN_TOP_RL_VARIANTS`);
  `tests/navigation/test_proc_room_vanilla_golden.py`; `write_stub_scene_corpus` in `test_sim/common/scenes.py`.
- Instruments: `scripts/measure_path_statistics.py` (`measure_procroom`) and `tools/path_statistics.py`, which need
  `room2_primitives` in the stub env and a crossing tag.
- No test freezes the enriched generator's output ("What training sees today"; PR-0 in "Implementation split").

## Out of scope

- **Training on the house** (decision 4), and its acceptance ("Doorway competence after training" in the analysis).
- **Any change to v3's IDs or goldens, to `GRID_SIZE` 80 or to the shared proximity reward; any curvature target.**
- **An aperture-aware off-path tolerance** for doors under 1.2 m: a shared-reward change, filed to v3 under ruling F-3.
- **The lane's outside goals and SLAM refinement drift:** the goals are grid floor under sky, not a second room.
- **Open-plan connections, corridors, L-shapes, houses of more than two rooms, and the lane's tools.**
