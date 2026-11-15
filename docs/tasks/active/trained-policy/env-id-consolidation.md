# Consolidate the navigation gym IDs: changes in place, and the depth IDs on the enclosed room

**Type:** task (a convention, a Kit baseline, then two code changes behind Kit gates)
**Owner:** DGX; the questions under "Open for the maintainer" are the maintainer's
**Priority:** P2. Nothing here gates v4's go: v4's change set lands in place under the rule whatever the collapse's
timing.
**Estimate:** M–L (about 10–15 agent hours to write, and about 3 h of Kit gates)
**Branch:** `task/env-id-consolidation`

## Story

As **the maintainer reading the env registry**, I want **every gym ID to name only its axes, every change to a
variant to land in place, and the `Enriched` split folded back into the depth IDs**, so that **the ID list stops
growing with each retrain and each improvement to a room, and an older policy's environment is found from its
record's commit.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md): the axes, the goldens, and
  "Changing a variant", the rule this brief applies.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md): the observation layout the two
  policy-layout goldens pin.
- [`depth-subgoal-v4-scoping`](depth-subgoal-v4-scoping.md): v4's list of changes, landed in place (task 3).
- [`procroom-depth-enrichment`](procroom-depth-enrichment.md): its "Vanilla-variant retirement trigger".
- [`procroom-two-room-house`](procroom-two-room-house.md): filed on new `House` IDs.

## Context

- **The registry.** `tasks/navigation/__init__.py` registers 29 IDs, and nine carry `Enriched`:
  - `RLDepth` and `RLDepth-Subgoal`, each in `Real` and `Robust`, each with its `-Play` (eight);
  - `Capture-Bridge-ProcRoom-Enriched`.
  - Their classes are the eight `StraferNavCfg_RLDepth…Enriched…` RL classes and
    `StraferNavCfg_BridgeAutonomy_ProcRoomEnriched`, each composed with `SceneSourceCfg(enrich_depth=True)`.
- **Why the word exists.** The enclosed, Infinigen-like ProcRoom went onto new IDs on 2026-07-20 so that the open-top
  depth checkpoints kept their contract. The enrichment brief's retirement trigger planned to fold it back: once the
  enriched retrain passed its acceptance, the open-top depth IDs retire, and the enriched variants become the depth
  variants. The trigger's text names v2; v3 is the policy that passed the sim gate.
- **What the tree did since.** v3's training ID (`…RLDepth-Subgoal-Enriched-Robust-v0`) kept its name while its
  environment changed. On `main` its contract golden changed in 15 commits: its creation, then 14 re-freezes in place
  (#162, #163, #164, #167, #195, #199, #200, #201, #218, #219, #221, #222, #224, and one direct commit on 2026-07-20).
  v3 trained at `615fc14`, contract `faf86756…` (`depth-subgoal-v3-retrain-2026-09-21`). "Changing a variant" in
  `env-composition-contract` writes that practice down.
- **What prompted this brief.** #268 first proposed a V4 ID pair for v4's measured intrinsics. In review the
  maintainer asked what versions in IDs are for, why `Enriched` persists, and where a collapse is planned. The pair
  was withdrawn, and #268 lands the measured focal length as two shared constants only. Its body answers the three
  questions, the second and third as proposals in this brief.
- **The goldens and what they cannot see.** `test_sim/env/test_composition_contract.py` freezes one contract hash per
  composed RL class (`_COMPOSED_RL`) and the two policy-layout goldens (`_POLICY_OBS_LAYOUT_GOLDENS`, depth and
  nocam). The contract serializer takes only `num_envs` and `env_spacing` off the scene, so a camera's focal length
  moves no hash; `tests/navigation/test_policy_camera_intrinsics.py` pins it instead.

## Tasks, in order

1. **Baseline the two Kit modules on `main`.** Run `test_sim/env/test_composition_contract.py` and
   `test_sim/env/test_env_registration.py` on `main`, before anything lands on them, and read the JUnit XML, never the
   exit code. Expected: 256 cases, 0 failures (the 272 of #268's earlier snapshot, less its 16 on the withdrawn IDs).
   A different result is a stop, because tasks 2 and 3 are judged against it. One short Kit launch; it may run before
   anything else here.
2. **The collapse, if its form is decided (below).** In the recommended form:
   - the enrichment gate keys on the sensor profile alone at both sites (`_select_scene` and
     `_ComposedStraferNavEnvCfg.__post_init__` in `composed_env_cfg.py`), and `SceneSourceCfg.enrich_depth` goes;
   - the eight `Enriched` RL classes and `BridgeAutonomy_ProcRoomEnriched` go, with their nine registrations;
   - the eight plain depth classes and `BridgeAutonomy_ProcRoom` keep their names and bodies, and now compose the
     enclosed room. NoCam is untouched: its profile is camera-free, so it keeps the open-top room;
   - `EXPECTED_ENVS` loses nine entries. The contract table loses eight keys, and each plain depth key takes its
     deleted twin's literal. `_OPEN_TOP_RL_VARIANTS` is derived from the cfg, not from the substring `"Enriched"`;
   - the pure tests and scripts that name a deleted class follow, the live docs follow, and the old-to-new table goes
     beside the existing one in `docs/INTEGRATION_SIM_IN_THE_LOOP.md`. The hashed building blocks keep their names;
   - 29 IDs become 20.

   *What it breaks:* every command that names an `Enriched` ID fails with an unknown ID, the intended loud failure.
   The eight plain depth strings and the open-top bridge string build a different room with no error.
   - One of them, `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0`, is in 14 files of the evidence repository (and in
     30 logs that list every registered ID).
   - Three script defaults name plain depth IDs, where the new room is the one wanted:
     `train_strafer_navigation.py` and `test_strafer_env.py` (`--env`), and `export_policy.py`'s `DEPTH` and
     `DEPTH_SUBGOAL` entries.
   - The open-top depth room and bridge can then be built only from a worktree of the commit before.
3. **v4's change set, in place, one PR.** The content is
   [`depth-subgoal-v4-scoping`](depth-subgoal-v4-scoping.md)'s. Its commits come in this order, so each can be
   attributed alone:
   1. the bridge publishes `CameraInfo` from the perception camera's cfg, with no value changed;
   2. the envelope mechanism with its range field at `None`: every contract golden moves by the field's presence,
      shown by re-hashing with the field dropped;
   3. the range (0.255, 1.0) in the robust tier's action parameters, through `get_action_config_params`;
   4. both camera factories build from `D555_MEASURED_FOCAL_LENGTH_MM`, and the intrinsics test's expected value
      flips with them;
   5. `num_rerenders_on_reset`, as [`episode-start-stale-depth`](episode-start-stale-depth.md) decides.

   The PR's base commit is v3's comparator tree.
4. **v4's record: the comparator and the fields no golden sees.** The v3 arm of each same-session comparison runs in
   v3's environment (decided below). Before the session, a Kit-free diff of the two arms' cfgs names every differing
   leaf. For each arm the record states:
   - the commit and the tree, and the ID and class as labels;
   - the contract hash and the layout hash;
   - the policy camera's focal length, aperture and resolution;
   - `num_rerenders_on_reset`, the envelope range, and every override, read back from the built env.

   Nothing is added to the tree for the comparator.
   [`training-run-provenance-manifest`](training-run-provenance-manifest.md) is this record's in-tree home later, and
   not a precondition.
5. **Later, optional, each separable:**
   - drop the two depth pairs nobody runs (fixed-goal robust, subgoal real) and move the `DEPTH_SUBGOAL` export
     default;
   - replace the `Play` IDs by a per-profile `--num_envs` default;
   - replace the `GRU` IDs by an agent entry-point argument;
   - rename the hashed building blocks that still carry `Enriched`, inside the module split's one planned re-freeze.

## Decided (2026-11-15)

- **The envelope is set in the robust tier's action parameters.** It then reaches the robust NoCam-subgoal and
  fixed-goal depth IDs too; none has a consumer today.
- **At v4's change set, both camera factories build from the measured constant,** so the bridge renders it. This is
  the bridge option the v4 brief recommends; the go confirms it.
- **v4's v3 comparator is a worktree of the change set's base commit, at its defaults.** The fallback, where one tree
  is required, is the current tree with the three fields set back to v3's values, each read back from the built env.
- **A Kit-free reading of the contract hashes is a pre-check on tasks 2 and 3, never the gate.**
- **The collapse runs before v4's change set if its form is decided by then,** so that v4's record stores the lasting
  names; otherwise after. It never gates v4's go.
- **The `Play` and `GRU` IDs and the two unused depth pairs wait** until after v4 (task 5).

## Open for the maintainer

| Question | Recommended answer |
|---|---|
| Is the enrichment brief's retirement trigger met? Its text names v2; v3 is the policy that passed the sim gate | Yes, rule it met |
| The collapse's form: the plain depth names take the enclosed room and the nine `Enriched` strings go (20 IDs); or the open-top IDs retire and the word stays (21); or new strings for the survivors | The first, knowing that nine strings then build a different room in silence and one is in 14 evidence files. It makes the script defaults right and adds no grammar |
| The two-room house: a default-off generator parameter in place, or `House` IDs as filed? | In place. Its brief's owner rewrites the brief before any of its code is written |

## Acceptance criteria

- [ ] **Task 1:** both Kit modules on `main`, from the JUnit XML: 256 cases, 0 failures.
- [ ] **The open questions answered,** each recorded in this brief with its date.
- [ ] **Task 2, if its form is decided:** 20 IDs registered.
  - Kit-free: the pure suite, and a whole-cfg comparison across two worktrees. Each surviving class must equal its
    deleted twin at the base commit, but for the removed key and the worktree path.
  - Under Kit: both env modules from the JUnit XML. Any golden value that is not already in the file is a stop.
  - One boot of the bridge ID whose seed-42 grid dump reads sha256 `36929c68…`, the check
    [`procroom-two-room-house`](procroom-two-room-house.md) registers.
- [ ] **Task 3:** the pure suite, and under Kit both env modules from the JUnit XML, with a preimage diff by field name
      for every moved key.
  - Both policy-layout goldens are unmoved.
  - The focal length is read from the camera prim at 321.522 ± 0.01 px, on one training ID and on the bridge's
    `CameraInfo`.
- [ ] **Task 4:** v4's record states each arm's fields as listed there.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or
      guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-lab-pure` passes, and the Kit gates read from the JUnit XML.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py`: `SceneSourceCfg`, `_select_scene`,
  `_ComposedStraferNavEnvCfg.__post_init__`, and the `…Enriched…` classes.
- `source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`: the registrations.
- `source/strafer_lab/test_sim/env/test_composition_contract.py`: `_POLICY_OBS_LAYOUT_GOLDENS`, `_COMPOSED_RL`,
  `_OPEN_TOP_RL_VARIANTS`.
- `source/strafer_lab/test_sim/env/test_env_registration.py`: `EXPECTED_ENVS`.
- `docs/INTEGRATION_SIM_IN_THE_LOOP.md`: the existing old-to-new gym-ID table.
- Script defaults: `source/strafer_lab/scripts/train_strafer_navigation.py` and
  `source/strafer_lab/scripts/test_strafer_env.py` (`--env`), `source/strafer_lab/scripts/export_policy.py`
  (`DEPTH`, `DEPTH_SUBGOAL`).

## Out of scope

- The deploy side: nothing deployed reads a gym ID.
- The content of v4's change set, which is the v4 brief's; this brief fixes only its form and order.
- The two-room brief's rewrite, which belongs to its owner.
