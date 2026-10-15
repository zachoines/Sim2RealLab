# Scope the v4 depth-subgoal retrain: every change it carries, and its acceptance

**Type:** task (scoping, then one training run and the evaluations that accept it)
**Owner:** DGX
**Priority:** P1. Since v3 was trained, the records have decided one training-side change (the
measured intrinsics) and named others (the speed envelope, the stale first frame, the noise
structure). Each brief that found one defers it to "the next depth-policy training". This brief is
that training's one list, so that no change rides in unlisted and none is forgotten.
**Estimate:** L (a pilot, a 1000-iteration run of about 26.5 h at v3's rate, then the evaluations
and the sim-bridge gate)
**Branch:** `task/depth-subgoal-v4-scoping`

**The retrain starts only on the maintainer's go.** Filing this brief starts nothing.

## Story

As **the depth-subgoal deployment**, I want **one retrain that carries every training-side change
the records since v3 have decided, each with its source and its decision status, against an
acceptance written before the run**, so that **v4 is compared with v3 on numbers fixed in advance,
and every difference between them is one this brief lists.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [`depth-subgoal-v3-retrain`](depth-subgoal-v3-retrain.md) and its record
  [`depth-subgoal-v3-retrain-2026-09-21`](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md):
  the run this one repeats, and the same-pose tables v4 is re-read on.
- [`goal-a-rig-gate-v3-2026-09-25`](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md):
  the sim-bridge gate v4 re-runs.
- [`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md):
  the bound, the stop-short shares and the speed cap's effect (§5, §9).
- [`vfov-gap-v3-2026-10-03`](../../../measurements/vfov-gap-v3-2026-10-03/README.md): the
  intrinsics.
- [`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md):
  the noise structure.
- [`lane-term-speedcap-ab-2026-10-14`](../../../measurements/lane-term-speedcap-ab-2026-10-14/README.md)
  and [`lane-term-scene-axis-2026-10-14`](../../../measurements/lane-term-scene-axis-2026-10-14/README.md):
  the two lane-term measurements this brief folds in.
- The briefs whose decisions land here:
  [`depth-noise-real-structure`](depth-noise-real-structure.md),
  [`episode-start-stale-depth`](episode-start-stale-depth.md),
  [`policy-time-to-goal-objective`](policy-time-to-goal-objective.md).

## Context

**v4 starts from v3's distribution.** v3 trained at `615fc14` on
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`, composition contract `faf86756…`. From
`615fc14` to `6cd6fd7` the training sources changed only in the command terms' debug markers
(`debug-markers-out-of-cameras`), and the contract moved by exactly
`commands.goal_command.debug_vis` to `c98d18ba…`. v3's markers never left the world origin, so its
observations did not change. A from-scratch run on `main` with none of the changes below therefore
trains on v3's distribution, and every difference between v4 and v3 is a row of the table below.

### What v4 carries

| # | Change | Source | Status | In v4 |
|---|---|---|---|---|
| 1 | Policy camera at fx = fy = 321.522 px (VFOV 58.48°), from 335.652 px (56.41°) | [`vfov-gap-v3-2026-10-03`](../../../measurements/vfov-gap-v3-2026-10-03/README.md); plan decision 30; `depth-noise-real-structure` item 7 | decided | **enters**, on v4's IDs only; the bridge's camera is decided at the go (§1) |
| 2 | `num_rerenders_on_reset` (each environment's first episode starts on a pre-reset frame) | [`episode-start-stale-depth`](episode-start-stale-depth.md), its Price and Decision boxes | open: the boxes have not run | **what that brief's Decision box decides**, run before the launch |
| 3 | Speed envelope: per-episode randomisation of the command envelope, unobserved | [`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md) §9; rulings 55 and 57; A3 and A4 | carried on ruling 55's ground; A3 reads the cap as not the lane's main term (§3); the pilot picks the form | **enters**, in the form the pilot picks |
| 4 | Noise structure, `depth-noise-real-structure` items 1–6 and 8 | [`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md) | item 1 decided (no change); items 2–6 and 8 open, gated on a capture not yet taken | **no change**, unless that brief decides before the go (§4) |
| 5 | The objective's time and speed terms | [`policy-time-to-goal-objective`](policy-time-to-goal-objective.md) | a variant gated on the lane term; A4 is void, so the term is still unidentified | **not a v4 default** (§5) |

**Not in v4.** The referent snap (`subgoal-generator-goal-as-final-waypoint`, ruling 53) and the
planner's direction schema ([`planner-translate-direction-schema`](../reliability/planner-translate-direction-schema.md),
ruling 56). Both are deploy-side and ship in paired sessions. Neither touches the training env.

**v3's IDs keep v3's contract.** v4 registers its own ID pair (Run), and each v4 change is set in
v4's composition only. Two of the changes would otherwise reach v3's IDs silently:
- the focal length is a shared constant that every policy camera reads (§1);
- the robust action cfg is shared by every robust variant (`ActionsCfg_Robust`, through
  `get_action_config_params` in `sim_real_cfg.py`), so an envelope range set in the robust training
  contract would change v3's ID too (§3).

### 1. Intrinsics

- **The measurement.** The real unit's intrinsics are fx = fy = 321.522 px at 640×360. The policy
  camera renders 335.652 px (`D555_FOCAL_LENGTH_MM` 1.93 over `D555_HORIZONTAL_APERTURE_MM` 3.68,
  `strafer_shared/constants.py`:160-161).
  - On v3's closed-loop eval at the real intrinsics, the tick-0 command turns 0.146° further right
    (z 3.2 across 8 seeds). That is the mean of a slight pull toward the subgoal bearing, not a fixed
    bias. The tick-0 frame at those first-episode poses is the floor-only frame rendered before the
    reset.
  - Over the first second the difference stays inside the registered yardstick. Of the outcome
    metrics, only off-path divergence moves past its standard error (1.1 / 1.2).
  - By its registered rule the gap matters, and it is small (plan decision 30). v3 is not changed.
- **The change.** v4's composition sets the policy camera's focal length to 321.522 × 3.68 / 640 =
  1.8487515 mm, as the VFOV record's arm B did through `scene.d555_camera.spawn.focal_length`.
  - The constant itself does not change. Both camera factories in `d555_cfg.py` read it, and every
    policy scene builds from them, v3's IDs included.
  - The composition contract takes only `num_envs` and `env_spacing` off the scene, so no golden
    would notice.
  - The change therefore carries two assertions on composed variants: v4's at 1.8487515 mm, and v3's
    at 1.93 mm. #222's pin on the camera factory's resolution (`tests/contracts/test_depth_deploy_reduction.py`)
    is the precedent for a field the golden misses.
- **Isaac Lab forces the principal point to the image centre.** `camera.py` sets `c_x = width/2` and
  `c_y = height/2`, and the spawner ignores aperture offsets. The real (318.44, 180.32) stays
  unmatched, as in every training so far.
- **Decided at the go: the sim bridge's camera.** The bridge publishes from the perception camera
  (`d555_camera_perception`), built from the same constant, so it renders 335.652 px.
  - **Measured intrinsics on the bridge** (the perception camera's focal length overridden in the
    bridge's composition): the lane renders what the real camera delivers and what v4 trained on.
    The same-session v3 arm (Acceptance) then runs at the real intrinsics, as the VFOV record's arm
    B did in training. **Recommended.**
  - **Bridge unchanged:** v4 runs the G-sim gate 4.4 % off its training focal length (3.7 % in VFOV;
    arithmetic), and v3's 4 of 6 stays like-for-like on the bridge.
- **What reads the constant and goes stale with a change.** Tests: `test_d555_perception_cfg.py`
  (:124), `test_depth_subgoal_rewards.py` (:52-65), `test_depth_noise_resolution_sensitivity.py`
  (:80). Docstrings: `d555_cfg.py`:41-43 ("differ ONLY in resolution and prim path") and :128-131
  ("56.4 deg … matching the real D555"); `test_depth_subgoal_rewards.py`:47-49.

### 2. The stale first frame

- **Exposure.** In a headless launch with no Kit visualizer and no camera-reading reward or
  termination, only each environment's first episode starts on a pre-reset frame: 96 of about 20 200
  episodes in v3's run (0.47 %). With a Kit visualizer or a camera-reading term, every episode does
  ([`episode-start-stale-depth`](episode-start-stale-depth.md); plan decision 31 as corrected on
  2026-10-12).
- **This brief's choice: v4 enables no camera-reading term.** `depth_obstacle_proximity` stays at
  weight 0, and the run is headless. So v4's exposure is v3's.
- **The decision comes from that brief's Price box and Decision box.** Neither has run. They run
  before v4's launch, and v4 sets `num_rerenders_on_reset` as the Decision box decides. If v4 is
  changed to enable a camera-reading reward, the field is decided with that change, because every
  episode would then start stale.
- **No golden hashes the field.** If it is set, it carries an assertion on v4's composed variant.

### 3. Speed envelope

- **Why it is in v4.** The node caps |vx| + |vy| at 0.7841 m/s and |ω| at 2.0933 rad/s
  (`NAV_VEL_SCALE` 0.5). Training never sees that envelope; uncapped, v3 commands more than the cap
  on 60 % of its rows. The evidence on the cap's effect in v3's training env
  ([`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md) §9):
  - **By the registered rule, a minor term.** Capped, 4.00 % of band entries stop short before the
    0.30 m crossing; the rule needed 4.47 %.
  - **Computed after the rule (2026-10-13, labelled post hoc in the record):** pooled over four
    launches, 44 of 688 band entries capped against 21 of 714 uncapped, RR 2.17 (1.31–3.62). "The
    node's speed cap roughly doubles the rate of the stop-short the policy already has" (ruling 55).
  - Read per band entry, the lane's rate is 7 of 10, 23.8-fold the pooled uncapped rate (record §9;
    ruling 57). A doubling is a factor of about two of that gap.
- **A3's verdict: on the lane the cap is not the main term**
  ([`lane-term-speedcap-ab-2026-10-14`](../../../measurements/lane-term-speedcap-ab-2026-10-14/README.md)).
  - The video set's goals were run alternating the deployed cap with training's envelope, six reaches
    each.
  - By its pre-registered rule the cap stays "a doubling, not the main term". 4 of 6 uncapped reaches
    went quiet before the 0.30 m crossing, against 5 of 6 capped. The uncapped band was not shorter:
    median 2.65 s against 1.93 s.
  - So the envelope enters v4 on the training-env ground, ruling 55's doubling and the exposure. It is
    not expected to move the lane's stop-short rate, and v4's lane read (Acceptance) is not a test of
    it.
- **A4 is void** (§5) and changes nothing in this section.
- **The form, in order of preference** (plan §6, item 6):
  - **(b) Per-episode randomisation of the envelope, unobserved.**
    - *This brief's design, for the pilot to test:* at each reset each environment draws one scale s
      from U[0.255, 1.0] and applies it to both caps, |vx| + |vy| ≤ s × 1.5683 m/s (about 0.40 to
      1.57 m/s) and |ω| ≤ s × 4.1866 rad/s. The deploy cap is s = 0.5. No observation changes.
    - *The plan's rationale, untested:* the recurrent policy reads the regime from its last action
      against its body velocity, as it would read a weak motor. This would also cover chassis
      variability. The pilot's (c) clause tests it.
  - **(a) A fixed clamp at the deploy cap.** The minimal change. It removes the envelope difference
    between training and the node; whether it removes the doubling is not measured. **The
    fallback.**
  - **(c) An observed target-speed input** with a tracking reward. An operator knob, but a contract
    change across the node, the parity tools and the dumpers. **Only if the pilot shows the policy
    needs it.**
- **Where it goes.** `MecanumWheelAction.process_actions` (`mdp/actions.py`:302-331) scales the
  action, then clamps with `l1_clamp_twist_batched(…, vel_cap_linear_m_s=self._max_linear_vel,
  vel_cap_angular_rad_s=self._max_angular_vel)`.
  - The per-environment caps are new tensors drawn in `reset()` and passed to the clamp only.
    `_max_linear_vel` and `_max_angular_vel` also build `_velocity_scale`, which must stay at the
    full envelope: the node scales the action by the full envelope (`policy_interface.py`,
    `interpret_action`) and then caps (`inference_node.py`:1203-1208).
  - `last_action` is the raw action (`mdp/observations.py`), so it stays unclamped, as on the node.
  - The clamp broadcasts a per-environment tensor, but its annotations say `float` and its parity
    test is scalar. A per-environment cap needs its own test.
  - The range field lives on the action cfg, defaults to off, and is set only in v4's composition. A
    new field must actually reach the action term: `motor_time_constant_range` is declared on the
    actuator cfg (`sim_real_cfg.py`:125, set at :549 and :626) and is absent from
    `get_action_config_params` and `MecanumWheelActionCfg`, so it is inert. Do not repeat that.
- **Every evaluation pins s.** The time-to-goal eval runs on the training ID, not PLAY, so on v4's
  ID:
  - the uncapped design (run 1's) pins s = 1.0;
  - the capped design (run 3's, and the capped open-dwell run's) pins s = 0.5, or s = 1.0 with the
    deposited `install_deploy_cap`. A drawn s below 0.5 would clamp tighter than the node does.

  This applies to the pilot's evaluations and to v4's acceptance evaluations.
- **The pilot** (before the full run, on v3's camera with no other change, so it isolates the
  envelope). Two warm-start legs from v3's `model_999.pt` at v3's settings (96 environments, seed
  42), 200 iterations each, about 5.3 h each at v3's rate:
  - **E**, with (b);
  - **K**, continuing on the unchanged action term.

  Each is evaluated with the time-to-goal eval of `policy-time-to-goal-baseline-2026-10-11`, seed 42,
  576 episodes, twice: uncapped (run 1's design) and under the deploy cap (run 3's design). The
  reads are the pre-crossing stop-short share per band entry, the completion share, and the
  deploy-cap bound T(L) over L in [2, 6] m. **The rule, stated before the pilot:**
  - If E's capped share is at least twice E's uncapped share, the cap still doubles the stop-short
    after training under the envelope, and the pilot goes to the maintainer with (c) proposed. No
    form is adopted without that ruling.
  - Otherwise (b) is adopted if E's capped share is at most K's capped share and E's capped T(L) is
    at most K's at every L in [2, 6] m.
  - Otherwise (a).

  **Power** (binomial at the run-1 and run-3 design sizes, n ≈ 535 and 513; arithmetic):
  - The (c) clause fires about 51 % of the time under an exact doubling, about 68 % at v3's
    observed 7.21 % against 3.18 %, and about 2.5 % if the cap has no effect.
  - A share comparison between equal true rates passes about 52 % of the time.
  - So the pilot is a direction read that picks a form. v4's acceptance is the test.
- **The floor and the time-out** (this brief's arithmetic, not measured). At 0.4 m/s a 6 m path takes
  15 s of the 20 s episode before any approach, and run 1's longest path (7.24 m) 18.1 s. The floor
  may push episodes into the time-out, which the learner bootstraps. The pilot's completion share
  reads it.

### 4. Noise structure

[`depth-noise-real-structure`](depth-noise-real-structure.md) items 1–8:

| Item | Finding | Record | In v4 |
|---|---|---|---|
| 1 | Amplitude passes gate (B) at σ_d 0.08 in every band; the sensor is 1.11–1.88× noisier | [`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md) §5; plan decision 21 | no change, by decision |
| 2 | The robust band [0.002, 0.16] is log-uniform; the all-band passing window [0.075, 0.178] is about 17 % of its draws | real-D555 §5; plan decision 22 | measured-only. The band is never resized alone; it and the structure are one decision |
| 3 | The floor is 1.5–2.2× other surfaces, depth-normalised | real-D555 §6 | measured-only |
| 4 | Specular phantoms on 2–2.5 % of cells | real-D555 §6 | measured-only |
| 5 | Adjacent cells correlate, anisotropically: 0.37–0.69 horizontally, 0.33–0.40 vertically | real-D555 §2 | measured-only |
| 6 | Real depth sits on a disparity lattice; some Z16 codes never occur | real-D555 | measured-only |
| 7 | The intrinsics | [`vfov-gap-v3-2026-10-03`](../../../measurements/vfov-gap-v3-2026-10-03/README.md) | **enters** (§1) |
| 8 | The noise term's stereo constant is 673 px at 1280 wide. The sim prim gives 671.30 px; the real unit 643.044 px (2 × 321.522, derived from its 640-wide intrinsics). At a given σ_d training's σ_z is about 4.5 % low | [`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md) (671.30 px, 673 shipped); real-D555 (321.522 px); plan decision 33 | measured-only. The band decision is made on the 643 px basis |

- **Why items 2–6 and 8 do not enter as written.** Any change to a noise field is gated by
  `noise-texture-parity-2026-09-17` §8. Gate (C) needs §8's Gate 0, a matched sim capture through the
  640×360 perception camera, and that capture has not been taken (plan decision 29).
- **If `depth-noise-real-structure` decides before the go**, its per-item decisions enter here, with
  the gate readings that accept them. Otherwise v4 trains v3's noise, and v4's record lists items
  2–6 and 8 as known differences from the sensor.

### 5. The objective, and what the lane terms decide

- **The dwell geometry is a separate question from the envelope.** The stop-short exists uncapped:
  in v3's training env 2.94 % of band entries pooled, 3.18 % in run 1. The envelope is meant to act
  on the cap's doubling; whether it moves that base rate is not known.
- **`policy-time-to-goal-objective`'s scheme is a variant gated on the lane term, not a v4 default.**
  Its time and speed terms wait until the lane term is identified.
- **A4's verdict: void**
  ([`lane-term-scene-axis-2026-10-14`](../../../measurements/lane-term-scene-axis-2026-10-14/README.md)).
  - One boot produced four scored reaches of the twelve its rule needs. SLAM's `map→odom` drifted
    1.35 m during the fourth slot's transit, the next transit stalled on both attempts, and the series
    was stopped.
  - Descriptively, 2 of the 4 stopped short before the 0.30 m crossing.
  - The scene axis is unanswered, the lane term is still unidentified, and the objective's terms stay
    gated.
  - Across that evening's four boots, `map→odom` moved more than 0.4 m from its first sample in
    three. So the SLAM referent is a lane condition in its own right; whether it raises the stop-short
    is not measured.

## Run

| item | value |
|---|---|
| task | a new ID pair registered beside v3's (below), composed from `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` plus the changes above |
| start | from scratch |
| environments | 96 |
| seed | 42 |
| iterations | 1000 (`num_steps_per_env` 48, learning rate 3e-4 fixed, γ 0.99, as v3) |
| expected wall clock | about 26.5 h at v3's measured 95.25 s per iteration; the intrinsics and the envelope are not priced |
| launch | `tools/kit_boot_watchdog.sh` around `train_strafer_navigation.py --env <v4 ID> --num_envs 96 --seed 42 --headless --max_iterations 1000`, as v3's deposited `launch_cmd.sh` |
| comparison | v3 (`model_999.pt`; exported ONNX sha256 `c866bfd5…`), on v3's IDs, evaluated in the same sessions with the same evaluations |

- **New IDs, not an in-place change.** `tasks/navigation/__init__.py` registered the enriched
  variants as new IDs with new goldens "so the open-top depth checkpoints keep their contract". v4
  does the same, so v3's IDs keep v3's contract and the v3-against-v4 evaluations need no worktree
  of an older commit.
- **Golden effects.** v4's IDs get new goldens. If the envelope's range field is added to
  `MecanumWheelActionCfg`, every contract golden that hashes `actions` moves by its presence.
  Attribute every moved golden by diffing preimages by field name. The policy-layout goldens must
  not move: v4's observation contract is v3's, `obs_dim` 3619.

## Acceptance criteria

Pre-registered here, before the run. Each comparison names its v3 figure and the design that
produced it. Every evaluation of v4 pins the envelope as §3 says.

- [ ] **The table above is complete when the go is given.** Rows 2 and 4 state what their briefs
      decided, and the bridge's camera (§1) is decided.
- [ ] **The pilot** runs, and its rule picks (a) or (b), or sends (c) to the maintainer. It is
      recorded with its evaluations.
- [ ] **The run** completes 1000 iterations at 96 environments, seed 42, through
      `tools/kit_boot_watchdog.sh`, with every attempt line kept and no NaN in the losses.
      Per-iteration collection and learning times are recorded against v3's 95.25 s.
- [ ] **The export** has `obs_dim` 3619, `action_dim` 3 and `is_recurrent`, exported with v4's new
      PLAY ID passed by `--env` (the `DEPTH_SUBGOAL` default is a non-enriched ID), and a one-episode
      play reaches an action.
- [ ] **G-sim, on the same lane and goals as v3's gate:** ≥ 4 of 6 within 0.30 m. The goals are G1,
      R1, L1, R2, L2 and R3 from start (0.075, −0.035, 130°), through `gate_mission.py`, 60 s
      bound, as in `goal-a-rig-gate-v3-2026-09-25`.
  - The lane has changed since v3's gate: #235–#238 merged after it, and the bridge's camera
    changes if §1 decides so. **A same-session v3 arm on the same six goals** is run and reported
    descriptively beside v4.
  - Power: if v4 reaches each goal with v3's 4-in-6 rate, it passes about 68 % of the time. A miss
    is a reading to examine, not by itself a regression.
- [ ] **The deploy-cap bound is no worse than v3's.** v3 reads T(L) = 11.87 + 1.53 L s, covering 501
      of 506 completions (the record's run 3: seed 42, 576 episodes, the node's caps on the
      executed twist, 60 s episode cap). v4 is read with the same evaluation and Reading 4. "No
      worse" means T_v4(L) ≤ T_v3(L) at every L in [2, 6] m, on the point fits.
  - The bound covers completions only, so the completion and collision shares are reported beside
    it, with v3's.
  - No power is stated for this clause; it has no distributional model.
- [ ] **The training-env pre-crossing stop-short share is no higher than v3's**, per band entry,
      each read in its comparator's design:
  - uncapped: ≤ 3.18 % (run 1: 17 of 535; dwell termination on, 20 s);
  - under the deploy cap: ≤ 4.00 % (the capped open-dwell run: 7 of 175; dwell termination off,
    40 s, rows 0–599).

  Power: if v4's true rates equal v3's observed ones, the clauses pass about 56 % and 60 % of the
  time. These are direction reads (ruling 55's lesson), and the record says so beside the outcome.
  The like-for-like capped run with the dwell on is run 3, at 7.21 % (37 of 513); it is reported
  beside them.
- [ ] **The lane's like-for-like stop-short rate is measured and reported** for v4 and the
      same-session v3 arm: per band entry, quiet before the first 0.30 m crossing, on the G-sim
      reaches. v3's lane figure is 7 of 10 (`policy-time-to-goal-baseline-2026-10-11` §9).
- [ ] **v3's depth-content behaviour is re-read for v4 on the records that established it:**
  - the same-pose tables (a1)–(a3) of `depth-subgoal-v3-retrain-2026-09-21` §4, with the deposited
    `table_a_samepose.py` and `table_a_node_base.py` and `--model v3=… --model v4=…`. v3 gives no
    rig-class frame in any row, and "answers the depth field through yaw rate and speed". The node
    rows use the bridge capture's frame at 335.652 px, which v4 did not train on; the record says so;
  - the depth-against-other-fields attribution of `goal-a-attribution-2026-08-22`, which established
    depth content for v2 (depth alone moved the command 95.8°, every other field at most 0.570°).
    No record has run it on v3, so it runs on v3 and v4 in the same session, descriptively, with no
    threshold.
- [ ] Record `depth-subgoal-v4-retrain-<date>` under `docs/measurements/`, deposited per
      [`docs/measurements/README.md`](../../../measurements/README.md).
- [ ] A change to the trained contract updates the composition-contract goldens in the same
      commit, and the brief lists each.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-lab-pure` passes.

## Investigation pointers

- Intrinsics: `source/strafer_shared/strafer_shared/constants.py`:160-161; the two camera factories
  in `source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py`; the bridge's camera publisher
  in `source/strafer_lab/scripts/run_sim_in_the_loop.py` (`d555_camera_perception`); Isaac Lab's
  `isaaclab/sensors/camera/camera.py` (principal point at the centre).
- Stale start: `num_rerenders_on_reset` defaults to 0 in `isaaclab/envs/manager_based_env_cfg.py`,
  and no strafer cfg sets it. The camera-reading reward is `depth_obstacle_proximity` in
  `strafer_env_cfg.py`.
- Envelope: `mdp/actions.py`:127-133 (the caps), :166-170 (`_velocity_scale`), :256 (`reset`),
  :302-331 (`process_actions` and the clamp); `strafer_shared/mecanum_kinematics.py`:141
  (`l1_clamp_twist_batched`); `strafer_shared/constants.py`:76-78 and 211-215; the node's order of
  scale then cap, `inference_node.py`:1203-1208; the deposited prior art for a capped executed
  twist, `install_deploy_cap` in `time_to_goal_eval.py`
  (`policy-time-to-goal-baseline-2026-10-11/record-files/` in the evidence repository).
- Noise: `sim_real_cfg.py`:246, 572, 648 (`focal_length_px` 673.0); the gates in
  `noise-texture-parity-2026-09-17` §8.
- v3's run: `depth-subgoal-v3-retrain-2026-09-21/record-files/train/launch_cmd.sh` in the evidence
  repository.

## Out of scope

- Starting the retrain. It starts on the maintainer's go.
- The referent snap and the planner's direction schema (deploy-side, paired sessions).
- Changing `NAV_VEL_SCALE`, the node's caps or the executor's policy budget. Those are Jetson-lane
  decisions in `strafer_shared`, `strafer_inference` and the executor.
- A noise-field change that has not passed `noise-texture-parity-2026-09-17` §8.
- The objective's time and speed terms, unless the lane term gates them in.
- G-real.
