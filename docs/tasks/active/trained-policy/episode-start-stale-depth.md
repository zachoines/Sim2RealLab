# Render a live depth frame at each episode start, and price it

**Type:** investigation (training env: confirm when tick 0 is stale, price a render after reset,
measure v3's response, then decide)
**Owner:** DGX
**Priority:** P2. Nothing is broken and v3 is not changed. In v3's headless training only each
environment's first episode in a run started on a pre-reset depth frame, at most about 0.3 % of
its episodes. Two things make it P2 rather than P3:
- the next depth-policy training is being scoped now, and Isaac Lab's protection is one cfg field
  whose cost in a headless launch should be close to zero;
- a camera-reading reward or termination with a nonzero weight, or a run with `--video`, would make
  every episode in that training start stale; the depth-proximity penalty ships at weight 0 with a
  note to re-enable it in a hardened env (`strafer_env_cfg.py`:1988).
**Estimate:** M (one raw-camera eval, two 96-environment training benches, eight eval launches,
then a decision)
**Branch:** `task/episode-start-stale-depth`

## Story

As **the next depth-policy training**, I want **when an episode's first depth frame predates its
reset, what rendering it live costs, and v3's response to a live first frame measured**, so that
**whether training starts each environment's first episode — and, under some launches, every
episode — on a pre-reset frame is decided on numbers, with the other changes that training
carries.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md): the training
  variant and its snapshot contract.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md): the node
  resets the policy at each mission boundary.
- [`vfov-gap-v3-2026-10-03`](../../../measurements/vfov-gap-v3-2026-10-03/README.md) §4 and its
  2026-10-11 correction: the finding, and the deposit this brief measures against.
- [`deploy-resolution-depth-2026-09-19`](../../../measurements/deploy-resolution-depth-2026-09-19/README.md)
  §5 and [`depth-noise-coverage-2026-09-18`](../../../measurements/depth-noise-coverage-2026-09-18/README.md)
  §8: how render work is priced at 96 environments.
- [`depth-noise-real-structure`](depth-noise-real-structure.md) items 7–8 and
  [`policy-time-to-goal-objective`](policy-time-to-goal-objective.md): the other changes the next
  training weighs.
- [`inference-last-action-mission-reset`](inference-last-action-mission-reset.md): the other
  first-tick difference between training and the node, whose priority is read alongside this one.

## Context

**The symptom.** From the first-episode observation dumps of 18 launches of v3's closed-loop eval
([`vfov-gap-v3-2026-10-03`](../../../measurements/vfov-gap-v3-2026-10-03/README.md) §4):
- In every environment's first episode, tick 0 does not show the start pose. It shows a
  floor-only frame rendered before the room existed: the far clamp above the horizon and a bare
  floor below. Within a launch it is the same render in every environment, up to each one's noise
  draw. Ticks 0 and 1 are floor-only in all 288 first episodes.
- Room geometry first reaches the policy at tick 2 in 80 first episodes, at 3 in 82, at 4 in 110,
  at 5 in 14 and at 6 in 2: ticks 2–6, median 3.
- Later episodes are different. The record read their tick 0 as the previous episode's last view,
  from the cfg. The deposit's depth sums say otherwise
  (`vfov-gap-v3-2026-10-03/stale-start-check/later_episode_tick0.txt` in the evidence repository):
  - in later episodes the largest single-tick change of the depth-block sum over ticks 1–8 is
    2.4 % in the median (90th percentile 5.3 %; 12 of 1 515 above 20 %);
  - at a first episode's room arrival the same statistic is 53.2 %, and two random episode starts
    differ by 46.2 %;
  - where the previous episode is recorded to its end (12 cases), tick 0 differs from that
    episode's last tick by 32.5 % and from its own tick 8 by 2.9 %.

  So in these headless launches a later episode's tick 0 already shows the new episode. That is
  read from sums, not from frames; the first acceptance box checks it on the camera's raw output.

**Why, read from Isaac Lab v3.0.0-beta2.patch1.**
- `num_rerenders_on_reset` defaults to 0 (`isaaclab/envs/manager_based_env_cfg.py`:131; its
  docstring, :132-139: "When this is 0, no render step will be performed after reset. Data
  collected from sensors after performing reset will be stale…"). No strafer cfg, script or
  Makefile target sets it (`git grep num_rerenders_on_reset -- source tools Makefile` is empty at
  `9f7d207`).
- **What renders a camera.** Sensors update lazily: a camera's data is computed when it is read
  (`isaaclab/sensors/sensor_base.py`:212-213; `lazy_sensor_update` defaults to True). A read pumps
  Kit's `app.update()` once per key of physics step count and render generation
  (`isaaclab_physx/renderers/isaac_rtx_renderer_utils.py`:124-127, 141-149). With no Kit visualizer
  attached and no `--video`, `sim.render()` only advances the render generation
  (`isaaclab/sim/simulation_context.py`:772-776).
- **The first reset.** The observation manager reads every term once while it is built, which
  renders the camera before any ProcRoom room has been placed (the room is generated by a reset
  event, `strafer_env_cfg.py`:1769-1771). The initial reset steps no physics and, at 0 re-renders,
  advances no render generation (`isaaclab/envs/manager_based_env.py`:422-437), so the first
  observation reuses that floor-only render.
- **A reset inside a step.** `ManagerBasedRLEnv.step` steps physics, computes terminations and
  rewards, resets terminated environments, then computes the observation
  (`isaaclab/envs/manager_based_rl_env.py`:226-227, 236, 248-253, 270). Nothing in the strafer
  training env reads the camera before the reset: the depth-proximity penalty has weight 0.0
  (`strafer_env_cfg.py`:1987-1990), and the reward manager skips zero weights
  (`isaaclab/managers/reward_manager.py`:146). So the step's first
  camera read is the observation's, after the reset, on a new key.
- **When a reset inside a step would be stale.** If anything pumps Kit or reads the camera before
  the reset, the observation's read reuses that render: a Kit visualizer, `--video`, or a
  camera-reading reward or termination with a nonzero weight. `num_rerenders_on_reset` ≥ 1 then
  advances the render generation after the reset (`manager_based_rl_env.py`:256-258, gated on
  `render_enabled`, `is_rendering` and `has_rtx_sensors`), so the observation's read renders anew.
- **The delay buffer** primes its ring with an environment's first frame after its reset ("Until
  an env's ring has been written it holds that env's first frame since its own reset",
  `mdp/noise_models.py`:48-49; priming at :147-156, reset at :117-128, called from
  `DepthNoiseModel.reset`, :609-622). On the robust tier the depth latency is drawn per environment
  from 1–3 steps (`sim_real_cfg.py`:614-615). A stale tick-0 frame therefore persists for the first
  1–3 ticks.

**Training's exposure.** v3 trained on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` at 96
environments, headless with no `--video`
([`depth-subgoal-v3-retrain-2026-09-21`](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md)).
Each environment's first episode started on the floor-only frame: 96 episodes against at least
about 28 000 (1 000 iterations × 96 environments × 48 steps is 4.6 M environment steps, about
28 000 episodes at the final 163.9-step mean, and earlier iterations ran shorter ones), so at most
about 0.3 %. The eval
in the record runs the same way: each launch's first 16 episodes start on the floor-only frame,
and those are the episodes its tick-0 statistics use.

**Deploy does not.** The inference node keeps the newest frame the camera delivered
(`inference_node.py`:884-887). Each tick reads it (:1078-1087, 1460-1465). A mission boundary resets
only the policy's recurrent state (:968-972). A frame older than `depth_timeout_s`, 0.5 s (:212),
trips the watchdog instead. So a mission's first tick sees where the robot stands.

**What a re-render costs, read from the code (not measured).**
- In a headless launch without `--video`, the extra `sim.render()` calls only advance the render
  generation. The observation's camera read pumps Kit once, as on every step. So
  `num_rerenders_on_reset = 1` should add one update per run, at the first reset, and nothing per
  step that carries a reset. A count above 1 adds nothing over 1.
- With a Kit visualizer, `--video`, or a camera-reading reward with a nonzero weight, each call
  pumps Kit. Then `num_rerenders_on_reset = 1` adds one update of every camera per step that carries
  a reset. With 96 environments, a mean episode length of *L* steps and independent resets, that is
  about 1 − e^(−96/*L*) of steps (arithmetic):

  | mean episode length *L* | where it comes from | share of steps with a reset |
  |---|---|---|
  | 61.7–194.8 | iterations 2–8 of #222's 20-iteration smoke (rsl_rl's trailing mean, so early iterations read short) | 0.79–0.39 |
  | 163.9 | v3's mean over its final 100 updates | 0.44 |

**The composition contract does not hash this field.** The snapshot covers `seed`, `decimation`,
`episode_length_s`, the managers, `sim.dt`, `sim.render_interval`, and the scene's `num_envs` and
`env_spacing` (`test_sim/env/test_composition_contract.py`:46-51, 84-90). Setting
`num_rerenders_on_reset` moves no golden. A training cfg that sets it needs its own assertion, the
way #222 pinned the camera resolution (`tests/contracts/test_depth_deploy_reduction.py`:199-217).

**What the record's deposit holds** (`vfov-gap-v3-2026-10-03/record-files/` in the evidence
repository, commit `b4c70ae`):
- `eval/A_s42` … `A_s49`, `eval/B_s42` … `B_s49` and the replicates `A_s42_r2`, `B_s42_r2`, each with a
  results JSONL and a `*_first_episode_obs.npz`. The npz holds every environment's first episode,
  ticks 0–7: `obs`, 16 environments × 8 ticks × 3619 float32 (19 scalars, then the 45×80 depth block
  scaled by 1/6 m), and `actions`, 16 × 8 × 3. `A` launches use v3's shipped camera (fx 335.652
  px), `B` launches the real unit's (321.522 px).
- Every JSONL episode carries its start state (`env_origin`, `root_pos_env`, `root_quat_w`,
  `subgoal_env`) and its first 30 ticks (`t`, the action, `bearing_rad`, `offset_rad`,
  `offset_valid`, `depth_sum`).
- Only the stale half of the first-episode comparison exists. Every launch ran with
  `num_rerenders_on_reset` = 0, so the dumps hold the floor-only tick-0 frame at 128 start poses per
  camera, and no live frame at those poses.
- The start poses reproduce. They are drawn in the initial reset, from seeded streams, before any
  policy step; they were identical across the two cameras (record §2), and replicate launches have
  byte-identical tick-0 observations (`stale_start.txt` §4).
- The run-to-run floor at identical observations: two launches of the shipped camera differ at tick
  0 by 0.005° in steering and 0.00004 in speed per pose (record §3).
- `onnx_parity.py` replays the dumps through the exported artifact on CPU (worst direction
  difference 0.3149°). The eval's dump flags (`vfov_eval_flag.patch`, which applies to `9f7d207`
  with a 2-line offset) and the launch (`vfov_run.sh`) are deposited too.

## Acceptance criteria

- [ ] **Resets inside a step.** Under the unchanged cfg and the training launch (headless, no
      `--video`), tick 0 after a reset inside a step shows the new episode. Read the camera's raw
      output (`env.scene["d555_camera"].data.output["distance_to_image_plane"]`, before noise and
      delay), dumped by a scratch eval patch for the step before and the step of each reset. The
      reset step's frame differs from the step before by more than the 90th-percentile
      step-to-step change, and from the next step's frame by less. If it does not, name the read
      or pump that makes it stale; the in-step case then enters the price and response boxes.
- [ ] **The first reset, with one re-render.** With `num_rerenders_on_reset = 1` set by an
      eval-only override, `tick0_frames.py`'s "tick-0 frames matching env 0's" falls below 16/16 in
      every launch, and a pose's tick-0 frame is floor-only only where its tick-1 raw frame is too
      (walls past the 6 m clamp). If one re-render leaves tick 0 on the pre-reset render, report
      it; do not raise the count in a headless launch, where extra calls collapse into the one
      update at the camera read.
- [ ] **Price.** Measure `num_rerenders_on_reset = 1` on
      `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` against the unchanged cfg, in the same
      session, set with a scratch patch that is deposited and not merged. Use the method #222 used:
      `train_strafer_navigation.py --num_envs 96 --seed 42 --headless`, 20 iterations, through
      `tools/kit_boot_watchdog.sh`, reporting the mean of iterations 2–8 for collection, learning
      and iteration time beside each iteration's mean episode length.
  - The collection ratio is the firm figure; the learning phase does the same work in both arms
    yet drifts by about 10 % (#221 §8).
  - The code predicts no difference beyond the run-to-run spread in this launch; the record says
    whether the collection ratio is inside it. If the next training is to run with `--video` or a
    camera-reading reward, price that configuration too.
  - The price holds at 96 environments only.
- [ ] **v3's response at identical poses.** One launch per seed 42–49 with the deposited launch
      (`vfov_run.sh` arm A: `--expect-fx-px 335.6522`, `--onset-ticks 30 --onset-obs-ticks 8`,
      v3's `model_999.pt`), on `9f7d207` with `vfov_eval_flag.patch`, plus the eval-only override
      the record names.
  - Void unless, at all 128 poses, the start state equals the deposit's arm-A launch at that seed,
    `obs[:, 0, :19]` is byte-identical to the deposit's, and the first-reset box passed for these
    launches.
  - Per pose, live − stale at tick 0, in steering (the signed direction offset, wrapped to ±180°)
    and in speed. Report the per-seed means of the signed difference and of its absolute value,
    their means, and the standard error across seeds, beside the floor of 0.005° and 0.00004.
  - Report the change in |offset| (live − stale), so whether the live frame turns the tick-0
    command toward the subgoal bearing or away from it. Later ticks diverge in closed loop (record
    §3), so tick 0 is the comparison.
- [ ] **Decision.** Decide for the next depth-policy training, together with the items in
      [`depth-noise-real-structure`](depth-noise-real-structure.md) and
      [`policy-time-to-goal-objective`](policy-time-to-goal-objective.md): set
      `num_rerenders_on_reset` to 1, or leave it at 0. Give the reason from the price, the response
      and the exposure (one episode per environment per run in a headless launch without a
      camera-reading term; every episode otherwise). If it is set, the change carries an assertion
      on the composed training variant, since the contract golden does not cover it. v3 is not
      changed.
- [ ] Record `episode-start-stale-depth-<date>` under `docs/measurements/`, deposited per
      [`docs/measurements/README.md`](../../../measurements/README.md).
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No tree change. v3, the training cfg and every default stay as they are, and the overrides
      live in the deposit.

## Investigation pointers

- Isaac Lab v3.0.0-beta2.patch1:
  - `isaaclab/envs/manager_based_env_cfg.py`:131-139, the field and its default.
  - `isaaclab/envs/manager_based_rl_env.py`:226-227 (the render in the step), 248-258 (the reset,
    then the re-render) and 270 (the observation).
  - `isaaclab/envs/manager_based_env.py`:422-437, the initial reset.
  - `isaaclab/sim/simulation_context.py`:745-800, `render()` and the render generation.
  - `isaaclab_physx/renderers/isaac_rtx_renderer_utils.py`:93-152, the once-per-key render update.
  - `isaaclab/sensors/sensor_base.py`:200-215, the lazy sensor update.
- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`:1174-1176 and 1196-1200 (the
  render cadence), 1769-1771 (the ProcRoom reset event), 1987-1990 (the depth-proximity weight).
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py`:44-175 (`DelayBuffer`)
  and 609-622 (`DepthNoiseModel.reset`).
- `source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`:614-615, the robust depth
  latency.
- `source/strafer_lab/scripts/train_strafer_navigation.py`:210-215. The cfg is parsed here, and no
  CLI flag reaches this field.
- `source/strafer_lab/scripts/eval_cadence_emulation.py`, the record's eval.
- `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`:884-887, 968-972,
  1078-1087 and 1460-1465, the deploy side.
- The deposit of `depth-noise-coverage-2026-09-18`, `bench/bench_direction_a.sh`: the timed
  96-environment training launch, reading `Iteration time:` and `Collection time:` from the runner
  log.

## Out of scope

- Setting the field on the shipped training cfg or any default. If adopted, it lands with the next
  training.
- v3, its artifact and its gate results.
- The delay buffer's priming and the depth latency band. Both are correct once the first frame is
  live.
- The bridge lane. It sets `render_enabled = False`
  (`source/strafer_lab/scripts/run_sim_in_the_loop.py`:909), which skips the re-render inside a step.
- Environment counts above 96.
