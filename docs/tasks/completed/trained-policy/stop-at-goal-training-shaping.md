# Stop-shaping the subgoal task: success is parking at the goal, not touching it

**Status:** Shipped (code + unit tests) 2026-07-04 in `c5aeef7` (DGX). The
subgoal task's `path_complete` is now **dwell-gated**: it fires only after the
robot holds inside `dwell_radius_m` (0.3 m) at/under `dwell_speed_max_m_s`
(0.1 m/s) for `dwell_steps` (10) consecutive control steps — parking, not
touching, collects the +200 completion bonus. The reward and termination read
the one shared flag, so both pay for parking. The per-step transition is a pure
`dwell_step` torch function with a Kit-free truth table; the counter is a per-env
`int32` on the command term, reset on env reset and on any radius/speed break.
`make test-lab-pure`: **696 passed, 1 skipped** (baseline 687/1; +9 dwell/config
tests, zero regressions).
**PR:** _(open on `task/stop-at-goal-training-shaping`)_
**MERGE GATE (operator, not yet met):** a full retrain of
`Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0` must converge **and** the play
episode (`…-Subgoal-Real-Play-v0`) must show the policy **parking** — enters the
radius, decelerates, holds; no overshoot-and-orbit. **Do not merge before the
play gate passes.** If the first retrain does not park cleanly, the fix is an
operator param **sweep** (`dwell_steps` / `dwell_speed_max_m_s` / `dwell_radius_m`
at the subgoal command call site), not a code change — the knobs are exposed for
exactly this.

**Type:** training shaping (env cfg + mdp)
**Owner:** DGX agent — every touched surface (`source/strafer_lab/…/mdp/`,
`strafer_env_cfg.py` subgoal command cfg) is DGX-lane.
**Priority:** P2 — restores the original meaning of goal-(a) Prereq 5 (the play
episode must show *parking*), and removes the deployment overshoot / target-
contact that instant-touch success trains in by construction. A runtime velocity
governor was rejected (sim-real gap); the fix is the policy.
**Estimate:** S — one shaping mechanism, one variable at a time (`action_smoothness`
stays 0.0; dwell alone forces deceleration).
**Branch:** task/stop-at-goal-training-shaping

## Why (reward-structure evidence)

The policy arrived at speed **by construction**. `RewardsCfg_ProcRoom_Subgoal`
pays `path_complete` **+200** and `TerminationsCfg_ProcRoom_Subgoal` ends the
episode the instant `path_complete` fires — and `path_complete` was
instant-touch (`state.end_distance < path_complete_threshold`). No dwell / taper
/ arrival-speed term existed (the subgoal reward set is along-track-progress +10,
cross-track −2, `path_complete` +200, off-path −50, collision terms,
`backward_motion` −2, `obstacle_proximity` −1, with `action_smoothness` and
`energy_penalty` at 0.0). So max-speed arrival was optimal, and deployment
overshoot / target contact on the rig follow directly.

## The change — success = PARK, not touch

**Mechanism: dwell-based success.** `path_complete` fires only once the robot has
held inside the completion radius at low speed for N consecutive control steps.

1. **Pure predicate** — `mdp/commands.py::dwell_step(counter, within_radius,
   below_speed, dwell_steps) -> (new_counter, parked)`. Envs holding both
   conditions increment; any break resets to zero; `parked` is True when the
   counter reaches `dwell_steps`. No env/module state → unit-testable truth
   table on the Kit-free path.
2. **Dwell state** — `SubgoalCommand.dwell_counter`, a per-env `int32` living
   with the other per-env path state on the command term. Reset on env reset
   (`_resample_command`), reset on radius/speed break (inside `dwell_step`), and
   **decimation-aware** — `_update_command` runs once per control step, so the
   count is in control steps, not sim substeps.
3. **Shared flag** — `_update_command` computes `within_radius = end_distance ≤
   dwell_radius_m` and `below_speed = ‖body_lin_vel_xy‖ ≤ dwell_speed_max_m_s`
   (body-frame planar speed, the deployment odometry the runtime governor was
   rejected in favor of), calls `dwell_step`, and writes the result to
   `path_complete`. The `path_complete` **termination** and the
   `path_complete_reward` both read that one flag, so the +200 pays for parking,
   not touching. No other reward/termination consumes an instant-touch predicate
   (verified — see below).
4. **Params on `SubgoalCommandCfg`** (documented; the operator sweep surface,
   also surfaced explicitly at the subgoal training call site next to
   `min_goal_distance`):
   - `dwell_radius_m: 0.3` — the former instant-touch completion threshold
     (`path_complete_threshold` folds into this; only the *dwell* requirement is
     new, not the radius).
   - `dwell_speed_max_m_s: 0.1` — parked-speed cap.
   - `dwell_steps: 10` — ≈ 0.33 s at 30 Hz control.

**Both realism tiers + PLAY get it for free.** Terminations/rewards/commands are
composed per scene-source (`_TERMINATIONS_BY_SOURCE_SUBGOAL` etc.), not per
realism tier, so Real and Robust share the edit, and the PLAY variants share it —
which is what makes the play-episode a true parking gate.

## Out-of-lane consumer found + protected: the coverage-capture driver

The brief's "verify no other consumer" check surfaced a **third** reader of
`path_complete` beyond the subgoal reward/termination:
`scripts/coverage_capture.py` breaks its per-leg step loop the instant
`subgoal_term.path_complete` fires (leg-advance), and `CaptureSubgoalCommand`
**inherits** `_update_command` from `SubgoalCommand`. Dwell-gating the flag
unconditionally would stall every coverage leg — the non-parking capture policy
drives through each leg end without stopping, so the dwell would never satisfy.

**Fix:** `CaptureSubgoalCommandCfg` overrides to instant-touch
(`dwell_steps=1`, `dwell_speed_max_m_s=float("inf")` → fire on the first
in-radius step, speed gate disabled). This reproduces the former
`end_distance ≤ dwell_radius_m` completion exactly. Guarded by
`test_capture_command_completes_on_touch_without_parking` and the full
`test_coverage_capture` suite (green). The dwell shaping is a training-only
change to the trained-policy task.

## Goal-noise synergy check (finding — no change made)

The subgoal command's truncated-Gaussian **waypoint perturbation**
(`waypoint_noise_std_m`, `perturb_waypoints`) is the subgoal-task analog of goal
noise. Findings against the current code:

- **Robust-tier value = `MAP_RESOLUTION / 2 = 0.025 m`** — the `SubgoalCommandCfg`
  default, *not* overridden per tier (`composed_env_cfg` only sets the per-tier
  lookahead band, not the waypoint noise). It is **nonzero**, so the brief's
  conditional ("if effectively zero, set it into the ±0.2–0.5 m band") is **not
  triggered** → left unchanged.
- **Structural finding (the load-bearing one):** `perturb_waypoints` **pins both
  endpoints exact** (`offsets[0]=offsets[-1]=0`; only interior waypoints move,
  and each is kept only if it stays in the disc-inflated free space). So it models
  planner **corridor-shape** disagreement — **not** goal-location / VLM-grounding
  error. The robot parks against the path's exact final point (`end_distance`),
  which the perturbation never moves. Bumping `waypoint_noise_std_m` to
  0.2–0.5 m would therefore **not** cover the grounding-error band no matter the
  value; it would only (a) get mostly rejected inside the 0.3 m-inflated corridor
  and (b) risk the along-track/lookahead corruption its own docstring warns
  about.
- **Conclusion:** the VLM-grounding-error robustness the brief gestures at
  belongs to a **goal-endpoint** perturbation (a different, out-of-scope
  mechanism), not this knob. `goal-noise-training.md` still owns endpoint noise
  for the DEPTH-direct baseline. Recommend a follow-up if true subgoal-endpoint
  noise is wanted; **not** in this one-variable-at-a-time stop-shaping PR.

## Verification

- `dwell_step` truth table: accumulates while inside+slow; fires exactly on the
  N-th consecutive step; resets on fast-inside, on outside, and re-accumulates
  from zero after a break (vectorized across envs).
- Integration through the command term: arriving at speed does **not** complete;
  parking completes on the `dwell_steps`-th hold; the counter resets on a single
  over-cap step and on leaving the radius; env reset clears the counter.
- Config-level: `SubgoalCommandCfg` defaults carry the three dwell params;
  `CaptureSubgoalCommandCfg` keeps instant-touch; the composed
  `CommandsCfg_ProcRoom_Subgoal.goal_command` carries dwell ON (10 / 0.1 / 0.3).
- `make test-lab-pure`: **696 passed, 1 skipped** (baseline 687/1). Kit-boot
  half (`run_tests.py`) not run here — no GPU/Kit on this host.
- Training smoke (episodes now END via dwell; length histogram shifts up
  slightly, no regression to non-termination): **not run** — operator, or here if
  GPU frees up. A few hundred iters on Robust.

## Operator hand-off (you run these; documented, not executed here)

- **Train:** `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0` (decision: train
  Robust). **Evaluate / play:** `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0`.
- **Export** via the export tooling as **`strafer_nocam_subgoal_v1.{onnx,pt,json}`**
  (v1 alongside v0 — rollback stays trivial); rsync + `.json` sidecar to the
  Jetson; update `STRAFER_INFERENCE_MODEL_PATH` to the v1 file.
- **Convergence / merge gate:** the play episode must show **parking** — enters
  the radius, decelerates, holds; no overshoot-and-orbit. This is goal-(a)
  Prereq 5's original intent, restored.
- **If it doesn't park:** sweep `dwell_steps` / `dwell_speed_max_m_s` /
  `dwell_radius_m` at the `CommandsCfg_ProcRoom_Subgoal.goal_command` call site —
  no code change needed.

## Docs housekeeping shipped in this PR

- Retired `parked/trained-policy/strafer-hybrid-sim-validation.md` (operator
  decision) → `completed/`, superseded by goal-(a).
- Updated goal-(a) Prereq 5 (`strafer-nocam-subgoal-singleroom-sim-validation.md`):
  with the stop-shaped v1 artifact, the play check now **requires parking**.

## Out of scope

- Re-tuning any other reward weight (`action_smoothness` stays 0.0 — dwell alone
  forces deceleration; one variable at a time).
- Goal-**endpoint** (VLM-grounding) noise — a separate mechanism; see the
  synergy-check finding above.
- The retrain, export, and rig validation themselves (operator; goal-(a)).
