# Train time to goal as an objective, and derive the policy goal bound from it

**Type:** investigation (training objective: measure, compare variants, then decide)
**Owner:** DGX
**Priority:** P2 — the node's flat 60 s bound stays in place, and nothing is broken. It is P2
rather than P3 because the bound was chosen, not derived. v3's slowest rig reaches spent
7.0–13.9 s short of the arrival radius at the gate (18.7 s once with the livestream on), two gate
missions ended at the bound, and the next retrain should carry whatever objective this brief
adopts.
**Estimate:** L (two evaluations of v3, training runs per variant at a stated budget, then a
decision)
**Branch:** `task/policy-time-to-goal-objective`

## Story

As **the inference node and the executor bounding a trained-policy goal**, I want **the policy
trained so that its time to goal has a known upper bound for a given path length**, so that
**the per-goal bound comes from what the policy was trained to do, not from a flat 60 s.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md) — the bound itself is
  Jetson-lane (`strafer_shared`, `strafer_inference`, the executor).
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`executor-policy-nav-budget`](../../completed/executor-policy-nav-budget.md), "Open question:
  the 60 s bound itself" — the question this brief answers, with the reach figures.
- [`stop-at-goal-training-shaping`](../../completed/trained-policy/stop-at-goal-training-shaping.md)
  — why completion is dwell-gated.
- [`goal-a-rig-gate-v3-2026-09-25`](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)
  — the terminal approach on the rig.
- [`depth-subgoal-v3-retrain-2026-09-21`](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md)
  — v3's training statistics.

## Context

**The symptom.** On the trained-policy backends, a goal ends at the inference node's
`POLICY_MISSION_TIMEOUT_S`, 60 s on the node clock. That value has been the node's default since
the DEPTH MVP, and nothing derives it. On the rig:
- **v3, 2026-09-25 gate.** The longest reach was 19.0 s sim, 13.9 s of it between 0.42 m and the
  0.30 m radius. Three of seven reaches spent 7.0–13.9 s in that band. R2 held 0.347–0.420 m
  for 52.7 s and, like R3, ended `ABORTED` at the bound.
- **v3, 2026-10-09 video set** (`goal-a-cli-video-2026-10-09`, in #237). This set ran through the
  executor with the livestream on, at RTF about 0.11. G1 reached in 24.98 s sim, 18.7 s of it in
  the band. The record does not attribute the longer approach.
- **v2.** Its one rig reach took 53.6 s sim for 3.03 m.

**The trained horizon.** A training episode is one goal, capped at 20 s (600 steps at 30 Hz). The
node's 60 s allows up to 1800 steps of the recurrent policy, and G1's 24.98 s reach was already
about 750. [`goal-a-rig-gate-2026-08-17`](../../../measurements/goal-a-rig-gate-2026-08-17/README.md)
lists that regime as untested.

**What the DEPTH_SUBGOAL objective has today.** v3 trained on it unchanged
(`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`).
- **Reward terms** (`RewardsCfg_ProcRoom_Subgoal_Depth`, `strafer_env_cfg.py`:1946-1998, which
  inherits `RewardsCfg_ProcRoom_Subgoal`, :1908-1942):

  | term | weight |
  |---|---|
  | along-track progress, per metre of path | +10 |
  | cross-track error | −2 |
  | completion | +200 |
  | off-path | −50 |
  | collision | −10 |
  | sustained collision | −5 |
  | backward motion | −2 |
  | obstacle proximity | −1 |
  | action smoothness, energy, depth proximity | 0 |

- **There is no target speed.**
  - Progress pays per metre (`mdp/rewards.py`:880), so its total over a path does not depend on
    speed.
  - The subgoal sits a lookahead ahead of the path cursor (`path_planner/cursor.py`:195). It is
    not a point that moves at a set speed. The lookahead is drawn per path, from 0.7–1.3 m on the
    robust tier v3 trained on (`strafer_env_cfg.py`:1878-1879; `mdp/commands.py`:673-676).
  - The observation carries neither a target speed nor time. Its goal terms, read from
    `goal_command`, say where the subgoal is, not how fast to reach it, and the only velocity
    command it sees is its own previous action, `last_action` (`strafer_env_cfg.py`:787-799).
- **The approach slowdown is the completion gate.**
  - The +200 pays, and the episode ends, only after the robot holds inside 0.30 m at or under
    0.1 m/s for 10 consecutive steps (`strafer_env_cfg.py`:1895-1903; `mdp/commands.py`:495-497,
    710-719).
  - That is a sparse signal at the radius. No dense term tapers speed on the approach itself.
  - `speed_near_goal_penalty` (`mdp/rewards.py`:408) exists, but only in the goal-directed reward
    set, at weight 0.
  - The slowdown's knobs today are `dwell_steps` and `dwell_speed_max_m_s` at the
    `CommandsCfg_ProcRoom_Subgoal.goal_command` call site. `dwell_radius_m` stays at
    `GOAL_ARRIVAL_RADIUS_M` (0.30 m): it is also the deploy arrival radius, so every variant and
    the bound are measured at it.
- **Time enters through the discount and the dense penalties.**
  - γ is 0.99 per 30 Hz step (`agents/rsl_rl_ppo_cfg.py`:191), an effective horizon of about
    3.3 s. A completion 5.5 s away is worth 0.19 of its value on arrival; one 13.9 s away is worth
    0.015.
  - There is no constant per-step cost. The cross-track, obstacle-proximity and backward-motion
    penalties are charged on every step they are non-zero, so lingering where they apply costs
    reward. Cross-track is a distance, so it is rarely exactly zero.
  - The 20 s cap (`_DEFAULT_NAV_EPISODE_LENGTH_S`) is a time-out the learner bootstraps, so
    running out of time carries no penalty of its own.
- **v3's last 100 updates:** mean episode 163.9 steps (about 5.5 s), completion share 0.870,
  time-out share 0.0010.
- **The speed caps differ.** Training scales the action to 1.568 m/s. The deployed node caps it
  at `NAV_LINEAR_VEL`, 0.7841 m/s (`strafer_inference/inference_node.py`:156-157, applied at
  :1203-1208). A trained speed above 0.7841 m/s cannot be reached on deploy.

**Not measured yet:** v3's time to goal in its training env, and whether the 0.30–0.42 m dither
seen on the bridge lane happens there at all. Training already discounts a late arrival and
charges for lingering, so the rig's slow terminal approaches may come from a difference between the
training env and the bridge lane rather than from a missing objective. The two success rules are
not that difference: deploy ends a goal on the first tick within 0.30 m and training only after
the 10-step dwell, but the policy acts the same under both until that first tick, and every band
dwell on the rig came before it. The baseline criterion shows which it is.

*2026-10-11: measured in
[`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md);
see the Baseline and Deploy-cap boxes.*

**The scheme to evaluate.** Make time to goal an objective of its own, in addition to a target
speed, and tune it together with the approach slowdown, so that travel time has an upper bound.
- **The combined scheme:**
  - a time term;
  - a speed-tracking term toward a set speed along the path, tapered inside an approach band
    (`speed_near_goal_penalty` is one existing taper);
  - both swept jointly with the dwell knobs above.

  The design travel time is then d / v_target plus the approach.
- **Variations on the time term:**
  1. A constant per-step cost. It needs no observation change.
  2. A completion bonus that decays with elapsed time. This needs elapsed time in the critic's
     observation, or a time-out that is not bootstrapped. Neither holds now.
  3. A per-episode deadline set from the path length, with time remaining in the observation, so
     that the policy learns the bound directly.
- **Variations on the scheme:** a time term alone, or a speed target alone, as ablations of the
  combined scheme.

## Acceptance criteria

- [x] **Baseline.** v3, evaluated in its training env with a fixed seed over at least 500
      episodes, has these recorded, each against path length:
  - time to completion;
  - time from 0.42 m to completion;
  - the share of episodes that end on time-out within 0.42 m.

  Before the run, the brief states the attribution rule: the time from 0.42 m to completion at
  the 99th percentile, and the time-out share within 0.42 m, at or under which the rig's band
  dwell is attributed to a difference between the training env and the bridge lane rather than
  to the objective.

  *2026-10-11 ([`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md)):*
  - *Met. The rule was stated before the run in the record's pre-registration rather than in
    this brief: `PREREGISTRATION.md` in the record's deposit, committed to the evidence
    repository at `675c0c5` (19:13:20 CDT), 13 s before the first scored launch.*
  - *The rule: the band dwell goes to a difference between the training env and the bridge lane
    if the 99th percentile of the time from 0.42 m to completion is at most 7.0 s and the share
    of episodes ending on time-out within 0.42 m is at most 1.0 %; otherwise to the objective.
    7.0 s is the shortest of the rig's long band dwells.*
  - *v3 in its training env, seed 42, 96 envs, 576 episodes, deterministic actions: the 99th
    percentile is 6.505 s over the 524 completions that entered 0.42 m, and 2 of 576 (0.35 %) end
    on time-out within 0.42 m. Verdict: **a training-env/bridge-lane difference**. The three
    statistics per path-length bin are in the record's §2.*
  - *The margin is one episode: computed after the runs, one more band time above 7.21 s would
    have put the 99th percentile past 7.0 s (record §2). There is one launch per run.*
  - *Under the node's speed cap and 60 s goal bound (the record's run 3) the same statistic reads
    11.15 s, above 7.0 s. The record separates this neither from launch-to-launch spread nor the
    speed cap from the 60 s cap (§2). Computed after the runs, 1.98 % of run 3's band times exceed
    7.0 s (run 1: 0.95 %) and 3 of 506 exceed 13.9 s (§6).*
  - *Two reads were added, each with its own rule stated before the run.*
    - *Quiet-stop geometry, with the dwell's termination removed and without the node's caps: v3
      goes quiet a median 0.174 m from the path end, at or under the rule's 0.25 m. Verdict: **a
      bridge-lane difference**. That median is not like-for-like with the lane: 121 of its 146
      stops come after the dwell, and the node ends a goal at the first 0.30 m crossing.*
      - *The like-for-like statistic, computed after the runs, is the share of band entries that
        go quiet before their first 0.30 m crossing: 3.18 %, 2.23 % and 7.21 % in runs 1–3,
        against 8 of 11 rig reaches (record §3).*
    - *Post-arrival behaviour, in the same run: the rig-like limit cycle appears in 1 of 178
      episodes (0.56 %), at or under the rule's 2 %. Verdict: **left with the lane's feedback
      path**. Post-arrival vx is not quiet: its half-range at +2 s is 0.327 m/s at the 90th
      percentile, and each condition of the cycle is common on its own (§4).*
      - *The lane never enters that post-dwell regime. Computed after the runs, the same detector
        before the first crossing fires on 0 of 25, 7 and 71 windows in runs 1–3, and on 5 of the
        8 rig windows long enough to read (record §4).*
  - *Read together, with the post-hoc reads (record §6): **the lane raises the rate of a terminal
    behaviour the policy already has; the record does not identify which lane condition.** The
    stop-short and its slow resolve occur in the training env at the rig's distances (run 1:
    0.303–0.410 m, median 0.342 m), and run 1's time-outs within 0.42 m are R2-like holds, but in
    3.18 % of band entries against the rig's 8 of 11.*
- [x] **Deploy-cap evaluation.** v3, and any adopted variant, is evaluated again with the same
      seed and episode count, with its linear command capped at `NAV_LINEAR_VEL` as the node
      caps it, and with an episode cap long enough that arrivals past 20 s are seen. The bound
      is read from this evaluation.

  *Met for v3 2026-10-11 ([`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md) §5);
  an adopted variant's evaluation follows the Decision.*
  - *Seed 42, 576 episodes, the node's linear and angular caps on the executed twist, episode cap
    60 s. Completion 87.85 %, time-out 0.35 %. The linear cap binds on 64.57 % of steps.
    Completions past 20 s are seen; the longest is 30.87 s.*
  - *Bound read: T(L) = 11.87 + 1.53 L s for path length L in metres, covering 501 of 506
    completions (99 %, in sample). In policy steps that is 356 + 46 L. It passes the 600-step
    trained episode at L = 5.30 m. It covers completions only.*
  - *Without the speed cap, on 20 s episodes (the baseline run), the same read is
    8.45 + 1.03 L s, which passes 600 steps at L = 11.21 m (an extrapolation: run 1's longest
    path is 7.24 m).*
- [x] **Decision.** Unless the baseline attributes the band dwell to the env difference, the
      brief trains the combined scheme and at least one variation with a different time term.
      Before the first run it states the training budget that each run and its comparison
      share: either all warm-start from v3 for the same number of iterations, against a
      continuation on the unchanged objective, or all train from scratch at v3's settings. It
      compares each run with that comparison on four measures:
  - completion share;
  - time to goal;
  - collision share;
  - off-path share.

  It names the variant or combination adopted, or says why none is.

  *Decided 2026-10-12 by the rule of the capped open-dwell read
  ([`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md) §9),
  not by training runs:*
  - *The baseline attributes the band dwell to a difference between the training env and the
    bridge lane, as a rate the lane raises. So the brief trains no variant now, and none is
    adopted.*
  - *The capped open-dwell read asked whether the node's speed cap is that difference. In v3's
    training env under the node's caps, with the dwell's termination off, 7 of 175 band entries
    (4.00 %) go quiet before the first 0.30 m crossing, against 2.23 % without the caps. Its rule
    needed twice that, 4.47 %, so the cap reads as **a minor term**, by one episode. Run 3, under
    the same caps, read 7.21 %; at these counts the cap's effect is not settled either way.*
  - *By that rule the next measurement controls the scene and depth content at the lane's goals.
    The lane conditions no run has controlled are the body-velocity source (wheel kinematics in
    training, the root velocity on the bridge), the actuation tier (`real` on the bridge, `robust`
    in training) and the referent drift (off on the bridge).*
  - *The time and speed variants above wait for that measurement and stay with the next training's
    scoping.*
- [ ] **The bound.** The brief gives the time-to-goal bound from the adopted objective, or from
      v3 if none is adopted, read from the deploy-cap evaluation:
  - as a function of path length;
  - with the share of evaluation completions it covers;
  - in policy steps, against the 600-step trained episode. It says for which path lengths the
    bound exceeds the trained horizon, and what follows: a cap at the horizon, a longer training
    episode, or an evaluation of the recurrent policy beyond 600 steps.

  It then states the per-goal bound the node and the executor should use.

  *2026-10-11: v3's deploy-cap read is in
  [`policy-time-to-goal-baseline-2026-10-11`](../../../measurements/policy-time-to-goal-baseline-2026-10-11/README.md)
  §5; see the Deploy-cap box.*
- [ ] A change to the trained contract updates the composition-contract golden in the same
      commit and is listed in the brief.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-lab-pure` passes.

## Out of scope

- Changing `POLICY_MISSION_TIMEOUT_S`, the node or the executor. Once the bound is derived, that
  is Jetson-lane work in `strafer_shared`, `strafer_inference` and the executor's policy budget.
- The deploy success rule (an instantaneous 0.30 m crossing against training's dwell), and the
  arrival radius itself. That is a terminal-approach parity question, separate from the
  objective.
- The retrain that ships an adopted variant, and its gate.
