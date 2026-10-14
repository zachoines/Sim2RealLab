# v3's time to goal in its training env, 2026-10-11

On the rig, some of v3's terminal approaches are slow. At the 2026-09-25 gate, three reaches of
eight band entries spent 7.0–13.9 s sim between 0.42 m and the 0.30 m arrival line, and R2 held
0.347–0.420 m for 52.7 s sim until the bound
([`goal-a-rig-gate-v3-2026-09-25`](../goal-a-rig-gate-v3-2026-09-25/README.md)). The 2026-10-09
video set ran with the livestream on, at RTF about 0.1; there the 0.42 m → result approach took
18.72 and 11.18 s sim ([`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/README.md)).
[`policy-time-to-goal-objective`](../../tasks/active/trained-policy/policy-time-to-goal-objective.md)
asks whether v3 does the same in the env it trained in, or whether the slow approaches come from a
difference between that env and the bridge lane. This record runs v3 in its training env in three
launches and scores them against readings written down and committed before the first scored
launch.

**The lane raises the rate of a terminal behaviour the policy already has; this record does not
identify the main lane condition.** By the registered rules all three readings attribute the rig's
terminal behaviour to a difference between the training env and the bridge lane. Computed after the
runs (§6), the same stop-short and slow approach occur in the training env, at the rig's distances
but at a far lower rate. Computed after the addendum's rule (§9), the node's speed cap roughly
doubles that rate, about a factor of two of the gap to the lane.
- **Band dwell (Reading 1, the brief's rule).** Over the 524 completions that entered 0.42 m, the
  99th percentile of the time from 0.42 m to completion is 6.505 s. 2 of 576 episodes (0.35 %) end
  on time-out within 0.42 m. Both are inside the rule's 7.0 s and 1.0 %, so the rig's band dwell
  is attributed to **a difference between the training env and the bridge lane**. The margin is
  one episode: computed after the runs, one more band time above 7.21 s would have put the
  percentile past 7.0 s (§2).
- **Quiet-stop geometry (Reading 2).** With the dwell's termination removed, 146 of the 179 band
  entries go quiet by row 599, a median 0.174 m from the path end. That is at or under the rule's
  0.25 m, so the stop-short is attributed to **a bridge-lane difference**.
  - **Not like-for-like with the lane.** 121 of the 146 stops behind that median come after the
    dwell, inside 0.30 m. The deployed node ends a goal at the first 0.30 m crossing, so the lane
    can only record a quiet stop made before it.
  - **The like-for-like statistic, computed after the runs.** The share of band entries that go
    quiet before their first 0.30 m crossing is 3.18 %, 2.23 % and 7.21 % in runs 1, 2 and 3.
    On the rig, 7 of the 10 reaches that entered the band did; 8 of 11 went quiet in the band,
    0.295–0.411 m from their subgoal, median 0.337 m (§3).
    The direction is the same as the registered read.
- **Post-arrival behaviour (Reading 3).** In the same run, 1 of 178 episodes shows the rig-like
  limit cycle after the dwell fires (0.56 %). That is at or under the rule's 2 %, so the
  oscillation is left with **the lane's feedback path**.
  - **The registered window never occurs on the lane.** It starts at the dwell's firing, a regime
    the deployed node never lets the policy enter. The rig's oscillation came before the crossing.
  - **The same detector on the pre-crossing windows, computed after the runs.** From band entry
    to the first 0.30 m crossing, it fires on 0 of 25, 0 of 7 and 0 of 71 windows in runs 1, 2 and
    3. On the rig's own records it fires on 5 of the 8 windows long enough to read (gate G1, L1,
    L2 and video G1, L2; not gate R1, R2 or F1), each before the crossing, so it can detect the
    rig's cycle (§4).
  - So the oscillation is a lane finding by the like-for-like read as well. The 1 of 178 is the
    registered post-dwell figure only.
- **The bound (Reading 4, not a verdict).** Under the inference node's speed cap and 60 s goal
  bound, 501 of 506 completions (99 %) finish within T(L) = 11.87 + 1.53 L s, L the path length in
  metres. The coverage is in sample: the intercept is set so that 501 sit under the line. In policy
  steps the bound is 356 + 46 L. It passes the trained 600-step episode at L = 5.30 m. It covers
  completions only; the 68 collisions and 2 time-outs of the 576 episodes are outside it (§5).

**What the verdicts would otherwise hide.**
- **The stop-short and its slow resolve exist in the training env, at the rig's distances.**
  Computed after the runs (§6, `posthoc_terminal_modes.txt`):
  - In run 1, 17 of 535 band entries (3.18 %) go quiet before their first 0.30 m crossing. They do
    so 0.303–0.410 m from the path end, median 0.342 m, a median 0.77 s after band entry. On the
    rig, the 7 band entries that went quiet before the crossing did so 0.295–0.411 m from the
    subgoal, median 0.343 m (§9).
  - 14 of them resolve into the dwell, a median 2.27 s later (90th percentile 6.92 s, maximum
    11.77 s). The other 3 never resolve: one collision and two time-outs.
  - All five of run 1's band times above 7 s are among these episodes.
  - In run 3, under the node's caps, 37 of 513 (7.21 %) do so, median 0.325 m, and two never
    resolve.
  - Counting instead every quiet stop at 0.30 m or more up to the dwell gives 18 and 38. The extra
    episode in each run dipped inside 0.30 m first, then went quiet outside it.
- **The time-outs within 0.42 m are R2-like holds.**
  - Run 1, episodes 506 and 556: quiet 0.394 and 0.342 m out, in the band 14.5 and 14.9 s from
    entry to the 20 s cap, and 0.300 and 0.324 m out at the cap.
  - Run 3, episodes 572 and 575: quiet 0.350 and 0.338 m out, in the band 55.9 and 55.6 s to the
    60 s bound, and 0.320 and 0.339 m out at the bound.
  - On the rig, R2 held 0.347–0.420 m for 52.7 s to the bound.
- **What differs is the rate:** 3.18 % of band entries in run 1 and 7.21 % in run 3, against 7 of
  10 band entries on the lane (§3). The speed cap accounts for about a factor of two of that gap
  (§9).
- **Under the node's caps, Reading 1's statistic reads past its threshold.** Run 3 applies the
  node's speed cap and its 60 s goal bound, which the rig runs under and training does not. There
  the same statistic reads 11.15 s, above 7.0 s. With one launch per run this is not separated from
  launch-to-launch spread (§7), nor the speed cap from the 60 s cap (§2). The caps are themselves
  training-env/bridge-lane differences of the kind the verdict names.
  - Long band times stay rare there. Computed after the runs, 1.98 % of run 3's band times exceed
    7.0 s (run 1: 0.95 %), and 3 of 506 exceed 13.9 s (run 1: 0 of 524).
  - Those band times run to the dwell's firing; the rig's end at the 0.30 m crossing. On that
    like-for-like band, run 3's 99th percentile is 8.78 s against 11.15 s (§6).
  - At 1.98 %, at least four of eight band entries at 7.0 s or more, as at the gate, has a binomial
    probability of about 1e-5. That is a scale, not a test: the gate's reaches are not independent
    draws (§6).
- **Under the node's caps the stop-short share stays under its rule's margin; pooled, the cap
  roughly doubles it.** The capped open-dwell run (§9, an addendum registered before its launch)
  removes the dwell's termination under the caps. There 7 of 175 band entries (4.00 %) go quiet
  before the first 0.30 m crossing, against run 2's 4 of 179 (2.23 %).
  - Its rule needed twice run 2's share, 4.47 %. It names the cap **a minor term**, by one episode:
    8 of 175 would have named it.
  - Run 3, under the same caps, read 7.21 % over 513 band entries. Run 3's dwell termination cannot
    censor a stop made before the crossing.
  - On the capped open-dwell run and run 2 alone, the cap's effect is not resolved (RR 1.79, 95 %
    CI 0.53–6.01); the rule had 52 % power against a true doubling (§9).
  - **Pooled across the four launches, computed after the rule (§9),** 44 of 688 band entries stop
    short under the caps against 21 of 714 without, RR 2.17 (95 % CI 1.31–3.62). The cap roughly
    doubles the rate. The launches share their first-episode starts; without those, RR 2.00
    (1.11–3.61).
  - Post-arrival behaviour under the caps is not read: the dwell ends run 3's episodes, and the
    addendum reads only the pre-crossing side.
- **Reading 2's censoring is visible.** In run 1 the dwell ends an episode as soon as it fires.
  There 460 of 535 band entries ended that way before any quiet run, and only 67 have a quiet stop
  (median 0.199 m, also at or under 0.25 m). In run 2, 146 of 179 band entries have one. The
  reading was taken from run 2 for that reason, as stated before the runs (§3).
- **Post-arrival is not quiet.** The first vx sign flip comes a median 0.30 s after the dwell
  fires. vx's half-range at +2 s is 0.041 m/s in the median and 0.327 m/s at the 90th
  percentile. Each condition of the limit cycle is common on its own (§4): a 1.5 s window with
  vx half-range ≥ 0.20 m/s in 39.9 % of episodes, a 5–7 Hz dominant vx in 81.5 %, a second axis at
  5–7 Hz in 55.6 %, five-fold growth in 17.4 %. Only the joint signature is rare. Condition (b) on
  its own, found in some one of up to 556 overlapping windows, is weak evidence. 32 of the 178
  windows were cut short, 22 by a collision and 10 by off-path divergence.
- **The second clock is thin.** It starts at a quiet stop at or beyond 0.30 m made before the dwell
  fires: 4 windows in run 2 (of its 6 such stops, 2 came after the dwell and are excluded) and 18
  in run 1. Two of run 2's four windows and 6 of run 1's 18 are shorter than one 45-row window, so
  they cannot show the cycle. None of the windows shows the rig-like limit cycle.
- **Actions are deterministic.** The eval runs the Beta mean, as the deployed node does. Training's
  statistics came from sampled actions. Completion is 90.97 % in run 1 and 87.85 % in run 3.
- **The A\* fallback paths do not change the verdict.** Without them, run 1 reads the same.
  Run 3's descriptive Reading-1 read says "objective" with them left out, as it does with them in.
  It is descriptive only: the rule reads run 1.
- **Each environment's first episode starts on a floor-only depth frame**
  ([`vfov-gap-v3-2026-10-03`](../vfov-gap-v3-2026-10-03/README.md) §4): 96 of 576 episodes in
  runs 1 and 3, 96 of 192 in run 2. The env leaves `num_rerenders_on_reset` at 0 (read back in
  every launch). None is excluded. That record as merged also reads every later episode's tick 0
  as the previous episode's last view. A correction on
  [#239](https://github.com/zachoines/Sim2RealLab/pull/239), unmerged when this was written, reads
  from depth sums that later episodes in a headless launch start live; #239's brief checks that on
  raw frames. If the merged reading holds, every scored episode starts on a pre-reset frame (§7).

- **Lane conditions that differ from training, none controlled by any run here.** The deploy speed
  cap is measured (above and §9). Three more differ:
  - **The body-velocity observation.** In training it comes from the wheels' forward kinematics
    (`observations.py`, `body_velocity_xy`). On the bridge it is the root's body velocity,
    published as the odometry twist the node feeds the policy (`bridge/async_publisher.py`;
    `inference_node.py`).
  - **Actuation.** The bridge composes the `real` actuation tier
    (`RealismCfg(level="real", localization_drift=False)` in `composed_env_cfg.py`), where v3
    trained on `robust`.
  - **Referent drift.** That `localization_drift=False` also removes `randomize_subgoal_drift`, the
    referent drift v3 trained with. On the lane the referent moves only with the real SLAM.

  Scene and depth content at the lane's goals is not compared either.

This record supplies the measurements the brief's Baseline box asks for, and its Deploy-cap box's
for v3. The addendum's rule (§9) decides the brief's Decision box. The Bound box stays open.

## 1. Setup

**Checkpoint.** v3's `model_999.pt` (sha256 `725fc6bf…`), the source checkpoint of the exported
`strafer_depth_subgoal_v3_999.onnx` (`c866bfd5…`,
[`depth-subgoal-v3-retrain-2026-09-21`](../depth-subgoal-v3-retrain-2026-09-21/README.md)). It is
loaded with `OnPolicyRunner.load` and run through `get_inference_policy`, which returns the Beta
mean. Every launch checks the sha256 before it boots.

**Env.** `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`, the ID v3 trained on, at its
training count of 96 envs and seed 42 (v3's training seed). It runs headless with cameras on,
observation corruption on and subgoal drift on, everything else as composed. The only differences
from training are the deterministic actions and the per-run overrides below.

**Eval.** `time_to_goal_eval.py`, a standalone script in the deposit (sha256 `877d158b…`), run on a
detached worktree of `9f7d207` (`main`) with no tree change. Each launch reads its cfg back from
the built env into `meta.json`: the episode cap, the termination terms, the cap hook,
`num_rerenders_on_reset` (0), corruption and drift, and the recurrent policy in eval mode.

**Runs.** One completed Kit boot each, in this order. Runs 1 and 3 booted on the watchdog's third
and last default attempt (§8). `nvidia-smi` listed no compute process before each launch
(`scored/logs/*.gpu_before.txt`, `scored/chain.log`).

| run | override | episode cap | quota per env | scored episodes | rows | vector steps | end causes |
|---|---|---|---|---|---|---|---|
| 1 `baseline` | none | 20 s (600 steps), as trained | 6 | 576 | 83 579 | 1 333 | 524 completed, 45 collision, 5 off-path, 2 time-out |
| 2 `open-dwell` | `path_complete` termination removed | 40 s (1200 steps) | 2 | 192 | 184 730 | 2 400 | 131 time-out, 42 collision, 19 off-path |
| 3 `deploy-cap` | the node's speed cap on the executed twist | 60 s (1800 steps), the node's `POLICY_MISSION_TIMEOUT_S` | 6 | 576 | 112 641 | 2 923 | 506 completed, 68 collision, 2 time-out |

- **Quota.** Episode *j* of an env is scored only if *j* is below the quota. The run steps until
  every env has finished its quota and discards the episodes past it. Which episodes are scored
  then does not depend on how long any of them lasts, so the slow tail is not under-sampled.
- **Seed.** All three runs use seed 42. Each env's first episode starts from the same pose, path
  end, lookahead and arc in every pair of runs, with a recorded difference of 0.000000 m
  (`summary/summary.txt`, "paired first episodes"). Later episodes diverge in closed loop and are
  compared as distributions.
- **Run 2** removes only the termination. The command term still computes the dwell each step, and
  nothing resamples the path, so after the dwell fires the subgoal stays on the path end (0.0 m at
  every row, `summary/summary.txt`). Off-path divergence, sustained collision, flipping and
  time-out stay on.
- **Run 3's cap** is the inference node's, as applied at `9f7d207`
  (`strafer_inference/inference_node.py`:1203-1208, defaults at :156-157). The twist is
  (a₀, a₁, a₂) × (1.568 m/s, 1.568 m/s, 4.187 rad/s), the last being `MAX_ANGULAR_VEL`; the
  action term's own angular scale is 4.186 rad/s. Its |vx| + |vy| is scaled down to
  `NAV_LINEAR_VEL`, 0.7841 m/s, when above it, and ω is clipped to `NAV_ANGULAR_VEL`,
  2.0933 rad/s.
  - The cap is hooked into the action term's `process_actions` on the instance, so the
    `last_action` observation still carries the raw action, as the node's does.
  - The executed twist matched the logged capped twist to 1.19e-7 at every step. A startup check
    matched the node's own scalar path to 1.96e-7. Over that check's 20 127 synthetic actions,
    which include the corners where both caps bind, the sim's wheel-speed clamp binds by at most
    5.77e-5 relative (`summary/summary.txt`, deploy-cap; `scored/eval/deploy-cap/meta.json`,
    `deploy_cap_check`).
  - The check loaded the node's `obs_pipeline.py` from the worktree and confirmed that its
    `l1_clamp_velocity` is `strafer_shared`'s `l1_clamp_twist`. The worktree's `inference_node.py`
    has the same sha256 as `9f7d207`'s (`44a2950a…`, `scored/eval/deploy-cap/meta.json`).
- **Episode caps.** Runs 2 and 3 lengthen `episode_length_s`. As the pre-registration notes, no
  observation or reward reads episode time, so the cap decides only when `time_out` fires.

**Definitions** (the pre-registration's, abridged).
- A **step** is one policy step, 1/30 s. **Row *i*** is the state before action *i*, and action *i*.
- The **path end** is the planned path's final point, the training env's completion referent.
  ***d*** is the robot's planar distance to it.
- **Path length *L*** is the path's arc length. Bins: [2, 3), [3, 4), [4, 5), [5, 6) and 6 m or
  more, as registered; the scorer adds a bin below 2 m (25 episodes in run 1, 17 in run 3).
- The **dwell fires** at the first row whose `path_complete` flag is set: after ten consecutive
  steps inside 0.30 m at a true planar speed of at most 0.1 m/s. **Time to completion *T*c** is
  that row / 30 s.
- **Band entry** is the first row with *d* ≤ 0.42 m. **Band time *T*b** runs from band entry to the
  dwell's firing. The **deploy scorer's band** runs from band entry to the first row with
  *d* ≤ 0.30 m, where the node would end the goal.
- **Planar command *c*** is hypot(a₀, a₁) × 1.568 m/s (run 3: the capped twist).
- A **quiet stop** is the 10th row of the first run of 10 rows with *c* < 0.05 m/s, from band entry
  on. **Quiet-stop distance *Q*** is *d* there.

**What was written down first.** The readings, their rules, the eval, the scoring script
`ttg_summary.py` with its self-test (150 of 150 checks pass, `selftest_out.txt`) and three plumbing
launches were committed to the evidence repository at `675c0c5`, at 19:13:20 CDT on 2026-10-11. The
pre-registration is stamped "written 2026-10-11T19:13:08-05:00". The first scored launch started at
19:13:33 (`scored/chain.log`).

**Void conditions.** A run was void if its seed was not 42, the checkpoint's sha256 differed, the
cfg read-back disagreed with its row of the run table, or it did not finish its quota within
(quota + 1) × its episode cap in steps. None of the three is void: each `meta.json` reads
`complete` with no void reason, and the scorer reports no protocol deviation.

## 2. Reading 1 — band dwell, from run 1

**The rule, as committed.** "The rig's band dwell is attributed to a difference between the
training env and the bridge lane if the 99th percentile of *T*b is at most 7.0 s and the time-out
share within 0.42 m is at most 1.0 %. Otherwise it is attributed to the objective."

7.0 s is the shortest of the rig's long band dwells (gate: L1 7.0, L2 7.8, G1 13.9, R2 52.7 at the
bound; video set: 18.72, 11.18). The pre-registration argues that if at most 1 % of training
approaches take longer, four of the gate's eight band entries at 7.0 s or more has a binomial
probability of at most 7.0e-7.

**Result** (`summary/summary.txt`, Reading 1):

| statistic | run 1 | rule |
|---|---|---|
| *T*b, 99th percentile, over 524 completions that entered 0.42 m | **6.505 s** (195.16 steps) | ≤ 7.0 s |
| episodes ending on time-out within 0.42 m | **2 of 576, 0.35 %** | ≤ 1.0 % |

**Verdict: training-env/bridge-lane difference.**

- 535 episodes entered 0.42 m. 524 completed, 9 then ended on a collision (8) or off-path (1)
  without completing, and 2 timed out. The 9 enter neither statistic. No completion co-fired with
  another termination, and no time-out also completed.
- *T*b over the 524: median 0.97 s, 95th percentile 2.63 s, maximum 12.07 s.
- The deploy scorer's band over the same set: median 0.13 s, 99th percentile 6.14 s, maximum
  11.77 s.
- Time to completion over all 524 completions: median 4.50 s, 99th percentile 13.14 s, maximum
  18.37 s.

**The margin** (computed after the runs, from `summary/summary.json`, `runs.baseline.per_episode`,
band time = (*i*f − *i*in) / 30).
- The 99th percentile interpolates between the 7th- and 6th-largest of the 524 band times, 6.300
  and 6.567 s. The 5th-largest is 7.500 s.
- Had the 6th-largest been above 7.21 s, the rule would read "objective".
- The time-out conjunct would need 4 more time-outs within 0.42 m to fail (6 of 576, 1.04 %).
- With one launch and no replicate (§7), the verdict rests on that margin.

**By path length** (run 1; times in s, shares of the bin; `summary/summary.txt`):

| *L* (m) | n | straight p50 (m) | completion | *T*c p50 / p95 / p99 / max | *T*b n | *T*b p50 / p95 / p99 / max | deploy band p50 | time-out within 0.42 m |
|---|---|---|---|---|---|---|---|---|
| < 2 | 25 | 1.70 | 88.00 % | 3.15 / 4.59 / 5.42 / 5.63 | 22 | 1.07 / 1.91 / 2.41 / 2.53 | 0.15 | 0.00 % |
| [2, 3) | 273 | 2.34 | 93.77 % | 3.88 / 6.14 / 9.22 / 18.37 | 256 | 0.97 / 2.67 / 6.29 / 12.07 | 0.13 | 0.37 % |
| [3, 4) | 170 | 3.27 | 89.41 % | 4.83 / 6.67 / 11.55 / 14.83 | 152 | 0.95 / 2.44 / 6.62 / 8.90 | 0.13 | 0.00 % |
| [4, 5) | 75 | 4.12 | 85.33 % | 5.83 / 7.44 / 10.30 / 12.17 | 64 | 1.00 / 2.35 / 4.16 / 6.30 | 0.17 | 1.33 % |
| [5, 6) | 26 | 4.89 | 92.31 % | 6.90 / 10.59 / 13.12 / 13.73 | 24 | 1.10 / 3.42 / 4.86 / 5.27 | 0.17 | 0.00 % |
| ≥ 6 | 7 | 5.24 | 85.71 % | 8.83 / 13.93 / 14.73 / 14.93 | 6 | 1.20 / 6.30 / 7.42 / 7.70 | 0.23 | 0.00 % |

- Within a bin, the *T*b 99th percentile passes 7.0 s only at 6 m or more, which holds 6
  band-entering completions. The [4, 5) bin's one time-out is 1.33 % of 75. The rule is read over
  all episodes.
- The share ending on time-out after entering 0.42 m at any row equals the share within 0.42 m at
  the end, in every bin.
- End causes per bin are in `summary/summary.txt`.

**The A\* fallback paths.** The training env falls back to a straight segment on some resets.
Run 1 has 6 (1.04 %), run 3 has 12 (2.08 %). Without run 1's: *T*b 99th percentile 6.511 s
(n 522), time-out within 0.42 m 2 of 570. The rule reads the same.

**Under the node's caps (run 3, descriptive).** The same statistic reads 11.153 s over 506
completions, above 7.0 s, and the time-out share is 2 of 576 (0.35 %). *T*b: median 1.07 s, 95th
percentile 3.66 s, maximum 24.73 s. The deploy scorer's band: median 0.27 s, 99th percentile
8.78 s, maximum 24.43 s. With run 3's 12 fallback paths left out the statistic is 11.163 s, and
the rule would read "objective". The rule is defined on run 1; run 3 is reported beside it.

Run 3 differs from run 1 in two of the node's settings: the speed cap, and the 60 s episode cap in
place of 20 s. The longer cap lets slow approaches complete instead of timing out. 3 of run 3's 506
completions fire their dwell after row 600; the other 503 form its dwell-clock set (§4). This
record does not separate the effects of the two caps on Reading 1 and the completions. The
pre-crossing stop-short, read on rows 0–599, is separated in §9.

## 3. Reading 2 — quiet-stop geometry, from run 2

**Deviation from the protocol as first specified, stated before the runs.** The protocol as first
specified read this from run 1. In run 1 the dwell ends an episode that parks inside 0.30 m while
still commanding 0.05 m/s or more, before any 10-row quiet run exists, and the dwell can never fire
outside 0.30 m. So the quiet stops that survive in run 1 lean toward stops outside 0.30 m, which
biases the median toward "objective". Run 2 removes only the termination. The reading is taken from
run 2, rows 0–599 (the trained 20 s horizon), with run 1's read under the first-specified
definition reported beside it.

**The rule, as committed.** Over run 2's episodes that entered 0.42 m and have a quiet stop by row
599, the median of *Q*. "A median of *Q* at or above 0.30 m attributes the stop-short to the
policy's own approach behaviour (the objective); at or below 0.25 m, to a bridge-lane difference;
between, undecided — both tails reported."

**Result** (`summary/summary.txt`, Reading 2): median *Q* **0.174 m** over 146 quiet stops.
**Verdict: bridge-lane difference.**

| of run 2's 179 band entries, rows 0–599 | episodes | share |
|---|---|---|
| quiet stop at *Q* ≥ 0.30 m | 6 | 3.35 % |
| quiet stop at 0.25 m < *Q* < 0.30 m | 14 | 7.82 % |
| quiet stop at *Q* ≤ 0.25 m | 126 | 70.39 % |
| no quiet stop by row 599 | 33 | 18.44 % |

- On the rig, the policy went quiet 0.295–0.411 m from its subgoal, median 0.337 m, on 8 of 11
  runs. These are re-derived from the gate's and the video set's deposited records with
  `planend.py`'s definition (`posthoc_terminal_modes.txt`; `planend.py` itself is in the evidence
  repository, in the referent-offset A/B's deposit,
  `subgoal-referent-offset-ab-2026-10-11/record-files/tools/`). The rig figure is measured to the
  subgoal; the pre-registration pairs it with *Q* to the observed referent, below.
- The median distance to the referent the policy observes, its drifted `goal_position`, is
  0.172 m.
- A quiet stop comes a median 2.5 s after band entry (90th percentile 10.7 s, maximum 15.5 s;
  `summary/summary.json`, `reading2.quiet_stop_row_offset_from_band_entry_s`).
- **The order of events.** In 153 of the 179 (85.47 %) the dwell fired before any quiet run. In 25
  a quiet stop came first: 4 at *Q* ≥ 0.30 m, 3 between, 18 at *Q* ≤ 0.25 m. One had neither. So
  121 of the 146 quiet stops in the median come after the dwell fired. Of the 25 that came first,
  18 are at or under 0.25 m, so their median is too.
- **vx sign runs** from band entry to the quiet stop: the median run is 1 row for both signs, and
  16.51 % of consecutive row pairs flip (per episode: median 11.32 %, 10th–90th percentile
  3.04–26.10 %). The pre-registration gives the rig's dithering as a median run of 2 rows with
  23–34 % flips.

**Beside the rule, the like-for-like statistic, computed after the runs** (`posthoc_terminal_modes.txt`):
the share of band entries that go quiet before their first 0.30 m crossing.

| run | band entries | quiet before the first crossing | share |
|---|---:|---:|---:|
| 1 (as trained) | 535 | 17 | 3.18 % |
| 2 (dwell termination off, rows 0–599) | 179 | 4 | 2.23 % |
| 3 (the node's caps) | 513 | 37 | 7.21 % |
| capped open-dwell (§9, rows 0–599) | 175 | 7 | 4.00 % |
| the rig (gate and video set) | 10 (of 11 reaches) | 7 | 70 % |

The rig's row is read the same way (`posthoc_pooled_cap.txt` §6, computed 2026-10-13): 8 of the 11
reaches went quiet in the band, but gate F2's quiet stop came on the pose sample of its first
0.30 m crossing, and gate R3 never entered the band. Earlier versions of this record set 8 of 11 in
this row.

- The registered median is not observable on the lane. The node ends a goal at the first 0.30 m
  crossing, so a quiet stop made after it, as 121 of run 2's 146 are, never appears there.
- The share above is the statistic both sides can read. It points the same way as the registered
  verdict, by a wide margin.
- Counting every quiet stop at 0.30 m or more up to the dwell instead adds one episode in run 1
  (18) and one in run 3 (38). In each, the robot first dipped inside 0.30 m.

**Run 1's literal read** (whole episodes, the dwell ending each completion):

| of run 1's 535 band entries | episodes | share |
|---|---|---|
| quiet stop at *Q* ≥ 0.30 m | 18 | 3.36 % |
| quiet stop between | 10 | 1.87 % |
| quiet stop at *Q* ≤ 0.25 m | 39 | 7.29 % |
| parked inside 0.30 m and ended by the dwell before a quiet run | 460 | 85.98 % |
| neither | 8 | 1.50 % |

The 67 quiet stops have a median *Q* of 0.199 m (0.188 m to the observed referent). That is also at
or under 0.25 m.

**Under the node's caps (run 3, descriptive).** The dwell ends run 3's episodes too, so its quiet
stops are censored as run 1's are. Over rows 0–599, of 513 band entries, 429 parked inside 0.30 m
before a quiet run, and 78 have a quiet stop, median *Q* 0.299 m. Reading 2's rule would read that
as undecided (`summary/summary.json`, `run3_descriptive.reading2_rows_0_599.rule_outcome`). 37 of
the 513 (7.21 %) go quiet before their first 0.30 m crossing, against 17 of 535 (3.18 %) in run 1.
The capped open-dwell run (§9) removes the dwell's termination under the caps.

## 4. Reading 3 — post-arrival behaviour, from run 2

**The set.** Run 2's episodes whose dwell fires at or before row 600, so a full 20 s window fits
inside the 40 s cap: 178. The window *W* runs from the dwell's row for 600 rows, cut short if the
episode ends first.

**The rig-like limit cycle**, per episode, is some 45-row (1.5 s) window inside *W* where all four
hold:
- (a) vx's half-range is at least 0.20 m/s;
- (b) vx's dominant frequency (mean removed, Hann window, 512-point FFT, searched over 0.5–15 Hz) is
  in 5.0–7.0 Hz, where a bin of the 512-point grid counts if it lies within half a bin (30/1024 Hz)
  of the band (4.980 and 6.973 Hz count, 7.031 Hz does not); the rig's five-row period is 6.0 Hz;
- (c) the same test passes for vy or ω in the same window, on an axis whose half-range there is at
  least 0.05 m/s (vy) or 0.10 rad/s (ω);
- (d) vx's half-range in that window is at least five times its half-range over the first 30 rows
  of *W*.

**The rule, as committed.** "If at least 10 % of these episodes show a rig-like limit cycle, the
oscillation is attributed to the policy, in a regime it never trained in. If at most 2 % do, it is
left with the lane's feedback path (the odometry twist source, latency). Between, undecided."

**Result** (`summary/summary.txt`, Reading 3): **1 of 178, 0.56 %.** **Verdict: lane feedback
path.**

| over the 178 episodes | value |
|---|---|
| first vx sign flip after the dwell, median / 90th percentile | 0.30 s / beyond *W* (25 episodes, 14.04 %, have no flip in *W*) |
| vx half-range at +1.5–2.5 s, median / 90th percentile (n 175) | 0.041 / 0.327 m/s |
| some window with (a) | 39.89 % |
| some window with (b) | 81.46 % |
| some window with (c) | 55.62 % (86.52 % without its amplitude floor) |
| some window with (d) | 17.42 % |
| some window with all four | 1, 0.56 % |

- Each condition is counted on its own; the joint signature needs all four in the same window.
  (b) has no amplitude floor, and a full *W* holds 556 overlapping windows, so (b) in some window
  is weak evidence on its own.
- The one episode is env 33's first. Its dwell fired at row 112, and its first rig-like window
  starts 534 rows (17.8 s) later. It ended later on a sustained collision
  (`summary/summary.json`, `reading3.per_episode_dwell_clock`).
- **Cut-short windows.** 146 windows are full. 32 are cut short: 22 by a sustained collision and 10
  by off-path divergence. Three windows end before +2.5 s, so the +2 s amplitude has 175 values.

**The second clock (descriptive).** The same reads, with *t*₀ at a quiet stop outside 0.30 m and
the window running to *t*₀ + 20 s or the dwell's firing, whichever comes first. Training does visit
"quiet outside the radius", and the rig's oscillation began about 4 s after such a stop.

| run | windows | shorter than one 45-row window | ended by | first flip, median | some window with (a) / (b) / (c) / (d) | rig-like |
|---|---|---|---|---|---|---|
| 2 | 4 | 2 | the dwell, 4 | none in 3 of 4 | 25.00 / 0.00 / 0.00 / 25.00 % | 0 |
| 1 | 18 | 6 | the dwell 15, episode end 3 | none in 10 of 18 | 5.56 / 33.33 / 11.11 / 5.56 % | 0 |
| 3 | 38 | 15 | the dwell 35, 20 s 3 | 1.30 s | 0.00 / 18.42 / 7.89 / 0.00 % | 0 |

- Run 2 has 6 quiet stops at or beyond 0.30 m by row 599. In 2 the dwell had already fired, so they
  are excluded, which leaves 4 windows.
- A window shorter than 45 rows holds no test window, so it cannot show the cycle
  (`summary/summary.json`, `windows_shorter_than_45_rows_n` under `reading3.quiet_clock`,
  `reading3_run1_quiet_clock` and `run3_descriptive.reading3_quiet_clock`).
- No run 2 or run 1 window reaches row 600. Four of run 3's do, and none is rig-like there.
- Run 3's dwell clock is empty by construction. The dwell ends run 3's episodes, so each of its 503
  dwell-clock windows ends on the row the dwell fires.

**Beside the rule: the detector before the crossing, computed after the runs**
(`posthoc_terminal_modes.txt`). The registered window starts at the dwell's firing. The deployed node
never lets the policy enter that regime: it ends a goal at the first 0.30 m crossing, and the rig's
oscillation came before that crossing. So the same detector, conditions (a)–(d) unchanged with (d)'s
baseline on the window's first 30 rows, is read on each band entry's window from band entry to the
first crossing, or to the episode's end.

| | windows of at least 45 rows | rig-like limit cycle |
|---|---:|---:|
| run 1 | 25 | 0 |
| run 2 (rows 0–599) | 7 | 0 |
| run 3 | 71 | 0 |
| capped open-dwell (§9, rows 0–599) | 24 | 0 |
| the rig, from band entry to the crossing | 8 | 5 |

- **On the rig's own records**, the gate's and the video set's, the detector fires on gate G1, L1
  and L2 and video G1 and L2, each before the crossing. It does not fire on gate R1, R2 or F1.
- **Too short to read.** Gate F2 and F3 cross within 41 and 39 ticks. Gate R3 never entered
  0.42 m.
- **So the detector can see the rig's cycle,** and in the training env it finds none before the
  crossing. The oscillation stays a lane finding by this read too.

## 5. Reading 4 — the bound under the deploy cap, from run 3

**The read, as committed.** Over run 3's completions: the least-squares slope *b* of *T*c on *L*.
For coverage *q*, *a*q is the smallest intercept with *T*c ≤ *a*q + *b*·*L* on a share *q* of
completions. The bound is *T*(*L*) = *a*0.99 + *b*·*L*. It is not a verdict.

**Result** (`summary/summary.txt`, Reading 4), over 506 completions:
- The least-squares fit is *T*c = 1.843 + 1.532 *L* s.
- Intercepts at the fitted slope:
  - *q* = 0.95: 4.568 s;
  - *q* = 0.99: 11.874 s, covering 501 of 506 (in sample: the intercept is set to cover them);
  - *q* = 1.0: 27.226 s.
- **The bound** is *T*(*L*) = 11.8745 + 1.5319 *L* s, or 356.23 + 45.96 *L* policy steps. Each
  further metre of path adds 1.53 s, a marginal 0.65 m/s (1/*b*, computed here) against the
  0.7841 m/s cap.
- **It passes 600 steps at *L* = 5.30 m.** For longer paths the bound runs past the trained
  episode. In run 3, 29 episodes have *L* in [5, 6) and 11 have 6 m or more.

| *L* (m) | n | completion | time-out | *T*c p50 / p95 / p99 / max (s) | straight p50 (m) |
|---|---|---|---|---|---|
| < 2 | 17 | 76.47 % | 0.00 % | 3.97 / 5.31 / 5.41 / 5.43 | 1.62 |
| [2, 3) | 288 | 90.62 % | 0.69 % | 5.17 / 7.40 / 13.51 / 30.87 | 2.37 |
| [3, 4) | 150 | 86.67 % | 0.00 % | 6.73 / 10.45 / 16.63 / 18.73 | 3.26 |
| [4, 5) | 81 | 87.65 % | 0.00 % | 8.27 / 11.60 / 19.24 / 22.07 | 4.08 |
| [5, 6) | 29 | 86.21 % | 0.00 % | 9.63 / 12.85 / 13.23 / 13.27 | 5.04 |
| ≥ 6 | 11 | 54.55 % | 0.00 % | 11.72 / 12.23 / 12.31 / 12.33 | 5.70 |

- **At the 60 s cap**, completion is 87.85 % and time-out 0.35 % (2 of 576). Both time-outs are
  in [2, 3). Collisions end 68.
- Completions past 20 s are seen. The longest is 30.87 s, in [2, 3).
- **The caps bind.** The linear cap binds on 64.57 % of run 3's 112 641 rows, the angular cap on
  8.56 %, and both on 3.77 %.
- **The bound covers completions only.** The 68 collisions and 2 time-outs are outside it, and the
  ≥ 6 m bin completes 6 of 11.
- **Run 1's fit, for comparison** (no speed cap, 20 s episodes, 524 completions):
  *T*c = 1.624 + 1.030 *L* s. Its completions all finish inside the 20 s cap, so it cannot see a
  slower approach.
  - *a*0.99 is 8.453 s, so the bound is 8.4526 + 1.0304 *L* s (253.58 + 30.91 *L* steps).
  - It passes 600 steps at *L* = 11.21 m (an extrapolation: run 1's longest path is 7.24 m,
    `summary/summary.json`, `runs.baseline.per_episode.L`).

## 6. Computed after the runs: the band-time tail against the rig's

Not pre-registered. `posthoc_band_tail.py` takes Reading 1's band times from the scorer's own loader
and definitions, for run 1 and run 3. For each run it reports the share of band-entering
completions above 7.0 s and above 13.9 s. For the share above 7.0 s it then gives the binomial
probability of at least 4 of 8 independent draws above 7.0 s. The 8 are the gate's band entries:
L1 7.0, L2 7.8, G1 13.9, R2 52.7 at the bound, and R1 2.3, F1 5.1, F2 1.3, F3 1.3.

| run | *T*b > 7.0 s | *T*b > 13.9 s | P(≥ 4 of 8 > 7.0 s) |
|---|---|---|---|
| 1, as trained | 5 of 524, 0.95 % | 0 of 524 | 5.63e-7 |
| 3, the node's caps | 10 of 506, 1.98 % | 3 of 506, 0.59 % | 1.00e-5 |

(`posthoc_band_tail.txt`.) The gate's reaches are not independent draws from this distribution. G1
is one goal run four times, and the rig is one room. The probability is a scale, not a test.

- **The end points differ.** The gate's times end at the 0.30 m crossing, while *T*b ends at the
  dwell's firing, after ten steps inside 0.30 m. Each episode's *T*b is therefore longer than its
  deploy-scorer band, which is the like-for-like.
- On the deploy scorer's band, run 3's 99th percentile is 8.78 s against *T*b's 11.15 s, and run
  1's is 6.14 s against 6.505 s (`summary/summary.txt`).
- So the shares in the table, and the probabilities computed from them, are upper bounds on what
  the rig's definition would give.
- No band time in either run is exactly 7.0 s (210 steps), so the shares above 7.0 s equal the
  shares at 7.0 s or more (`summary/summary.json`, `runs.<run>.per_episode`).

**The stop-short-and-slow-resolve mode** (`posthoc_terminal_modes.py` → `posthoc_terminal_modes.txt`,
not pre-registered). It counts band entries whose first quiet stop comes before their first 0.30 m
crossing. The quiet stop is the 10th row of the first run of ten rows below 0.05 m/s, counted from
band entry; that is Reading 2's definition, and `planend.py` reads the rig the same way.

| | run 1 | run 2 (rows 0–599) | run 3 |
|---|---|---|---|
| band entries | 535 | 179 | 513 |
| quiet before the first crossing | 17 (3.18 %) | 4 (2.23 %) | 37 (7.21 %) |
| their distance from the path end | 0.303–0.410 m, median 0.342 m | 0.314–0.344 m, median 0.322 m | 0.302–0.410 m, median 0.325 m |
| time after band entry, median | 0.77 s | 0.65 s | 1.00 s |
| resolve into the dwell | 14, a median 2.27 s later (90th percentile 6.92 s, max 11.77 s) | 4, median 2.18 s (max 3.43 s) | 35, median 2.13 s (90th percentile 7.57 s, max 24.43 s) |
| never resolve | 3: one collision, two time-outs | 0 | 2, both time-outs |
| band times above 7.0 s among them | all 5 of run 1's | — | 6 of run 3's 10 |

**The time-outs within 0.42 m** are the four episodes behind Reading 1's time-out share and its
run-3 descriptive read:

| run | episode | quiet at | in the band, from entry to the cap |
|---|---|---|---|
| 1 | 506 | 0.394 m | 14.5 s, to the 20 s cap |
| 1 | 556 | 0.342 m (after 9.6 s in the band) | 14.9 s |
| 3 | 572 | 0.350 m | 55.9 s, to the 60 s bound |
| 3 | 575 | 0.338 m | 55.6 s |

On the rig, R2 held 0.347–0.420 m for 52.7 s to the bound (the gate record).

## 7. What this does not show

- **Run-to-run spread.** One seed, one launch per run. No replicate launch was run, so the spread of
  each statistic across launches is not measured.
- **Sampled actions.** The eval runs the Beta mean. Training's statistics came from sampled actions
  (v3's last 100 updates: completion 0.870, time-out 0.0010,
  [`depth-subgoal-v3-retrain-2026-09-21`](../depth-subgoal-v3-retrain-2026-09-21/README.md)). They
  are context, not a comparison arm.
- **The bridge lane.** Everything here runs in the training env. There is no ROS bridge, odometry
  twist source, latency, executor or SLAM.
  - The verdicts say that the rig's behaviour is not typical of v3's deterministic behaviour in
    this env at seed 42, by the registered thresholds. They do not say which lane difference
    produces it.
  - **The node's caps.** The speed cap and the 60 s goal bound are the only lane settings run here,
    besides the deterministic (Beta-mean) action that every run uses. They are applied together in
    run 3, so the cap and the bound are not separated for Reading 1's band times and Reading 4. On
    the pre-crossing reads, rows 0–599, the bound cannot act (§9). The capped open-dwell run (§9)
    applies the speed cap with a 40 s cap.
  - **What is read under the caps.** The pre-crossing reads are, by §9. Post-arrival behaviour
    under the caps is not.
  - **The other lane conditions** named in the summary are not controlled by any run: the
    body-velocity source, the actuation tier and the referent drift. Neither is the scene and depth
    content at the lane's goals.
- **Stale first frames.** The env leaves `num_rerenders_on_reset` at 0. Each env's first episode
  starts on a floor-only depth frame rendered before its room existed
  ([`vfov-gap-v3-2026-10-03`](../vfov-gap-v3-2026-10-03/README.md) §4). Those episodes are 96 of
  576 in runs 1 and 3 and 96 of 192 in run 2, and they are scored.
  - That record as merged reads every later episode's tick 0 as pre-reset too. A correction on
    [#239](https://github.com/zachoines/Sim2RealLab/pull/239), unmerged when this was written,
    reads from depth sums that later episodes in a headless launch start live, and its brief checks
    that on raw frames. If the merged reading holds, every scored episode starts on a pre-reset
    frame.
  - Reading 3's one rig-like episode is env 33's first, and 91 of its 178 episodes are first
    episodes (`summary/summary.json`, `reading3.per_episode_dwell_clock`, `j` = 0).
  - The eval logs no depth, so this record does not check those frames itself.
- **Other limit-cycle definitions.** Reading 3's limit cycle is the pre-registered joint
  definition, written from the rig's description: a five-row period, three axes together, growth
  from about ±0.02 to ±0.5 m/s. Any one condition alone reads far higher (§4). Read on the rig's own
  deposited logs, from band entry to the crossing, it fires on 5 of the 8 windows long enough to
  read (§4).
- **The rig's figures.** The quiet-stop distances (8 of 11 quiet in the band, 0.295–0.411 m from the
  subgoal, median 0.337 m; the 7 of 10 band entries that went quiet before the crossing, the same
  range, median 0.343 m) and the detector's reads on the rig's windows are re-derived from the
  gate's and the video set's deposited records (§3, §4). The rest are the pre-registration's, and
  this record does not re-derive them: the dithering figures, the five-row period, the growth from
  about ±0.02 to ±0.5 m/s, and the oscillation starting about 4 s after a stop outside the radius.
- **A per-goal bound for the node.** Reading 4 is v3's bound in this env under the node's caps, over
  completions only. The per-goal bound belongs to the brief's Decision and Bound boxes.

## 8. Deviations, plumbing and boot stalls

- **Reading 2 is read from run 2, not run 1.** That was stated in the pre-registration, with its
  reason, before the runs (§3). Run 1's literal read is reported.
- **Plumbing launches came first.** Three launches at seed 7 (not a scored seed), 8 envs and quota
  1, one per run type, ran from 19:08 to 19:12 CDT, before the pre-registration was committed
  (`plumbing/`). They checked the logging, the quota loop, the read-back, the cap hook and the
  scorer's refusals. Their numbers enter nothing. They ran the same eval file as the scored
  launches (sha256 `877d158b…` in every binding file).
- **Interim scoring.** `scored/scored_chain.sh` ran the three scored launches back to back
  without a stop (`scored/chain.log`). The scorer was also run after run 1 and after run 2, while
  the next launch ran. Those interim outputs are not deposited, and nothing in this record reads
  them. The verdicts they had computed (Reading 1 after run 1; Readings 1–3 after run 2) match the
  final summary's.
- **Boot stalls.** Run 1 and run 3 each stalled at boot on their first two watchdog attempts and
  booted on the third. Run 2 and the three plumbing launches booted on the first attempt.
  - Each stall was killed at 61 s, at about 80 MB resident with no CPU or output for 60 s
    (`scored/logs/baseline.log`, `scored/logs/deploy-cap.log`).
  - So 4 of the 10 Kit boots in the 2026-10-11 series stalled. The addendum's one launch
    (2026-10-12) booted on its first attempt. The pre-registration counts a relaunched
    stall as a fresh start, not a void. The hang is tracked in
    [`kit-boot-hang-2026-09-11`](../kit-boot-hang-2026-09-11/README.md).
- **The scored summary's run directories.** `summary/summary.txt` and `summary/summary.json` name
  each run's directory by the absolute path it was scored at, the original eval directory. The
  deposited eval files are byte-identical to it. With numpy 2.3.1, re-scoring the deposit
  reproduces both files byte for byte except those three paths (three lines of `summary.txt`, three
  `dir` fields of `summary.json`). Another numpy version also changes line 2 of `summary.txt`,
  which names the version, and can change the last digits of the fit in `summary.json`.

## 9. Addendum 2026-10-12: the capped open-dwell run

**The question.** Does the node's speed cap raise the rate of the pre-crossing stop-short in v3's
training env? The cap scales |vx| + |vy| to 0.7841 m/s and clips |ω| to 2.0933 rad/s, and v3 never
trained under it. This addendum was registered with its reads script before the launch
(`addendum-2026-10-12/PREREGISTRATION.md`, committed at `9b045ab…`, stamped 20:22:27 CDT). The
launch followed at 20:22:53.

**The run.** One Kit boot of run 2's design with run 3's cap hook:
- the dwell's termination removed, a 40 s cap, quota 2 (192 episodes);
- 96 envs, seed 42, v3 deterministic;
- the same eval and worktree as runs 1–3.

It booted on the first attempt and completed in 2400 vector steps (1102 s), with no void. The
record's scorer, under a fourth run-table row, applied every read-back, log and twist check:
- the logged twist matches the node-capped twist of the logged action to 2.0e-7, with no
  cap-flag mismatch;
- no episode ended on `path_complete`;
- the subgoal stayed on the path end after every dwell;
- each env's first episode started from the start state its `meta.json` records
  (`meta_start_states_vs_episode0_max_abs` 0.0 in `addendum-2026-10-12/capped_open_dwell_reads.txt`).

**The reads,** on rows 0–599 with Reading 2's definitions:

| | run 2, as trained | capped open-dwell |
|---|---|---|
| band entries | 179 | 175 |
| 1. quiet before the first 0.30 m crossing | 4, 2.23 % | **7, 4.00 %** |
| 2. their distance from the path end | 0.314–0.344 m, median 0.322 m | 0.303–0.410 m, median 0.335 m |
| 3. pre-crossing detector, rig-like windows | 0 of 7 | 0 of 24 |

**The rule, as committed** (`addendum-2026-10-12/PREREGISTRATION.md`). If the capped pre-crossing
stop-short share is at least twice run 2's (2 × 4/179 = 4.47 %, about 8 of 179 band entries):
- the speed cap is the first named lane term;
- the time-to-goal brief's scope gains the cap;
- v4 scoping carries "train under the deploy cap, or raise `NAV_VEL_SCALE` with the chassis limits
  re-read".

Otherwise the cap is a minor term, and the next measurement controls scene and depth content at the
lane's goals.

**Verdict: the speed cap is a minor term**, 4.00 % against 4.47 %.
- **The margin is one episode.** 8 of 175 would have named the cap.
- **Beside it.** Run 3, under the same caps with about three times the band entries (513), read
  7.21 %. Both
  capped reads sit above run 2's, and run 2's own 2.23 % rests on four episodes.
- **On this run and run 2 alone the cap's effect is not resolved** either way. By the rule as
  registered, the next measurement is the scene and depth content at the lane's goals. The pooled
  read below, computed after the rule, puts the cap on the list as well.

**Computed after the rule, 2026-10-13: the cap's effect, pooled** (`posthoc_pooled_cap.py` →
`posthoc_pooled_cap.txt`, not pre-registered).
- **What the rule could tell.** It names the cap at 8 or more of 175 band entries.
  - Taking run 2's 4 of 179 as the true uncapped rate, it had 52 % power against a true doubling
    and a 4 % false-positive rate. Taking run 2's count as the draw it is, the false-positive rate
    is 21 %.
  - Its threshold sits at the count a doubling is expected to give, so a rule of this form names a
    true doubling about half the time at any sample size.
  - On this run and run 2 alone the effect is not resolved: RR 1.79 (95 % CI 0.53–6.01), Fisher
    p 0.375.
- **The four launches pooled.** Uncapped is runs 1 and 2; capped is run 3 and this run; the
  stop-short is the strict pre-crossing one on rows 0–599. Each launch's stop-shorts are the same
  episodes with and without the row limit, and no band entry or pre-crossing quiet stop falls at
  row 600 or later, so pooling the four on rows 0–599 is exact.
  - **Pooled:** 21 of 714 band entries (2.94 %) against 44 of 688 (6.40 %), RR 2.17 (1.31–3.62),
    Fisher p 0.002.
  - **Run 3 against run 1,** the dwell's termination on in both: 37 of 513 against 17 of 535, RR
    2.27 (1.29–3.98), p 0.003.
  - Run 3 also carries the 60 s bound where run 1 has 20 s, and the bound cannot act on this read.
    The policy observes no time. Apart from the time-out itself, which is the bound, the env's
    terms read the episode's length only to mark its first step, and the subgoal command resamples
    every 10^6 s. On rows 0–599, the speed cap is the
    only setting in which run 3 differs from run 1.
  - The intervals are Katz log intervals and the p values are two-sided.
- **The shared starts.** All four launches start each environment's first episode from the same
  state. Launches under the same cap setting stop short on the same starts more often than chance:
  the two capped launches (run 3 and this run) share 4 where 0.41 are expected, and the two
  uncapped launches share 1 where 0.04 are. The figures above treat every episode as independent.
  - Without the shared starts: RR 2.00 (1.11–3.61), p 0.024.
  - With each start counted once: RR 2.07–2.12, p 0.007–0.012.
- **Paired on the shared starts,** with the dwell's termination matched (run 3 with run 1, this
  run with run 2). 9 pairs stop short only under the cap, and 1 only without it: p 0.021 if the
  192 pairs were independent. They cover 96 starts, and per start the count is 7 against 1 (sign
  test p 0.070).
- **Where the cap binds.** Run 1's uncapped command exceeds the cap on 60.2 % of its rows and on
  8.3 % of its rows within 0.42 m. In run 3 the cap binds on 64.6 % of rows and on 2.8 % of rows
  within 0.42 m.
- **Read together:** "The node's speed cap roughly doubles the rate of the stop-short the policy
  already has; it accounts for about a factor of two of the roughly 25-fold gap to the lane's 8 of
  11; the main lane term is still unidentified."
  - The lane's 8 of 11 counts every quiet stop in its band. Read the way the training env is read,
    per band entry and before the first 0.30 m crossing, it is 7 of 10. That is 23.8-fold the
    pooled uncapped rate (11.8–31.7 on the lane's exact interval) and 10.9-fold the capped one.
  - The cap's share of the gap is the pooled RR on either count.
  - The rig's 7 pre-crossing stop-shorts went quiet 0.295–0.411 m from the subgoal, median
    0.343 m; its 8 quiet stops in the band, median 0.337 m.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `policy-time-to-goal-baseline-2026-10-11/record-files/` |
| deposit commit | `d8749170d492f36a47e5da7623348a0aadd734d5` |

The pre-registration, the eval, the scoring script, its self-test, the launch script and the three
plumbing launches are in the deposit's first commit, `675c0c59f159bf9cb2e8a16335461a9cf6bca848`. It
was made at 19:13:20 CDT, before the first scored launch at 19:13:33. None of those files has
changed since. The addendum's pre-registration and reads script went in at
`9b045aba2ed6e760a92ac98f56d07e7e740f0a3a`, before its launch at 20:22:53 on 2026-10-12, and
neither has changed since. The post-hoc terminal reads' script, first committed there, later gained
two printed lines (the deposit's `DEPOSIT.md` says which). `f80345c` added the run and its reads.
`f69886b` (2026-10-13) adds the pooled read, `posthoc_pooled_cap.*`. `da0a55d` adds two lines to
its output (the rig's subgoal distances) and renames one of its labels, and the deposit commit
above corrects `DEPOSIT.md`'s description of that label.

The deposit holds:
- the launch inputs:
  - `PREREGISTRATION.md`, the readings and their rules as committed before the runs;
  - `time_to_goal_eval.py`, the eval;
  - `run_ttg.sh`, one launch, and `scored/scored_chain.sh`, the scored order;
- the raw results:
  - `scored/eval/<run>/`, each run's `meta.json` (read-back, wall time, vector steps),
    `episodes.jsonl` and `rows.npz`;
  - `scored/logs/`, each launch's console log, package binding, empty GPU check and watchdog
    attempts, and the sampling log;
  - `scored/chain.log` and the wrapper outputs;
- the analysis:
  - `ttg_summary.py`, with `summary/summary.txt`, `summary/summary.json` and
    `summary/summary.stdout`;
  - `ttg_selftest.py` and `selftest_out.txt`, the scorer's self-test;
  - `posthoc_band_tail.py` and `posthoc_band_tail.txt`, computed after the runs;
  - `posthoc_terminal_modes.py` and `posthoc_terminal_modes.txt`, computed after the runs, which
    read the gate's and the video set's deposited records in this repository as well;
  - `posthoc_pooled_cap.py` and `posthoc_pooled_cap.txt`, computed after the addendum's rule,
    which read the same rig records;
- `addendum-2026-10-12/`: the capped open-dwell run's pre-registration, its reads script and
  output, and its run (`scored/eval/capped-open-dwell/`, logs, wrapper output);
- `plumbing/`, the three seed-7 launches and their summary.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/policy-time-to-goal-baseline-2026-10-11/record-files/. \
      docs/measurements/policy-time-to-goal-baseline-2026-10-11/
```

Re-derive the readings and the post-hoc tail with Python 3.8 or later and numpy 1.20 or later (the
deposited outputs: Python 3.12, numpy 2.3.1):

```
cd docs/measurements/policy-time-to-goal-baseline-2026-10-11
PYTHONPATH=. python3 -B ttg_summary.py --baseline scored/eval/baseline \
    --open-dwell scored/eval/open-dwell --deploy-cap scored/eval/deploy-cap --out /tmp/ttg-rederive
diff /tmp/ttg-rederive/summary.txt summary/summary.txt
PYTHONPATH=. python3 -B posthoc_band_tail.py scored/eval/baseline scored/eval/deploy-cap \
    | diff - posthoc_band_tail.txt
```

With numpy 2.3.1 the first `diff` shows only lines 5, 10 and 15, the run directories (§8); another
numpy version also changes line 2, which names it. The second shows nothing.

The post-hoc reads that also read the rig's records run in the evidence repository, where those
records sit beside this deposit; the pooled read needs scipy as well (1.17.0 here). Both `diff`s
show nothing:

```
cd Sim2RealLab-Artifacts/policy-time-to-goal-baseline-2026-10-11/record-files
PYTHONPATH=. python3 -B posthoc_terminal_modes.py | diff - posthoc_terminal_modes.txt
PYTHONPATH=. python3 -B posthoc_pooled_cap.py | diff - posthoc_pooled_cap.txt
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/policy-time-to-goal-baseline-2026-10-11/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
2955bb2bf078aef70a70bbe61d804074a5bfbd9aec1db6dca319a917d5df3118  PREREGISTRATION.md
47b1e769014d722cf132467dbefb36b532d72ec3a7e3f94c346d218e3ddb9c20  addendum-2026-10-12/PREREGISTRATION.md
065385f50b07bc88265d958d4f5662a4ac9c0812a3799e3ce8644057f9931430  addendum-2026-10-12/capped_open_dwell_reads.py
6213abb164f7a0b36a20bb3d3b7cd323ffde94e262642415f3970c26eb157edc  addendum-2026-10-12/capped_open_dwell_reads.txt
19eaf43821a7660ec323a87c8457bf74823beb296c39f5e01aa8a683aa50f061  addendum-2026-10-12/scored/a2.exit
6191640abf0a4894d7ee361a22a4302181ad10fce0074c939815fee553ec68a5  addendum-2026-10-12/scored/a2.wrapper.out
2fc34ced3d8b072a842320d1d9aad8dcffb90b673b3a585d145bd217409ebe8c  addendum-2026-10-12/scored/eval/capped-open-dwell/episodes.jsonl
7a6a75d9f48654fa1210bc1ce489f48901ab86baabb3876241ed877689551afc  addendum-2026-10-12/scored/eval/capped-open-dwell/meta.json
cb23370990398b2c6df3ddcecfad50e457c5d0d51985a4cdcccb503dacfa01fd  addendum-2026-10-12/scored/eval/capped-open-dwell/rows.npz
f434dcb72a5ff8a9bace2b657b14f2215d663a00cb9f426dd290c56fab226667  addendum-2026-10-12/scored/logs/capped-open-dwell.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  addendum-2026-10-12/scored/logs/capped-open-dwell.gpu_before.txt
69a395230a5955a3bb8fee4b3a8bf63def50c553f566c99f9634ad2e048395eb  addendum-2026-10-12/scored/logs/capped-open-dwell.log
6bbca11dda6d5e2cad0e7c13d135625d729a5664ac0161a2339375b70d6719db  addendum-2026-10-12/scored/logs/sampling.log
afa433881f2152539b1f6a2a3eb09cd61562d7ceaf1d4015de95f819bfd27c93  addendum-2026-10-12/scored/logs/watchdog_capped-open-dwell.log
afa433881f2152539b1f6a2a3eb09cd61562d7ceaf1d4015de95f819bfd27c93  addendum-2026-10-12/scored/logs/watchdog_capped-open-dwell.log.attempt1
1fbe72fa3d576e2e1f10abe721a773488a07798c2f77f33327339fcb300549c5  plumbing/eval/plumb-baseline/episodes.jsonl
668dccd32427261e3f27018e560c1acd9f33ccd81cae0cd51334579afcb0f1c3  plumbing/eval/plumb-baseline/meta.json
561d2954a90a11dc5a60b035fd96b89a313b317fffa4811333f893b9bc339f12  plumbing/eval/plumb-baseline/rows.npz
6bb6db8102875adac86d18fe8b66c452263eb23264523599081d9ea4b911771b  plumbing/eval/plumb-deploy-cap/episodes.jsonl
fb36e8ace63eb261fb414d69fb272e2bd86f657c0580c24a81c7f597a841e424  plumbing/eval/plumb-deploy-cap/meta.json
642408d07074b1036d56d52977fd8decf9f653ee6a8239b5f41c22ca876784c0  plumbing/eval/plumb-deploy-cap/rows.npz
bc9903170656bdfb9765d2bfff88a142a1ca4389316e7fbc9ae686f8ab06034d  plumbing/eval/plumb-open-dwell/episodes.jsonl
ab00c5aa8c192d1e15309487822c5b89e5e2626df506598b7b73334c8f140070  plumbing/eval/plumb-open-dwell/meta.json
a99abe5f26557652ea4a9dd3fbfee528768d6a00232daee195810d22cc01cdd4  plumbing/eval/plumb-open-dwell/rows.npz
c25a45de91846b79e4ecfb75a09e6349cf98815a3fc76ca7762e343f9997c293  plumbing/logs/plumb-baseline.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  plumbing/logs/plumb-baseline.gpu_before.txt
5bdbd53db6c5cc42ee0ae9d5fac7429c9ab1299c3a9e888fd504d892e47a2501  plumbing/logs/plumb-baseline.log
163a34ff2c8a2724e4c52bff0f2a6d5da6a6d60d9e3354f62d5622718214c276  plumbing/logs/plumb-deploy-cap.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  plumbing/logs/plumb-deploy-cap.gpu_before.txt
6cd1c96d0a09f30bce1f0e483d0da3b00de53ec6a5103308c605c6fb6888aeb9  plumbing/logs/plumb-deploy-cap.log
ced071dde7496b890a9af6960a9caca7f8df01523091bd197ab4ceabcb5f0538  plumbing/logs/plumb-open-dwell.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  plumbing/logs/plumb-open-dwell.gpu_before.txt
d007c3af41273c8c7c498246c5d250675f4ecbc098bad2017c569c62305e61fa  plumbing/logs/plumb-open-dwell.log
73c25c310e4b96c76c43e71ac32eeb6e0513da8bb4e82b744d1b48bb7daaa304  plumbing/logs/sampling.log
866f4b7f1c26a938ce4850ade6604efb08279caaf6a821702f25352b14b8e567  plumbing/logs/watchdog_plumb-baseline.log
866f4b7f1c26a938ce4850ade6604efb08279caaf6a821702f25352b14b8e567  plumbing/logs/watchdog_plumb-baseline.log.attempt1
4154c93d9d1069c663e74ce243a55bfd948f852f93b75e5886915b8bc0f40ced  plumbing/logs/watchdog_plumb-deploy-cap.log
4154c93d9d1069c663e74ce243a55bfd948f852f93b75e5886915b8bc0f40ced  plumbing/logs/watchdog_plumb-deploy-cap.log.attempt1
93b09468ce30dddd676c203bbcfa13bd1a7375f01f2b89b35f73989961e3d3d6  plumbing/logs/watchdog_plumb-open-dwell.log
93b09468ce30dddd676c203bbcfa13bd1a7375f01f2b89b35f73989961e3d3d6  plumbing/logs/watchdog_plumb-open-dwell.log.attempt1
19eaf43821a7660ec323a87c8457bf74823beb296c39f5e01aa8a683aa50f061  plumbing/plumb-baseline.exit
8f72e35f71d353c15c36412fdd57b7f25c365f676c62730cbd89d2b0127d2a3f  plumbing/plumb-baseline.wrapper.out
93a1c50d299754842e8ae51e15ac7e7af6d6c79446b1874a2dbe0e29c1e82cb6  plumbing/plumb-deploy-cap.wrapper.out
d701f57d7ff2712448e42bdaaae6e2ad4432f0f29789f31f87440237eeb7d5cc  plumbing/plumb-open-dwell.wrapper.out
aa81e63d652040aff8c6b65f4070f2ef29dad192a071d4469f4d2f0cb720d285  plumbing/summary.stdout
2dd01f2e4d977d0c01c17938dd65c037e1afc4694c1c16867f4fa17f39a7b360  plumbing/summary/summary.json
aa81e63d652040aff8c6b65f4070f2ef29dad192a071d4469f4d2f0cb720d285  plumbing/summary/summary.txt
5e8e550e678a2d25548927182f7d19d9f588d2538e751bb77e3aa34d898fb6d1  posthoc_band_tail.py
daae4f445d806ae11cf6c13ec751489ad11f60975bc0eb7949c427e4badf8a49  posthoc_band_tail.txt
3bfc2bce1ce36f778af74db0388a66f2e5d8a7d36985c5c1f1eaef8232a5b770  posthoc_pooled_cap.py
2cc77dafab897753301f3a88352ac86ffbb30b99137b5ca0e2b6ad7de226055b  posthoc_pooled_cap.txt
9fbd01b4a12a44a07c4c37f7eab3bc257247f0dc9923e9112623a56e06ed7167  posthoc_terminal_modes.py
08f165dc700d61f332ff6e3ff4fb08f47bfff6fa6b99ba12d31c1a5655283b90  posthoc_terminal_modes.txt
0e0e5aca4ea687b49cdbbde0ff49b79124b76de295e32bf011673b17bc979619  run_ttg.sh
c476fe506dba2a0ff47929d121025febaee2e966ed50f04e1bbcbed18d0ee55d  scored/baseline.wrapper.out
875814c7a3fc5bdead71bf9a5f5ad7a145b0e20b14333adf62502669bb02e190  scored/chain.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  scored/chain.out
c0a764e235fda7b13e954ba34526e20f10662c8271107c56a041a795d54e1e2a  scored/deploy-cap.wrapper.out
f1c591805b2065d81042fd7f1f5286abb12036b09f9b0db41bf80747c9dc7f09  scored/eval/baseline/episodes.jsonl
6816bfaad580eea3ad7997915feaac86221f929c0b43a20a7f9241340c933dfb  scored/eval/baseline/meta.json
7b013e1c3685ac5e5db8641d5b15db8f66857de129f0534b798244b915d80e38  scored/eval/baseline/rows.npz
5ce9ad10e699d1cadaaef43c0cd787addeb8011a17a4f810dc27c48108c0eeb4  scored/eval/deploy-cap/episodes.jsonl
409e39feb93df2e76e0ca4819d4d1c2adbdb88e75e7b9f7168a9718c6278c911  scored/eval/deploy-cap/meta.json
146de7a30bc73362fccfbd464b7d679f8ce966731e1dbf840b75965dfee32851  scored/eval/deploy-cap/rows.npz
28e5f4f6e24dc13663d9fbd4aefb0a02991e09348651caea0bf2a122c8c36f35  scored/eval/open-dwell/episodes.jsonl
163ebae3760ea2ac849ceb32e245e790425903e42397d0cf71fc5fca25770472  scored/eval/open-dwell/meta.json
f4b4048c86b821fc1cb9d999d5308e2643c2f59793aec8c86c0a01b890ccb821  scored/eval/open-dwell/rows.npz
9d80f9601451e959bbd245114197c0d8b33679a1d1f804e77510bc9c50ca76b3  scored/logs/baseline.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  scored/logs/baseline.gpu_before.txt
91c3f7eae1a71db9cd36e3ba239076caf5b84db1e051606eac66e2071c48e377  scored/logs/baseline.log
7e9535ede42cd01caf006c569bed5da9c6ddc86ab123bd1158178890fe776bc4  scored/logs/deploy-cap.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  scored/logs/deploy-cap.gpu_before.txt
b5003b84d89be383555d18fdbaac04a5758ec42c165841fd41b75fa7ff75a386  scored/logs/deploy-cap.log
41a9ab061e5d78180958cfc3c3bc6f3a77b51216ae85c299b2a6d2223230be38  scored/logs/open-dwell.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  scored/logs/open-dwell.gpu_before.txt
11cdc0a1bd005e2ed1d294987210ce5dd690b921b8c32cd559bf09eade6416c7  scored/logs/open-dwell.log
ec8cf4d79576c669cdadedfa6be04b228219e15a22180754987c5a08f9c58130  scored/logs/sampling.log
b11f3c418a2a4f6287c813fd77c0a13790847a80a0aa2f290c5424c1dbf5cfad  scored/logs/watchdog_baseline.log
aa3f8cd2fe9d9340f50a98b92d71d061ffb0dd460d13b400bdfb7fd588cfd544  scored/logs/watchdog_baseline.log.attempt1
aa3f8cd2fe9d9340f50a98b92d71d061ffb0dd460d13b400bdfb7fd588cfd544  scored/logs/watchdog_baseline.log.attempt2
b11f3c418a2a4f6287c813fd77c0a13790847a80a0aa2f290c5424c1dbf5cfad  scored/logs/watchdog_baseline.log.attempt3
f67e42eaabd8ef84855e3d3f1806b249ab13b49a3df2635a908a745463570080  scored/logs/watchdog_deploy-cap.log
aa3f8cd2fe9d9340f50a98b92d71d061ffb0dd460d13b400bdfb7fd588cfd544  scored/logs/watchdog_deploy-cap.log.attempt1
aa3f8cd2fe9d9340f50a98b92d71d061ffb0dd460d13b400bdfb7fd588cfd544  scored/logs/watchdog_deploy-cap.log.attempt2
f67e42eaabd8ef84855e3d3f1806b249ab13b49a3df2635a908a745463570080  scored/logs/watchdog_deploy-cap.log.attempt3
35901ae81823ccf5521349e261d46b3b6a1f6b5a90c9f9183cefab5c6a13213f  scored/logs/watchdog_open-dwell.log
35901ae81823ccf5521349e261d46b3b6a1f6b5a90c9f9183cefab5c6a13213f  scored/logs/watchdog_open-dwell.log.attempt1
73b8023e83b75918d2f0fe68141cf0de1c4b4db7c68be9769126578f8bdb582d  scored/open-dwell.wrapper.out
21da91bbf441d372c264e525f4217dbc751fb095135df4a89e8b3ec574b92c42  scored/scored_chain.sh
c04c3b9d74548fe1cf74f105728b57a2cdc217d77c143394918c7488015d60ac  selftest_out.txt
08eb85e6e87d93beeb2b1d42bd8cc5740999a241c68939b0f84864127af1113c  summary/summary.json
d1d7aaf262c4d29252b11ddf8fe732a18dc10c620b1f6b2a114ffcd17974615d  summary/summary.stdout
d1d7aaf262c4d29252b11ddf8fe732a18dc10c620b1f6b2a114ffcd17974615d  summary/summary.txt
877d158bcea76324ddb251e7d1ea2dae21ce631eafeeb1dfcca827c27b68d5c5  time_to_goal_eval.py
10a73335a73d0d8b198f7bca4e76133feaf0facdee5a4c41b5f5c8410d574b68  ttg_selftest.py
a2b7c519f97b5ce4c92f8c6fe311c670e4b75a54ebf9ec7715eb9e512a1d7eae  ttg_summary.py
```
