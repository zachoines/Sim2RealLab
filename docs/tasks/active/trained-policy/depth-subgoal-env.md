# Build the strafer_lab subgoal-following training env for `DEPTH_SUBGOAL` policy

**Type:** task / new training environment
**Owner:** DGX (`strafer_lab` lane — env config + reward shaping + training run; reuses the path planner from
[`subgoal-env`](../../completed/subgoal-env.md))
**Priority:** P3 — closes the 2×2 variant matrix (direct/hybrid × NOCAM/DEPTH). The architecturally maximally-useful hybrid corner: Nav2 plans the global route, the policy handles local control *and* sees depth so it can leave the path to avoid late-arriving obstacles. Not blocking any current mission shape; pre-filed so the design questions stay visible while the NOCAM_SUBGOAL path lands and teaches us what we actually need.
**Estimate:** L (~2–3 weeks: design questions resolved + reward shaping iterated + DEPTH-rate training run)
**Branch:** task/depth-subgoal-env

## Un-park trigger (satisfied — un-parked 2026-07-04)

**Un-parked** by explicit operator decision (epic decision 5): the goal-c env
composition deliberately overlaps the goal-a retrain. The soft trigger below was
also arguably met — NOCAM_SUBGOAL's observed corner-contact behavior is exactly
the depth-recoverable failure class this brief names. The gating conditions,
retained as historical record:

The brief was parked until **all** of the following had shipped:

1. [`inference-package`](../../completed/inference-package.md) ✅ shipped (provides the DEPTH observation pipeline + the recurrent-policy infrastructure this brief's checkpoint will deploy through).
2. [`subgoal-env`](../../completed/subgoal-env.md) ✅ shipped (provides the `SubgoalCommand` term, the path planner — Option A or B from its Phase 1 — and the registered NoCam-Subgoal task IDs this brief extends).

A third soft prerequisite worth waiting on: NOCAM_SUBGOAL **deployed via** [`hybrid-mode`](../../completed/hybrid-mode.md) and the [`strafer-hybrid-sim-validation`](../../completed/trained-policy/strafer-hybrid-sim-validation.md) brief shipped. The deployment evidence is what tells us whether NOCAM_SUBGOAL's failure modes (dynamic obstacles, costmap staleness — the things the variant is documented unsafe in) actually bite in practice, and therefore whether DEPTH_SUBGOAL's training cost is justified.

Un-parked via `git mv parked/trained-policy/depth-subgoal-env.md active/trained-policy/depth-subgoal-env.md` in the picking-up PR.

## Story

As a **mission operator running cross-room navigation in environments where Nav2's global plan is geometrically correct but late-arriving obstacles exist** (moved chair, person walking through, dynamic-object encounter, costmap that hasn't refreshed since the last update), I want **a hybrid execution mode where Nav2 plans the route AND the policy can see depth**, so that **obstacle-aware local control and costmap-aware global planning compose into the maximally-capable hybrid backend** — and the NOCAM_SUBGOAL caveats in [`subgoal-env`](../../completed/subgoal-env.md) (trusts costmap absolutely, unsafe in dynamic-obstacle scenarios) are lifted.

## Context bundle

Read these before starting:

- [`subgoal-env.md`](../../completed/subgoal-env.md) — the NOCAM_SUBGOAL predecessor. This brief reuses its path planner, `SubgoalCommand` term, reward functions (cross-track / along-track), and task-registration patterns. The only conceptual delta is "the policy now sees depth and the reward shaping must trade off path-tracking against obstacle-avoidance".
- [`completed/inference-package.md`](../../completed/inference-package.md) — the DEPTH sibling. Useful as a reference for the conv-aware actor architecture (rsl_rl GRU 1×128 over a depth-aware backbone) and the noise model on the depth field. This brief's trained checkpoint deploys through the same `load_policy()` path inference-package shipped.
- [`hybrid-mode.md`](../../completed/hybrid-mode.md) — the consumer-side runtime brief; documents the variant-agnostic plumbing that [`depth-subgoal-hybrid-runtime`](depth-subgoal-hybrid-runtime.md) extends.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md) — DEPTH_SUBGOAL inherits the recurrent contract verbatim from DEPTH; the variant change does not touch hidden-state shape, initial state, threading, reset trigger set, or determinism semantics.
- [`subgoal-env`](../../completed/subgoal-env.md)'s `## Context > NOCAM_SUBGOAL safety` section — the trust-boundary discussion this brief's variant explicitly lifts.

## Design questions (must be resolved during the picking-up PR)

**Resolved (Phases 1–4 PR, un-parked 2026-07-04):**

- **Q1 → A** (same architecture as DEPTH-direct, fresh PPO run). The four
  registered ids pair with `STRAFER_PPO_DEPTH_RUNNER_CFG` (already recurrent:
  CNN + GRU 1×128) — no new runner cfg. The fine-tune-from-checkpoint comparison
  is a Phase-5 ablation, not the default.
- **Q2 → B** (explicit depth-aware obstacle-avoidance reward). Started from
  `RewardsCfg_ProcRoom_Subgoal` unchanged (its dwell-gated `path_complete` and
  geometric `obstacle_proximity` carry over) and *added*
  `depth_obstacle_proximity_penalty` — the saturating `1/(min(depth)+ε)` form,
  zeroed beyond a clearance threshold and floored at a saturation depth. The
  geometric term (ground-truth shaping) and the depth term (the sensed channel)
  coexist deliberately; documented in `RewardsCfg_ProcRoom_Subgoal_Depth`. Depth
  makes obstacles *observable*, so the proximity gradient is real signal here
  (unlike NOCAM_SUBGOAL). Fall-back to Option A stays available if Phase-5
  coefficient sweeps don't converge.
- **Q3 → A** (reuse `SubgoalCommand` unchanged, fixed lookahead). The dwell
  params ride in automatically. Depth-modulated lookahead stays out of scope
  (see the follow-up below).

**Composition mechanics:** the codebase moved from hand-written
`StraferNavEnvCfg_*` classes to composed cfgs since this brief was filed. The
depth×subgoal corner composes through the existing machinery with **no
structural extension** — `_ComposedStraferNavEnvCfg` with a depth sensor stack +
`objective="subgoal"` already selects the depth observation and the subgoal
command/termination blocks, and the goal-shaped depth-obs scalars re-bind to the
subgoal command via the shared `goal_command` name. The one addition is a leaf,
profile-keyed override of the subgoal-reward selector
(`_REWARDS_BY_SOURCE_PROFILE_SUBGOAL`) so the depth stack gets the depth-sensed
penalty while the NOCAM_SUBGOAL path stays byte-identical — mirroring how the
observation table is already keyed on `(realism, profile)`. No design consult
was needed; the NOCAM_SUBGOAL two-arm run is untouched.

### 1. Architecture: share backbone with DEPTH-direct or train from scratch?

DEPTH-direct's actor is conv-backbone → GRU 1×128 → Gaussian head, trained against the goal-directed reward landscape. DEPTH_SUBGOAL has the same input shape (4819 dims) but a fundamentally different reward landscape (path-tracking, not goal-converging).

Three options:

- **A. Same architecture, fresh PPO run.** Reuse the runner config, retrain from scratch. Slowest but architecturally cleanest.
- **B. Same architecture, fine-tune from DEPTH-direct checkpoint.** Initialize from converged DEPTH weights, then retrain on the subgoal reward. Faster convergence; risk of negative transfer if the goal-vs-subgoal distributional gap is large.
- **C. Separate architecture (e.g. larger / smaller backbone).** Justify only with evidence — the default is "use the architecture DEPTH-direct uses unless there's a reason."

Recommendation: A. The PR description records the fine-tune-from-checkpoint comparison as an ablation, not a default.

### 2. Reward shaping: how does cross-track penalty interact with obstacle avoidance?

The load-bearing design question. Two reward terms now want different things:

- **Cross-track penalty** (from NOCAM_SUBGOAL): "stay close to the path".
- **Obstacle avoidance** (new): "leave the path when there's something in the way".

Three options:

- **A. Collision-only termination + cross-track penalty.** Discourage off-path drift, terminate on contact. Architecturally clean; slow convergence because the policy gets no gradient signal toward avoidance until it actually collides.
- **B. Explicit depth-aware obstacle-avoidance reward.** Negative reward proportional to the closest depth-field value below some threshold. Faster but couples the two reward terms — the path-tracking coefficient and the obstacle-avoidance coefficient interact in non-obvious ways during tuning.
- **C. Path-as-attractor + depth-as-repulsor potential.** Geometric reward composed from two potential fields. The mathematically elegant choice; least common in practice for RL setups.

Recommendation: start with B (matches DEPTH-direct's obstacle-aware shaping pattern), fall back to A if coefficient sweeps don't converge.

### 3. SubgoalCommand reuse vs DEPTH-specific variant

NOCAM_SUBGOAL's `SubgoalCommand` (from [`subgoal-env`](../../completed/subgoal-env.md) Phase 2) computes a rolling subgoal at fixed `lookahead_m`. DEPTH_SUBGOAL with explicit obstacle awareness might want to *vary* the lookahead based on perceived clutter: shorter lookahead in cluttered scenes, longer in open. Two options:

- **A. Reuse `SubgoalCommand` unchanged.** Fixed lookahead. Variant difference is purely in obs assembly (depth in, same goal referent).
- **B. New `DepthAwareSubgoalCommand` with depth-modulated lookahead.** More expressive; commits to a design choice the NOCAM_SUBGOAL training run did not test.

Recommendation: A for the MVP. File the modulated version as a follow-up only if A's deployment surfaces a failure mode tighter lookahead would solve.

### 4. Training cost estimate

DEPTH training on `Isaac-Strafer-Nav-RLDepth-Real-v0` takes substantially longer per wall-clock hour than NOCAM (the 4800-dim depth + conv backbone is the bottleneck). DEPTH_SUBGOAL trains over the same input distribution but with a more complex reward landscape, so expect ≥ the same wall-clock budget. The PR description records the actual training-time + GPU-hours so future briefs can plan against measured cost, not assumed cost.

## What's missing

(All deltas relative to the post-subgoal-env state.)

1. **`PolicyVariant.DEPTH_SUBGOAL`** in `strafer_shared.policy_interface`. Same 4819 fields as DEPTH; only the goal-related referent changes (rolling subgoal vs final goal). Same docstring contract as NOCAM_SUBGOAL re: the goal-field referent. Same recurrent semantics as DEPTH (point 4.5 of recurrent-policy-contract.md inherits verbatim).

2. **Reward functions** for the trade-off resolved per design question 2. If Option B, a new `depth_obstacle_proximity_penalty` term in `mdp/rewards.py`. If Option A, no new reward functions — just reuse NOCAM_SUBGOAL's set with the existing `collision` termination handling avoidance.

3. **Env config** for `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0` (and `Robust` + `Play` variants) composing the existing ProcRoom-Depth scene + DEPTH observations + `SubgoalCommand` (or its successor) + the chosen reward.

4. **Registered task IDs** in `navigation/__init__.py`, alongside the existing NoCam-Subgoal variants.

5. **A trained checkpoint** at convergence. Exported via `source/strafer_lab/scripts/export_policy.py --variant DEPTH_SUBGOAL` (the export path already supports any registered variant via the sidecar JSON contract; no new export-side work needed).

## Approach

Five phases, sequenced. Phases 1–4 are dev work; Phase 5 is the training run that gates the runtime consumer.

**Phases 1–4 landed (un-park PR, 2026-07-04).** Two places the implementation
diverged from the sketch below, both because the codebase moved on after this
brief was filed:

- Phase 4 composes through `_ComposedStraferNavEnvCfg`
  (`StraferNavCfg_RLDepthSubgoal_{Real,Robust}[_PLAY]`) rather than hand-written
  `StraferNavEnvCfg_*` classes — see the composition-mechanics note under Design
  questions. The depth reward is added via `RewardsCfg_ProcRoom_Subgoal_Depth`
  (a subclass of the shared subgoal rewards) selected by a profile-keyed
  override, so the NOCAM_SUBGOAL reward stays byte-identical.
- The reward unit tests live at `tests/navigation/test_depth_subgoal_rewards.py`
  (Kit-free, `SimpleNamespace`-mocked depth sensor — the repo convention for MDP
  reward-function unit tests), not the `tests/test_depth_subgoal_rewards.py`
  path the sketch names.

Phase 5 (training + converged checkpoint + export round-trip) is operator-gated;
the brief stays active until it closes.

### Phase 1 — Reuse subgoal-env's path planner (½ day)

No new path-planning work. Reuse whatever planner subgoal-env shipped (Option A: Nav2 offline, or Option B: custom A* + noise). Document the choice this brief inherited and the implication for deployment parity (see [`strafer-hybrid-sim-validation`](../../completed/trained-policy/strafer-hybrid-sim-validation.md)'s subgoal-pose parity bound — same logic, same bound).

### Phase 2 — `PolicyVariant.DEPTH_SUBGOAL` (½ day)

In `strafer_shared.policy_interface`:

- Add `_DEPTH_SUBGOAL_FIELDS` mirroring `_DEPTH_FIELDS` exactly (same shapes, same scales).
- Add `PolicyVariant.DEPTH_SUBGOAL = _DEPTH_SUBGOAL_FIELDS`.
- Docstring contract: same shape as DEPTH, but the goal-related fields refer to a *rolling subgoal pose*, not a final goal pose. Same caveat as NOCAM_SUBGOAL. Consumers must use the matching command term at training and inference time; mixing DEPTH and DEPTH_SUBGOAL silently produces garbage.

### Phase 3 — Reward shaping per design-question resolution (3–5 days)

The phase that determines whether this whole brief converges. Resolve design question 2 (cross-track vs obstacle-avoidance) at PR-opening time, ship the reward functions, then tune coefficients in Phase 5.

If Option B (explicit depth-aware reward): add `depth_obstacle_proximity_penalty` in `mdp/rewards.py`. Negative reward proportional to `1 / (min(depth) + ε)` clipped at some minimum depth (so the penalty saturates rather than explodes at near-contact).

Unit tests in `source/strafer_lab/tests/test_depth_subgoal_rewards.py`:

- Synthetic depth field with a near obstacle: penalty fires.
- Synthetic depth field clear of obstacles: penalty is zero.
- Path-tracking reward composes additively with avoidance reward (no double-counting).

### Phase 4 — Env config + task registration (1 day)

In `strafer_env_cfg.py`: `StraferNavEnvCfg_Real_ProcRoom_Subgoal_Depth` (and `_Robust_` + `_PLAY` variants). Composes the existing ProcRoom-Depth scene + DEPTH observations + `SubgoalCommand` (Phase 1's chosen variant) + new rewards from Phase 3.

In `navigation/__init__.py`: register `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0`, `-Play-v0`, `Robust` variants.

Smoke test (operator Kit gate): `$ISAACLAB -p source/strafer_lab/scripts/test_strafer_env.py --env Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0 --num_envs 1 --pattern circle --duration 15 --video` runs without errors; the recorded clip (or a headed viewport — drop `--video`, add nothing else) shows the planned path + rolling-subgoal markers (both `debug_vis=True`) and the robot moving through the ProcRoom depth scene. The script's flag is `--env` (not `--task`), it launches via `isaaclab.sh -p`, and it auto-enables cameras for depth envs.

### Phase 5 — Training run + checkpoint (1–3 weeks wall, depending on architecture choice from design question 1)

Train against `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0` per PPO. Target convergence metrics (record in PR description):

- Episode reward at convergence.
- Median cross-track error at convergence (target: ≤ 15 cm; looser than NOCAM_SUBGOAL's ≤ 10 cm because depth-driven detours from the path are expected).
- Obstacle-avoidance success rate on a held-out scene with seeded obstacles (target: ≥ 95 % reach-without-collision).
- Median time-to-subgoal (target: subgoal stays at ~lookahead_m distance with no oscillation).
- Wall-clock training time + GPU-hours (records the actual cost for the next brief to plan against).

If `goal-noise-training` has shipped by this point, run a noise-resilience pass on top of the converged baseline. Otherwise file the noise pass as a separate `policy-depth-subgoal-noise-training.md`.

#### Phase-5 decision (2026-07-06): ship the depth *tracker*, defer reactive avoidance

After the diagnostic arc below, the depth **penalty** is **shipped inert (weight 0.0,
term kept wired — a one-float re-enable)**. What was proven and what wasn't:

- **Proven and shipped:** the depth *observation* is the win — the weight-0 depth policy
  path-follows the rolling-subgoal task at ~90 % completion, **beating the NOCAM_SUBGOAL
  baseline**. That is DEPTH_SUBGOAL's defensible deliverable.
- **Not achievable in this env at any penalty weight:** reactive sensed-obstacle avoidance.
  The A* planner line-of-sight-shortcuts paths straight and inflates by the robot radius, so a
  planned path is a pre-cleared near-straight shot — the forward camera almost never faces an
  on-path obstacle to steer around, so the penalty's improvement slope is shallow (raw
  ~0.285 → 0.21). Raising the weight −0.25 → −1.0 left the sensed obstacle distance unchanged
  (~0.43 m either way) and destabilized late; the residual ~10 % collisions are **blind rear
  bumps** (random spawn yaw forces a turn-around; the forward-only camera can't see behind) —
  a sensing/spawn artifact no forward penalty can touch. The 0.3 m off-path corridor also
  *terminates and −50-penalizes* the exact lateral avoidance the penalty rewards.
- **Follow-up:** [`depth-subgoal-reactive-avoidance`](../../parked/trained-policy/depth-subgoal-reactive-avoidance.md)
  re-enables the penalty in a hardened env (authored on-path gate corridor, widened
  depth-only off-path bound, detour-economics rebalance, spawn-heading fix; moving obstacles
  as a staged sub-phase). Operator-confirmed value: Nav2 pre-clears static geometry but
  updates the costmap slowly for dynamics and can't give a blind agent margin in tight
  passages (doorways) — so the penalty earns its place on **dynamic/late-arriving obstacles +
  tight clearances**, not on the static pre-cleared paths this env produces.

#### Phase-5 attempt 1 (2026-07-04/05): entropy collapse — diagnosed, reward repaired

The first training run (`logs/rsl_rl/strafer_navigation/depth_subgoal_baseline/run_20260704_233901`,
124 iters, ~4 h; + a 66-iter resume, ~2.5 h — ~6.5 GPU-hours total on the GB10) collapsed:
policy entropy dove 0.62 → −0.75 (resume: → −1.18), episode length pinned at 22–28 of 600
steps, 77–90 % of episodes terminated via `off_path_divergence`, and `path_complete` never
fired in 190 logged iterations. TB forensics + reward arithmetic isolated the cause: the
level-mounted camera (optical center ~0.35 m, VFOV ~71°) always sees the floor at ~0.5 m
z-depth in its lower FOV, so `depth_obstacle_proximity_penalty` (then weight −1.0, min over
all pixels, 1.0 m threshold) ran at 76–96 % of its saturation cap from iteration 1 —
an unavoidable ambient tax. With one-shot terms realizing at `weight × step_dt` (off-path
−50 → −1.67, completion +200 → +6.67), the dense stream made terminating early
return-optimal beyond a ~21–37-step horizon — exactly the observed episode-length pin. The
NOCAM subgoal contrast run (same terminations, no depth term) recovered from the identical
early off-path spike and reached 29 % path completion. Two repairs landed before restart:

1. **Floor-plane exclusion** in `depth_obstacle_proximity_penalty`: per-pixel expected
   floor depth from the camera's ground-truth pose (privileged, train-time-only; pitch/roll
   absorbed exactly), pixels within `floor_margin` (0.07 m) of the prediction excluded from
   the min. A bare floor now reads exactly zero; small floor clutter stays detected via its
   occlusion deficit, and the 0.5–1.0 m band the floor used to min-clip is restored. Weight
   retuned −1.0 → −0.25 so even a saturated reading stays inside the task economics.
2. **DeFM input scale**: the encoder (train + export mirrors) now un-scales the obs-normalized
   [0, 1] depth back to metric meters before DeFM's preprocess — its global log channels are
   anchored to absolute distance and were previously compressed ~1σ off the pretraining
   distribution. Invalidates the collapsed checkpoints (which were not salvageable anyway);
   pre-fix exported artifacts remain self-consistent and unaffected.

**Restart gates (watch in the first ~50–100 iters; alarm ⇒ stop early, don't ride it out):**

- `Episode_Reward/depth_obstacle_proximity` per-step ≈ 0 at iters 0–5 under the random
  policy (was 76–96 % of cap) — the single number that proves the ambient tax is gone.
- `Train/mean_episode_length` climbing through 150–300 by iter 50 (NOCAM: 265@25, 352@50);
  alarm if pinned in the 22–40 band past iter 30.
- `Episode_Termination/off_path_divergence` declining after the normal early spike (≤0.5 by
  iter 50); alarm if it plateaus >0.75 past iter 30. `time_out` must stay nonzero past iter 20.
- `Loss/entropy` holding near its ~+0.62 init or declining gently (Beta-head entropy goes
  negative as it sharpens — the alarm is a monotone dive crossing 0 by iter 75).
- Pre-approved second lever ONLY if entropy still dives after the reward fix: Beta
  `init_noise_std` 0.3 → 0.5 and/or more envs; keep all other runner knobs frozen for
  attribution.

## Acceptance criteria

### Variant + command

- [x] `PolicyVariant.DEPTH_SUBGOAL` defined with `obs_dim == PolicyVariant.DEPTH.obs_dim` (4819) and a docstring that explicitly contrasts the goal-field semantics vs DEPTH-direct. *(Phases 1–4 PR; `test_policy_variant_depth_subgoal.py`.)*
- [x] `_DEPTH_SUBGOAL_FIELDS` shares scale and dim with `_DEPTH_FIELDS` per field (re-use the constants — don't redefine). *(Built as `_NOCAM_SUBGOAL_FIELDS + _DEPTH_FIELDS[len(_NOCAM_FIELDS):]` — the depth `ObsField` is reused verbatim, asserted `is` the DEPTH one.)*

### Reward shaping

- [x] Reward functions chosen per design question 2; unit-tested against synthetic depth + path inputs. *(Option B — `depth_obstacle_proximity_penalty`; `test_depth_subgoal_rewards.py` covers fires/zero/saturates/inf + additive-with-path-tracking.)*
- [x] Termination set inherits from NOCAM_SUBGOAL (off-path divergence + path-complete) plus the existing DEPTH-side collision termination. *(`TerminationsCfg_ProcRoom_Subgoal`: path_complete + off_path_divergence + sustained contact-force collision + robot_flip/timeout.)*

### Env + registration

- [x] Four task IDs registered (`Real` / `Robust` × non-play / play). `gym.make(<id>)` succeeds. *(`Isaac-Strafer-Nav-RLDepth-Subgoal-{Real,Robust}[-Play]-v0` on `STRAFER_PPO_DEPTH_RUNNER_CFG`; `EXPECTED_ENVS` + composition-contract goldens updated.)*
- [ ] Smoke test on the play variant runs to completion with path + subgoal + depth visualization visible. *(Operator Kit smoke — the Phase-4 merge gate.)*

### Training run — redefined to the *validated* deliverable (see decision below)

The Phase-5 deliverable is the depth **path-follower**, not reactive obstacle
avoidance. The latter was un-achievable in this env at any penalty weight and is
deferred to [`depth-subgoal-reactive-avoidance`](../../parked/trained-policy/depth-subgoal-reactive-avoidance.md).

- [x] Converged depth-tracking checkpoint. *(Weight-0 depth run: ~90 % path-completion, **beating the NOCAM_SUBGOAL baseline** — the depth observation tracks the rolling-subgoal path better than proprioception. `STRAFER_PPO_DEPTH_RUNNER_CFG`, DeFM/GRU/Beta.)*
- [ ] Export through `source/strafer_lab/scripts/export_policy.py --variant DEPTH_SUBGOAL` produces a working `.onnx` (+ `.engine` if the rig is available) that `load_policy()` consumes. *(Operator, on the tracking checkpoint.)*
- [→] Reactive obstacle avoidance (≥ 95 % reach-without-collision on seeded obstacles; policy visibly deviates to clear an obstacle). **Deferred** — [`depth-subgoal-reactive-avoidance`](../../parked/trained-policy/depth-subgoal-reactive-avoidance.md) re-enables the depth penalty in a hardened ProcRoom (authored on-path gates + widened depth-only corridor). Justification: Nav2 pre-clears static geometry, so the penalty's real value is dynamic/late-arriving obstacles + tight-clearance passages a costmap can't give a blind agent margin in.

### Recurrent contract preservation

- [ ] Cross-format parity test pattern from `test_recurrent_contract_e2e.py` applies to DEPTH_SUBGOAL exports unchanged (same GRU shape, same reset semantics). Re-confirm with the new artifact if the test fixture parametrizes over variant; otherwise file a follow-up to parametrize.

### Maintenance

- [x] If your work invalidates a fact in any referenced context module or in `subgoal-env.md`'s "NOCAM_SUBGOAL safety" section (which now coexists with DEPTH_SUBGOAL's lifted-trust-boundary story), update those in the same commit. *(No facts invalidated — `subgoal-env.md` already frames DEPTH_SUBGOAL as the depth-seeing successor. The lifted-trust-boundary story is stated in the `PolicyVariant` docstring; the moved-brief links in `subgoal-env.md` / `hybrid-mode.md` / BOARD were repointed parked→active.)*

## Investigation pointers

- [`subgoal-env.md`](../../completed/subgoal-env.md) — the NOCAM_SUBGOAL predecessor. Phases 1, 4, and most of 2 transfer verbatim; Phase 3 (reward shaping) is the meaningful design work.
- [`completed/inference-package.md`](../../completed/inference-package.md) — DEPTH-direct's runtime is the deployment target. The variant difference is at training-config and reward-function level, not network or contract level.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py) — `depth_image` term and the existing depth-aware preprocessing this brief inherits.
- [`completed/policy-export-tooling.md`](../../completed/policy-export-tooling.md) — export path. No new export-side work needed; the variant flows through via the sidecar's `policy_variant` field.

## Out of scope

- **The Jetson-side runtime extension.** That's [`depth-subgoal-hybrid-runtime`](depth-subgoal-hybrid-runtime.md). This brief produces the trainable env + the deployable checkpoint; that brief makes the inference node consume them.
  - **Known deploy blocker for the DEPTH family (handed off to [#140](https://github.com/zachoines/Sim2RealLab/pull/140), the runtime PR):** the real-robot obs pipeline double-scales depth — `downsample_depth` normalizes to [0, 1] (`× 1/DEPTH_MAX`) and then `assemble_observation` applies `DEPTH_SCALE` again, so the deployed network receives depth 6× smaller than sim (a 3 m surface → 0.083 deployed vs 0.5 in sim). Pre-existing (predates this brief) and it does not affect sim training, but the DeFM metric-scale channels make it load-bearing for any real-robot DEPTH / DEPTH_SUBGOAL run. Fix: `downsample_depth` returns raw meters (mirroring `mdp.depth_image`) so the single `DEPTH_SCALE` matches sim; pin the assembled depth *value* in `test_obs_pipeline.py`. Changes deployed DEPTH input distribution → needs real-robot re-validation.
- **Sim validation of the deployed DEPTH_SUBGOAL path.** File `depth-subgoal-sim-validation.md` once both this brief and the runtime extension ship. Same shape as [`strafer-hybrid-sim-validation`](../../completed/trained-policy/strafer-hybrid-sim-validation.md) with hybrid-specific acceptance + DEPTH-shaped parity bounds.
- **Real-robot DEPTH_SUBGOAL validation.** Files later, gated on sim validation passing.
- **Depth-modulated lookahead** (design question 3 Option B). File as `depth-subgoal-modulated-lookahead.md` follow-up if deployment shows fixed lookahead bites in cluttered scenes.
- **Hyperparameter sweep over reward coefficients.** Coefficients are Phase 5 tuning artifacts, not durable code.
- **`DEPTH_SUBGOAL_RECURRENT_OFFICIAL` variant** or any other new variant name. The set is closed: NOCAM, NOCAM_SUBGOAL, DEPTH, DEPTH_SUBGOAL closes the 2×2 deployment matrix. Future variants get filed as their own briefs with their own deployment story.
