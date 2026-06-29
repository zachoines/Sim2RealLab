# Build the strafer_lab subgoal-following training env for `DEPTH_SUBGOAL` policy

**Type:** task / new training environment
**Owner:** DGX (`strafer_lab` lane — env config + reward shaping + training run; reuses the path planner from
[`subgoal-env`](../../completed/subgoal-env.md))
**Priority:** P3 — closes the 2×2 variant matrix (direct/hybrid × NOCAM/DEPTH). The architecturally maximally-useful hybrid corner: Nav2 plans the global route, the policy handles local control *and* sees depth so it can leave the path to avoid late-arriving obstacles. Not blocking any current mission shape; pre-filed so the design questions stay visible while the NOCAM_SUBGOAL path lands and teaches us what we actually need.
**Estimate:** L (~2–3 weeks: design questions resolved + reward shaping iterated + DEPTH-rate training run)
**Branch:** task/depth-subgoal-env

## Un-park trigger

This brief is parked until **all** of the following have shipped:

1. [`inference-package`](../../completed/inference-package.md) ✅ shipped (provides the DEPTH observation pipeline + the recurrent-policy infrastructure this brief's checkpoint will deploy through).
2. [`subgoal-env`](../../completed/subgoal-env.md) shipped (provides the `SubgoalCommand` term, the path planner — Option A or B from its Phase 1 — and the registered NoCam-Subgoal task IDs this brief extends).

A third soft prerequisite worth waiting on: NOCAM_SUBGOAL **deployed via** [`hybrid-mode`](../../active/trained-policy/hybrid-mode.md) and the [`strafer-hybrid-sim-validation`](strafer-hybrid-sim-validation.md) brief shipped. The deployment evidence is what tells us whether NOCAM_SUBGOAL's failure modes (dynamic obstacles, costmap staleness — the things the variant is documented unsafe in) actually bite in practice, and therefore whether DEPTH_SUBGOAL's training cost is justified. If the operator can confirm a deployment failure NOCAM_SUBGOAL cannot recover from, un-park immediately. If NOCAM_SUBGOAL works fine in all observed deployments, this brief stays parked.

Un-park by `git mv parked/trained-policy/depth-subgoal-env.md active/trained-policy/depth-subgoal-env.md` in the PR that picks it up.

## Story

As a **mission operator running cross-room navigation in environments where Nav2's global plan is geometrically correct but late-arriving obstacles exist** (moved chair, person walking through, dynamic-object encounter, costmap that hasn't refreshed since the last update), I want **a hybrid execution mode where Nav2 plans the route AND the policy can see depth**, so that **obstacle-aware local control and costmap-aware global planning compose into the maximally-capable hybrid backend** — and the NOCAM_SUBGOAL caveats in [`subgoal-env`](../../completed/subgoal-env.md) (trusts costmap absolutely, unsafe in dynamic-obstacle scenarios) are lifted.

## Context bundle

Read these before starting:

- [`subgoal-env.md`](../../completed/subgoal-env.md) — the NOCAM_SUBGOAL predecessor. This brief reuses its path planner, `SubgoalCommand` term, reward functions (cross-track / along-track), and task-registration patterns. The only conceptual delta is "the policy now sees depth and the reward shaping must trade off path-tracking against obstacle-avoidance".
- [`completed/inference-package.md`](../../completed/inference-package.md) — the DEPTH sibling. Useful as a reference for the conv-aware actor architecture (rsl_rl GRU 1×128 over a depth-aware backbone) and the noise model on the depth field. This brief's trained checkpoint deploys through the same `load_policy()` path inference-package shipped.
- [`hybrid-mode.md`](../../active/trained-policy/hybrid-mode.md) — the consumer-side runtime brief; documents the variant-agnostic plumbing that [`depth-subgoal-hybrid-runtime`](depth-subgoal-hybrid-runtime.md) extends.
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md) — DEPTH_SUBGOAL inherits the recurrent contract verbatim from DEPTH; the variant change does not touch hidden-state shape, initial state, threading, reset trigger set, or determinism semantics.
- [`subgoal-env`](../../completed/subgoal-env.md)'s `## Context > NOCAM_SUBGOAL safety` section — the trust-boundary discussion this brief's variant explicitly lifts.

## Design questions (must be resolved during the picking-up PR)

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

### Phase 1 — Reuse subgoal-env's path planner (½ day)

No new path-planning work. Reuse whatever planner subgoal-env shipped (Option A: Nav2 offline, or Option B: custom A* + noise). Document the choice this brief inherited and the implication for deployment parity (see [`strafer-hybrid-sim-validation`](strafer-hybrid-sim-validation.md)'s subgoal-pose parity bound — same logic, same bound).

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

Smoke test: `python source/strafer_lab/scripts/test_strafer_env.py --task Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0` runs without errors, path + subgoal markers + depth visualization visible in the Kit viewport.

### Phase 5 — Training run + checkpoint (1–3 weeks wall, depending on architecture choice from design question 1)

Train against `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0` per PPO. Target convergence metrics (record in PR description):

- Episode reward at convergence.
- Median cross-track error at convergence (target: ≤ 15 cm; looser than NOCAM_SUBGOAL's ≤ 10 cm because depth-driven detours from the path are expected).
- Obstacle-avoidance success rate on a held-out scene with seeded obstacles (target: ≥ 95 % reach-without-collision).
- Median time-to-subgoal (target: subgoal stays at ~lookahead_m distance with no oscillation).
- Wall-clock training time + GPU-hours (records the actual cost for the next brief to plan against).

If `goal-noise-training` has shipped by this point, run a noise-resilience pass on top of the converged baseline. Otherwise file the noise pass as a separate `policy-depth-subgoal-noise-training.md`.

## Acceptance criteria

### Variant + command

- [ ] `PolicyVariant.DEPTH_SUBGOAL` defined with `obs_dim == PolicyVariant.DEPTH.obs_dim` (4819) and a docstring that explicitly contrasts the goal-field semantics vs DEPTH-direct.
- [ ] `_DEPTH_SUBGOAL_FIELDS` shares scale and dim with `_DEPTH_FIELDS` per field (re-use the constants — don't redefine).

### Reward shaping

- [ ] Reward functions chosen per design question 2; unit-tested against synthetic depth + path inputs.
- [ ] Termination set inherits from NOCAM_SUBGOAL (off-path divergence + path-complete) plus the existing DEPTH-side collision termination.

### Env + registration

- [ ] Four task IDs registered (`Real` / `Robust` × non-play / play). `gym.make(<id>)` succeeds.
- [ ] Smoke test on the play variant runs to completion with path + subgoal + depth visualization visible.

### Training run

- [ ] Converged checkpoint at `logs/rsl_rl/strafer_navigation/depth_subgoal_baseline/model_<step>.pt`. PR description records all five Phase 5 metrics.
- [ ] Play-script rollout in a scene with a seeded obstacle: the robot reaches the goal *without colliding*, visibly deviating from the path to clear the obstacle. Operator records video / screenshots.
- [ ] Export through `source/strafer_lab/scripts/export_policy.py --variant DEPTH_SUBGOAL` produces a working `.onnx` (+ `.engine` if the rig is available) that `load_policy()` consumes.

### Recurrent contract preservation

- [ ] Cross-format parity test pattern from `test_recurrent_contract_e2e.py` applies to DEPTH_SUBGOAL exports unchanged (same GRU shape, same reset semantics). Re-confirm with the new artifact if the test fixture parametrizes over variant; otherwise file a follow-up to parametrize.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context module or in `subgoal-env.md`'s "NOCAM_SUBGOAL safety" section (which now coexists with DEPTH_SUBGOAL's lifted-trust-boundary story), update those in the same commit.

## Investigation pointers

- [`subgoal-env.md`](../../completed/subgoal-env.md) — the NOCAM_SUBGOAL predecessor. Phases 1, 4, and most of 2 transfer verbatim; Phase 3 (reward shaping) is the meaningful design work.
- [`completed/inference-package.md`](../../completed/inference-package.md) — DEPTH-direct's runtime is the deployment target. The variant difference is at training-config and reward-function level, not network or contract level.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py) — `depth_image` term and the existing depth-aware preprocessing this brief inherits.
- [`completed/policy-export-tooling.md`](../../completed/policy-export-tooling.md) — export path. No new export-side work needed; the variant flows through via the sidecar's `policy_variant` field.

## Out of scope

- **The Jetson-side runtime extension.** That's [`depth-subgoal-hybrid-runtime`](depth-subgoal-hybrid-runtime.md). This brief produces the trainable env + the deployable checkpoint; that brief makes the inference node consume them.
- **Sim validation of the deployed DEPTH_SUBGOAL path.** File `depth-subgoal-sim-validation.md` once both this brief and the runtime extension ship. Same shape as [`strafer-hybrid-sim-validation`](strafer-hybrid-sim-validation.md) with hybrid-specific acceptance + DEPTH-shaped parity bounds.
- **Real-robot DEPTH_SUBGOAL validation.** Files later, gated on sim validation passing.
- **Depth-modulated lookahead** (design question 3 Option B). File as `depth-subgoal-modulated-lookahead.md` follow-up if deployment shows fixed lookahead bites in cluttered scenes.
- **Hyperparameter sweep over reward coefficients.** Coefficients are Phase 5 tuning artifacts, not durable code.
- **`DEPTH_SUBGOAL_RECURRENT_OFFICIAL` variant** or any other new variant name. The set is closed: NOCAM, NOCAM_SUBGOAL, DEPTH, DEPTH_SUBGOAL closes the 2×2 deployment matrix. Future variants get filed as their own briefs with their own deployment story.
