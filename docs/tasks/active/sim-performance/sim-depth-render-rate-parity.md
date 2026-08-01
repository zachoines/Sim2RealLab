# Establish whether the sim depth camera renders at 15 Hz or 30 Hz

**Type:** investigation
**Owner:** DGX agent
**Priority:** P2 — it does not break deployment (train and deploy currently
agree), but it silently halves the depth novelty the whole DEPTH family sees.
**Estimate:** S
**Branch:** `task/sim-depth-render-rate-parity`

## Story

As the **DEPTH policy family**, I want **to know whether the sim depth images I
train and deploy on are genuinely new every step**, so that **"one inference per
fresh depth frame" means one inference per fresh IMAGE and not per fresh
timestamp.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

Measured on the 2026-07-31 parity artifacts while diagnosing
[`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md):

- **Deploy side, from the bag.** The sim bridge publishes depth on a strict
  30.000 Hz sim stamp grid with zero gaps, but the *content* repeats: over the
  join window there are exactly **583 duplicate runs, every one of length 2** —
  every image published twice, bit-identical. Over the whole 2245-frame bag,
  1213/2244 consecutive pairs (54.06%) are byte-identical. That is a **15 Hz**
  render behind a 30 Hz stamp.
- **Training side, from the gym dump.** The same structure appears, but not from
  the start: identical-consecutive fraction is **0.0%** for records 0–~2000
  (t_sim ≲ 66 s) and then ~50–72% for the rest of the file.

So the two sides currently agree at 15 Hz, which is why this is P2 rather than
P0 — but the *transition* on the training side is unexplained, and if it is
load-dependent then train and deploy agree only by coincidence.

## Acceptance criteria

- [ ] State which it is: a render-rate config, a decimation, a load-dependent
      effect (the renderer failing to keep up and the bridge republishing the
      last frame), or an artifact of how the dump is written.
- [ ] Explain the gym-side 30 Hz → 15 Hz transition at t_sim ≈ 66 s
      specifically. If it is load-dependent, say what the load was.
- [ ] State whether the policy camera's effective render rate is a *stated*
      contract anywhere, and if not, make it one (a constant plus an assert or
      a bridge-side counter), so a future divergence between the two lanes
      fails loud rather than silently halving depth novelty on one side.
- [ ] If train and deploy can diverge, say which one is authoritative and how
      the other is brought to it.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the bridge smoke path (`bridge_harness_smoke`).

## Investigation pointers

- The deploy-side counter that made this visible is `depth_repeat_content` in
  the inference node's periodic `cadence:` line — it counts consecutive
  inferences fed a bit-identical downsampled block.
- Preserved artifacts: `~/strafer_v2_validation/gym_obs_parity.jsonl` (11136
  records — note the session's "14189" figure is wrong) and
  `~/strafer_v2_validation/parity_capture/parity_bag/`.
- Bridge camera config: `source/strafer_lab/strafer_lab/bridge/config.py`, and
  the policy/perception camera definitions in the navigation task's `d555_cfg`.

## Out of scope

- Any Jetson-side change. The deploy node already reports the rate it sees.
- Re-opening the depth *geometry* question — that is settled, see
  [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md).
