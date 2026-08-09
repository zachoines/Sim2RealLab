# Establish whether the sim depth camera renders at 15 Hz or 30 Hz

**Type:** investigation
**Owner:** DGX agent
**Priority:** P2 — it does not break deployment (train and deploy currently
agree), but it silently halves the depth novelty the whole DEPTH family sees.
**Estimate:** S
**Branch:** `task/sim-depth-render-rate-parity`

## Story

As the **DEPTH policy family**, I want **to know whether the sim depth images I
train and deploy on are genuinely new every step**, so that **a fresh timestamp
is not mistaken for a fresh IMAGE by anything downstream of the publisher.**

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
- **Training side, from the gym dump** (workstation re-run over the full
  **14,189-record** file). Overall **46.4%** consecutive-identical, but not from
  the start: only **30 of 6582** records before t_sim = 66 s are duplicates, with
  the onset at **t_sim ≈ 64 s**. And past the onset it is worse than the
  length-2 pairs the deploy side shows — later stretches contain **freezes of up
  to 28 identical frames (~0.9 s)**.

  *Provenance note:* an earlier Jetson-side reading of this file gave 11,136
  records and a ~50–72% post-onset fraction. That copy was an `scp` truncated
  mid-write; both counts were correct for the file each side actually held. The
  14,189-record figures above are the ones to work from.

So the two sides broadly agree at 15 Hz, which is why this is P2 rather than
P0 — but the *transition* on the training side is unexplained, the 0.9 s freezes
have no deploy-side counterpart, and if either is load-dependent then train and
deploy agree only by coincidence.

**What the deploy-side tick semantics changed, and what it did not.** These
figures were read when an inference ran only on a depth frame whose sequence had
advanced, so a duplicate image with a fresh stamp bought an inference on stale
pixels. The node now ticks on a timer and reuses the newest cached frame, which
decouples the inference rate from arrival and makes the duplicate-content share
the whole of what a halved render rate costs. That *sharpens* this brief rather
than retiring it: the novelty question is now the only one on this axis, and a
closed-loop sweep put duplicate content at the cheaper half of the two — so what
is left to establish is whether the 15 Hz render is a stated contract or a
coincidence, not whether it gates inference.

## Acceptance criteria

- [ ] State which it is: a render-rate config, a decimation, a load-dependent
      effect (the renderer failing to keep up and the bridge republishing the
      last frame), or an artifact of how the dump is written.
- [ ] Explain the gym-side onset at t_sim ≈ 64 s specifically (only 30 of the
      first 6582 records are duplicates). If it is load-dependent, say what the
      load was.
- [ ] Account for the **28-frame (~0.9 s) freezes** in later stretches of the
      gym dump. A 15 Hz render explains length-2 runs; it does not explain a
      0.9 s hold, and nothing of that length appears on the deploy side.
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
- Preserved artifacts: `gym_obs_parity.jsonl` (**14,189 records** — verify the
  line count before analysing; a truncated copy has circulated) and the
  companion rosbag2 under `parity_capture/`.
- Bridge camera config: `source/strafer_lab/strafer_lab/bridge/config.py`, and
  the policy/perception camera definitions in the navigation task's `d555_cfg`.

## Out of scope

- Any Jetson-side change. The deploy node already reports the rate it sees.
- Re-opening the depth *geometry* question — that is settled, see
  [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md).
