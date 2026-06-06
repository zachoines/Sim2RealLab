# Revert `gpu_max_num_partitions` to default (or document why it must stay 1)

**Type:** investigation + small refactor (PhysX solver config)
**Owner:** DGX agent
**Priority:** P3 — likely a training-throughput win, not blocking. The
ProcRoom PhysX config pins `gpu_max_num_partitions=1` (default is 8), which
forces a single GPU solver partition and caps solver parallelism at high env
counts.
**Estimate:** S (one-line config change + one high-env-count training run to
validate; the only real cost is the GPU run).
**Branch:** task/gpu-solver-partitions-default

## Story

As a **DGX operator running RL training at production env count**, I want
**the PhysX solver to use its default partitioning unless a single partition
is provably required**, so that **training isn't leaving GPU solver
parallelism on the table for a setting that may be a stale troubleshooting
leftover.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)

## Context

`_apply_procroom_physx_buffers` in
[`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
sets `gpu_max_num_partitions=1` (PhysX default is **8**). It was introduced in
commit `0437019` ("fix(physics): stabilize mecanum rollers at high env
counts") — the high-env-count **flip-prevention** work, bundled with the
`gpu_*_capacity` buffer increases that address GPU-buffer-overflow →
dropped-contacts → asymmetric wheel support → robot flipping at ~256 envs.

Two facts make this worth revisiting:

- **It is not the high-yaw-bounce knob.** That was fixed by the PGS solver
  switch ([`roller-contact-high-omega-bounce`](../../completed/roller-contact-high-omega-bounce.md));
  PGS does not depend on the partition count and the bounce does not return
  if this reverts. So "set it back now that the bounce is fixed" is sound
  reasoning — it was never load-bearing for the bounce.
- **Whether it is load-bearing for *flip* stability is unverified.** The real
  flip fix is the buffer-capacity increase; `gpu_max_num_partitions=1` was
  added in the same commit and may be belt-and-suspenders rather than
  necessary. A single partition serializes the constraint solve across the
  partition graph, which only costs at scale (256/4096 envs) — single-env
  probes show no difference.

So this is a scoped experiment: revert to the default and confirm flip
stability holds at production env count, banking the parallelism.

## Acceptance criteria

- [ ] Remove `gpu_max_num_partitions=1` from `_apply_procroom_physx_buffers`
      (let it fall back to the PhysX default of 8).
- [ ] Run a training pass at production env count and confirm **no rise in
      `Episode_Termination/robot_flipped`** vs the `=1` baseline, and record
      the throughput delta (steps/s).
- [ ] Land the outcome: if stable, keep the default (and delete the now-stale
      "single solver partition" comment); if flips return, restore `=1` **with
      a comment stating the flip rationale** (the current comment only says
      "single solver partition", not *why*).
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- `_apply_procroom_physx_buffers` (the `gpu_max_num_partitions=1` line + the
  flip-prevention comment block above it) in `strafer_env_cfg.py`.
- Commit `0437019` — the change that introduced it; check whether the buffer
  increases alone resolved the flips in that work or whether the partition
  pin was independently required.
- `solver_type` in the same block is PGS (shipped with the roller-bounce fix);
  re-validate flip stability under PGS, not the old TGS default.

## Out of scope

- The PGS solver switch and the high-yaw bounce — shipped separately.
- The other `gpu_*_capacity` buffer increases in the same block — those are
  the documented flip fix; this brief only revisits the partition count.
