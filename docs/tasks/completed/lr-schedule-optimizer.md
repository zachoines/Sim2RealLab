# Drive `--lr_schedule` through the optimizer

**Status:** Shipped 2026-08-15 in `49fbdd4` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/206

**Type:** bug fix (one mechanism + the regression test that was missing)
**Owner:** DGX (`strafer_lab` lane — training entry point)
**Priority:** P1 — the flag is a prerequisite for any annealing leg, and its
failure mode is silent: a run using it produces a clean decay curve in
TensorBoard and a constant learning rate in Adam, so the leg reads as a
completed null result rather than a broken one.
**Estimate:** S (one session, no training run, no GPU)
**Branch:** `task/lr-schedule-optimizer`

## Story

As a **DGX operator preparing an annealing leg**, I want **`--lr_schedule` to
change the rate Adam actually steps at**, so that **a leg that reports a decay
curve is a leg that ran one, and a null result from it is evidence about the
recipe rather than about the plumbing.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)

## Context

`--lr_schedule` was added on 2026-03-29 in `b366b6d` and its central line has
not changed since. The block forces `agent_dict["algorithm"]["schedule"] =
"fixed"` so RSL-RL's KL-adaptive controller cannot fight the decay, then
monkey-patches `runner.alg.update` to assign `runner.alg.learning_rate` each
iteration.

That assignment reaches nothing. In rsl-rl-lib 5.0.1 there is exactly one site
that writes `optimizer.param_groups[...]["lr"]` — `algorithms/ppo.py:293-294`
— and it sits inside the guard at `ppo.py:269`, `if self.desired_kl is not None
and self.schedule == "adaptive"`. Forcing `"fixed"` makes that branch dead for
the whole run, and `torch.optim.Adam` reads `group["lr"]` and nothing else. The
same guard shape is mirrored in the auxiliary PPO copy at
`agents/strafer_ppo.py:182`, so the propagation path is closed there too.

The failure is worse than a no-op because `on_policy_runner.py:122` logs
`self.alg.learning_rate` — the attribute the schedule does write. TensorBoard's
`Loss/learning_rate` therefore shows a textbook cosine decay from 3e-4 to 1e-5
across a run in which every gradient step was taken at 3e-4.

A second, independent defect sits in the same block. Progress was
`min(it / max(max_iterations, 1), 1.0)` with `it =
runner.current_learning_iteration`. `learn()` iterates
`range(start_it, start_it + num_learning_iterations)`, so on a resume the
numerator is an absolute iteration index and the denominator is the budget for
one leg. Resuming from `model_9999.pt` with `--max_iterations 6000` gives
`progress = 1.0` on the first update and `lr_min` for every iteration after it.
The counter compounds this: the runner assigns `current_learning_iteration`
after `alg.update()` returns, so a schedule reading it from inside the patched
update sees the previous iteration's index.

Nothing recorded is invalidated. A grep across `docs/`, the cheatsheet, the
READMEs and the full git history finds no run, measurement or brief that ever
passed the flag; the only mention is a proposed fallback in
[`procroom-depth-enrichment`](../active/trained-policy/procroom-depth-enrichment.md)
that was never executed. Every recorded leg trained at a constant 3e-4, which
is what the configs say and what everyone believed.

## Resume semantics

`--lr_schedule` with `--resume` continues the curve rather than restarting it.
Progress is measured against `start + max_iterations` — the last iteration the
invocation will reach — so a leg resumed from 4000 with `--max_iterations 6000`
traverses the tail of a 10000-iteration curve and lands on `lr_min` at its end.

The alternative reading — anneal from `lr_init` across the new segment — was
rejected because it makes a resumed decay non-monotone: the rate jumps back to
the initial value at every resume, which is a warm restart rather than a decay,
and it makes the rate at a given absolute iteration depend on how the run
happened to be split. The chosen reading keeps the schedule monotone across
resumes and reproduces the intended curve exactly for the crash-resume case.

For the same reason the decay is anchored at the configured rate, not at the
checkpoint's. The anchor is the curve's value at absolute iteration 0; taking
it from the restored optimizer state would re-anneal an already-decayed rate
and step the LR down discontinuously at each resume.

## Acceptance criteria

- [x] The rate the schedule computes is written to
      `optimizer.param_groups`, not only to `alg.learning_rate`, and the two
      agree so the logged value is the applied value.
- [x] The rate is applied before the update it governs, so the first update of
      a run — including a resumed one, where `load()` has restored the
      checkpoint's optimizer rate — runs at its scheduled rate.
- [x] Progress is measured against the iterations the invocation will execute,
      and the iteration index does not lag the loop.
- [x] A pure test (no Kit, no GPU) drives the patched update against a real
      `torch.optim.Adam` and asserts on `optimizer.param_groups`, never on
      `alg.learning_rate`. It fails against the pre-fix logic and passes after.
- [x] The rest of the CLI override block is audited for the same defect class —
      a value written where the framework reads it only under a condition the
      override itself disables — and the result recorded whether or not it is
      empty.
- [x] Pure suite and the contract two-file gate hold at their counts.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Audit of the override block

Every CLI override in the script was traced to the code that consumes it. The
`--lr_schedule` block is the only one with an inert write. Findings that are
not defects but are worth having on the record:

- **`schedule="fixed"` is redundant against the config but still load-bearing.**
  All three PPO cfg sites already ship `schedule="fixed"`, so the CLI line
  changes nothing today. It stays because it is what makes the schedule
  authoritative if a config ever moves to `"adaptive"`: the KL controller would
  otherwise overwrite `param_groups` from inside the same update.
- **`desired_kl=0.01` is inert config at all three PPO cfg sites.** It is read
  only under `schedule == "adaptive"`, which none of them selects, so it reads
  as an active KL target and is not one. Deliberately not touched here — an
  in-flight PR owns that file's comment block and the rationale for `fixed`.
- **`install_strafer_ppo()` composes correctly but the ordering is undefended.**
  It patches the `PPO.update` class attribute and must run before the runner is
  built; it does, so the bound method the LR wrapper captures is the
  aux-patched one and the layers nest correctly. Nothing enforces the order —
  moving the install below the runner construction would silently drop every
  auxiliary loss from the gradient with no error and no log line.
- **Auxiliary anneal state is not in the checkpoint.** `DAPGAuxiliary` keeps
  its own `update_count` as the sole clock for the BC weight anneal, and GAIL's
  discriminator and its optimizer are neither saved nor loaded, so a resumed
  `--aux` leg restarts the anneal at full weight and trains a fresh
  discriminator against a mature policy. Moot rather than filed: the auxiliary
  modules are being removed.
- **The runner mutates the config dict it is handed.** `OnPolicyRunner` holds
  `train_cfg` by reference and `construct_algorithm` pops keys out of it, so
  reads of `agent_dict` after construction see a mutated dict. Algorithm,
  actor and critic sub-dicts are frozen into PPO attributes at construction;
  `num_steps_per_env`, `save_interval` and `check_for_nan` are read lazily in
  `learn()` and stay editable. The schedule now takes its anchor from
  `runner.alg.learning_rate` rather than the dict.
- **Resuming replays an iteration.** `save()` stores the last completed index
  and `load()` restores it as `start_it`, so resuming from `model_N.pt` re-runs
  iteration N. Upstream behaviour, unchanged here, noted because it shifts a
  resumed schedule by one iteration.

## Out of scope

- `rsl_rl_ppo_cfg.py`. The `desired_kl` finding above is reported, not fixed;
  an in-flight PR is editing that file and the rationale for `schedule="fixed"`
  is that PR's residual.
- Any change to a default learning rate, schedule choice or `max_iterations`.
  This repairs the mechanism; the recipe is a separate lane.
- The auxiliary resume gap, which the auxiliary-module removal supersedes.
- Restoring `alg.learning_rate` from a checkpoint. It is not serialized by
  `PPO.save()`, and making it so would change the checkpoint format.
