# Record which task a training run actually used

**Type:** task (tooling / reproducibility)
**Owner:** DGX
**Priority:** P2 — nothing is broken, but the gap costs a multi-agent
investigation every time a policy's provenance is questioned, and it already has
once.
**Estimate:** S
**Branch:** `task/training-run-provenance-manifest`

## Story

As **anyone asking "what distribution was this policy trained on?"**, I want
**the run directory and the export sidecar to answer it**, so that **the answer
is a lookup rather than an archaeology exercise across Kit logs and commit
dates.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The 2026-08-01 anchoring re-validation turned on whether v2 trained on the
vanilla or the enriched ProcRoom generator — the difference decides whether the
session's v1-vs-v2 contrast is confounded. **That question could not be answered
from the repo at all.** It took two verification agents and was ultimately
settled from Isaac Sim Kit logs on the DGX, outside version control.

Three concrete defects, each independently sufficient to lose the answer:

1. **The export sidecar's `env_id` cannot identify a training run.**
   `export_policy.py` writes the *export-time* `--env`, which defaults to
   `_DEFAULT_ENV_BY_VARIANT[...]`. Every depth-subgoal artifact therefore carries
   the identical string `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0` — the
   **vanilla** Play env — including `v0`, whose `obs_dim` 4819 predates the
   depth-subgoal env rework. The field's docstring calls it *"the Gym task ID the
   checkpoint trained against"*, which is not what the code writes.
2. **The run directory carries no manifest.** `train_strafer_navigation.py`
   prints `env_name` to stdout and persists nothing; `log_dir` is a bare
   `run_<%Y%m%d_%H%M%S>`. Resume chains are likewise unrecorded, so a
   two-leg run's final checkpoint does not name the leg it continued.
3. **`run_sim_in_the_loop.py`'s `--task` help omits the enriched bridge task.**
   It names only `…ProcRoom-v0` although `…ProcRoom-Enriched-v0` is registered.
   This is a likely contributor to the 2026-08-01 session running vanilla.

**Kit logs are the recovery path for historical runs** — they capture each run's
full command line and match run directories to the second. That worked here
(`kit_20260726_221941.log` ↔ `run_20260726_221955`), but it depends on logs
living outside the repo, on one machine, and it does not survive a wipe.

**A second, better recovery path exists and is in the repo's own tree.** The
training stdout logs sitting beside the run directories record the task id
outright, and the run directory they belong to, a few lines apart:

```
logs/rsl_rl/strafer_navigation/depth_subgoal_cprime_stdout.log
:56   Environment: Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0
:213  Logging to: .../logs/rsl_rl/strafer_navigation/run_20260812_230811
```

The same pairing maps the other legs (`depth_subgoal_vfov8045_stdout.log`,
`depth_subgoal_v2_stdout.log`, `depth_subgoal_v2b_stdout.log`). So provenance for
historical runs is recoverable **today**, without Kit logs, for any run whose
stdout was captured — which is stronger than this brief assumed when it was
filed. Two consequences: the recovery procedure this brief promises to record
should name these files first, and the manifest work should consider simply
formalising what the stdout header already prints rather than inventing a new
schema beside it.

**Why this is not hypothetical.** On 2026-09-13 a checkpoint from
`run_20260812_230811` was loaded into `Isaac-Strafer-Nav-RLDepth-Real-Play-v0` for a
visual check. Both sides are 3619-dim observation / 3-dim action, so
`runner.load()` succeeded in silence; the policy was a rolling-subgoal,
enriched-scene, robust-DR one being asked to drive a fixed-goal, vanilla,
realistic env, and it simply performed badly with no diagnostic. The mismatch was
identified only by reading the stdout log above. A manifest plus a load-time check
would have caught it at the first line.

## Acceptance criteria

- [ ] `train_strafer_navigation.py` writes `<log_dir>/train_manifest.json` at run
      start, carrying at minimum: **task id**, **git SHA** (plus dirty flag),
      **seed**, **num_envs**, **max_iterations**, and the **resume chain** (the
      `--resume` checkpoint path, if any) so multi-leg runs are traceable to
      their first leg.
- [ ] `export_policy.py` records the **training env and the export env as
      separate fields**, sourcing the training env from the checkpoint's
      `train_manifest.json` when present. Existing sidecars have no training
      field; absent-and-explicit is better than present-and-wrong.
- [ ] The `env_id` docstring is corrected to describe what the code actually
      writes, or the field is renamed to make its meaning unambiguous.
- [ ] `run_sim_in_the_loop.py`'s `--task` help lists **every** registered bridge
      task, including `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`.
- [ ] A note in the brief (or a context module) records the **provenance recovery
      path for runs predating the manifest**, with the matching procedure —
      naming the in-tree `logs/rsl_rl/**/*_stdout.log` headers first (they carry
      `Environment: <task id>` and `Logging to: <run dir>`) and the machine-local
      Kit logs as the fallback.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Investigation pointers

- `source/strafer_lab/scripts/export_policy.py` — `_DEFAULT_ENV_BY_VARIANT`, the
  `env_id` docstring, and the sidecar payload assembly.
- `source/strafer_lab/scripts/train_strafer_navigation.py` — `log_dir`
  construction; `env_name` is printed but never persisted, and the runner
  receives only the agent dict.
- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — the `--task` argument
  help text.
- Worked example of the gap and its Kit-log resolution:
  [`enriched-scene-anchoring-addendum`](enriched-scene-anchoring-addendum.md).

## Out of scope

- Back-filling manifests for historical runs. Kit logs cover those; the point of
  this brief is that future runs should not need them.
- Any change to training behaviour, configs or defaults — this is provenance
  recording only.
