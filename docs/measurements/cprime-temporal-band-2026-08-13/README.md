# Temporal-band leg — selection sweep and acceptance grid, 2026-08-13

Scores for the leg trained with the robust stream law held to the realistic
tier, on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, 16 envs
x 100 episodes, deterministic policy. The leg is `run_20260812_230811`; the
reference is `run_20260727_171735/model_998.pt`.

The run does not plateau — completion falls about 0.19 over its last hundred
iterations while entropy keeps dropping — so the checkpoint is selected rather
than taken as the final one. Selection ran on **seed 7**, scoring on **seed
42**, disjoint so the selection cannot flatter the score.

## Selection (seed 7)

| file | checkpoint | completion |
|---|---|---|
| `select-seed7-model100.jsonl` | `model_100` | 0.180 |
| `select-seed7-model200.jsonl` | `model_200` | 0.560 |
| `select-seed7-model300.jsonl` | `model_300` | 0.660 |
| `select-seed7-model400.jsonl` | **`model_400`** | **0.770** |
| `select-seed7-model499.jsonl` | `model_499` | 0.660 |

## Acceptance grid (seed 42, `model_400`)

| file | arm | completion |
|---|---|---|
| `grid-seed42-REF-v2-model998.jsonl` | reference, clean | **0.810** |
| `grid-seed42-cprime-model400-clean-band-degraded.jsonl` | clean / band / degraded | **0.640** / 0.710 / 0.450 |
| `grid-seed42-drift1x-v2-model998.jsonl` | reference, fixed-gain 1x drift | **0.644** |
| `grid-seed42-drift1x-cprime-model400.jsonl` | leg, fixed-gain 1x drift | **0.530** |
| `grid-seed42-T2D2-cprime-model400.jsonl` | leg, stale 0.76 run 4 | 0.750 |

Three results this set carries.

The selected checkpoint scores **0.770 on the selection seed and 0.640 on the
scoring seed** — about 2.7 standard errors apart at this sample size. Selecting
the maximum over five checkpoints inflates the estimate, and the disjoint
scoring seed is what exposes it.

The **fixed-gain 1x arms are the first true fixed-gain measurement on a tier
that drifts natively**: both carry `env_drift_active: false`, so the harness's
gain is the only drift present rather than riding on the environment's own
band. The policy trained with drift scores **below** the policy that never
trained with it, 0.530 against 0.644.

The `degraded` arm runs a hold fraction of 0.583, above the 0.35 this leg
trained on, so that row measures behaviour outside the training band.

Every record carries `env_drift_active`, `harness_drift_gain` and `seed`, so a
file states which quantity it holds without reference to this table.
