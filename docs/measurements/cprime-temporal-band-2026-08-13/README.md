# Temporal-band leg — selection sweep and acceptance grid, 2026-08-13

Scores for the leg trained with the robust stream law held to the realistic
tier, on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, 16 envs
x 100 episodes, deterministic policy. The leg is `run_20260812_230811`; the
reference is `run_20260727_171735/model_998.pt`.

The run does not plateau — completion falls about 0.19 over its last hundred
iterations while entropy keeps dropping — so the checkpoint is selected rather
than taken as the final one. Selection ran on **seed 7**, scoring on **seed
42**, disjoint so the selection cannot flatter the score.

## Evidence deposit

The ten result files this record scores are not in this directory. They are in
the companion evidence repository
`https://github.com/zachoines/Sim2RealLab-Artifacts`, a private repository
holding the evidence behind these records, under
`cprime-temporal-band-2026-08-13/record-files/`, deposited at commit
`ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12`. That directory mirrors this one,
so a filename below names the same file there; restoring it puts the files back
where this record names them:

```
cp -a <clone>/cprime-temporal-band-2026-08-13/record-files/. \
    docs/measurements/cprime-temporal-band-2026-08-13/
```

The deposit's own `DEPOSIT.md` comes across with the files and is not part
of the record.

sha256 of every deposited file:

```
09b5d665d2a6141d8359f25cd7787e4b59fcede44ed37fa4b55d4eaf894fc07c  grid-seed42-REF-v2-model998.jsonl
4326f45c81fb9e816b3024e8e68a7294a1f12862cf902459197f030c6ef070b7  grid-seed42-T2D2-cprime-model400.jsonl
8e9facd9d6a75d5fcec7823f01dbf73f880a62fae267032392b68f0999f46451  grid-seed42-cprime-model400-clean-band-degraded.jsonl
88d35838929fd43d2cb4a48a90c8580071b6ef54250636488d170a8f958f78ea  grid-seed42-drift1x-cprime-model400.jsonl
6f739c394ad9d90aa75191e69a517c989ccbb64ff57e42d36696b4230a3734bb  grid-seed42-drift1x-v2-model998.jsonl
bae64659a6a6227ae7b211f4576bc7b113b4bd1bd2b5a21794488cc0047e91fa  select-seed7-model100.jsonl
8b1a15fba2b0f2f7a99c0522f25c16e25a7be38ab79b000f5bff066af8bba12a  select-seed7-model200.jsonl
23bd5a5bd12667ea342c26f32260dec8d7097f45c8a4e6ca11151c16aeba208b  select-seed7-model300.jsonl
3efb23fad48818e7351e592abbd864f28a02810070d5392ea2353130107417e7  select-seed7-model400.jsonl
6708c4d1ce79944453ef3505d5087abb2a0bf2841cac9680ba410fece68988e8  select-seed7-model499.jsonl
```

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
