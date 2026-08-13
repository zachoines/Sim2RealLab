# v2.1(a) narrow-band leg — acceptance grid and checkpoint sweep, 2026-08-12

Scores for the leg trained on the narrowed referent-drift band, its
reference re-measure, and the checkpoint sweep that established the
instability finding. All arms on the **`(0.0, 0.5)` drift env** —
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, `--profile
clean`, 16 envs x 100 episodes, seed 42, deterministic policy.

| file | checkpoint | trained with | completion | offset | left |
|---|---|---|---|---|---|
| `v2-model998-clean-REF.jsonl` | `run_20260727_171735/model_998.pt` | no holds, no drift | **0.870** | +3.32° | 0.591 |
| `v2-model499-clean.jsonl` | `run_20260726_221955/model_499.pt` | no holds, no drift | 0.730 | +2.84° | 0.558 |
| `v21-model400-clean.jsonl` | `run_20260809_171025/model_400.pt` | holds + drift (0, 1.25) | 0.550 | −14.50° | 0.198 |
| `v21-model499-clean.jsonl` | `run_20260809_171025/model_499.pt` | holds + drift (0, 1.25) | 0.350 | +3.14° | 0.563 |
| `v21a-model200-clean.jsonl` | `run_20260810_192018/model_200.pt` | holds + drift (0, 0.5) | 0.370 | +13.01° | 0.720 |
| `v21a-model300-clean.jsonl` | `run_20260810_192018/model_300.pt` | holds + drift (0, 0.5) | 0.380 | +2.89° | 0.600 |
| `v21a-model400-clean.jsonl` | `run_20260810_192018/model_400.pt` | holds + drift (0, 0.5) | 0.330 | +0.72° | 0.523 |
| `v21a-model499-clean-band-degraded.jsonl` | `run_20260810_192018/model_499.pt` | holds + drift (0, 0.5) | **0.140** | +27.45° | 0.925 |

The last file carries three arms (clean 0.140 / band 0.160 / degraded 0.180).
Those ratios exceed 1.0 against their own clean because the baseline is at the
floor — read them as uninformative, not as robustness.

Two results this set establishes. The adopted band costs a competent policy
about 0.03 (REF 0.870 against ~0.900 undrifted), so it is not the constraint.
And the runs that train on it do not converge to a stable policy: completion
swings 0.20 within a run across 99 iterations, and the directional bias swings
sign (−14.5° right to +27.5° left), while the reference is steady at +2.8° to
+3.3° at every checkpoint.

Earlier figures for these same checkpoints measured on the pre-narrowing
`(0.0, 1.25)` env live in `../v21-drift-band-2026-08-10/` and are a different
distribution — do not mix the two sets.
