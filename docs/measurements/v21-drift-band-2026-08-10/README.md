# v2.1 leg-1 drift-band evaluation — 2026-08-10

Deterministic clean-arm scores that set the referent-frame drift band to the
off-path corridor bound: a policy trained on the `(0.0, 1.25)` robust band
scored **0.360** completion where the drift-naive reference scored **0.723** at
matched budget on the same distribution.

| file | checkpoint | trained on drift | completion |
|---|---|---|---|
| `v21-leg1-model499-clean.jsonl` | `run_20260809_171025/model_499.pt` | yes, `(0.0, 1.25)` | 0.360 |
| `v2-model998-clean.jsonl` | `run_20260727_171735/model_998.pt` | no | 0.720 |
| `v2-model499-clean.jsonl` | `run_20260726_221955/model_499.pt` | no | 0.723 |

The third row is the control: matched at 499 iterations against the first, it
rules out training budget as the explanation. The near-identical 998-vs-499
reference scores also record that v2's second leg bought nothing on this
distribution.

Produced by `scripts/eval_cadence_emulation.py --profile clean --num_envs 16
--episodes 100 --seed 42` on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`.
`clean` is env-native DR only — the harness drift knob defaults off, so nothing
double-applies. One JSON object per arm; `episodes[]` carries per-episode cause,
progress fraction and tick accounting.
