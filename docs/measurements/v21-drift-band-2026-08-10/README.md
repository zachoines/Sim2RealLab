# v2.1 leg-1 drift-band evaluation — 2026-08-10

Deterministic clean-arm scores that set the referent-frame drift band to the
off-path corridor bound: a policy trained on the `(0.0, 1.25)` robust band
scored **0.360** completion where the drift-naive reference scored **0.723** at
matched budget on the same distribution.

## Evidence deposit

The three result files this record scores are not in this directory. They are
in the companion evidence repository
`https://github.com/zachoines/Sim2RealLab-Artifacts`, a private repository
holding the evidence behind these records, under
`v21-drift-band-2026-08-10/record-files/`, deposited at commit
`ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12`. That directory mirrors this one,
so a filename below names the same file there; restoring it puts the files back
where this record names them:

```
cp -a <clone>/v21-drift-band-2026-08-10/record-files/. \
    docs/measurements/v21-drift-band-2026-08-10/
```

The deposit's own `DEPOSIT.md` comes across with the files and is not part
of the record.

sha256 of every deposited file:

```
1eeaa306112f8d39d4263b39b180f98ad5aa1b6c63c42e8810a4fdf107e76325  v2-model499-clean.jsonl
40656e055f32309584ece0d3df7d1741c8d6ae6049af06b8290033a330c38a1d  v2-model998-clean.jsonl
e56a7d973dd59c4bdcb62af94686a9491a8727e7db286f64f6969fb54fff06f4  v21-leg1-model499-clean.jsonl
```

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
