# v2.1(a) narrow-band leg — acceptance grid and checkpoint sweep, 2026-08-12

Scores for the leg trained on the narrowed referent-drift band, its
reference re-measure, and the checkpoint sweep that established the
instability finding. All arms on the **`(0.0, 0.5)` drift env** —
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, `--profile
clean`, 16 envs x 100 episodes, seed 42, deterministic policy.

## Evidence deposit

The eight result files this record scores are not in this directory. They are
in the companion evidence repository
`https://github.com/zachoines/Sim2RealLab-Artifacts`, a private repository
holding the evidence behind these records, under
`v21a-narrow-band-2026-08-12/record-files/`, deposited at commit
`ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12`. That directory mirrors this one,
so a filename below names the same file there; restoring it puts the files back
where this record names them:

```
cp -a <clone>/v21a-narrow-band-2026-08-12/record-files/. \
    docs/measurements/v21a-narrow-band-2026-08-12/
```

The deposit's own `DEPOSIT.md` comes across with the files and is not part
of the record.

sha256 of every deposited file:

```
becb97729cf9403a82477af0e568d523dfae481a1442989f995b5f66e2999d7f  v2-model499-clean.jsonl
d2db8596888aac10abca3ac73528e93c245377329d4a8dca16977d539db560fb  v2-model998-clean-REF.jsonl
429fd8d0131df46bfe3e0336114f52168b401c5b83d1ef83d7542d0b0c5d058a  v21-model400-clean.jsonl
ca38431b4fba43f0b4aa8af0b7615c16b800c0294bc16436471b6cb5594c9b56  v21-model499-clean.jsonl
4d5ab84f2201a7576d0e297e4040d22c1c88c1b8b26cd3e645896ec14aa2cbb0  v21a-model200-clean.jsonl
490a9f4fa7d2b1301dab7bc975f8f070d377492b2fecdb8709944daaa81e40f8  v21a-model300-clean.jsonl
c1ed2217f1841388bf846d9eff9617cf52f032641e53c6d4f410f0556341bbd4  v21a-model400-clean.jsonl
5b91d2f537e82d90502fb263b5383fdd06f6bf5cd6463c98c90104d1a2e89ebe  v21a-model499-clean-band-degraded.jsonl
```

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
