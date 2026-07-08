# Train↔deploy parity JSONL schema

The contract both sides of the observation-parity check emit against: the
Jetson inference node (via the `obs_dump_path` parameter) and the DGX-side gym
dumper (to be written against this file). One JSON object per line; UTF-8; no
trailing commas.

## Obs-dump record

```json
{"t_sim": 12.3667, "variant": "NOCAM_SUBGOAL", "obs": [/* obs_dim floats */], "referent": {"x": 1.4, "y": -0.2, "frame": "map"}}
```

| Field | Type | Meaning |
|---|---|---|
| `t_sim` | float | **Sim-time seconds** — the join axis. On the Jetson this is the node clock under `use_sim_time` (i.e. the bridge `/clock`). The gym side must emit the same sim time as the step the obs belongs to. |
| `variant` | string | `PolicyVariant` name (`NOCAM`, `DEPTH`, `NOCAM_SUBGOAL`, `DEPTH_SUBGOAL`). Every record in a file must share one variant. |
| `obs` | float array | The **full** assembled, normalized obs vector — length `PolicyVariant[variant].obs_dim`. Never truncated (a DEPTH variant carries the whole depth block). Dimension order is exactly `PolicyVariant.fields`. |
| `referent` | object or null | The map-frame pose the goal-shaped obs fields were computed against — the rolling subgoal for `*_SUBGOAL` variants, the final goal otherwise. `{x, y, frame}`. Auxiliary (the obs already contains the derived triplet); used for debugging and by the self-check to source an exact referent. |

Both sides derive `obs_dim` and the field layout from `PolicyVariant` — never
from a hardcoded dimension. The depth block's size and resolution follow the
variant's `depth_image` field, so a resolution change (e.g. 80×60 → 80×45) needs
no change here.

## Join

Records are matched **nearest-timestamp on `t_sim` within ±POLICY_PERIOD_S/2**
(one half policy period; `POLICY_PERIOD_S = 1/30 s`). Node ticks with no
reference tick inside the tolerance — and reference ticks with no node tick —
are counted and reported as unmatched; they are never dropped from the
denominator. Below `MIN_MATCHED_FRACTION` (0.6) matched coverage, the parity
result is marked FAIL on coverage alone: a parity claim over a minority of ticks
is not a parity claim.

## Bounds

| Block | Bound | Rationale |
|---|---|---|
| Scalar dims (everything but `depth_image`) | ≤ **1e-5** max-abs-delta | float32 assembly noise. |
| Depth dims (`depth_image`) | ≤ **1e-3** max-abs-delta | renderer nondeterminism budget; reported separately. |
| Rolling subgoal position | ≤ **MAP_RESOLUTION·2 = 0.10 m** | half a costmap cell each side. |

## Depth spatial-residual report (depth variants)

The matched per-dim depth residual is reshaped to the variant's depth
resolution (H×W) and scored:

- **Row-structured** residual (per-row means vary far above the noise floor) →
  vertical-FOV geometry-mismatch signature.
- **Unstructured, time-varying** residual (large per-tick variation, flat
  spatial map) → frame-freshness-lag signature.

This is the discriminator between the two diagnosed DEPTH_SUBGOAL train↔deploy
depth-parity root causes. The verdict is a heuristic hint; the raw per-row /
per-column means are reported so an operator can eyeball the map.

## Cadence report

From the node dump's `t_sim` column, the inter-inference sim-time delta
histogram. Expected: a spike at `POLICY_PERIOD_S` for every variant — the
empirical proof that the depth freshness gate delivers training's
one-fresh-depth-per-step cadence in sim time regardless of wall RTF. A shifted
mode, gaps, or bursts is a cadence-parity concern.

## The two CLIs

- `obs_parity.py --node-dump NODE.jsonl --gym-dump GYM.jsonl` — the strict gate
  (scalar ≤1e-5, depth ≤1e-3). `--self-check --bag DIR` instead re-assembles the
  reference from a rosbag2's raw topics (`/d555/imu/filtered`,
  `/strafer/joint_states`, `/strafer/odom`, `/tf`, and depth for camera
  variants) through the node's own `obs_pipeline`, pinning assembly wiring with
  no DGX involvement. The self-check re-samples the bag (it does not replay the
  node's exact cached inputs) and takes the referent + `last_action` from the
  node dump, so it isolates the sensor→obs pipeline. Because re-sampling adds
  temporal deltas far above the strict bounds, the self-check defaults to
  **advisory wiring tolerances** (scalar ≤0.1; depth report-only — the depth
  spatial-residual structure, not the numeric bound, is the depth signal there)
  and prints a mode banner; the gym-dump join is the authoritative numerical
  gate. The node's obs dump is truncated per launch, so re-capture one run per
  file — a concatenated multi-run dump is flagged (non-monotonic `t_sim`) but
  its verdict is not trustworthy.
- `subgoal_parity.py --bag DIR` — bag-replay self-consistency of the rolling
  subgoal (≤0.10 m), replaying recorded `/plan` + `/tf` through the deployed
  numpy generator.

## Notes for the DGX gym dumper (goal-(a) deliverable, DGX lane)

Step the gym env alongside the bridge and, per env step, evaluate the same
`mdp/observations.py` terms the training group assembles, emit one record above
with the step's sim time as `t_sim` and the env's `SubgoalCommand`/goal pose as
`referent`. Match `variant` to the env id's policy variant. No depth field for
`NOCAM*`; the full depth block for `DEPTH*`.
