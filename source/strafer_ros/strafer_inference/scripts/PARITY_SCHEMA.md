# Train↔deploy parity JSONL schema

The contract both sides of the observation-parity check emit against: the
Jetson inference node (via the `obs_dump_path` parameter) and the
workstation-side gym dumper (to be written against this file). One JSON object
per line; UTF-8; no trailing commas.

## Arming the node side

`obs_dump_path` is read **once at node init**, so it must be set at launch —
`ros2 param set` is too late and does nothing. Empty (the default) disables it
with zero per-tick cost.

| Lane | How |
|---|---|
| bare-metal / `ros2 launch` | `ros2 launch strafer_inference inference.launch.py obs_dump_path:=/tmp/node_obs.jsonl` |
| containerized (`inference` service) | `STRAFER_OBS_DUMP_PATH=/obs_dumps/node_obs.jsonl` → `inference_policy.launch.py` → `inference.launch.py` → the node param. Uncomment it plus the writable bind in `deploy/docker-compose.override.sim-bridge.yml`, then `up -d --force-recreate inference` (**not** `restart` — that reuses the old container env). |

Confirm with `docker logs strafer_inference 2>&1 | grep 'obs dump ENABLED'`.

There is no separate dump-variant knob: the node stamps every record with its
loaded `policy_variant` (`STRAFER_POLICY_VARIANT`), so a dump cannot disagree
with the artifact that produced it.

**Not for normal missions.** A `DEPTH`/`DEPTH_SUBGOAL` variant writes a full
depth vector per tick at 30 Hz (~2–3 MB/s of JSONL). Arm it for a capture, then
unset and force-recreate. The file is truncated per launch, so capture one run
per file.

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
| Depth dims (`depth_image`) | ≤ **1e-3** max-abs-delta | renderer nondeterminism budget; reported separately. **Known not to pass and not currently achievable**: a max-abs bound over 3600 dims is set by single pixels at depth discontinuities, which no reduction removes (measured worst per-dim 6.34e-01 with the shipped block median, 7.81e-01 with the block mean it replaced). Judge depth on the distributional figure — overall mean \|Δ\|, currently **6.19e-04** — until this bound is replaced by a percentile one. |
| Rolling subgoal position | ≤ **MAP_RESOLUTION·2 = 0.10 m** | half a costmap cell each side. |

## Depth spatial-residual report (depth variants)

The matched per-dim depth residual is reshaped to the variant's depth
resolution (H×W) and scored:

- **Row-structured** residual (per-row means vary far above the noise floor) →
  vertical-FOV geometry-mismatch signature.
- **Unstructured, time-varying** residual (large per-tick variation, flat
  spatial map) → frame-freshness-lag signature.

This distinguishes a depth geometry mismatch from a frame-freshness lag. The
verdict is a heuristic hint; the raw per-row / per-column means are reported so
an operator can eyeball the map.

**Both structure scores are RATIOS to the overall mean residual, so they are
scale-free — do not use them to compare two runs.** Fixing the dominant residual
makes every score go *up*. Measured when the deploy reduction changed from a
block mean to a block median (2026-07-31 capture, 2245 joined ticks): overall
mean |Δ| **4.268e-03 → 6.187e-04** (6.9× better) and the absolute std of the
per-row means **4.765e-03 → 1.960e-03** (2.4× better), while `row_structure`
went **1.116 → 3.168**. Compare `overall_mean` and the absolute per-row spread;
read the structure scores only to locate *where* a residual lives, never to
judge whether it shrank.

Below an overall mean |Δ| of `_RESIDUAL_FLOOR` (1e-3) the verdict says so
explicitly and reports no signature, so the tool stops naming a geometry
mismatch at a residual an order of magnitude under the cross-camera floor.

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
  no workstation involvement. The self-check re-samples the bag (it does not replay the
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

## Notes for the workstation gym dumper

The gym half is implemented in bridge mode of `run_sim_in_the_loop.py`
(`--obs-dump-path` / `--obs-dump-variant`), backed by
`strafer_lab.bridge.obs_dump`. Per env step it evaluates the same
`mdp/observations.py` terms the training group assembles — IMU accel/gyro
(`d555_imu`), wheel-encoder velocities and body velocity (`robot`), and, for
`DEPTH*`, the depth term against the policy camera (`d555_camera`) — runs them
through `assemble_observation`, and emits one record above stamped with the
step's `/clock` sim time. Match `variant` to the loaded artifact's variant.

The referent-derived triplet (`goal_*` / `subgoal_*`) and `last_action` are
**NaN-filled**: in bridge mode the rolling subgoal is picked by the Jetson
generator off its own planned path (not by this env), and `last_action` is
node-internal. `referent` is therefore `null`. The join masks NaN dims from the
pass/fail bound and reports them as masked — those dims are accounted for, not
silently passed. No depth field for `NOCAM*`; the full depth block for `DEPTH*`.
