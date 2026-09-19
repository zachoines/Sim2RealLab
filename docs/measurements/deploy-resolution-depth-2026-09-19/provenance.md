# Provenance — deploy-resolution depth in training, 2026-09-19

captured_date: 2026-09-19

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim / training | gx10-d1d8 (DGX Spark, GB10, aarch64, 124543 MiB unified) | 7.0.0-1019-nvidia | 580.173.02 |

`nvidia-smi` reports no GPU memory total on this part, so memory is sampled from
`--query-compute-apps=used_memory` and `/proc/meminfo`.

## Trees

| item | value |
|---|---|
| base commit | `b6975f5a4483b0e4519600f221b6d047dc1cea3d` (the #221 merge) |
| branch | `task/deploy-resolution-depth` |
| measurement head | `7a8778c4a9e545f5b399d05f81e9ccde74ccd78a` |
| mutation arm | a detached worktree at the measurement head, `torch.median` substituted for the even-count median; the live tree was verified clean afterwards |
| legacy-render arm | no tree change — the pre-change 80×45 camera is built in the probe's scratch cfg alongside the shipped one |

## Interpreter

| item | value |
|---|---|
| conda env | `env_isaaclab3` |
| python | 3.12.13 |
| numpy | 2.3.1 |
| torch | 2.11.0+cu130 |
| onnxruntime | 1.25.1 |
| Isaac Sim | 6.0.1.0 |
| Isaac Lab | v3.0.0-beta2.patch1 (`~/Documents/repos/IsaacLab`) |
| LD_PRELOAD | `/lib/aarch64-linux-gnu/libgomp.so.1` |

CPU-only work ran with `PYTHONPATH` set to the measurement tree's
`source/strafer_lab` and `source/strafer_shared`; both packages were confirmed to
resolve there rather than into an editable install pointing elsewhere.

Kit-booting work ran through `tools/kit_boot_watchdog.sh`, one actor at a time,
with `nvidia-smi --query-compute-apps` confirmed empty before each boot and the
GPU released afterwards.

## Inputs read from other records

| input | from | what it is |
|---|---|---|
| `bench/direction_a.patch` | `depth-noise-coverage-2026-09-18` | the scratch implementation the reduction is taken from, deposited and never merged |
| `bench/summary_baseline80x45.txt`, `bench/summary_directionA640x360.txt` | `depth-noise-coverage-2026-09-18` | the 90.039 s and 103.003 s per-iteration means, and their memory peaks |
| `coverage/coverage_curve.json` | `depth-noise-coverage-2026-09-18` | the capture's own per-band residual column, used for scale in §4 |
| `probes/golden_attribution.py` | `depth-noise-coverage-2026-09-18` | the attribution walker, carried forward unchanged |
| `texture/texture_structure.json` | `noise-texture-parity-2026-09-17` | the 0.000524 whole-frame valid-only residual the pre-registration named |
| `same-pose-probe/` | `goal-a-attribution-2026-08-22` | the 30-frame anchor harness whose pose could not be re-rendered |

None of these are re-deposited here; each is cited through its own record's
deposit.

## Artifact

The 20-iteration smoke checkpoint and its export are in this record's deposit,
not in the model registry — they are a path check, not a trained artifact.

| file | sha256 |
|---|---|
| `smoke/export/depth_subgoal_v3_smoke.pt` | see `DEPOSIT.md` |
| `smoke/export/depth_subgoal_v3_smoke.onnx` | see `DEPOSIT.md` |
| `smoke/runs/run_20260919_174852/model_19.pt` | see `DEPOSIT.md` |

`DEPOSIT.md` carries the sha256 of every one of the 111 deposited files; the
record's evidence section cites the deposit commit that fixes those bytes.
