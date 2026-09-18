# Provenance — near-field depth convention reconciliation, 2026-09-13

captured_date: 2026-09-13

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim, probes, gates, replay | `gx10-d1d8` (DGX Spark, aarch64 GB10) | 7.0.0-1019-nvidia | 580.173.02 |

Everything in [`README.md`](README.md) was produced on this host. Nothing in this
record needed the robot; the one robot-side input is the 2026-08-22 observation
capture, reused rather than re-recorded.

## Trees

| item | value |
|---|---|
| Sim2RealLab, base of the work | `d1002a05564b7cc3139c210cb8614c044f4b1915` (`main`, merge of #218) |
| branch | `task/depth-nearfield-convention-mismatch` |
| Isaac Lab clone | `/home/zachoines/Documents/repos/IsaacLab` @ `v3.0.0-beta2.patch1` |

The canonical pair is the one #218 landed: Isaac Sim 6.0.1.0 with Isaac Lab
`v3.0.0-beta2.patch1`. The retired pair is not used anywhere in this record, and
§5 of the README is the consequence of that — the scene the 2026-08-22 probe
reproduced belongs to the retired pin's generation.

## Interpreter

`env_isaaclab3` on `gx10-d1d8`: python 3.12.13, numpy 2.3.1, torch 2.11.0+cu130,
onnxruntime 1.25.1, isaacsim 6.0.1.0, isaaclab 6.1.14. Every inference in this
record ran on `CPUExecutionProvider`.

`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` is set for the Kit-free runs, as
`env_setup.sh` does for this host. Both Kit-booting probes ran through
`tools/kit_boot_watchdog.sh`; relaunch counts are in the README's gate section.

The golden attribution and the convention A/B need only this interpreter and the
files below. `probes/golden_attribution.py` imports the contract test module by
path so the goldens' serialization is the shipped one; `probes/convention_ab.py`
imports the production noise model for the same reason.

## Artifacts

| artifact | sha256 |
|---|---|
| `models/strafer_depth_subgoal_v2_998.onnx` | `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165` |
| `models/strafer_depth_subgoal_v2_998.pt` | `03871cb5a09bffd2383994bd00f8245ea68a4aec1cac33f98a5ac6ae5d637b37` |
| `models/strafer_depth_subgoal_v1.pt` | `e8c55a890aa2fafd90ce68af3c87b4910ac4e7e40fe01b10d27109efbf1f13cc` |

The ONNX digest matches the one the attribution record published, so the patch
replays in §2 run against the same bytes that record measured. `models/` is
untracked repo content.

## Inputs this record reads from another record

The convention A/B's fixed scene is the 2026-08-22 attribution record's own
clean observation-term output, and its policy input is that record's robot-side
tick-0 observation. Both are cited by digest there and neither is re-derived
here:

| input | source |
|---|---|
| `same-pose-probe/gym_obs.jsonl`, `same-pose-probe/env_obsbuf.jsonl` | `goal-a-attribution-2026-08-22`, deposit `ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12` |
| `arm3-obs-capture/node_obs.jsonl` record 0 | same record's sibling deposit; extracted verbatim into `ab/node_obs_rec0.json` |

Restoring that record from its deposit is what makes §2 re-runnable:

```
cp -a <clone>/goal-a-attribution-2026-08-22/record-files/. \
    docs/measurements/goal-a-attribution-2026-08-22/
```

The restored copy is not part of any commit. The 2026-08-22 files were restored,
read, and left byte-identical; the same-pose probe re-run of §5 overwrites them,
so it was run against a copy taken aside first and the pristine deposit restored
afterwards — verified by digest.

The depth fingerprint tool is likewise another record's file, run unmodified:
`isaac-lab-upgrade-baseline-2026-08-14/record-files/depth_obs_stats.py`, sha256
`d7a1f32a2dc10dcda3b8b95f0002e933966ef077fcaae43a19ea77c5c1fd2567`. Its
pre-flip baseline numbers, quoted once in README §4 for comparison, come from
that record's own `render/` JSONs.
