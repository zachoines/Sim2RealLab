# Stage 3 provenance

| item | value |
|---|---|
| host | `gx10-d1d8`, NVIDIA GB10, driver 580.82.09 / CUDA 13.0 |
| date | 2026-08-23, 10:31–13:54 local |
| Sim2RealLab | `ab2daedff174b9557f447546a7b412cbd948133d` (`main`) |
| **NEW pair** | conda `env_isaaclab3beta2` (Python 3.12.13) / `~/Documents/repos/IsaacLab-3beta2` |
| new clone SHA | `ffff603eafc6b74264a5261cc0183d6a65390d78` = tag `v3.0.0-beta2.patch1` |
| new clone state | **pristine** — zero modifications before and after the session |
| `isaaclab.__version__` (new) | 6.1.14 |
| new stack | torch 2.11.0+cu130 (CUDA available), isaacsim 6.0.1.0, rsl-rl-lib 5.4.2, onnxscript 0.7.1, onnxruntime 1.25.1, warp-lang 1.13.0, tensordict 0.14.0, torchcodec 0.16.0 |
| **OLD pair** (control only, read-only) | conda `env_isaaclab3` / `~/Documents/repos/IsaacLab` @ `ae41e2aca68bcf06cb6ea02dd6618e4ddb16e1da` |
| old stack | torch 2.10.0+cu130, isaaclab 4.6.12, isaacsim 6.0.0.0 |
| `.env` | **not modified** — still points at the old pair for the whole session |
| preserved anchors | `~/Documents/upgrade_baseline_artifacts/`, 20/20 verify against `SHA256SUMS.txt` |
| new artifacts | `~/Documents/upgrade_stage3_artifacts/` (Gate B re-export) |

Full interpreter state: [`pip-freeze-env_isaaclab3beta2.txt`](pip-freeze-env_isaaclab3beta2.txt).

## Binding proof

Every Kit run in this record logs `sys.executable` and `isaaclab.__file__` at the top of its
output. A mixed-pair run is the silent failure the two-shell design guards against, and one was
produced deliberately during setup to confirm the guard is needed — see the README's
"Binding" section.

- new pin: `…/envs/env_isaaclab3beta2/bin/python` + `…/IsaacLab-3beta2/source/isaaclab/isaaclab/__init__.py`
- old pin: `…/envs/env_isaaclab3/bin/python` + `…/repos/IsaacLab/source/isaaclab/isaaclab/__init__.py`
