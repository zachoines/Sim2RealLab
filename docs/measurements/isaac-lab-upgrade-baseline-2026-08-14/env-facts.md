# Environment facts at the pre-bump pin

State of the machine-local Isaac stack as measured on 2026-08-14, recorded so a
later migration inherits these as known conditions rather than rediscovering
them as surprises. Nothing here was changed by the capture.

## Interpreter and pins

| item | value |
|---|---|
| conda env | `env_isaaclab3` |
| python | 3.12.13 |
| Isaac Lab checkout | `~/Documents/repos/IsaacLab` @ `ae41e2aca68` (branch `develop`, committed 2026-04-23) |
| checkout `VERSION` file | `3.0.0` |
| `isaaclab.__version__` | `4.6.12` |
| `importlib.metadata.version("isaaclab")` | **`4.5.24`** — resolves to the stale dist-info, not the live one |
| isaacsim | 6.0.0.0 (all sub-packages) |
| torch / torchvision / torchaudio | 2.10.0+cu130 / 0.25.0+cu130 / 2.10.0+cu130 |
| rsl-rl-lib | 5.0.1 |
| onnx / onnxruntime / onnxscript | 1.21.0 / 1.25.1 / **0.6.2** |
| numpy | 2.3.1 |
| warp-lang | 1.12.0 |
| newton | 1.0.0 |
| gymnasium | 1.2.1 |
| driver / CUDA | 580.82.09 / 13.0 |
| GPU | NVIDIA GB10, 93414 MB reported to Kit, 20 logical cores, 124552 MB system |

Full freeze: [`pip-freeze-env_isaaclab3.txt`](pip-freeze-env_isaaclab3.txt) (334 entries).

`onnxscript` at 0.6.2 is the version whose `aten_gru` omits `linear_before_reset=1`.
The dynamo export path must not be enabled on this env.

## Duplicate `isaaclab` dist-info

Two editable installs of the same package coexist, both pointing at the same
source directory:

```
site-packages/isaaclab-4.5.24.dist-info   direct_url -> .../IsaacLab/source/isaaclab
site-packages/isaaclab-4.6.12.dist-info   direct_url -> .../IsaacLab/source/isaaclab
__editable__.isaaclab-4.5.24.pth + __editable___isaaclab_4_5_24_finder.py
__editable__.isaaclab-4.6.12.pth + __editable___isaaclab_4_6_12_finder.py
```

Imports resolve to the checkout and report 4.6.12; metadata lookups return
4.5.24. The consequence for a rebuild is that pip's resolver consults the stale
`Requires-Dist`. Neither dist-info was removed by this capture.

## Editable install from a deleted worktree

`strafer_autonomy` is pip-visible at 0.1.0 but does not import:

```
direct_url -> file:///home/zachoines/Workspace/Sim2RealLab-bridge-driver/source/strafer_autonomy
python -c "import strafer_autonomy"  ->  ModuleNotFoundError
```

The worktree it was installed from no longer exists. `strafer_lab` and
`strafer_shared` both import correctly from the live checkout.
`strafer_navigation` is a ROS package and is not installed in this env, which is
expected.

## Local modifications inside the Isaac Lab checkout

A fresh clone at the same SHA has neither of these; both must get an explicit
repro-or-obsolete decision when a new pair is built.

1. `omni.kit.telemetry` deleted from the extension list of **both**
   `apps/isaaclab.python.kit` and `apps/isaaclab.python.headless.kit`
   (one removed line each; nothing else differs in those files).
2. An untracked no-op extension shim at
   `source/omni.kit.pip_archive/config/extension.toml`, version 1.0.0,
   described in its own manifest as a "Local no-op shim for stale
   omni.kit.pip_archive dependencies in pip-installed Isaac Sim."

## `.env` pointer values

The three lines a rollback flips, plus the interpreter the pure suites take:

| key | value |
|---|---|
| `STRAFER_ISAACLAB_PYTHON` | `/home/zachoines/miniconda3/envs/env_isaaclab3/bin/python` |
| `CONDA_ENV` | `env_isaaclab3` |
| `ISAACLAB` | `/home/zachoines/Documents/repos/IsaacLab/isaaclab.sh` |
| `CONDA_ROOT` | `/home/zachoines/miniconda3` |
| `ISAACSIM_PATH` | `/home/zachoines/Workspace/IsaacSim/_build/linux-aarch64/release` |

`make test-lab` sources `env_setup.sh` but never conda-activates, so its Kit half
binds to whatever `CONDA_PREFIX` the calling shell carries. Every Kit command in
this capture was run from a shell with `env_isaaclab3` activated.

## Runtime warnings that are present at the current pin

These are the pre-bump baseline for "what the stack already complains about", so
a post-bump warning set can be diffed rather than read cold.

- **199 × `DeprecationWarning: TiledCameraCfg is deprecated. Use CameraCfg
  directly`** from the contract suite alone. Present-and-deprecated, not removed.
- **torch does not list this GPU's compute capability.** Every Kit boot prints:
  `Found GPU0 NVIDIA GB10 which is of cuda capability 12.1. Minimum and Maximum
  cuda capability supported by this version of PyTorch is (8.0) - (12.0).`
  torch 2.10.0+cu130 runs anyway; the warning is a standing condition, not a new
  symptom.
- `roller_bounce_probe.py` triggers six deprecation warnings for direct root-state
  reads and the non-indexed `write_*_to_sim` / `set_joint_velocity_target` forms
  (`root_state_w` is slated for removal in Isaac Lab 4.0).
- Two Fabric warnings per env boot:
  `FabricManager::initializePointInstancer mismatched prototypes on point
  instancer: /Visuals/Command/{goal_heading,goal_sphere}`.

## Asset-data API shape at this pin

`Articulation.data.root_state_w` already returns a **`warp.array`**, not a torch
tensor: a probe written against `.detach()` fails with
`AttributeError: 'array' object has no attribute 'detach'`. The in-tree
`roller_bounce_probe.py` already carries the accommodating idiom
(`wp.to_torch(x) if isinstance(x, wp.array) else x`), and the pose-trace probe in
this directory uses the same one. Any new code reading asset data must not assume
a torch tensor.

## Container and ROS lane availability on this host

- No native ROS toolchain: `colcon` absent, `/opt/ros` absent.
- No `strafer-cpu:humble` image is built (`docker images` lists only unrelated
  `ros:humble` and `ryan-assignment:humble` tags).
- `tools/run_ros_tests.sh ros` therefore fails loudly with
  `no native ROS toolchain and image 'strafer-cpu:humble' is missing`, and the
  `test-ros` / `test-driver` counts are unknown here. They need a Jetson run or a
  built container.
