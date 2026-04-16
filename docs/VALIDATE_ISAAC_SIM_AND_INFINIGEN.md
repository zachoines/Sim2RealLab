# Isaac Sim + Infinigen install validation

A runbook for validating the DGX Spark Isaac Sim source build and the
Infinigen + bpy 4.2.0 aarch64 install before touching any scene-generation
or Task 8 work. Run the commands one-by-one in order. Each phase gates
the next — if a command in Phase A fails, do not skip ahead to Phase B.

Before starting, make sure the shell is configured:

```bash
cd /home/zachoines/Workspace/Sim2RealLab
source env_setup.sh
```

This exports `LD_PRELOAD`, `STRAFER_BLENDER_BIN`, `INFINIGEN_ROOT`,
`ISAACSIM_PATH`, `ROS_DOMAIN_ID`, and friends. Skipping it is the #1
cause of "works on my laptop, fails here" failures on this host.

---

## Phase A — Install smoke tests (~10 min total)

Minimal import / binary checks. Any failure here means the install is
broken at the foundation and later phases will not help you diagnose it.

### A1. Activate `env_phase15` and verify Python

```bash
conda activate env_phase15
python --version
```

**Expect:** Python 3.11.x. The env was built against Isaac Sim 5.1's
Python 3.11 bundle, so a 3.12 here means the wrong env.

---

### A2. Isaac Sim Python module import

```bash
python -c "from isaaclab.app import AppLauncher; print('isaaclab.app OK')"
```

**Expect:** `isaaclab.app OK` and no traceback.

**If it fails** with a missing `omni`/`pxr` module, the Isaac Sim release
directory (`$ISAACSIM_PATH`) is not on `PYTHONPATH` or the `.pth` file
that wires `pxr` in is missing. Check `$ISAACSIM_PATH` points at
`.../IsaacSim/_build/linux-aarch64/release` and the release dir exists.

---

### A3. Isaac Sim boot + exit

```bash
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print('Isaac Sim booted:', app.is_running())
app.close()
print('Isaac Sim closed cleanly')
"
```

**Expect:** boots in ~30-60 s, prints `Isaac Sim booted: True` then
`Isaac Sim closed cleanly`. Ignore Kit's verbose startup log noise.

**If it hangs** longer than ~3 min, kill it. The first boot after a
source build can be slow while Omniverse warms its shader cache; a
second run of the same command should be noticeably faster (~15 s).

---

### A4. `strafer_lab` task registration

`strafer_lab.tasks` transitively imports `isaaclab.managers`, which
imports `omni.timeline` — a Kit-runtime module that only exists after
`AppLauncher` boots Isaac Sim. So this test has to boot Kit before
touching the task package.

```bash
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import gymnasium as gym
import strafer_lab.tasks  # noqa

ids = sorted(k for k in gym.envs.registry if 'Strafer' in k)
print(f'{len(ids)} Strafer envs registered')
for i in ids[:5]: print(' ', i)
assert 'Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0' in ids, \
    'perception env not registered'
print('perception env registered OK')
app.close()
"
```

**Expect:** at least 20 envs registered, including
`Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`. Takes ~45-60 s
(most of it is the Isaac Sim boot).

**If 0 envs register** but no traceback, check that
`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`'s
`gym.register(...)` calls were actually executed — a silent early
return in the task package would produce this.

**If you see `ModuleNotFoundError: No module named 'omni.timeline'`**,
you're running `import strafer_lab.tasks` without `AppLauncher` — use
the command above, not a bare `python -c "import strafer_lab.tasks"`.

---

### A5. Blender 4.2 binary

```bash
"$STRAFER_BLENDER_BIN" --version
```

**Expect:** `Blender 4.2.x` (source build). If it prints `4.0.2` or
`command not found`, `.env` is wrong — `STRAFER_BLENDER_BIN` must point
at `/home/zachoines/Workspace/blender-build/build_blender/bin/blender`.

---

### A6. `bpy` import from `env_infinigen`

```bash
conda activate env_infinigen
python -c "import bpy; print('bpy', bpy.app.version_string)"
```

**Expect:** `bpy 4.2.x`. This is the aarch64 source-built wheel — if
it fails, the wheel install in `env_infinigen` is broken and Infinigen
cannot run, period.

---

### A7. Infinigen package import

```bash
python -c "import infinigen; print('infinigen', infinigen.__file__)"
```

**Expect:** path ends with `/home/zachoines/Workspace/infinigen/infinigen/__init__.py`.
Any other path means `env_infinigen` picked up a stale pip-installed
copy and you need to reinstall with `pip install -e .` from
`$INFINIGEN_ROOT`.

---

### A8. Infinigen `launch_blender` sanity

```bash
python -m infinigen.launch_blender --help 2>&1 | head -5
```

**Expect:** `launch_blender.py`'s own argparse help, starting with
`usage: launch_blender.py [-h] [-m MODULE] [-s SCRIPT]`. The `--help`
is intercepted by `launch_blender.py`'s own argument parser before
Blender itself is invoked — that is why you see the wrapper's usage
text, not Blender's. What this command actually proves is that the
`infinigen.launch_blender` module imports cleanly (no `bpy` import
error, no missing-binary crash). If you see a Python traceback, the
`$INFINIGEN_ROOT/blender → blender-build/.../bin` symlink is broken
or the wrapper's imports are failing.

---

**Phase A gate:** all eight commands green → move to Phase B. Any red →
stop and fix the failing item first.

---

## Phase B — Isaac Sim functional tests (~30 min total)

Exercise the real sim loop on a known-good env that does NOT depend on
Infinigen. If this phase passes, we know Isaac Sim + Isaac Lab +
strafer_lab work end-to-end regardless of any Infinigen issues.

Switch back to the Isaac Sim env:

```bash
conda activate env_phase15
```

### B1. strafer_lab full unit test suite

`source/strafer_lab/run_tests.py` is the right entry point here — a
direct `pytest` run is drowned by Isaac Sim's startup log noise, and
the root `conftest.py` calls `os._exit(...)` before pytest can print
its own summary. The wrapper runs each suite in a subprocess with
`--junit-xml`, parses the XML, and prints a clean per-suite + overall
table.

It also knows how to run the suites that need process isolation
(`depth_noise`, `rewards`, `imu`) where each test file instantiates
its own `ManagerBasedRLEnv` and Isaac Sim only allows one
`SimulationContext` per process. Running those suites with plain
`pytest` produces singleton-conflict errors.

```bash
cd /home/zachoines/Workspace/Sim2RealLab/source/strafer_lab
python run_tests.py all
```

**Expect:** a clean per-suite progress log followed by a summary table
like:

```
SUMMARY
Suite                 Tests   Pass   Fail    Err
--------------------  -----   ----   ----    ---
+ terminations           12     12      0      0
+ events                  8      8      0      0
...
+ noise_models           55     55      0      0
+ depth_noise             9      9      0      0
+ imu                    14     14      0      0
--------------------  -----   ----   ----    ---
  TOTAL                  N      N      0      0

ALL PASSED
```

Runtime is **~30-45 min** on DGX Spark — `noise_models` alone can take
~15 min (55 tests doing heavy GPU observation rollouts) and
`depth_noise` runs three files sequentially in separate subprocesses
(~3-5 min each). The wrapper's per-suite timeouts are generous; if a
suite hits the timeout it's marked as an error rather than hanging.

**Running a subset** instead (faster iteration during debugging):

```bash
# Just the noise + sensor stack — the pieces most at risk from a Kit
# version bump or a driver change.
python run_tests.py noise_models depth_noise imu sensors

# Just the fast suites — smoke-check after a small change.
python run_tests.py terminations events commands observations curriculums
```

Available suite names: `terminations events commands observations
curriculums rewards sensors actions env noise_models depth_noise imu`,
or `all`.

**If anything fails:** the wrapper prints the failing testcase name
and a truncated traceback inside the suite's `FAIL` rows. Re-run just
that one suite for full output, e.g. `python run_tests.py noise_models`.
A single flaky test is OK to note and move on; anything structural
(import errors, fixture failures, timeouts) is a blocker.

---

### B2. Short rollout on a no-camera env

`AppLauncher` must boot **before** `import strafer_lab.tasks` or
`import gymnasium as gym` — importing the task package first pulls in
`isaaclab.managers`, which chains into `omni.timeline` and explodes
because the Kit runtime is not up yet. Same ordering rule as A4.

```bash
cd /home/zachoines/Workspace/Sim2RealLab
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app

import gymnasium as gym
import strafer_lab.tasks  # noqa
import torch
from isaaclab_tasks.utils import parse_env_cfg

task = 'Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0'
env_cfg = parse_env_cfg(task, device='cuda:0', num_envs=8)
env = gym.make(task, cfg=env_cfg)

obs, _ = env.reset()
print('reset OK, obs keys:', list(obs.keys()))
for step in range(20):
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env.step(action)
print('20 steps OK on 8 envs')
env.close()
app.close()
"
```

**Why `parse_env_cfg` instead of `num_envs=8`:** Isaac Lab registers its
envs with an `env_cfg_entry_point` kwarg pointing at the config class,
but the actual `ManagerBasedRLEnv.__init__` takes a concrete `cfg=`
instance as its first positional arg. Passing `num_envs=8` to
`gym.make` forwards it as an extra env-creator kwarg instead of
setting it on the config, which is why you see
`__init__() missing 1 required positional argument: 'cfg'`.
`parse_env_cfg` is the canonical Isaac Lab helper that loads the
registered config class, applies `num_envs` / `device` to the
instantiated config, and returns the concrete object `gym.make` needs.

**Expect:** prints `reset OK, obs keys: ['policy']` then
`20 steps OK on 8 envs`. Takes ~60 s including Isaac Sim boot. This
validates the physics-only path — no renderer, no camera, just the
articulation controller driving the wheels.

**If it fails** with a USD asset load error, the Strafer robot USD is
missing. Check `source/strafer_lab/strafer_lab/assets/strafer/` has
the expected `.usd` files.

---

### B3. Short rollout on the perception env (renderer exercise)

> **Requires:** at least one Infinigen scene USD under
> `Assets/generated/scenes/scene_*.usdc`. The
> `StraferNavEnvCfg_Real_InfinigenPerception_PLAY` config's
> `__post_init__` calls `_apply_infinigen_scene_setup`, which hard-fails
> with `FileNotFoundError: No valid scene_*.usdc files found` if no
> scenes exist. **Skip this step until Phase C (Infinigen generation)
> has produced at least one scene, or Task 8 has run the scene prep
> pipeline.** B4 below does not depend on Infinigen and can run now.

```bash
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import strafer_lab.tasks  # noqa
import torch
from isaaclab_tasks.utils import parse_env_cfg

task = 'Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0'
env_cfg = parse_env_cfg(task, device='cuda:0', num_envs=1)
env = gym.make(task, cfg=env_cfg)

obs, _ = env.reset()
cam = env.unwrapped.scene['d555_camera_perception']
rgb = cam.data.output['rgb']
depth = cam.data.output['distance_to_image_plane']
print('rgb shape:', tuple(rgb.shape), 'dtype:', rgb.dtype)
print('depth shape:', tuple(depth.shape), 'dtype:', depth.dtype)
print('depth range:', float(depth.min()), float(depth.max()))
env.close()
app.close()
"
```

**Expect:** `rgb shape: (1, 360, 640, 4)` and a non-zero depth range
(something like `0.01` to `5.5`). Takes ~90 s including sim boot.

**If depth range is `(0.0, 0.0)`** the camera rendered a single frame
of empty scene — fine, the room placeholder is minimal. The shape
check is what matters: it proves the 640×360 render path is wired.

**If the shape is wrong** (e.g. 80×60 instead of 640×360), Task 1's
work regressed — check `strafer_shared/constants.py` PERCEPTION_* values.

---

### B4. Short training run on the cheapest env (full RL loop)

Use the project's own training wrapper `Scripts/train_strafer_navigation.py`,
not Isaac Lab's stock `train.py`. The stock script only imports
`isaaclab_tasks`, so the Strafer envs never get registered and
`gym.spec()` raises `NameNotFound` on any `Isaac-Strafer-*` id. The
project wrapper imports `strafer_lab` after the AppLauncher boots,
registers the custom network, and handles auxiliary loss modules.

```bash
cd /home/zachoines/Workspace/Sim2RealLab
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0 \
    --num_envs 64 \
    --max_iterations 10 \
    --headless
```

**Expect:** 10 PPO iterations complete in ~3-5 min, each iteration
logs a mean reward and the policy loss is finite. A checkpoint gets
written to `logs/rsl_rl/strafer_navigation/<timestamp>/model_9.pt`.

**Note on flags:** the project wrapper uses `--env`, not `--task`,
and defaults to `Isaac-Strafer-Nav-Real-v0` if omitted. AppLauncher
flags (`--headless`, `--enable_cameras`, etc.) are still accepted
because the wrapper calls `AppLauncher.add_app_launcher_args`.

**If iterations crash** after the first one with a NaN or CUDA OOM,
reduce `--num_envs` to 32 and retry. NaN after several iterations is a
symptom of learning dynamics, not the install — that's a separate
debugging session.

**If it hangs on iteration 0** with no output for > 3 min, the observation
pipeline is blocking. Most likely cause is a camera update racing the
policy; since this env is NoCam, that shouldn't happen — check the
traceback after a ctrl-c.

---

### B5. Short training run with video capture (renderer + writer smoke)

Same idea as B4 but on a ProcRoom env that has a camera and with
`--video` so the pipeline also exercises `gym.wrappers.RecordVideo`
and the MP4 writer. 10 iterations is still fast but the render path
makes it noticeably slower, so drop to 32 envs. `train_strafer_navigation.py`
auto-enables `--enable_cameras` when the env variant needs it or
`--video` is set, so you don't have to pass it explicitly.

```bash
cd /home/zachoines/Workspace/Sim2RealLab
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 32 \
    --max_iterations 10 \
    --video \
    --video_length 100 \
    --video_interval 5 \
    --headless
```

**Expect:** 10 iterations in ~8-12 min. One MP4 gets written to
`logs/rsl_rl/strafer_navigation/<timestamp>/videos/train/` roughly
100 env-steps long, showing the random-initialized policy driving the
robot through the ProcRoom. Quality is bad (it's 10 iterations of PPO,
not a real policy), but the robot should move.

**If the video file is missing** but training reports success,
`gym.wrappers.RecordVideo` is silently catching an exception — check
the `logs/rsl_rl/strafer_navigation/<timestamp>/videos/train/`
directory exists and has write permissions.

**If training is ~5× slower than B4**, that is expected — the Depth
env renders a camera every env step and the video wrapper adds an
additional RGB readback.

---

### B6. (Optional) Full training run for a usable policy

Only run this when you actually want a policy that can navigate, not
as a validation check. Budget a few hours of DGX time. The exact
iteration count to reach a usable policy depends on the task; for
`ProcRoom-Depth-v0` a reasonable target is ~3000 iterations at 64
envs. The eventual plan is to train in ProcRoom (Infinigen-independent)
first, then move to InfinigenDepth once Phase C and Task 8 have
produced real scenes.

```bash
cd /home/zachoines/Workspace/Sim2RealLab
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 \
    --max_iterations 3000 \
    --video \
    --video_length 200 \
    --video_interval 500 \
    --headless
```

`--video_interval 500` records a clip every 500 env steps, so you get
periodic snapshots of policy improvement throughout the run. The
final clip (near iteration 3000) is the one that should look
recognizably "navigating toward the goal" if training converged.
All clips land in `logs/rsl_rl/strafer_navigation/<timestamp>/videos/train/`.

**Checkpoint playback** is not yet wired up as a separate script in
`Sim2RealLab/Scripts/`. To replay a specific checkpoint you currently
have two options:

1. **Resume training for zero iterations** — `train_strafer_navigation.py`
   accepts `--resume <checkpoint_path>` and will do a single evaluation
   pass if you also pass `--max_iterations 0`. This is the shortest
   path to "load this checkpoint, record one clip, exit".
2. **Use the last clip from B6's training run** — the `--video_interval`
   snapshots captured during training already include late-training
   clips, so if you just want to see the final policy you can grab the
   highest-numbered MP4 from the `videos/train/` directory without
   running a separate playback.

When a dedicated play script lands (likely paired with the sim-in-the-
loop harness work in Windows Task 5 / Jetson Task 11), this step will
get a cleaner replay command.

---

**Phase B gate:** B1-B4 green → Isaac Sim install is fully validated.
B5 green → the renderer + video writer path also works. B6 is not a
gate; it is the "eventually, run a real training job" step for when
you want a working policy.

Continue to Phase C only if you need Infinigen for Task 8.

---

## Phase C — Infinigen generation tests (~60 min total)

Infinigen on aarch64 is the most experimental piece of the stack. The
bpy 4.2.0 wheel was source-built specifically for this host and has
had near-zero real-world mileage. Start with the smallest possible
scene to validate the import chain end-to-end before trying anything
realistic.

```bash
conda activate env_infinigen
cd /home/zachoines/Workspace/infinigen
```

### C1. Smallest possible Infinigen scene (~1 min runtime)

This is the "floor layout, overhead view, no objects" recipe from
Infinigen's `HelloRoom.md` — the fastest generation path that still
exercises the constraint solver.

```bash
python -m infinigen_examples.generate_indoors \
    --seed 0 --task coarse \
    --output_folder /tmp/infinigen_smoke/coarse \
    -g no_objects.gin overhead.gin \
    -p compose_indoors.terrain_enabled=False
```

**Expect:** runs for ~30-90 s, final output contains
`/tmp/infinigen_smoke/coarse/scene.blend` (at least 1 MB). The console
logs step-by-step constraint solving — if it stops logging for more
than 2 min it is probably wedged; kill it.

**If it crashes on import** with a `bpy` error, the wheel is broken
and you need to rebuild it. Hard problem — document the error and
stop.

**If it crashes inside the solver**, try again with a different seed
(`--seed 1`). Some constraint configurations are flaky even on x86.

---

### C2. Verify the `.blend` is openable

```bash
"$STRAFER_BLENDER_BIN" --background /tmp/infinigen_smoke/coarse/scene.blend \
    --python-expr "import bpy; print('objects:', len(bpy.data.objects))"
```

**Expect:** prints something like `objects: 14` (exact count varies).
Any non-zero object count proves the file is valid.

**If Blender segfaults** loading the file, the solver wrote a corrupt
`.blend` — rerun C1 with a different seed and this step against the
new output.

---

### C3. Export the scene to USD for Isaac Sim

```bash
python -m infinigen.tools.export \
    --input_folder /tmp/infinigen_smoke/coarse \
    --output_folder /tmp/infinigen_smoke/export \
    -f usdc -r 512 --omniverse
```

**Expect:** ~1-3 min runtime. Output is a directory tree under
`/tmp/infinigen_smoke/export/` containing one `.usdc` file plus
textures. Note the exact path of the `.usdc` file — the next step
needs it.

```bash
find /tmp/infinigen_smoke/export -name "*.usdc" -type f
```

**If export fails** on a texture bake, drop `-r 512` to `-r 256` and
retry. Lower texture resolution is fine for a smoke test.

---

### C4. Load the exported USD in Isaac Sim

Switch back to `env_phase15` for this one — Isaac Sim cannot run from
`env_infinigen`'s Python.

```bash
cd /home/zachoines/Workspace/Sim2RealLab
conda deactivate
conda activate env_phase15

python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import glob
usds = glob.glob('/tmp/infinigen_smoke/export/**/*.usdc', recursive=True)
assert usds, 'No .usdc file found under /tmp/infinigen_smoke/export'
usd_path = usds[0]
print('loading', usd_path)

from pxr import Usd
stage = Usd.Stage.Open(usd_path)
assert stage is not None, 'Stage.Open returned None'

prim_count = sum(1 for _ in stage.Traverse())
print(f'stage loaded OK, {prim_count} prims')
app.close()
"
```

**Expect:** `stage loaded OK, <N> prims` where N is in the hundreds to
low thousands. This is the definitive proof that Infinigen output can
be consumed by Isaac Sim.

**If Stage.Open returns None** or prim traversal errors, the USD has
a schema Isaac Sim's USD runtime does not accept. Check whether
Infinigen's `--omniverse` flag was honored (it should inject
Omniverse-compatible schema). Rerun C3 with `--omniverse` explicitly
if you dropped it.

---

### C5. Render one frame through a TiledCamera on the loaded stage

This is the test that most closely simulates what Task 8 + perception
data collection will actually do: load an Infinigen USD as a scene,
place a camera, capture one RGB frame.

```bash
python -c "
import glob
import numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

import omni.usd
from pxr import Usd

usds = glob.glob('/tmp/infinigen_smoke/export/**/*.usdc', recursive=True)
usd_path = usds[0]

ctx = omni.usd.get_context()
ctx.open_stage(usd_path)
print('stage open:', usd_path)

# Let Kit run a few update ticks so the renderer catches up
for _ in range(30):
    app.update()

print('30 update ticks completed without crashing')
app.close()
"
```

**Expect:** `stage open: ...` then `30 update ticks completed without
crashing`. Takes ~60-90 s.

**If Kit crashes during updates** with a material / shader error, the
Infinigen export's materials are not Omniverse-compatible. The scene
geometry is still usable for Task 8 — Task 8 only cares about
semantic labels on the prims, not about PBR material bake quality.

---

**Phase C gate:** if C1-C4 pass, Infinigen can produce USDs Isaac Sim
can open. C5 passing is a stretch goal; C1-C4 green is enough to
unblock Task 8 since Task 8 is a prim-attribute pass, not a render
test.

---

## Phase D — Task 8 readiness gate (~5 min)

Once Phase C passes, verify the glue that Task 8 will actually exercise.

### D1. Inspect Infinigen-exported prim tags

Task 8's whole job is to walk the USD, read whatever semantic info
Infinigen already wrote, and re-stamp it as `semanticLabel` attrs the
Isaac Sim Replicator bbox annotator understands. So before writing any
Task 8 code, look at what Infinigen actually emits:

```bash
python -c "
import glob
from pxr import Usd
usds = glob.glob('/tmp/infinigen_smoke/export/**/*.usdc', recursive=True)
stage = Usd.Stage.Open(usds[0])

seen_attrs = set()
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if 'semantic' in name.lower() or 'label' in name.lower() or 'class' in name.lower():
            seen_attrs.add(name)
print('semantic-related attrs found:', sorted(seen_attrs))

# First 10 prim paths + types
for i, prim in enumerate(stage.Traverse()):
    if i >= 10: break
    print(f'  {prim.GetPath()} [{prim.GetTypeName()}]')
"
```

**Expect:** some set of semantic-related attribute names, and a sample
of prim paths that look like `/Root/Home/Kitchen/Chair_0` or similar.
The attribute names you see here dictate how Task 8 reads labels from
the export — document them in the Task 8 design before writing code.

**If no semantic attrs are found**, Infinigen's USD export is not
preserving semantic tags on this code path. Task 8 gets harder (must
cross-reference the `.blend` file's object names with the USD prim
tree). Not a blocker, but worth knowing early.

---

## Appendix — common failure modes

| Symptom | Fix |
|---------|-----|
| `command not found: blender` in `env_setup.sh` status | `.env` missing `STRAFER_BLENDER_BIN` — run `cp .env.example .env` and fill it in |
| Isaac Sim boots then segfaults on first render | Almost always `LD_PRELOAD` not set; re-`source env_setup.sh` |
| `ModuleNotFoundError: No module named 'omni.timeline'` | Trying to import `strafer_lab.tasks` without Isaac Sim running — only import after `AppLauncher` boots |
| `bpy` wheel broken on `env_infinigen` | Rebuild the wheel from `~/Workspace/blender-build/` (long, skip unless truly needed) |
| `infinigen.tools.export` OOMs | Lower `-r` to 256 or 128; the DGX's unified memory means Blender OOM takes down the whole host |
| Isaac Sim compat check hangs | Skip it — boot exit 0 (A3) is sufficient validation, the `isaac-sim.compatibility_check.sh` script is known-flaky on aarch64 |
