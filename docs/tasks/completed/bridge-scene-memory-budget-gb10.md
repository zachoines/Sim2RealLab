# Default the bridge/harness single-scene to the LIGHTEST registered scene (GB10 unified-memory OOM)

**Status:** Shipped 2026-07-08 in `8a98074` (DGX). The single-scene Infinigen
consumer (`_apply_infinigen_scene_setup`, which backs `make sim-bridge` /
`sim-harness` teleop + coverage capture) now binds the **lightest** registered
scene by resolved `.usdc` file size via the new `_lightest_scene_usd_path()`
helper, instead of `_get_scene_usd_paths()[0]` — the `sorted()`-first entry,
which is a ~29 GB `high_quality_dgx` room that OOM-kills the unified-memory GB10
during render init. `_get_scene_usd_paths()` keeps its name-sorted order
untouched (the multi-scene ordering invariant), and `SCENE_USD` / `--scene-usd`
remain the precision override. RL **training is unaffected** — it runs on
`kind="procroom"` / `"plane"` and never loads an Infinigen scene USD, so it does
not flow through the changed site. One decisive `[scene-select]` log line names
the picked scene + size. Hermetic tests cover size-pick, symlink-target
measurement, name tie-break, missing-manifest fallback, the sorted-order
invariant, and consumer routing. Cross-package ride-along: the `strafer_autonomy`
runtime-env mocks now derive the policy-cam shape from `strafer_shared` constants
(80×45), not a stale 80×60 literal.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/146
**Remaining (operator, ~2 min, after the GPU frees):** bare `make sim-bridge`
selects the single-room scene by default and boots without OOM — the
`[scene-select]` log names it (`scene_singleroom_000_seed0`, USDC ~0.5 GB; ~1.9 GB
on disk with textures). The runbook's explicit `SCENE_USD` pin stays the
documented practice; this is the safety net under it.

## Re-scope note (why this shipped smaller than estimated M)

The original M-sized scope paired a scene-selection knob **with** a GB10
texture/room budget (or a downscale-on-ingest re-bake). The larger half is now
**moot**:

- Genuine light scenes exist (`single-room-couch-scene-supply`, #128:
  `scene_singleroom_000_seed0`, `scene_tworoom_000_seed0`), so no `gb10` /
  `unified_mem` preset and no downscale-on-ingest pass are needed to obtain a
  loadable scene — a light scene is already in the corpus.
- `SCENE_USD` / `--scene-usd` pins work (#128) as the explicit precision override.
- Occupancy-derived spawn is correct per loaded scene (`bridge-spawn-from-occupancy`,
  #127), so the per-scene spawn/floor follow the picked scene automatically.

What remained was the **`sorted()[0]` trap**: the *default* still resolved
alphabetically to the heaviest `high_quality` room and OOM-killed the GB10 even
with the runbook in hand. Fixing that default is the entire residual scope; the
texture-budget / downscale-on-ingest work is **retired** — regenerate a lighter
corpus via the #128 `--quality low` / `--texture-res` flags if one is ever
wanted, rather than re-baking existing 29 GB scenes.

## torch sm_121 note (documented environment observation — ops item, no code)

Torch on the GB10 still warns `cuda capability 12.1 … max supported 12.0`: the
installed PyTorch is not built for sm_121 (GB10). It is **not** on the OOM
critical path — the lightest scene loads within the unified pool regardless — so
it needed no change here. **Ops item:** watch on the next torch bump; a genuine
sm_121 build may reduce / relocate allocations. No code change in this brief.

## What shipped

- **`_lightest_scene_usd_path()`** (`strafer_env_cfg.py`) — over the same valid
  set `_get_scene_usd_paths()` returns, resolves each top-level `<scene>.usdc`
  symlink to its real file, `stat`s the size, and returns the smallest; ties
  break by the sorted-first name. The single resolved `.usdc` size is a
  sufficient proxy — heavy vs light scenes differ by an order of magnitude at the
  file level (~10 GB vs ~0.5 GB), so there is no need to sum texture trees.
- **The single `[0]` consumer** (`_apply_infinigen_scene_setup`) now calls the
  helper. It was the *only* `_get_scene_usd_paths()[0]` site in the tree (the
  brief's earlier `:1202` reference was stale). `_get_scene_usd_paths()` itself
  is unchanged, so any multi-scene consumer keeps identical ordering.
- **Hermetic tests** (`tests/harness/test_lightest_scene.py`) — tmp scenes dir +
  fake manifest + real symlink layout; nothing reads the transient corpus.
- **Ride-along** — `strafer_autonomy/tests/test_sim_in_the_loop_runtime_env.py`
  policy-cam mocks derived from `strafer_shared.constants.DEPTH_WIDTH/HEIGHT`
  (80×45), so they can't restale after the #143 60→45 change.

## Story

As **a DGX operator bringing up the sim bridge on the GB10 (gx10-d1d8)**, I want
**`make sim-bridge` / `sim-harness` to load a representative Infinigen scene
without exhausting the unified memory pool**, so that **sim-in-the-loop and the
harness are runnable on this hardware instead of OOM-killing during scene load.**

## The problem (measured 2026-06-07 on gx10-d1d8)

`make sim-bridge` (env `StraferNavCfg_BridgeAutonomy`) launches, parses the
env cfg, builds the scene, starts physics — then is **SIGKILL'd by the OOM
killer** during scene/render init:

```
NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... (repeated)
systemd-journal invoked oom-killer ... global_oom
Out of memory: Killed process (python)
  total-vm: 305 GB,  anon-rss: 14 GB,  file-rss: 43 GB
```

Root cause — **the heaviest scene was always loaded, and it doesn't fit the
unified pool:**

- The single-scene Infinigen consumer chose room geometry by
  `_get_scene_usd_paths()[0]` — the `sorted()`-first scene in
  `scenes_metadata.json`, with no "pick the lightest" default.
- The corpus's `high_quality_dgx` rooms are 29 GB / 27 GB on disk (~10 GB at the
  `.usdc` file level); the preset is `base_indoors`, up to 5 rooms, 1024-px
  textures — sized for a discrete-VRAM DGX.
- Isaac Sim mmaps the heavy room (→ `file-rss: 43 GB`) and decompresses textures
  into the **same** 121 GB pool the GPU renders from; with the RTX perception
  path initializing and swap already deep, NVRM runs out → OOM.
- **Cameras are not the driver:** at `num_envs=1` the policy + perception cams
  are a few MB. The scene asset is the cost.

## Why lightest-by-default is the fix (not a knob the operator must remember)

The `SCENE_USD` pin is explicit and correct, but `scene_singleroom…` sorts
**after** `scene_high_quality…` (`s` > `h`), so it never becomes the alphabetical
`sorted()[0]` by accident — the default kept resolving to the 29 GB room and
OOM-killing the GB10 even when the runbook was in hand. Making the single-scene
default the *lightest* scene removes that footgun while leaving `SCENE_USD` /
`--scene-usd` as the precision override.

## Out of scope (unchanged)

- **RL training's** use of scenes — it runs on `procroom` / `plane`, never the
  Infinigen USD path, so it is untouched here.
- **The PhysX-bound throughput** work — `teleop-perf-architecture`'s lineage.
- **Regenerating the scene corpus wholesale** — `infinigen-scene-corpus` /
  `single-room-couch-scene-supply` own the corpus; this brief only needed a
  deterministic, memory-safe *default* over whatever scenes are registered.
