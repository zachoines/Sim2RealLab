# Bridge / harness scene memory budget on the GB10 (unified memory)

**Type:** investigation + fix (sim runtime resource budget)
**Owner:** DGX agent (lane: `source/strafer_lab/` — bridge/harness env cfgs, `run_sim_in_the_loop.py`, the Infinigen scene pipeline, top-level `Makefile`, workstation `docs/`)
**Priority:** P2 — blocks `make sim-bridge` / `sim-harness` / full-stack sim-in-the-loop on the GB10 with the current high-quality scenes. A light-scene workaround exists (below), so not a hard stop.
**Estimate:** M (measurement + a scene-selection knob is small; a GB10 texture/room budget or downscale-on-ingest path is the larger part).
**Branch:** task/bridge-scene-memory-budget-gb10

## Story

As **a DGX operator bringing up the sim bridge on the GB10 (gx10-d1d8)**, I want
**`make sim-bridge` / `sim-harness` to load a representative Infinigen scene
without exhausting the unified memory pool**, so that **sim-in-the-loop and the
harness are runnable on this hardware instead of OOM-killing during scene load.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md) — the DGX is the GB10 (Tegra, **unified memory**: CPU + GPU share one ~121 GB pool, unlike a discrete-VRAM DGX).
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md) — `Assets/generated/scenes/` are gitignored build artifacts (outputs of `prep_room_usds.py`); the bridge/harness env cfgs + `run_sim_in_the_loop.py` are the harness lane.
- [`../harness/harness-architecture.md`](../harness/harness-architecture.md) — **owns the bridge/harness env cfgs + `run_sim_in_the_loop.py`**; any scene-selection or env-cfg change here must coordinate.
- [`../../completed/teleop-perf-architecture.md`](../../completed/teleop-perf-architecture.md) — prior bridge-perf work (loop is PhysX-bound ~10 FPS); this brief is the memory sibling.
- [`../harness/infinigen-scene-corpus.md`](../harness/infinigen-scene-corpus.md) — the scene-generation corpus the bridge consumes.
- [`docs/DGX_SPARK_SETUP.md`](../../../DGX_SPARK_SETUP.md) — DGX Spark / CUDA provisioning (NVRTC + the torch CUDA build).

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

Root cause — **the heaviest scene is always loaded, and it doesn't fit the
unified pool:**

- `StraferNavCfg_BridgeAutonomy` ([`composed_env_cfg.py:496`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)) composes `cameras_required=(rgb_full, depth_full, depth_policy)` + `kind="infinigen"` → scene class `StraferSceneCfg_InfinigenPerception`.
- The room geometry is chosen by [`_get_scene_usd_paths()[0]`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) (`strafer_env_cfg.py:83` + `:1202`) — the **`sorted()`-first** scene listed in `Assets/generated/scenes/scenes_metadata.json`. There is no "pick the lightest" or `--scene` override.
- The only scenes present are two **`high_quality_dgx`** rooms (`scene_high_quality_dgx_000_seed{1,2}`), **29 GB and 27 GB** on disk. The `high_quality_dgx` preset ([`prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py)) is `base_indoors`, **up to 5 rooms, 1024-px textures** — sized for a discrete-VRAM DGX.
- Isaac Sim mmaps the 29 GB room (→ the `file-rss: 43 GB`) and decompresses textures into the **same** 121 GB pool the GPU renders from; with the RTX perception path initializing and swap already ~7 GB deep, NVRM runs out → OOM.
- **Cameras are not the driver:** at `num_envs=1` the policy cam (80×60) + perception cam (640×360) ([`d555_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py)) are a few MB. The scene asset is the cost.
- Secondary smell: torch warns `cuda capability 12.1 … max supported 12.0` — the installed PyTorch isn't built for the GB10 (sm_121), which can inflate / mis-place allocations.

## Immediate workaround (stopgap, already usable)

`_get_scene_usd_paths()` takes `sorted()[0]`, and `scene_fast_singleroom…`
sorts **before** `scene_high_quality…` (`f` < `h`). So generating a
`fast_singleroom` (512-px textures, 1 room) into the same dir makes it `[0]`
without deleting the heavy scenes:

```bash
python source/strafer_lab/scripts/prep_room_usds.py generate \
    --config fast_singleroom --num-scenes 1 --output Assets/generated/scenes
$ISAACLAB -p source/strafer_lab/scripts/generate_scenes_metadata.py
make sim-bridge
```

This is luck-of-the-alphabet, not a contract — hence this brief.

## Investigation pointers

- Peak unified-memory during scene load: monitor `free -g` / `tegrastats` while
  loading a `high_quality_dgx` vs a `fast_singleroom` scene; record the delta
  and the headroom needed below ~121 GB (minus OS + Kit baseline).
- `du -sh Assets/generated/scenes/*` (current: 29 GB / 27 GB) vs a
  `fast_singleroom` (expect single-GB) — the texture-res × room-count driver.
- `prep_room_usds.py` PRESETS: `high_quality_dgx` (1024 / 5 rooms) vs
  `fast_singleroom` / `windows_baseline` (512 / 1 room).
- Whether textures can be down-res'd post-export (re-bake / USD texture
  override) without regenerating the whole room.

## Approach (options — pick during the brief)

1. **Scene selection knob (cheap, do first).** Give `run_sim_in_the_loop.py`
   a `--scene <name>` / `--scene-usd <path>` override (harness mode already has
   `--scene-usd`; extend bridge mode), and/or make `_get_scene_usd_paths()`
   prefer the lightest valid scene rather than `sorted()[0]`. Coordinate with
   harness-architecture (it owns these files).
2. **GB10 scene budget.** Add a `gb10`/`unified_mem` preset (fewer rooms,
   ≤512-px textures) and/or a documented texture-resolution cap for scenes the
   GB10 bridge consumes; OR a downscale-on-ingest pass that re-bakes textures to
   a cap so existing 29 GB scenes become loadable.
3. **Memory headroom.** Confirm/replace the PyTorch build so it targets sm_121
   (GB10); reduce swap thrash; consider unloading non-active rooms.
4. **Measurement bar.** Document peak unified memory for the chosen
   scene/preset so future scenes are sized against the ~121 GB ceiling.

## Acceptance criteria

- [ ] `make sim-bridge` **and** `sim-bridge-gui` reach the mainloop on the GB10
      with a representative Infinigen scene (not a degenerate empty room) — no OOM.
- [ ] There is a **deterministic, documented** way to choose the bridge/harness
      scene (override flag and/or lightest-first), not alphabetical accident.
- [ ] A GB10-appropriate scene budget exists (preset and/or downscale path) with
      **measured** peak unified memory under the ceiling with headroom; recorded
      in the brief / a `docs/` note.
- [ ] `make sim-harness` and the full-stack sim-in-the-loop smoke run on the GB10.
- [ ] The torch `sm_121 > max sm_120` build mismatch is confirmed and either
      fixed or filed (link the follow-up).
- [ ] If your work invalidates a fact in a context module, README, or guide,
      update it in the same commit.

## Out of scope

- **RL training's** use of these scenes (headless, no perception camera, large
  `num_envs` but tiled-camera-light) — measure separately; fix here only if it
  shares the OOM.
- **The PhysX-bound throughput** work — that's `teleop-perf-architecture`'s lineage.
- **The script/tool layout** — `script-tooling-layout-consolidation` (shipped);
  this is unrelated (the crash reproduces on `main`).
- **Regenerating the scene corpus wholesale** — `infinigen-scene-corpus` owns the
  corpus; this brief only needs a loadable scene + a budget.
