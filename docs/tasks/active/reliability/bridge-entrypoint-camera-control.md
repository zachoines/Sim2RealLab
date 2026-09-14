# The bridge entrypoints cannot be watched

**Type:** task (tooling / observability)
**Owner:** DGX
**Priority:** P3 — nothing is blocked; it costs a diagnosis path whenever the bridge
misbehaves in a way that is visible but not logged.
**Estimate:** S — the pattern already exists in three sibling scripts and is proven
headless.
**Branch:** `task/bridge-entrypoint-camera-control`

## Story

As **whoever is debugging the sim bridge**, I want **to see what the bridge is
simulating**, so that **a wrong pose, a missing scene, or a robot driving through
geometry is caught by looking rather than inferred from topic rates.**

## Context bundle

- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`sim_bridge_autonomy_cheatsheet.md`](../../../sim_bridge_autonomy_cheatsheet.md) — the
  remote-view section this brief would make apply to the bridge.

## Context (measured 2026-09-13)

Isaac Sim's livestream makes any `AppLauncher` script watchable from another machine, and
that path is documented and verified. It gives a useful picture only if something aims the
camera, because the viewport camera controller does not exist headless
(`manager_based_env.py:187-191` leaves it `None` unless `sim.has_gui` or a visualizer is
registered), so `cfg.viewer` is ignored and the camera keeps Kit's default pose.

The three capture/training scripts each solve this with an anchor block that writes world
coordinates directly:

| script | anchors the camera | per-step follow |
|---|---|---|
| `train_strafer_navigation.py` | `:273-303` under `--video` | none — fixed over env 0 |
| `play_strafer_navigation.py` | `:177-189` under `--video` | none — fixed over env 0 |
| `coverage_capture.py` | `:527-561` under `--video` | yes, `_follow_overhead_camera` at `:738-749` |

The two bridge entrypoints have **no camera-positioning code at all** — grep for
`set_camera_view`, `anchor_capture_camera` and `cam_prim` returns zero hits across all
1415 lines of `run_sim_in_the_loop.py` and all of `bridge_harness_smoke.py`. Both accept
`--livestream` (they take `AppLauncher` args), so both will happily stream; they stream
Kit's default camera, which on a multi-env grid frames nothing in particular.

**The mechanism to copy is proven headless, not assumed.** `coverage_capture.py`'s follow
routes through `strafer_lab.isaacsim_compat.set_camera_view` → `ViewportManager` → a USD
`TransformPrimCommand`. Measured under `--livestream 2`, with and without `--viz kit`: two
successive calls moved `/OmniverseKit_Persp` to `(3,4,9)` and then `(-6,2,7)` exactly. The
teleop follow (`teleop_capture.py:429-463`), which additionally gates on
`get_active_viewport()`, also works — the viewport is live under livestream.

Note the sibling asymmetry that suggests where the code should live: three scripts now
carry three near-identical anchor blocks, and a fourth and fifth need the same thing.

## Acceptance criteria

- [ ] `run_sim_in_the_loop.py` can anchor its camera over a chosen env, behind a flag in
      keeping with its existing ones, and is watchable over the livestream.
- [ ] `bridge_harness_smoke.py` likewise, or an explicit note in the brief recording why a
      ~30 s smoke does not warrant it.
- [ ] The anchor is a **shared** helper rather than a fourth copy — the three existing
      blocks either call it or are recorded as deliberately left alone.
- [ ] Verified by looking: a livestreamed bridge run shows the robot, from a pose that was
      asked for. A pose readback is not required — the point of this brief is that the
      picture is the check.
- [ ] The cheatsheet's remote-view section names the bridge among the watchable
      entrypoints.
- [ ] If your work invalidates a fact in any referenced context module, package README, or
      guide under `docs/`, update those in the same PR.

## Out of scope

- Changing what the bridge simulates or publishes.
- The viewport-controller gap itself (`cfg.viewer` ignored headless) — that is upstream
  behaviour this brief works around, the same way the capture scripts already do.
- Any recording path. This is about watching, not writing MP4s.

## Triggered by

Filed 2026-09-13 while validating the Isaac Lab pair flip
([`isaac-lab-upgrade`](../../completed/isaac-lab-upgrade.md)). The livestream path was set
up to give the flip a visual check; it worked for the capture scripts and was found to be
useless for the bridge, which is the entrypoint most likely to need watching.
