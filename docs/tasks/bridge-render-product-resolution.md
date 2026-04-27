# Bridge render product resolution — IsaacCreateRenderProduct ignores camera prim's configured size

**Type:** task / bug
**Owner:** DGX agent
**Priority:** P1
**Estimate:** S (~half day; one OmniGraph builder edit + cam_info
verification + smoke test)

## Story

As a **bridge operator wiring the Strafer perception camera into
ROS2**, I want **the bridge's `IsaacCreateRenderProduct` to render at
the configured perception-camera resolution (640×360)** instead of
**Hydra's 1280×720 default**, so that **`/d555/color/camera_info`
matches the real D555 spec the rest of the stack assumes,
`/d555/color/image_raw` and `/d555/depth/image_rect_raw` pixel
indexing aligns with VLM-grounded bboxes, and the depth-aligned-to-
color contract holds without 4× bandwidth waste**.

## Context

The perception camera is configured in
`source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py` as a
`TiledCameraCfg(width=PERCEPTION_WIDTH=640, height=PERCEPTION_HEIGHT=360)`
with `D555_FOCAL_LENGTH_MM=1.93`, `D555_HORIZONTAL_APERTURE_MM=3.68`.

For that pinhole config, `fx` in pixels should be:

```
fx = focal_length_mm × width_px / aperture_mm
   = 1.93 × 640 / 3.68 = 335.65 px
```

But Isaac Sim's startup log under `make sim-bridge` reports:

```
[isaacsim.ros2.core.impl.camera_info_utils] Forcing fy to fx
(671.3043512313513 != 671.3043174040571) when computing CameraInfo,
as renderer assumes square pixels.
```

`fx ≈ 671.30 px` — exactly `1.93 × 1280 / 3.68 = 671.30`. **The
render product is rendering at 1280×720, not the configured 640×360.**
`/d555/color/camera_info.width` and `/d555/color/camera_info.height`
will publish 1280 / 720.

The bridge OmniGraph builder at
`source/strafer_lab/strafer_lab/bridge/graph.py:223-253` constructs
each stream's `IsaacCreateRenderProduct` without setting `inputs:width`
or `inputs:height`. The graph relies on the docstring claim:

> "the render product's resolution tracks the camera prim's resolution
> attribute"

That claim does NOT hold for camera prims spawned by `TiledCameraCfg`.
Tiled rendering uses Replicator's tiled-camera product separately
from the standalone `IsaacCreateRenderProduct` the bridge creates;
the spawn-time `width`/`height` set on the prim's USD attributes
isn't picked up by the bridge's render product, which falls through
to the Hydra default (1280×720).

### Downstream impact, observed today

- **Goal projection** (Jetson `strafer_perception`) multiplies a
  Qwen-normalized [0, 1000] bbox center by `cam_info.width` to get
  pixel coordinates. With `cam_info.width = 1280`, projected pixels
  range up to (1280, 720) — well outside the policy / sim contract.
  This is how a recent failure surfaced as `Invalid depth at pixel
  (801, 306)` — perfectly valid in 1280-wide camera coords; literally
  impossible from a 640-wide camera.
- **VLM grounding** is being fed 1280×720 RGB. The VLM was
  benchmarked on the real D555 stream (640×360) — recall and bbox
  precision past sim-vs-real are an open question.
- **RTAB-Map** is running visual odometry / loop closure on 1280×720
  RGB-D pairs, ~4× the pixel work the policy + real-robot deployment
  expect. Probably contributing to the sub-unity sim RTF the
  perception camera already dominates (rendering bottleneck called
  out in the goal-projection task and the velocity-mismatch chain).
- **/d555/color/image_raw bandwidth** is 4× over budget on the
  Jetson↔DGX LAN.

### Adjacent warning that may be related

```
[omni.syntheticdata.plugin] SdRenderVarPtr missing valid input
renderVar DistanceToImagePlaneSDbuff
```

This typically fires on the first tick before the synthetic-data
plugin has bound the depth render variable to the camera. If depth
images publish successfully a few ticks later it's benign cold-start
noise, but verify during this task — the resolution mismatch and
this warning could share root cause if the SD plugin is binding
against a render product the OmniGraph builder didn't fully
configure.

## Scope of impact

- **Sim contract correctness**: `cam_info.width = 640`, `height = 360`
  matches the real-robot D555 spec the rest of the stack assumes.
  Goal projection's pixel coords stay in-range; VLM sees the
  resolution it was benchmarked on.
- **Sim throughput**: render-product readback for color + depth at
  640×360 is ~4× cheaper than 1280×720. Combined with the async
  camera-publishers task, sim RTF should improve materially.
- **Real-robot deployment**: zero impact. Real D555 driver publishes
  640×360 natively; this brings sim into parity with that.

## Acceptance criteria

- [ ] `_attach_camera_stream` (or wherever `IsaacCreateRenderProduct`
      is configured in `bridge/graph.py`) explicitly sets
      `inputs:width` and `inputs:height` from the camera prim's
      configured resolution. Source those values from
      `strafer_shared.constants.PERCEPTION_WIDTH` /
      `PERCEPTION_HEIGHT` so there is one source of truth.
- [ ] After the fix, `ros2 topic echo --once /d555/color/camera_info`
      under `make sim-bridge` reports `width: 640`, `height: 360`,
      `k[0] (fx) ≈ 335.65`, `k[4] (fy) ≈ 335.65`.
- [ ] `ros2 topic echo --once /d555/color/image_raw` reports
      `width: 640`, `height: 360`.
- [ ] Same checks for `/d555/depth/image_rect_raw` —
      `width: 640`, `height: 360`. Encoding unchanged
      (`32FC1` per the bridge's `stream_type="depth"` setting).
- [ ] Investigate whether the
      `SdRenderVarPtr missing valid input renderVar
      DistanceToImagePlaneSDbuff` warning was a symptom of the same
      bug. If yes, document; if it's a separate cold-start race,
      file or note in `DEFERRED_WORK.md`.
- [ ] Unit / integration test under `source/strafer_lab/test/`:
      either an OmniGraph-config-only test that asserts the
      `inputs:width` / `inputs:height` fields are populated on the
      `RenderProduct*` node spec the builder emits, or a smoke test
      that brings up the bridge briefly and reads `cam_info` off the
      ROS2 graph (matching the existing bridge-test scaffolding).

## Investigation pointers

- `source/strafer_lab/strafer_lab/bridge/graph.py:195-253`
  — `_attach_camera_stream` (the function that builds each
  RenderProduct/CameraHelper/CameraInfoHelper trio). Add the
  `(f"{render_node}.inputs:width", PERCEPTION_WIDTH)` and
  `(..."inputs:height", PERCEPTION_HEIGHT)` lines under
  `keys.SET_VALUES`. Already imports under the OmniGraph block; just
  add the two key/value pairs.
- `source/strafer_lab/strafer_lab/bridge/config.py` —
  `CameraStreamConfig` currently carries prim path, topic, frame_id,
  and stream type. Consider extending with `width: int`,
  `height: int` fields rather than reading constants directly inside
  `graph.py`, to keep `graph.py` free of `strafer_shared` imports.
  Either approach is fine; pick the one that matches existing tone.
- `source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py:120-172`
  — confirm both `make_d555_camera_cfg` (80×60 policy camera, NOT
  bridged) and `make_d555_perception_camera_cfg` (640×360 perception
  camera, IS bridged) — only the perception camera flows through the
  bridge, so `PERCEPTION_WIDTH`/`HEIGHT` are the right constants
  here. The 80×60 policy camera is GPU-resident only.
- `strafer_shared.constants.PERCEPTION_WIDTH` (640),
  `PERCEPTION_HEIGHT` (360) — the source of truth.
- Isaac Sim's `IsaacCreateRenderProduct` node spec — the
  `inputs:width` / `inputs:height` fields control the render product
  resolution. Verify in the Kit-shipped node docs that these are the
  current attribute names (Isaac Sim 4.x → 5.x renamed a number of
  things; the bridge's `frameSkipCount` rename in graph.py:241 is a
  precedent for that kind of rename).

## Risks / open questions

- **Texture / Replicator alignment.** The perception camera is also
  used by Replicator for bbox extraction in
  `collect_perception_data.py` etc. Those use the tiled-camera
  product, NOT the bridge's render product. Verify they continue to
  render at 640×360 after this fix (they should — tiled and bridge
  render products are independent).
- **/d555 RGB visualization tooling.** If anything was implicitly
  expecting 1280×720 (e.g., a hard-coded crop in a preview script),
  it may break. Grep `source/` for `1280|720|1280x720` before
  closing the task.
- **Real D555 fx/fy.** The real RealSense D555 publishes its own
  intrinsics from the device EEPROM. The sim-bridge fx (335.65 after
  this fix) won't perfectly equal real fx (D555 at 640×360 typically
  reports fx ≈ 380-400 px depending on the calibration profile).
  Goal projection consumes whatever cam_info reports, so this is
  fine — the sim and real bridges deliver consistent contracts at
  matching resolution; the policy was trained against sim
  intrinsics, deployment uses real. No action required, just a
  reminder that intrinsics are camera-specific even after this fix.

## Out of scope

- Raising the perception camera resolution back to 1280×720 as a
  feature. The whole stack — policy, real D555, VLM benchmarks,
  RTAB-Map tunings — assumes 640×360. If a future task wants higher
  perception resolution, it's a stack-wide retune, not a one-line
  fix.
- Goal projection's `_DEPTH_MAX_M` cap. Tracked separately in
  [goal-projection-depth-range](goal-projection-depth-range.md).
  Both fixes can land independently and missions need both before
  they reliably succeed in sim.
- Async camera publishers (separate task,
  [async-camera-publishers](async-camera-publishers.md)) — that's an
  architecture move; this is a config bug. The two are
  complementary; either can land first.
