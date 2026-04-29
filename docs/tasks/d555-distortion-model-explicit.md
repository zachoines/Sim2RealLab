# D555 perception camera — set explicit `opencvPinhole` distortion model

**Type:** task
**Owner:** DGX agent
**Priority:** P2
**Estimate:** S (~half day; small d555_cfg / startup-event edit + smoke
verification that the warnings are gone and `/d555/color/camera_info`
still matches real-D555 intrinsics)
**Branch:** task/d555-distortion-model-explicit

## Story

As a **bridge operator running `make sim-bridge[-gui]` against
Isaac Sim 6**, I want **the perception camera prim to carry an
explicit `omni:lensdistortion:model = "opencvPinhole"` schema with
zeroed coefficients**, so that **the bridge's `CameraInfo` publisher
takes the supported code path instead of the deprecated
`physicalDistortionModel` fallback, sim startup stops emitting two
warning lines per camera, and a future Isaac Sim release that drops
the deprecated path doesn't silently change our camera_info contract**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

## Context

`make sim-bridge-gui` on Isaac Sim 6 emits these two warnings per
camera at startup:

```
[isaacsim.ros2.core.impl.camera_info_utils] ROS2 CameraInfo support
  for lens distortion models beyond opencvPinhole and opencvFisheye
  is deprecated as of Isaac Sim 5.0, and will be removed in a future
  release.
[isaacsim.ros2.core.impl.camera_info_utils] Unsupported physical
  distortion model 'None'. Using plumb_bob with default coefficients.
```

Tracing them to their source in
`Workspace/IsaacSim/source/extensions/isaacsim.ros2.bridge/python/impl/camera_info_utils.py`:

- `read_camera_info()` first checks the camera prim's
  `omni:lensdistortion:model` attribute (lines 41-75).
  - If it's `"opencvPinhole"` or `"opencvFisheye"`, the function reads
    coefficients from the matching `omni:lensdistortion:<model>:*`
    schema and publishes a clean `CameraInfo`. **No warnings.**
  - Otherwise (current state, attribute unset), it falls into the
    deprecated `else` branch (line 77+) which logs the deprecation
    warning, then tries `physicalDistortionModel` — that attribute is
    also unset, so the fallback emits the second warning and defaults
    to `plumb_bob` with zero coefficients.

So the current bridge publishes the correct `CameraInfo` (plumb_bob
with zeroed `d`), but via a path that is announcing its own removal.

The camera prims live at
`/World/envs/env_0/Robot/strafer/body_link/d555_camera` and
`d555_camera_perception`, spawned by:

- `make_d555_camera_cfg` in
  [`source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py)
  (policy 80×60 — internal only, not bridged).
- `make_d555_perception_camera_cfg` in the same file (640×360,
  bridged to `/d555/color/*` and `/d555/depth/*`).

Both currently use `sim_utils.PinholeCameraCfg(...)` with no
distortion field. `PinholeCameraCfg` (Isaac Lab) doesn't expose a
distortion schema in its config dataclass — distortion attributes
live directly on the USD prim. Recommended path: set them post-spawn
via `isaacsim.sensors.camera.Camera.set_opencv_pinhole_properties()`
(the helper in
`Workspace/IsaacSim/source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py:1873`,
`set_distortion_coefficients`) which writes the full
`omni:lensdistortion:opencvPinhole:*` schema in one call.

A natural hook is a startup event in the env (mirroring
`lift_ground_plane_to_floor`'s pattern in
[`strafer_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py))
that walks the camera prims after scene assembly and stamps the
schema. Setting it only on `d555_camera_perception` is sufficient if
we want to scope the change to the bridged camera; the policy camera
isn't published, so its warnings are cosmetic and could be left.

The intrinsics from
`strafer_shared.constants` (`D555_FOCAL_LENGTH_MM = 1.93`,
`D555_HORIZONTAL_APERTURE_MM = 3.68`, perception 640×360) imply
`fx = fy = 640 × 1.93 / 3.68 ≈ 335.65`, `cx = 320`, `cy = 180`,
`k1..k5 = 0`. These should be the values stamped onto
`omni:lensdistortion:opencvPinhole:{fx, fy, cx, cy, k1..k5,
imageSize}`.

## Acceptance criteria

- [ ] `make sim-bridge` and `make sim-bridge-gui` startup logs no
      longer contain the two `camera_info_utils` warnings for the
      perception camera (cosmetic warnings on the policy camera are
      acceptable to leave if the fix is scoped to the bridged
      camera).
- [ ] `/d555/color/camera_info` continues to publish:
      `distortion_model = "plumb_bob"`, `d = [0, 0, 0, 0, 0]`,
      `k = [335.65, 0, 320, 0, 335.65, 180, 0, 0, 1]`,
      `width = 640`, `height = 360`. (Same numbers as today; the goal
      is to take the supported code path, not change the contract.)
- [ ] `/d555/depth/camera_info` mirrors the color stream's `k`/`d`
      (same physical sensor on the real robot).
- [ ] Jetson-side RTAB-Map / `depthimage_to_laserscan` /
      goal-projection consumers continue to work unchanged.
- [ ] If a startup event is the chosen hook, it lives in
      [`strafer_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
      next to the existing `lift_ground_plane_to_floor` event, and
      the change is gated on the perception camera being present in
      the scene (NoCam variants must not crash).

## Investigation pointers

- Bridge-side warning source:
  `Workspace/IsaacSim/source/extensions/isaacsim.ros2.bridge/python/impl/camera_info_utils.py:41-100`.
- Intended Isaac Sim API to set the schema:
  `Workspace/IsaacSim/source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py`
  (`Camera.set_distortion_coefficients`, lines 1873+).
- The schema attribute names are predictable from
  `OPENCV_PINHOLE_ATTRIBUTE_MAP` in `isaacsim/sensors/camera/camera.py`
  if we'd rather set them directly via `prim.GetAttribute(...).Set(...)`
  without going through the Camera helper.
- Smoke check: the same `Forcing fy to fx (335.6521... != 335.6521...)`
  warning that fires today is informational (renderer assumes square
  pixels and the 1.7e-5 mm difference is float round-trip); not part
  of this task.

## Out of scope

- Modeling actual D555 lens distortion. The real D555 has a fisheye
  lens, but the project deliberately runs sim cameras with zero
  distortion — see the rationale in
  [`context/bridge-runtime-invariants.md`](context/bridge-runtime-invariants.md#camera-resolutions-sim-mirrors-real)
  about avoiding sim-only distortion that consumers would have to
  invert. The contract stays "plumb_bob with zeroed `d`"; we're only
  changing how the schema is stored on the prim.
- The `Forcing fy to fx` rounding warning. It's a 1.7e-5 mm
  difference between axes from the focal-length-to-pixel computation
  and the renderer rounds it cleanly; not worth chasing.
- The `SdRenderVarPtr missing valid input renderVar
  DistanceToImagePlaneSDbuff` warning — that's an Isaac Sim
  synthetic-data startup-ordering message and orthogonal to
  distortion. File separately if depth is observably broken on the
  Jetson side.
- The `Deprecated: direct use of ITimeline callbacks` warning — fires
  inside Isaac Sim's own modules, not our code.
