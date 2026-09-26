# Decode the D555's 16UC1 depth and carry validity through the reduction

**Type:** task (deploy runtime — parity fix)
**Owner:** Jetson
**Priority:** P1 — it blocks the depth policy on hardware outright, and the
second half of it silently inverts the meaning of a third of the observation.
Implementation waits for goal-b bringup; the brief exists now so the hazard is
owned in-repo rather than living only in a measurement note.
**Estimate:** S (one decode path, one mask threaded through, test vectors)
**Branch:** `task/d555-depth-decode-validity`

## Story

As the **deploy depth pipeline on real hardware**, I need **the D555's 16UC1
depth decoded and its invalid pixels carried as an explicit validity mask that
maps to the training convention**, so that **the policy receives the same
observation semantics it trained on instead of reading invalid depth as
near-obstacles.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

Two defects, both measured against the real sensor on 2026-08-04 (see
[`d555-invalid-pixel-statistics`](../../completed/d555-invalid-pixel-statistics.md)).
The first stops the policy; the second is worse, because it does not.

### 1. The node rejects every real depth frame

`inference_node.py` requires `32FC1` on its depth subscription and drops
anything else, counting `depth_bad_encoding`. The real RealSense driver
publishes `/d555/depth/image_rect_raw` as **`16UC1`** (Z16, millimetres). On
hardware the node therefore drops **100%** of depth frames and never infers.

It is invisible in the sim-bridge lane because the Isaac publisher emits
`32FC1` in metres: the 2026-08-02 arm logged `bad_encoding=0` against
`inferences=4387`.

Nothing currently bridges the gap. `depth_downsampler` does convert
16UC1→32FC1, but publishes 80×45 to `/d555/depth/downsampled`, whereas the node
consumes 640×360 and runs its own `downsample_depth` (which asserts the
full-resolution shape).

#### What that blocks beyond deployment

The consequence reaches further than "the policy cannot run on hardware", and
it was not written down here until
[`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md)
looked for real-sensor depth and found none. **Every depth inference on record
in either repository was made on Isaac-bridge depth** — 4 387 (2026-08-02),
24 892 (the [2026-08-17 mission gate](../../../measurements/goal-a-rig-gate-2026-08-17/README.md)),
1 799 (the [2026-08-22 capture](../../../measurements/goal-a-attribution-2026-08-22/README.md)).
There are no real-camera runs, and the encoding gate means there can be none
until this lands.

So the depth-subgoal line has no train-versus-real depth comparison available to
it at all. Anything phrased as "the robot's depth" in that line means the node's
pipeline run on renderer depth, which is a much weaker statement than it reads
as — the 2026-08-22 capture is arithmetically excluded from being 16UC1-derived
(0.19 % of its interior values land on the 0.5 mm grid a median of 64 integer
millimetres must produce, i.e. the continuous-float chance rate). Closing this
item is what makes the comparison possible, so its priority is not only the
deployment blocker.

### 2. A decode alone would invert the invalid convention

This is the part that must not be missed, because it fails **quietly**.
`obs_pipeline.downsample_depth` rescues only **non-finite** values:

```python
depth = np.where(np.isfinite(depth), depth, max_depth)   # 0 is FINITE -> untouched
depth = np.median(...)                                    # block median, dragged toward 0
depth = np.where(depth < nearfield_clip, nearfield_fill, depth)   # 0 < 0.4 -> 0.2 m
```

Z16 invalid is `0`, which is finite. A naive millimetre conversion therefore
lets invalid pixels through the `isfinite` rescue, drags the block median toward
zero, and the post-median nearfield rule converts the result into
**`DEPTH_NEARFIELD_FILL` = 0.2 m** — an obstacle just in front of the robot.

Training's observation term is unambiguous in the other direction: `depth_image`
rescues a non-finite pixel to **`DEPTH_MAX` = 6.0 m** before the reduction — the
same rescue the `32FC1` path applies to `+inf`. So the same reduction would mean
"far" in training and "blocked" on hardware. (The noise model's fills act after
the reduction and differ: its shipped `hole_fill="median"` writes the median of the
valid 3×3 neighbours, and `min_range` (0.2 m) where the whole neighbourhood is
invalid, with `too_close_fill="near"`. `noise_models.py` documents where it parts
company with the deploy far clamp. What this brief matches is the observation
term's rescue.)

**This is a parity fix, not a robustness preference.** Scale, from the same
capture: 33.9% of blocks were majority-invalid in one real room — that fraction
is pose-dependent, but the inversion mechanism is not.

## Decision — the invalid convention on deploy (2026-09-25)

Matched to training's `mdp/observations.py:depth_image`, which rescues
non-finite depth to `max_depth` (6.0 m) **before** `reduce_depth_to_policy_grid`
(the same block median, numpy even-count semantics), applies the nearfield
fill (`< DEPTH_MIN` 0.4 m → `DEPTH_NEARFIELD_FILL` 0.2 m) **after** it, then
clamps to `[0, max_depth]`. On deploy:

- the Z16 decode (`obs_pipeline.decode_depth_image`) returns an explicit
  `valid_mask = raw != 0` alongside metres;
- `downsample_depth(..., valid_mask=...)` sets every masked or non-finite pixel
  to `max_depth` before the same median, so a majority-invalid block
  (>= 33 of 64) reads 6.0 m, an exactly-32 block takes numpy's even median of
  the two middle ranks exactly as training would, and a genuine sub-0.4 m
  return still takes the 0.2 m fill;
- `valid_mask=None` is the 32FC1 (sim) path, byte-identical to before.

**Remaining gap (2026-09-25).** One sentinel cannot separate the two causes of
a Z16 `0`: "no return" (far, which this decision matches to training) and "too
close to measure". A surface nearer than the sensor's minimum range — about
0.24 m, the bottom of the bench capture's finite 0.244–0.399 m band — returns
`0` and so reads **6.0 m ("free") on deploy, where training renders it at its
real distance and fills it to 0.2 m ("blocked")**. The camera sits near the
chassis front face, so a wall approach reaches that range. Nothing regresses
(before this branch every Z16 frame was dropped), but the gap is open until a
close-wall capture on hardware measures it; see the hardware item below.

## Acceptance criteria

- [x] The node decodes **16UC1** (Z16, millimetres → metres) on its depth
      subscription, alongside the existing `32FC1` path. Encoding is detected,
      not assumed.
      *(2026-09-25: `_on_depth` routes every frame through
      `decode_depth_image` on its own `encoding`: `16UC1`/`mono16` → uint16 mm
      → float32 m (× 0.001, as `depth_downsampler` does), `32FC1` → the
      existing float path, anything else → `depth_bad_encoding`. Both honour
      `is_bigendian` and accept packed rows or `step`-padded rows; any other
      buffer length is `depth_bad_shape`. A frame that decodes cleanly at any
      resolution other than 640×360 (the driver's default profile, if it
      rejects the pinned one) is also dropped at the gate as
      `depth_bad_shape` (the counter counts every drop; its warning is
      throttled to once per 5 s), rather than cached for `downsample_depth` to raise on
      the tick (`TestZ16Decode::test_off_resolution_*`). `scripts/obs_parity.py`
      decodes the same way and hands the mask to
      `reassemble_obs_from_extracted`; `test_parity.py::TestReassembly::test_z16_mask_reads_an_invalid_block_as_depth_max`
      and `test_obs_parity_self_check.py` (the CLI's `_self_check_stream` over a
      synthetic one-tick Z16 bag) pin that an all-zero block reads 1.0
      normalized there, not the fill.)*
- [x] An **explicit validity mask** accompanies depth through
      `downsample_depth`, rather than validity being inferred from `isfinite` or
      from any sentinel value. Neither `0` nor `+inf` may be load-bearing.
      *(2026-09-25: `downsample_depth(..., valid_mask=)`; the Z16 decode
      produces it once, from the format's own "no return" code, and the
      node caches it in the same locked `(array, mask, stamp, rx_t, seq)`
      tuple as the frame, so a `timer_reuse` tick consuming the cached frame
      again gets that frame's mask (pinned by
      `test_the_mask_travels_with_a_reused_frame`). The 32FC1 path passes no
      mask and keeps its `isfinite` rescue deliberately — the byte-identity
      criterion below requires it.)*
- [x] **Z16 invalid maps to `DEPTH_MAX` (6.0 m)** — the training observation
      term's convention (its non-finite rescue before the reduction).
      *(2026-09-25: before the median; see the decision above.)*
- [x] **Genuine sub-0.4 m returns keep the nearfield fill.** The bench capture
      showed the sensor returns finite values in 0.244–0.399 m for 4.16% of
      pixels, so that rule is load-bearing and must survive: the fix must
      distinguish *invalid* (→ 6.0 m) from *too close* (→ `nearfield_fill`),
      which a single sentinel cannot.
      *(2026-09-25: the nearfield rule is unchanged and still runs after the
      median; a valid 0.3 m block reads 0.2 m.)*
- [x] **Required test vectors**, as unit tests on `downsample_depth`:
      an **all-zero block** and a **majority-zero block** each produce
      **6.0 m**, never 0.2 m. Plus a genuine sub-0.4 m block still producing
      `nearfield_fill`, so the two paths are proven distinct.
      *(2026-09-25: `test_obs_pipeline.py::TestDownsampleDepthValidityMask` —
      all-zero, 33-zero, 40-zero-beside-0.3 m (each 6.0), all-0.3 m (0.2),
      exactly-32-zero (numpy's even median, strictly between 2.0 and 6.0).
      Node level, `test_inference_runtime.py::TestZ16Decode`: a full-resolution
      16UC1 `sensor_msgs/Image` decodes with no `bad_encoding`, infers, and
      downsamples to those values; an unknown encoding still counts
      `bad_encoding`.)*
- [x] Sim-lane behaviour unchanged: a `32FC1` frame with `+inf` invalids
      produces byte-identical output to today (regression, not a rewrite).
      *(2026-09-25: `TestDownsampleDepthNoMaskRegression` compares against a
      verbatim copy of the pre-change function over 8 seeded 640×360 frames
      mixing +inf, −inf, NaN, 0, sub-0.4 m and >6 m values plus whole blocks of
      each, asserting `tobytes()` equality with and without an explicit
      `valid_mask=None`; the node's 32FC1 decode returns
      `np.frombuffer(..., float32)` exactly as before.)*
- [ ] Verified on hardware: `depth_bad_encoding` and `depth_bad_shape` stay 0
      and `inferences` advances with the real D555 attached. Include a
      close-wall check: face a flat wall from about 0.15 m, 0.25 m and 0.35 m
      and record what the policy cells covering it read, to measure the
      too-close gap in the decision above.
      *(2026-09-25: not run — the D555 enumerates as 8086:0bdc "Intel
      RealSense Generic Device" with no `/dev/video*` nodes and
      realsense2_camera 4.58.4 reports "No RealSense devices were found!", so
      no real depth stream exists to verify against. The cadence line now ends
      `| z16 frames=N majority_invalid_cells=P% over M frames`, which is what
      to read when it runs. The close-wall check was added on review the same
      day and is equally blocked on the camera.)*
      *(2026-09-25, later: partly met.*
      - *Measured, after a power cycle brought the D555 back on USB 3.2: the merged
        node (images `de3a865e5810`, v3 loaded) ran 170 s on the real `16UC1`
        stream. `depth_bad_encoding` and `depth_bad_shape` read 0 on all 18
        counter lines, `z16 frames` rose from 401 to 10588, and the log has no
        exception.*
      - *The rate was 59.9 frames a second rather than 30, because the driver
        publishes every depth frame twice on this host.*
      - *Not met: `inferences` stayed 0. The rig has no motor controller, so there
        was no goal, TF or odometry. The IMU is also disabled on this host's
        kernel, and `/d555/imu/filtered` is a watchdog source, so it would hold
        the node regardless.*
      - *The close-wall check was not run.*
      - *See
        [`real-d555-hardware-readback-2026-09-25`](../../../measurements/real-d555-hardware-readback-2026-09-25/README.md)
        and
        [`d555-l4t-stream-integrity`](../reliability/d555-l4t-stream-integrity.md).)*
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *(2026-09-25: `source/strafer_ros/README.md` now states the dual decode,
      the mask convention and the appended cadence counters; no context module
      or guide claimed 32FC1-only. [`real-d555-depth-texture-capture`](real-d555-depth-texture-capture.md)
      gains a dated note that an offline Z16 reduction must pass the mask.)*
- [x] No regression in the workflows the touched code supports.
      *(2026-09-25: `tools/run_ros_tests.sh ros` — 809 passed across the seven
      packages, strafer_inference 541 passed / 11 skipped, after the review
      fixes (536 before them). Each of the branch's commits passes the
      strafer_inference suite on its own. The new counters
      are appended after `stale_sources[...]`, the last field, so the existing
      `cadence:` token test holds unchanged; nothing under `tools/` parses the
      line.)*

## Adjacent — pin the driver's depth QoS while this lane is being made to work

Once the decode lands, the real lane's depth subscription becomes live for the
first time and its QoS starts to matter. Two things to settle here rather than
separately, because this brief is what makes them measurable:

- **Pin `depth_qos` explicitly** in `perception.launch.py`'s
  `launch_arguments` to `rs_launch.py`, alongside the stream profiles already
  pinned there. Depth currently inherits the wrapper's `SYSTEM_DEFAULT` (which
  resolves RELIABLE), so a RELIABLE subscriber works *by default rather than by
  contract*, and a driver-version change would break it silently — a reliable
  subscriber receives **nothing** from a best-effort publisher. Note that the
  `launch_arguments` dict is the surface (the never-loaded `d555_params.yaml`
  was deleted by [`d555-params-file-inert`](../../completed/d555-params-file-inert.md)).
  An argument name `rs_launch.py` does not declare does **not** fail the
  include: `realsense2_camera` 4.58.4 prints `Parameter '<name>' is not
  supported` and never forwards it to the node, so verify the name against the
  installed wrapper's `configurable_parameters`.
- **Then decide the real lane's `depth_reliability`.** The sim lanes subscribe
  RELIABLE via `STRAFER_DEPTH_RELIABILITY`
  ([`depth-qos-reliable-flip`](../../completed/depth-qos-reliable-flip.md));
  the real lane deliberately kept `best_effort` because it could not be
  measured while every frame died at the encoding gate. Parity is the goal, but
  the real lane has its own contention history on this topic — `timestamp_fixer`
  is a RELIABLE subscriber of the same stream and was recorded dropping
  fragmented Images under load — so this wants a measurement on hardware, not
  an assertion.

### Status of the adjacent items (2026-09-25)

**Open — pinning by launch argument is unavailable on the installed wrapper.**
`/opt/ros/humble/share/realsense2_camera/launch/rs_launch.py` in
`strafer-cpu:humble` (realsense2_camera **4.58.4**) declares 80
`configurable_parameters` and none of them is a QoS argument (`grep qos` on the
file is empty). The node library does carry a `%s_qos` format string
(`strings librealsense2_camera.so`), i.e. per-stream QoS parameters exist on
the node, but `launch_setup` forwards only the declared set plus the
`config_file` yaml, so a `depth_qos` launch argument would never reach it.
Correction to the first bullet, read from the file rather than executed: on
4.58.4 an undeclared argument does **not** fail the include — `launch_setup`
prints a "Parameter '…' is not supported" warning and drops it, a silent no-op
rather than a loud failure. The one untested route left
is `config_file`, whose yaml `launch_setup` passes to the node unfiltered
(with the same warning); whether the node honours `depth_qos` from it wants a
run against a streaming camera. QoS is unchanged by this branch, and the
`depth_reliability` decision still waits on a hardware measurement.

## Investigation pointers

- Encoding gate: `strafer_inference/strafer_inference/inference_node.py`, the
  depth callback's encoding check.
- The reduction and the inversion: `obs_pipeline.downsample_depth`.
- Constants: `strafer_shared/constants.py` — `DEPTH_MAX`, `DEPTH_MIN`,
  `DEPTH_NEARFIELD_FILL`.
- An existing 16UC1→metres conversion to mirror:
  `strafer_perception/depth_downsampler.py`; `goal_projection_node.py` also
  handles both encodings and documents the dual contract.
- The measured sensor behaviour this brief consumes:
  [`d555-invalid-pixel-statistics`](../../completed/d555-invalid-pixel-statistics.md).

## Out of scope

- Changing the block reduction. The median is settled permanently by the
  measurement brief; this changes only what invalid pixels *mean* on their way
  into it.
- Re-opening the depth geometry question — settled, see
  [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md).
- A calibrated sim-versus-real depth-realism comparison. Recorded as a
  possibility in the measurement brief, not filed; its trigger is real-robot
  deployment going live.
