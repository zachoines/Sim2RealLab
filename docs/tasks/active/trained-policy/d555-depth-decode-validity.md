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

Training's convention is unambiguous in the other direction: invalid maps to
**`DEPTH_MAX` = 6.0 m** in both the observation term and the noise model. So the
same reduction would mean "far" in training and "blocked" on hardware.

**This is a parity fix, not a robustness preference.** Scale, from the same
capture: 33.9% of blocks were majority-invalid in one real room — that fraction
is pose-dependent, but the inversion mechanism is not.

## Acceptance criteria

- [ ] The node decodes **16UC1** (Z16, millimetres → metres) on its depth
      subscription, alongside the existing `32FC1` path. Encoding is detected,
      not assumed.
- [ ] An **explicit validity mask** accompanies depth through
      `downsample_depth`, rather than validity being inferred from `isfinite` or
      from any sentinel value. Neither `0` nor `+inf` may be load-bearing.
- [ ] **Z16 invalid maps to `DEPTH_MAX` (6.0 m)** — the training convention.
- [ ] **Genuine sub-0.4 m returns keep the nearfield fill.** The bench capture
      showed the sensor returns finite values in 0.244–0.399 m for 4.16% of
      pixels, so that rule is load-bearing and must survive: the fix must
      distinguish *invalid* (→ 6.0 m) from *too close* (→ `nearfield_fill`),
      which a single sentinel cannot.
- [ ] **Required test vectors**, as unit tests on `downsample_depth`:
      an **all-zero block** and a **majority-zero block** each produce
      **6.0 m**, never 0.2 m. Plus a genuine sub-0.4 m block still producing
      `nearfield_fill`, so the two paths are proven distinct.
- [ ] Sim-lane behaviour unchanged: a `32FC1` frame with `+inf` invalids
      produces byte-identical output to today (regression, not a rewrite).
- [ ] Verified on hardware: `depth_bad_encoding` stays 0 and `inferences`
      advances with the real D555 attached.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Adjacent — pin the driver's depth QoS while this lane is being made to work

Once the decode lands, the real lane's depth subscription becomes live for the
first time and its QoS starts to matter. Two things to settle here rather than
separately, because this brief is what makes them measurable:

- **Pin `depth_qos` explicitly** in `perception.launch.py`'s
  `launch_arguments` to `rs_launch.py`, alongside the stream profiles already
  pinned there. Depth currently inherits the wrapper's `SYSTEM_DEFAULT` (which
  resolves RELIABLE), so a RELIABLE subscriber works *by default rather than by
  contract*, and a driver-version change would break it silently — a reliable
  subscriber receives **nothing** from a best-effort publisher. Note that
  `d555_params.yaml` is **not** the surface: its own header offers it as a
  `--params-file` and no launch file loads it, so anything written there is
  inert. An argument name `rs_launch.py` does not declare fails the include
  outright, so verify the name against the installed wrapper.
- **Then decide the real lane's `depth_reliability`.** The sim lanes subscribe
  RELIABLE via `STRAFER_DEPTH_RELIABILITY`
  ([`depth-qos-reliable-flip`](../reliability/depth-qos-reliable-flip.md));
  the real lane deliberately kept `best_effort` because it could not be
  measured while every frame died at the encoding gate. Parity is the goal, but
  the real lane has its own contention history on this topic — `timestamp_fixer`
  is a RELIABLE subscriber of the same stream and was recorded dropping
  fragmented Images under load — so this wants a measurement on hardware, not
  an assertion.

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
