# Load the D555 params file, or delete it and pin the filters in the launch

**Type:** task (deploy runtime — config hygiene)
**Owner:** Jetson
**Priority:** P2 — nothing is broken today, because the wrapper's defaults
happen to agree with the file. It is P2 rather than P3 because the file is
already being cited as evidence in merged measurement records, and because a
`realsense2_camera` version bump could change the defaults silently.
**Estimate:** S (one launch argument list, or one deletion plus the same list)
**Branch:** `task/d555-params-file-inert`

## Story

As the **deploy depth path**, I need **the camera's post-processing filter state
fixed by something that is actually read**, so that **"no smoothing reaches the
policy" is a property of the configuration rather than a property of whichever
wrapper version is installed.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

`source/strafer_ros/strafer_perception/config/d555_params.yaml` sets
`decimation_filter.enable`, `spatial_filter.enable`, `temporal_filter.enable`
and `hole_filling_filter.enable` to `false`, and its header offers itself as a
`--params-file`. **No launch file, compose service, Dockerfile or entrypoint
loads it.** `perception.launch.py` includes `rs_launch.py` with an explicit
`launch_arguments` dict carrying no `--params-file`, and pins only the depth and
colour profiles plus `global_time_enabled`. A repo-wide grep finds the filename
only in its own header comment. It is installed to `share/` by `setup.py`, so it
looks live.

The filters are therefore off by wrapper default, which is the same outcome by a
different mechanism — and the difference matters twice.

First, the deploy depth path's one guarantee is that it applies no spatial
smoothing: the only spatial operator between the sensor and the policy is the
8×8 block median in `obs_pipeline.downsample_depth`
([`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md) §2).
If a wrapper bump turned the spatial filter on, that guarantee would break with
nothing in the repo changing and no test failing.

Second, the file is already load-bearing in prose.
[`d555-invalid-pixel-statistics.md`](../../completed/d555-invalid-pixel-statistics.md)
justifies its per-pixel σ table as the sensor's **raw** structure on the grounds
that "post-processing filters are disabled in `d555_params.yaml`". That
justification is void as written, so the filter state during that capture is
unverified — and a temporal filter, had one been on, would have biased those σ
figures **down**. The record wants the correction either way; this brief is the
fix that stops it recurring.

The same file also sets `depth_module.enable_auto_exposure: true`, equally
inert. AE is on by driver default, so the behaviour is unchanged, but it is
another setting the repo believes it controls and does not.

## Acceptance criteria

- [x] The four post-processing filters and `depth_module.enable_auto_exposure`
      are set by something that is read at launch: either `d555_params.yaml`
      passed as `--params-file`, or the file deleted and the same values passed
      as explicit `launch_arguments` in `perception.launch.py`. Pick one; do not
      leave both.
      *Met 2026-09-25: file deleted; `perception.launch.py` passes
      `decimation_filter.enable`, `spatial_filter.enable`,
      `temporal_filter.enable`, `hole_filling_filter.enable` = `"false"` and
      `depth_module.enable_auto_exposure` = `"true"` to `rs_launch.py`, with a
      comment saying they are pinned by contract. All five are in the
      `configurable_parameters` list of `realsense2_camera` 4.58.4's
      `rs_launch.py` (read inside `strafer-cpu:humble`), so each is forwarded
      to the node. The file's `rgb_camera.enable_auto_exposure: true` was not
      carried over: colour is not on the policy path and the wrapper default is
      the same `true`.*
- [x] If the file is kept, its header stops describing a usage that is not the
      usage. If it is deleted, `setup.py`'s `data_files` entry goes with it.
      *Met 2026-09-25: the `config/*.yaml` entry is gone from
      `strafer_perception/setup.py`; it was the directory's only file.*
- [x] A test asserts the filter state the launch requests, in the same idiom as
      the existing inference-config assertions
      (`test_inference_config.py` pins the depth topic this way). A launch-arg
      assertion is enough; no camera needed.
      *Met 2026-09-25: `strafer_perception/test/test_perception_launch.py`
      captures the `IncludeLaunchDescription` arguments (the `_CaptureNode`
      idiom) and asserts each of the five values; checks the installed
      `rs_launch.py` still declares each name, since an undeclared name is
      warned about and dropped rather than failing the include; and checks the
      params file and its `setup.py` entry are gone. The five value
      assertions fail against the pre-change launch (5 failed, 8 passed).*
- [ ] Verified against the running node on hardware: the requested values are
      the values the node reports.
      *Open 2026-09-25: the D555 enumerates as 8086:0bdc "Intel RealSense
      Generic Device" with no `/dev/video*` nodes and `realsense2_camera`
      4.58.4 logs "No RealSense devices were found!", so there is no node to
      read back from.*
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *Met 2026-09-25: `source/strafer_ros/README.md` names the pinned filter
      state and its test. Two active briefs that described the file as live or
      pending were updated: `real-d555-depth-texture-capture` step 1, and
      `d555-depth-decode-validity`, whose claim that an undeclared argument
      "fails the include outright" was wrong; it also now notes that 4.58.4
      does not declare `depth_qos`. No context module cited the file.
      Completed records and `docs/measurements/` were left alone (the
      2026-08-04 correction is out of scope).*

## Investigation pointers

- `source/strafer_ros/strafer_perception/config/d555_params.yaml` — the file
  itself, including the header that offers it as a `--params-file`.
- `source/strafer_ros/strafer_perception/launch/perception.launch.py` — the
  `rs_launch.py` include and its `launch_arguments` dict; it pins the depth and
  colour profiles and `global_time_enabled`, and nothing else.
- `source/strafer_ros/strafer_perception/setup.py` — the `data_files` entry that
  installs the file to `share/`, which is what makes it look live.
- `source/strafer_ros/strafer_inference/test/test_inference_config.py` — the
  idiom for asserting a launch-time value, already used to pin the depth topic.
- [`completed/d555-invalid-pixel-statistics.md`](../../completed/d555-invalid-pixel-statistics.md),
  the "Measurement 2026-08-04" setup paragraph — the citation that rests on this
  file being loaded.

## Adjacent finding (2026-09-25, not fixed here)

**The three `global_time_enabled` pins in the same include are inert too.**
`perception.launch.py` passes `depth_module.global_time_enabled`,
`rgb_camera.global_time_enabled` and `motion_module.global_time_enabled` =
`"false"`, with a comment that global time is broken on Jetson. None of the three
is in `realsense2_camera` 4.58.4's `rs_launch.py` `configurable_parameters`
(read inside `strafer-cpu:humble`). `launch_setup` prints
`Parameter '<name>' is not supported` for each one, which matches the startup
log, and builds the node's parameters only from the declared list, so the
values never reach the node. The node sets these sensor options itself from
librealsense, so global time is at the driver default, which is probably
**on**. That is the opposite of what the comment says, and it changes the
timestamp semantics that `timestamp_fixer` was written to correct. The same
gap applies to `depth_qos`, which
[`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)
plans to pin.

Suggested follow-up: pass these through `rs_launch.py`'s declared
`config_file` argument. That YAML reaches the node unfiltered; unknown keys are
warned about but still passed. Two cautions. `launch_setup` builds the node
with `parameters=[params, params_from_file]` (`rs_launch.py:160` in 4.58.4), so
the file is applied second and would silently override the five filter and AE
pins if it set any of them; `test_no_params_file_or_config_file_override` in
`test_perception_launch.py` exists to block that, so adopting `config_file`
means replacing it with a test that loads the file and fails if it sets any
`PINNED` key. And the YAML must be a plain parameter dict (dotted keys such as
`depth_module.global_time_enabled: false`, or the equivalent nesting): the
`d555: ros__parameters:` wrapper the deleted file used would be flattened into
`d555.ros__parameters.<name>` and reach no real parameter. Then read the
running values back with
`ros2 param get /d555 depth_module.global_time_enabled` on hardware, and check
whether `timestamp_fixer`'s offset logic assumed global time was off. Also add
a test that every `launch_arguments` key is declared by the installed wrapper,
which would have caught all three. That test would fail today, which is why it
is not in this change.

## Out of scope

- **Deciding whether any filter should be on.** They should stay off — the
  training distribution carries no spatial smoothing, and the deploy reduction is
  the only spatial operator the policy's observation is built around. This brief
  makes the existing choice explicit, not different.
- **Correcting the 2026-08-04 record's prose.** Tracked separately; this brief
  removes the cause, not the citation.
- **The depth QoS setting**, which has its own by-default-not-by-contract
  history.
