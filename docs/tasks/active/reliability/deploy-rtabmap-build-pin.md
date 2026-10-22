# Pin the robot image's rtabmap build, so a rebuild cannot change SLAM unnoticed

**Type:** task (deploy reproducibility: the robot's SLAM build changes with every image rebuild, and nothing records
it)
**Owner:** Either (the Dockerfile change on any host; the rebuild and its check on the robot)
**Priority:** P2. The 2026-10-19 rebuild moved the robot's rtabmap from 0.23.7 to 0.23.13. The frame-health record
(2026-10-17) states 0.23.7; no earlier record states its build. The reprocessing record finds the same outcome under
both builds by its registered rule, but different behaviour: under 0.23.13 refinement varies far less from run to
run, and the deployed set's metre-scale outliers move (new on F5 and F6, gone on F2r2).
**Estimate:** S
**Branch:** `task/deploy-rtabmap-build-pin`

## Story

As **whoever compares a SLAM measurement with the robot it describes**, I want **the image to carry a known rtabmap
build, and to say which one it carries**, so that **a rebuild cannot change SLAM without a record of it.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`lane-rtabmap-reprocess-2026-10-21`](../../../measurements/lane-rtabmap-reprocess-2026-10-21/README.md) §5,
  and its provenance's image table.
- [`lane-slam-frame-health-2026-10-17`](../../../measurements/lane-slam-frame-health-2026-10-17/README.md)'s
  provenance (images labelled `9f7d20758d5b`, rtabmap 0.23.7).

## Context

Read at `72b9ba7` (`source/strafer_ros/deploy/docker/Dockerfile.cpu`, unchanged since `85cc34df`).

- **The image floats.**
  - `Dockerfile.cpu` starts `FROM ros:humble-ros-base`, a tag, not a digest.
  - It installs rtabmap through `rosdep install` after an `apt-get update` from the live ROS apt source.
  - It names no package version, holds nothing (`apt-mark hold`), and writes no package manifest into the image.
  - Its only provenance is the git revision it was built from: the label `org.opencontainers.image.revision`, the
    `STRAFER_IMAGE_REVISION` variable and the start line that prints it. None of these identifies the packages.
- **What changed unnoticed.**
  - The rtabmap databases of 2026-10-17 were written by rtabmap 0.23.7 (`Admin.version`).
  - The image rebuilt on 2026-10-19 carries `ros-humble-rtabmap 0.23.13-1jammy.20261002.091850`. Its rtabmap-ros
    wrapper packages are at `0.23.13-1jammy.20261008.*` (`-slam`, `-launch`, `-demos`, `-examples`, `-ros`) and
    `0.23.13-1jammy.20261002.*` (the rest), by that rebuild's log (`chassis-bringup-2026-10-19`'s deposit,
    `images/make_images.log`).
  - The image that ran on 2026-10-17 (revision label `9f7d20758d5b`) is gone from the robot host. Which 0.23.7 build
    it carried cannot now be read.
- **Why it matters.** The reprocessing record's source reading finds changes on refinement's path (the guess-window
  search becomes exact, bundle-adjustment outliers are counted per observation) and in the vocabulary's index
  rebuild rule. Its 0.23.13 replicate behaves differently from 0.23.7 on the same data.
- **What exists to pin against.** The ROS 2 Humble snapshot archive (`snapshots.ros.org/humble/<date>/`, signed by
  key `4B63 CF8F DE49 746E 98FA 01DD AD19 BAB3 CBF1 25EA`) publishes dated snapshots. The reprocessing record built
  its two rtabmap images from the snapshots of 2026-09-14 and 2026-10-12.

## Acceptance criteria

- [ ] **A fixed package source or fixed versions.** Either a dated snapshot apt source, or explicit
      `ros-humble-rtabmap*` versions with `apt-mark hold`. The choice is recorded, with how a deliberate upgrade
      is made.
- [ ] **The base image by digest** (`ros@sha256:…`), with the tag in a comment.
- [ ] **A build label carrying the pinned build string,** from the build argument the install line uses, and a build
      step that fails when `dpkg-query -W ros-humble-rtabmap` disagrees with it. `docker image inspect` then shows the
      build without starting a container.
- [ ] **The container start line names the build,** beside the revision line it already prints.
- [ ] **On the robot,** a rebuild with `--no-cache --pull` and no source change keeps the same rtabmap build string,
      checked from the label and from `dpkg -l` inside the container.
- [ ] **Standing rule for the measurement register:** every lane or robot SLAM record's provenance states the rtabmap
      package build string. Image digests do not survive rebuilds. Stated in `docs/measurements/README.md`, beside
      its description of `provenance.md`.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: the images build, and `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/deploy/docker/Dockerfile.cpu` (the `FROM`, the `rosdep install` layer, the label block).
- `source/strafer_ros/deploy/docker/Dockerfile.gpu` floats the same way, but does not install rtabmap.
- The reprocessing record's deposit, `image/<version>/Dockerfile`: a snapshot source pinned with its key.

## Out of scope

- **Which rtabmap build to run.** That is a measurement decision; this brief makes the build fixed and visible.
- **The rtabmap parameters** (`rtabmap_params.yaml`).
