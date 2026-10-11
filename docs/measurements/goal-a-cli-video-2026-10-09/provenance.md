# Provenance — two CLI-submitted missions on the v3 policy, recorded on video, 2026-10-09

captured_date: 2026-10-09

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (ASUS GX10, a GB10 DGX Spark system, aarch64) | 7.0.0-1019-nvidia | driver 580.173.02; the bridge, the planner and VLM services, the RTSP recording and the video processing |
| robot | strafer-nx (Jetson Orin NX 16 GB, L4T R36.4.3) | 5.15.148-tegra | the deploy stack in containers, the CLI set's harness, the first-person recorder and the analysis |

The two hosts were joined by the direct cable, which carried DDS and the planner and VLM calls.

Sources:
- `host_facts.txt` holds the host names, both hosts' architecture, the robot's module, L4T and
  kernel, and the sim host's DMI name and family (`GX10`, `DGX Spark`). These were read on
  2026-10-10; neither host had rebooted since September.
- Kit's banner (`logs/dgx/goal_a_video_bridge.log`, lines 13–23) also records the sim host's
  driver, GPU and kernel at session time.
- The robot's device-tree model string, `NVIDIA Jetson Orin Nano Seeed recomputer classic
  Robotics`, names an Orin Nano. Its module id (`p3767-0000`), the Seeed image name
  (`…orin-nx-16g…`) and its 16 GB of memory are the Orin NX 16 GB's.

## Tree and stack

| side | tree | stack |
|---|---|---|
| sim | `3b13ffa` (`git=3b13ffa` on line 1 of `logs/dgx/goal_a_video_bridge.log`), the checkout the CLI set's bridge ran from | Isaac Sim 6.0.1 (`VERSION` 6.0.1-rc.7+release.42383.32955d8d.gl, the `env_isaaclab3` package's) + Isaac Lab `ffff603ea` (`v3.0.0-beta2.patch1`, with the local `omni.kit.telemetry` deletion that `isaac-lab-upgrade-landing-2026-09-12` records), conda `env_isaaclab3`; Kit experience `isaaclab.python.rendering.kit` with the livestream; Kit log `kit_20261009_120627.log` (line 5 of the same console). The versions, the commit and the experience are in `host_facts.txt` |
| robot | images built at `9b80e99` (revision label `9b80e9976961`), the CLI set's images | `strafer-cpu:humble` `d475acb6b733`, `strafer-gpu:humble` `8dc409b0efb2` (`logs/stage0/images.txt`; tags in `host_facts.txt`) |

`run_sim_in_the_loop.py` ran unchanged at `3b13ffa`. The bridge console's first line records
`git=3b13ffa`. `host_facts.txt` shows that the checkout has not moved since 2026-09-24 and that the
file has no change against `3b13ffa`; it was last edited on 2026-09-19. `tools/bridge_fixed_view.py`
loads it as a module and wraps its `_run_bridge_mode`. On the second launch, the one the missions
ran on, that wrapper posed the viewport camera after the env's reset and applied the pose file's two
UI entries: it hid the editor UI and closed the "Simulation Output Settings" window.

## Artifact

| file | sha256 | matches |
|---|---|---|
| `strafer_depth_subgoal_v3_999.onnx` (in the `inference` container) | `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91` | the exported-artifact row of `depth-subgoal-v3-retrain-2026-09-21`, as in the CLI set |
| `strafer_depth_subgoal_v3_999.json` | `d22e3504e6aa2f0cefbcf2af41561c8acbab4ecf87a8c2e3a4b494294bf33ab5` | the same |

## GPU and Kit

There was one Kit actor on the sim host at a time. `session/dgx_up.sh` refuses to launch the bridge
while any compute process other than the planner and VLM services is listed (`session/dgx_up.sh`
lines 17–19); both launches went ahead (the console lines in `host_facts.txt`).
- **First launch:** 17:04:43 UTC. Kit's first log line is at 17:04:44 (`logs/dgx/bridge_launch1.log`
  line 5). It was stopped by PID once its stream showed the editor UI.
- **Second launch:** 17:06:26 UTC, 17:06:27 in `logs/dgx/goal_a_video_bridge.log` line 5. Both
  missions ran on it.

At the end the bridge and both services were stopped by PID.

## Timeline (2026-10-09 UTC)

| | |
|---|---|
| planner and VLM launched; both `model_loaded` | 17:03:35; by 17:04:38 (`host_facts.txt`) |
| robot stack up, fresh SLAM key | 17:07 |
| G1: command formed, accepted, result | 17:11:58, 17:12:12, 17:16:19 |
| L2, first attempt: transit failed, nothing submitted | 17:18:44 to 17:48:45 |
| L2: one manual move (`tools/nudge.py`) | between 17:48:45 and the second attempt's start at 17:51:13 |
| L2: command formed, accepted, result | 17:54:59, 17:55:44, 17:58:20 |
| teardown: the bridge and services stopped by PID | after L2's recording stopped at 17:58:43 |
