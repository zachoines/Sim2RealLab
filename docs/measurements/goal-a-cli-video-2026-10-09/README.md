# Two CLI-submitted missions on the v3 policy, recorded on video, 2026-10-09

Two missions of the CLI confirmation set (`goal-a-cli-confirmation-2026-10-03`) were run again on
the sim-bridge rig, on the same stack, through the autonomy CLI with the executor in the loop,
and recorded on video:
- **G1** (−2.00, 2.25), a straight reach;
- **L2** (−2.75, −2.25), where the executor first turns the robot in place and the policy then
  drives.

Both ended on the policy node's own `SUCCEEDED`, and the CLI reported each as `succeeded`, with no
executor cancel.

Each mission has two videos:
- **a third-person view of the sim**, the Isaac Sim viewport streamed over RTSP from a fixed,
  elevated camera, in real time and as an 8× time-lapse;
- **the robot's own camera**, with the mission's text, the node's status, the distance to the goal
  and the elapsed sim time drawn over it.

This record is descriptive evidence for people to watch. It does not re-score the CLI set or change
its reading. The livestream slows the bridge (session RTF 0.107, against 0.132 in the scored set),
and both reaches took longer in sim time than they did there.

## Setup

| | |
|---|---|
| change on the rig | `9b80e99` (the policy-backend navigate budget, PR #235), as in the CLI set; images `strafer-cpu:humble` and `strafer-gpu:humble` with revision label `9b80e9976961`, read from all five running containers |
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, `Environment seed : 42`, one bridge launch for both missions. The room is open-topped: its ceiling is parked below the floor (z −10) |
| cadence contract | `publish 30.00 Hz sim`, `frame_skip=3 (derived, derived 3)`, bridge tick 120 Hz, script defaults |
| bridge launch | the CLI set's launch line with three changes: `tools/bridge_fixed_view.py` runs the unchanged bridge script and holds the viewport camera at a fixed pose; `--livestream 2` with the RTSP extension enabled and selected; no `--headless`. With the livestream, Isaac Lab loads the `isaaclab.python.rendering.kit` experience, where the scored set ran `isaaclab.python.headless.rendering.kit` |
| viewport | world-frame eye (−7.6, −0.6, 4.6), target (−3.1, −0.6, 0.0); the robot spawned at world (−1.65, −0.55), and on the sim bridge the map frame starts at the spawn. The camera sits about 5 m west of the room's west wall, 4.6 m up, looking east and down, so the start, the doorway and both goals outside it are in frame. Kit's editor UI is hidden (`/app/window/hideUi`), so the stream carries only the viewport. The pose was set before the robot stack started and did not change |
| SLAM key | `enrich_goalavideo1` (fresh for this bridge launch) |
| policy | `strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91`, sidecar `d22e3504…33ab5`, verified inside the running `inference` container |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`, node `mission_timeout_s` 60 s, arrival radius 0.30 m; executor `use_sim_time` true on both nodes, policy budget 65.0 s |
| planner / VLM | Qwen3-4B and Qwen2.5-VL-3B on the sim host over the direct cable, `model_loaded` true in 2–5 ms from the `autonomy` container |
| transport | direct cable, the Cyclone pin selected in all five containers; one RELIABLE depth publisher and two RELIABLE subscribers |
| harness | the CLI set's tools, unchanged and checked against its pinned digests: `run_cli_one.sh` (transit to the nominal start (0.075, −0.035, 130°), the passive observer, the planner dry-check, then one `make submit-deploy`), and its analysis tools |
| recording | **third person:** `ffmpeg -rtsp_transport tcp … -c copy` on the sim host, from the moment the observer formed the command until 3 s wall after the first-person recorder ended (2 s sim after the node's result), about 22 s wall after the result. **First person:** `tools/fp_recorder.py` in the `autonomy` container, gated on the same moment. It wrote its mp4 (0.6 and 0.5 MB) inside the container; each file was streamed to the sim host, checked against the recorder's sha256, and then removed from the container, so no video stayed on the robot host |
| window | G1 accepted 17:12:12 UTC, result 17:16:19; L2 accepted 17:55:44, result 17:58:20 (2026-10-09) |

## The two missions

| | G1 | L2 |
|---|---|---|
| command (formed by the observer from the measured start) | `move 3.105 meters forward and 0.105 meters left` | `move 0.197 meters forward and 3.594 meters left` |
| start offset from nominal | 0.028 m / +0.65° | 0.013 m / +1.36° |
| planner dry-check | one `translate` step, matching to 0.001 m | the same |
| executor pre-rotation | none (the command's bearing is +1.9°) | **3.3 s sim** turning in place through 81.4° (rotation commands 310.09–313.36 s sim; the command's bearing is +86.9°), inside the 3.4 s from the mission's `EXECUTING` at 309.95 s to the policy goal's acceptance at 313.37 s |
| node result / CLI | `SUCCEEDED` / `succeeded`, no executor cancel | `SUCCEEDED` / `succeeded`, no executor cancel |
| reach (policy goal accepted → result) | 24.98 s sim, 246.8 s wall | 17.23 s sim, 156.6 s wall |
| final distance (TF) | 0.2993 m | 0.2995 m |
| terminal approach (0.42 m → result) | 18.72 s sim | 11.18 s sim |
| RTF with the livestream on (observer, per mission) | 0.1012 | 0.1100 |
| cadence p50 / inferences (fresh / reuse) | 30.01 Hz sim / 748 (748 / 0) | 30.00 Hz sim / 516 (516 / 0) |
| `depth_age` p50 / p95 / max | 0.025 / 0.025 / 0.033 s sim | 0.025 / 0.025 / 0.033 s sim |
| map→odom corrections in the window (`analysis/drift_summary.json`) | 13, largest 0.054 m | 19, largest 0.087 m |

**Comparison with the scored set:**
- **RTF.** The session RTF here is 0.1071 by the ride-along (`analysis/drift_summary.json`), against
  0.1324 by the same tool in the scored set; per mission the set's G1 and L2 ran at 0.1271 and
  0.1361.
- **Reach and terminal approach.** The scored set's G1 and L2 reached in 8.83 and 12.53 s sim,
  with terminal approaches of 3.46 and 6.50 s. Here, most of each reach is spent between 0.42 m and
  the 0.30 m radius, as on three of the v3 gate's seven reaches (7.0–13.9 s there). This record does
  not attribute the difference.

Each mission ran with one or two mid-mission watchdog skips in the inference node
(`analysis/scrape.stdout`). No subgoal anchor was in collision.

## What the videos show

| file | length | bytes | content |
|---|---:|---:|---|
| `video/G1_third_person.mp4` | 283.0 s | 285 229 079 | real time, h264 1440×900 at 60 fps, from the command to 22 s wall after the result |
| `video/G1_third_person_8x.mp4` | 35.4 s | 3 125 923 | the same at 8× |
| `video/G1_first_person.mp4` | 27.8 s | 607 359 | the robot's camera, 835 frames at 30 fps of sim time |
| `video/L2_third_person.mp4` | 222.7 s | 223 240 602 | real time, as above |
| `video/L2_third_person_8x.mp4` | 27.9 s | 2 436 013 | the same at 8× |
| `video/L2_first_person.mp4` | 23.4 s | 507 529 | the robot's camera, 702 frames at 30 fps of sim time |

The two real-time files are deposited in three parts each and reassemble to the digests in
`DEPOSIT.md`.

**Third person.** The robot starts inside the room, about 0.8 m (G1) and 1.0 m (L2) east of its
west doorway, and is seen through it. The start is the same nominal pose in the map frame for both
missions, but SLAM had moved that frame about 0.56 m by L2 (`map→odom` in the observer records'
`start.mo`), so in the sim L2 started 0.57 m further north-east than G1, partly behind the north
wall segment. The robot leaves through the doorway and drives to a goal outside the room:
north-west for G1, on the left of the frame, and south-west for L2, on the right. The approach then
slows as the robot closes on the radius.
- In L2 the robot first turns in place by 81°, which is the executor's pre-rotation (the
  observer's pose series). The policy takes over once that turn ends.
- The stream runs at wall speed, at about 9.5 wall seconds per sim second. The 8× time-lapse brings
  each mission under 40 s.
- Three stills per mission (`stills/`) are taken from the real-time file: at the submission, at the
  moment the robot had covered half its start distance, and just after the node's result
  (`analysis/video_times.txt`).

**First person.** The image is the bridge's colour stream (`/d555/color/image_raw`, rgb8 640×360,
stamped on sim time), the same frames the robot stack receives. Every frame was written: no gaps,
none out of order, at a median of 30.0 Hz of sim time. The video therefore plays at sim speed. The
overlay has two lines:
- the label, the command, and the node's status (`EXECUTING`, then `SUCCEEDED`);
- the distance from TF to the goal, and the sim seconds since the mission was accepted. Until the
  policy goal is seen, the goal is the intended one and is labelled so; then it is the dispatched
  goal, under 1 mm away.

During L2's pre-rotation the status reads "executor translate, no policy goal yet", because the
executor turns the robot inside its `translate` skill before it sends the policy its goal.

## Deviations

- **Record name and date.** The set was planned as `goal-a-cli-video-2026-10-05` and recorded on
  2026-10-09. The record takes the recording date. The two tools' docstrings still name the planned
  directory.
- **L2 needed a second attempt and one manual move.**
  - On the first attempt, the scripted transit back from G1's goal pinned the robot on the doorway's
    north jamb at map (−1.35, 0.13). Its first try ran out of its 150 s sim budget; its one retry
    stalled at the same place and was stopped (SIGINT). Nothing was submitted
    (`logs/runs/stale/L2.1791568274/`, `logs/attempts/L2.1/`).
  - `tools/nudge.py` then moved the robot 0.37 m south-west, off the jamb, closed-loop on TF, in
    3.3 s sim (`logs/L2_nudge.out`). It first checked that no policy goal was active and that nobody
    else was commanding `/cmd_vel`. Its target, (−1.55, −0.20) in the map frame, was the doorway's
    centre line in the map as first built. The map frame had since moved about 0.6 m, so the robot
    stopped at the opening's northern edge (`logs/frames/l2_after_nudge.png`).
  - The second attempt's transit reached the start in 23.0 s sim. Everything L2 reports is from that
    attempt.
  - The scored set's FX had a transit run out its budget the same way; there the retry succeeded.
- **The bridge was launched twice.** The first launch streamed Kit's whole editor window
  (`logs/frames/view_check_1.png`). It was stopped before the robot stack started and relaunched
  with the editor UI hidden and the camera moved closer (`logs/frames/view_check_2.png`). Between
  the launches the viewport tool gained the view file's UI entries (`settings`, `hide_windows`). The
  first launch ran an earlier version without them, which is not deposited. The missions ran on the
  second launch, and the SLAM key was used once.
- **G1's third-person recording overran.** The stop signalled the shell that had launched ffmpeg,
  not ffmpeg itself, so the capture ran on until ffmpeg was stopped directly. The deposited G1
  file is that capture cut with `-c copy` to the same tail as L2's: 22 s wall after the result. The
  cut-off 82 s, the robot standing at the goal, is held on the sim host. The stop was fixed before
  L2.
- **The wrapper's own join.** It reads `STALE_RECORD`, the CLI set's known defect, so it exits 3
  after a successful mission. The join was run by hand, as in the set (`logs/runs/<L>.prereg.json`),
  and re-runs from the deposit to the same reading; only the embedded input paths differ
  (`analysis/<L>.prereg_rerun.json`).
- **A stale log line.** On L2's second attempt, the orchestration log's "READY" line came from the
  first attempt's console, read before the new recorder rewrote it. The new recorder was in fact
  ready at 285.0 s sim, 24 s sim before the command (`logs/fpv/L2.fp.out`), so its recording was
  unaffected.
- **The wrapper's output directory** was pointed at this set's own (`CLI_CONFIRM_RUNS_DIR`, which
  its header describes as for tests), so the CLI set's files were not touched.

## What this record does not claim

- **A re-score.** The CLI set's reading stands as recorded. These two runs do not add to it or
  subtract from it.
- **The scored set's timing.** RTF is lower with the livestream on, Kit runs another experience
  file, and both reaches took longer in sim time. The longer terminal approaches are not attributed.
- **A neutral camera.** The third-person viewport was chosen to frame these two goals and held
  fixed for both missions. The goal itself is not drawn in the scene, because a prim in the stage
  would also be seen by the robot's camera.
- **Real-sensor footage.** The first-person video is the bridge's rendered colour stream, not the
  real D555.
- **Ground truth on the overlay.** The distance is from TF in the map frame, not from the
  simulator's own pose. SLAM moved that frame 0.08 m inside G1's window and 0.19 m inside L2's
  (the observer records' `mission.slam_map_odom_shift_m`), and about 0.56 m between the two
  missions. L2's final distance is 0.2995 m in the map frame and 0.157 m in its own start frame
  (`analysis/reach_terminal.stdout`).

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `goal-a-cli-video-2026-10-09/record-files/` |
| deposit commit | `198ad07639c2015f4d4765634614f36c711490fa` |

The deposit holds:
- the six videos and six stills;
- per mission, the command, the planner dry-check, the CLI's JSON transcript and timing, the
  observer's record and the joined reading;
- the ride-along, the five container logs, and the sim host's bridge, planner, VLM and ffmpeg logs;
- the stack checks before the first mission;
- the analysis with its exact commands (`analysis/COMMANDS.md`);
- the three tools written for this set, and the session scripts.

The CLI set's own tools ran unchanged. Their digests are in `cli_set_tools_sha256.txt`, a copy of
that deposit's list. Nine text files had rig network addresses replaced with placeholders before
deposit, and `DEPOSIT.md` lists them.

Restore and verify with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/goal-a-cli-video-2026-10-09/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
for L in G1 L2; do cat video/${L}_third_person.mp4.part-* > video/${L}_third_person.mp4; done
```

sha256 of every file in the deposit:

```
7b04c3361432bcb33d9d9ae7179fd0586925b53554f2b23053aff3c5e8ce659d  analysis/COMMANDS.md
f9f0a3ece8c43e996fa5fbf871ee5137e419b3c6d3baeb85e486f844cb364cbf  analysis/G1.prereg_rerun.json
235630866b6ed9d29d96abde9c76251987d824281ae0cfab9d0a99f1736f6b7c  analysis/L2.prereg_rerun.json
86b7e726586cb1904a0168990ad9249bde983efca17428392fc5ede0b9573803  analysis/drift_summary.json
265c7cfca9d1ec0791aa6e66125bc1bc32c539b7bbc663e31a2f506819e607e7  analysis/drift_summary.stdout
33f4eabd257b2c62d121f18968c18255b8632b4d9acee1274ce492e919c813a0  analysis/reach_terminal.json
95a4b31e4e45b3087b37ccea3c61ec4c3fb974b162fbb2fa6f13de2552bcd339  analysis/reach_terminal.stdout
c873c32ba3317874103cf8df448948b81c19cd8de0a98618fbc834b5ca018e87  analysis/scrape.json
ed9f559218f2cb7aac51d9d040114a68d5af40f026a0b61bad602a29af98e68b  analysis/scrape.stdout
98805b009c652cd92c557e3685cdbab4129c61ba494181d24f56e05bdc183dae  analysis/tools/video_times.py
b37c8aa0972788e1bc7551aa68e4a971c70d8a9211d9ba2176ead6e9a0870597  analysis/video_times.txt
73179029e156bb428fe16596d7be709b309ec01c2ffda825fdbdacb7675c0cd9  cli_set_tools_sha256.txt
65cd736f9e2043d451ef5ab0ec361fd5fe5a41d2ac87997b79275e56f278fb8f  logs/G1.video_one.log
65cd736f9e2043d451ef5ab0ec361fd5fe5a41d2ac87997b79275e56f278fb8f  logs/G1.video_one.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/G1.wrapper.out
9df9d18df6e5d94d18f3c2ab02dbdb64a3c4de428b47bd9e666b6c90bd4614ad  logs/L2.video_one.log
9df9d18df6e5d94d18f3c2ab02dbdb64a3c4de428b47bd9e666b6c90bd4614ad  logs/L2.video_one.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/L2.wrapper.out
052580ed7a0150634bfe2402b78a3e620e728cd6267ffd458854cfe532a0d551  logs/L2_nudge.out
e59ce471134001ead6cb3654b560ab1f40914ba3ddccdc58c68de93458f7a416  logs/attempts/L2.1/L2.fp.out
f2b12248526785c531424f9f947ddc1fa2ea253b736aff70ec46becdc8afbae1  logs/attempts/L2.1/L2.video_one.log
f2b12248526785c531424f9f947ddc1fa2ea253b736aff70ec46becdc8afbae1  logs/attempts/L2.1/L2.video_one.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/attempts/L2.1/L2.wrapper.out
5a3b6856c612094f8241d534e296652a4db493968e71e08226984dee0db24967  logs/attempts/L2.1/L2_first_person.mp4.json
92078399b2a97519771db548f1f284376adadb82ee1fc20dae463b7f55782d4e  logs/dgx/G1_ffmpeg.log
36039e1e83355e05d94f5a7df04a39eb6745feb3dc1d2b7442902a64aba41dd4  logs/dgx/G1_ffmpeg.pid
7a71bedcf82804cdd82d8bdf2b41628dd35f436969921844803f38482600b3f2  logs/dgx/G1_rec_start_epoch.txt
7f16274bdf04bc657d5b007f83c53fb55c2c627ccd392baf36e90296aca91d37  logs/dgx/G1_rec_start_utc.txt
b151d33742de07f71a0d58627113658e51b7c13ab93d3f1587dcad113113707b  logs/dgx/G1_rec_stop_actual_utc.txt
6e985b5d4e7dd90821e7708206e1d7fd7c2ec1df74b1ea7eb4d51b7273386a80  logs/dgx/G1_rec_stop_epoch.txt
3c354add23b70069f3b2d9b7041a8a3a6ab2bcffaf204475ec7fbf35fc49c332  logs/dgx/G1_rec_stop_utc.txt
02f775833463f8c82f93974251255e404254bfddef3fc0f3f81670593903f101  logs/dgx/L2_ffmpeg.log
4173668bd704c7d790c8f8c166fc22ef0a6c52b7e88149623ae728313949d2e2  logs/dgx/L2_ffmpeg.pid
f3985c1595ff97bff4b3f6f154d4c38a2e4e8cbcdbf86c20463604edcc3da844  logs/dgx/L2_rec_start_epoch.txt
9e164017d41f865fb1c1e8310d8155bd3d88f0b6e4c6ba9e57da2c238fefd2d1  logs/dgx/L2_rec_start_utc.txt
033133232c8c282d5dde3a7102c316bb4957c690dc5b8969ebc540ddc3709c02  logs/dgx/L2_rec_stop_epoch.txt
03deba81e965d91c3b8fffa9be8d93669a78aa212a286b572fc2f1774cf89b47  logs/dgx/L2_rec_stop_utc.txt
eaaa06f6b666f4d02a6bc5cd8b75815e44f88eb991eb2f48998d92d9d9d2d9fe  logs/dgx/bridge_launch1.log
545b8d5f72563fecec4c2c65fdc61a39272e23300490b5d34b3ab55caf4ed8f2  logs/dgx/goal_a_video_bridge.log
120590a338d1076e6321523c3e4998292d59038c0b99169af1169d7010f42250  logs/dgx/goal_a_video_envsetup.log
2dd3aefa5d00de4f4377b5144c918ad231b2daae9861e01b707cd60847591e18  logs/dgx/goal_a_video_serve_planner.log
4e6eedc11d3c45f32f3e314713450ebb9345539a00a391ebe17a8a11d26e7621  logs/dgx/goal_a_video_serve_vlm.log
7f0597b0bce852f51d094cf4ab013da8f59fee59d8d188fd2997af8e6fd0709f  logs/fpv/G1.fp.out
502c9f61960683df64530b6f269116a4386d8bb1f697f10e29c6ed0c552de08b  logs/fpv/G1_first_person.mp4.json
b86bbebc73eeb42856562b916487ef549aa314c4aa49995d51ccc615d3a92c45  logs/fpv/L2.fp.out
27ccfdb40bfb7c9652ff2ca55694be4d162101c2fba954a4954cf60f8f6aa0ee  logs/fpv/L2_first_person.mp4.json
bd551bd8fe617343adb8c8a006ccb5c8895d3128d378407c9be6c67aaa191f26  logs/frames/fp_g1_10s.png
5fb607b165be2501884ab7ac72dd41e09649b58fa87196ae6d49f0cc84f1b733  logs/frames/fp_l2_1s.png
d0b7a89896aaa61167e9641b4a8d3417b39f21f376cef70bc21255779e4a8f37  logs/frames/l2_after_nudge.png
f97452d65a2d91fa4be584fa79b45c6c74f353a1f0e9ac05f07d5d30c30654aa  logs/frames/l2_transit_stuck.png
b9a2f92ff2ad146b4062c113bb1bed1bbc24b926304f1f6398c56d5236948cdd  logs/frames/view_check_1.png
c28782e5044d9d496a2db21bf3816722eadebb7fdc03b820cded35e109589d65  logs/frames/view_check_2.png
e5f0772701eeb8f3e5cc94a37ea97c017cc0be06daa2c700e65fc066cdc940e9  logs/probe_recheck.out
7c2f6945580026b581ae40a298ea0eb288b2af6a97a7448858e24a744ff213ea  logs/probe_video_recheck_costmap.npz
893f28e49f74c060cf90a362b78c4d34ddaf904c573fa5b2a3f8b0fc6efa57d4  logs/runs/G1.cli.meta
1d5b2142276694f0b8b4446a40f560e6e15c357ac40c95062ebb8f6d4ab7c36d  logs/runs/G1.cli.out
da984c355db0ab11dbcc3b8c69494d08a495f893252a039d805366e8cb657a6a  logs/runs/G1.cmd
4f00c584f0de3c68e1aac2c0f6a139f7ce1f6b2d19fdd36b273ce14e3a753cc8  logs/runs/G1.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/G1.done
5bb811be419646c99f008b95ca2576bc6f3bea0279b25992eed6a12a6ed1aed9  logs/runs/G1.plancheck.out
2752d3c0d105777576686b9333325ba3b88fd5d4e20fc66a9a8713a6dbbeb3d8  logs/runs/G1.prereg.json
3a321f53d5a34252760aad75cee87a336864272f7745fc39a359ec0b49e02d0e  logs/runs/G1.prereg.out
41e6ba33b4c2c7e9333e70b4615f0594d41ce3edab425f676e0fedcf80648017  logs/runs/G1.run.log
0b90296c56602a392a775a45e6cfbedcfec1fd0a7d6c4f0161b36eb3b9e5af65  logs/runs/G1.to_start.out
4594b6c0890c8a703be07e37a9fad05485eb7b4192a566cc6c20442bb9145cf0  logs/runs/L2.cli.meta
ca0181e480d2975b05b0255f2f83066c8bf88d307b85454d4e9fbede57f210bc  logs/runs/L2.cli.out
b366487892481c1033ed61089c11ae74756e2f94439fdebbd67561d7489ec890  logs/runs/L2.cmd
18862a59b63e3c773ba7eea65028d2ae0549a955bbfc3078ed944b23cf33c7b7  logs/runs/L2.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/L2.done
df4fda4e38c743d02967ee6ccb60bc9c9a95322a85b6fd8060317462cc62b1a0  logs/runs/L2.plancheck.out
6ba0470b1b9bca81607baaca16cc1089e2c68d46a5ae412917e6833da919bf2a  logs/runs/L2.prereg.json
4f0819df0e96453fcff700d14dd0847db15c65b3535208c6ce188aad911ea3ce  logs/runs/L2.prereg.out
37598df65f0c823fe69093d906c51ced643da69d65634f0a4d92a778c4291018  logs/runs/L2.run.log
593e6722bf4dbffe9bf524dac4a3b2c1eb5d778affbe43d113b092a7bedaa57a  logs/runs/L2.to_start.out
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/G1.cli_exited
da984c355db0ab11dbcc3b8c69494d08a495f893252a039d805366e8cb657a6a  logs/runs/container_gate/G1.cmd
4f00c584f0de3c68e1aac2c0f6a139f7ce1f6b2d19fdd36b273ce14e3a753cc8  logs/runs/container_gate/G1.cmd.json
595fe29a2eb8afbefb65f1e95aba3a0b6d9528986a4037ef0b4a1069d324f329  logs/runs/container_gate/G1.json
c097663a6fcf471347d2a4d4e791c2d509372345a3613e4e04daf3fb59eca961  logs/runs/container_gate/G1.mission.out
8f65f74775d6c66e5dee2c4d9527ec495a707f2fe2bd3e538aa5c11c24c651b7  logs/runs/container_gate/G1.progress.jsonl
5dc0c19095fcccb18e1299186707a16554281bb900a61fd2ade15e9492ca9c9f  logs/runs/container_gate/G1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/L2.cli_exited
b366487892481c1033ed61089c11ae74756e2f94439fdebbd67561d7489ec890  logs/runs/container_gate/L2.cmd
18862a59b63e3c773ba7eea65028d2ae0549a955bbfc3078ed944b23cf33c7b7  logs/runs/container_gate/L2.cmd.json
eae899a4c8bbb04fe62b1fd9bbc6f1d5df704481c6223dd131f1aa64c3a12313  logs/runs/container_gate/L2.json
539243cfccc3eb2b0ebdbaba37271768071343af3d4e098183e1a48bb53f127a  logs/runs/container_gate/L2.mission.out
36343bb3805395f4a47a25c44ce83a7f56751ab353018328958c1b569cb3cd3f  logs/runs/container_gate/L2.progress.jsonl
2f5a0566836487853297f503b342ca14e0c01df4386cf7fb39595fc8171ec1ab  logs/runs/container_gate/L2.to_start.json
b0e64604bbc9163b86beabd178c32c16ab3d23d8825e2f4ae5b4c28449bfef33  logs/runs/container_gate/L2.to_start.json.1791568274.bak
e61bcaf4a4161940af2c17c4b034369170948f2c1ca02727084ac233e8e56c8c  logs/runs/container_gate/L2.to_start_retry.json.1791568274.bak
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/G1.cli_exited
da984c355db0ab11dbcc3b8c69494d08a495f893252a039d805366e8cb657a6a  logs/runs/container_gate_all/G1.cmd
4f00c584f0de3c68e1aac2c0f6a139f7ce1f6b2d19fdd36b273ce14e3a753cc8  logs/runs/container_gate_all/G1.cmd.json
595fe29a2eb8afbefb65f1e95aba3a0b6d9528986a4037ef0b4a1069d324f329  logs/runs/container_gate_all/G1.json
c097663a6fcf471347d2a4d4e791c2d509372345a3613e4e04daf3fb59eca961  logs/runs/container_gate_all/G1.mission.out
8f65f74775d6c66e5dee2c4d9527ec495a707f2fe2bd3e538aa5c11c24c651b7  logs/runs/container_gate_all/G1.progress.jsonl
5dc0c19095fcccb18e1299186707a16554281bb900a61fd2ade15e9492ca9c9f  logs/runs/container_gate_all/G1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/L2.cli_exited
b366487892481c1033ed61089c11ae74756e2f94439fdebbd67561d7489ec890  logs/runs/container_gate_all/L2.cmd
18862a59b63e3c773ba7eea65028d2ae0549a955bbfc3078ed944b23cf33c7b7  logs/runs/container_gate_all/L2.cmd.json
eae899a4c8bbb04fe62b1fd9bbc6f1d5df704481c6223dd131f1aa64c3a12313  logs/runs/container_gate_all/L2.json
539243cfccc3eb2b0ebdbaba37271768071343af3d4e098183e1a48bb53f127a  logs/runs/container_gate_all/L2.mission.out
36343bb3805395f4a47a25c44ce83a7f56751ab353018328958c1b569cb3cd3f  logs/runs/container_gate_all/L2.progress.jsonl
2f5a0566836487853297f503b342ca14e0c01df4386cf7fb39595fc8171ec1ab  logs/runs/container_gate_all/L2.to_start.json
b0e64604bbc9163b86beabd178c32c16ab3d23d8825e2f4ae5b4c28449bfef33  logs/runs/container_gate_all/L2.to_start.json.1791568274.bak
e61bcaf4a4161940af2c17c4b034369170948f2c1ca02727084ac233e8e56c8c  logs/runs/container_gate_all/L2.to_start_retry.json.1791568274.bak
cebc6e8be585d7cd8138ce9c2dadb004d77dcf4daa9254bc65d7728a17ee97df  logs/runs/container_gate_all/tf_logger_video.out
308f404cb73425a98d03c525deeedeba78f22b0ce67d6f948f3c7f40e93ff4f0  logs/runs/container_gate_all/tf_video.jsonl
b0e64604bbc9163b86beabd178c32c16ab3d23d8825e2f4ae5b4c28449bfef33  logs/runs/stale/L2.1791568274/L2.to_start.json
e61bcaf4a4161940af2c17c4b034369170948f2c1ca02727084ac233e8e56c8c  logs/runs/stale/L2.1791568274/L2.to_start_retry.json
ede850df02e833cce4e28a7b1685a9dea1346bca104c8ba395b8de3a947ae073  logs/stage0.out
417bcc69cead7640131a17e18eb4fc32acf2c4a366c12df9afd8b0a6a8da546f  logs/stage0/cyclone_pin_in_containers.txt
5f3b14836e6000bbb4551b9be0d7b553a1249ec515981207da47c154d5075a99  logs/stage0/depth_topic_info.txt
c38424f0046e3921fcb1bad0bf6296d4369682e77ecb273e4a45b34fc5482cd8  logs/stage0/executor_checks.txt
f93b884f474c388c24ebcd3b8962a1b95467fd9656c1cfe8953e5f46e0e7c9d6  logs/stage0/harness_sha256.txt
276f72bd1b7966757f57263b3b1316952d8bbed3716337d6844f97c2c9841ab6  logs/stage0/images.txt
8bdec51bfcd3525723ffeb1356e22fae9d594c2a50456e484413443667f15eed  logs/stage0/in_container_artifact_checks.txt
8eccb007a4b8e774df7240419d9a8a2d6899a8b8de22ac3204de00bdb03df361  logs/stage0/probe_container/probe_video_recheck.json
7c2f6945580026b581ae40a298ea0eb288b2af6a97a7448858e24a744ff213ea  logs/stage0/probe_container/probe_video_recheck_costmap.npz
94c75891a7fd6a940d3492c0833de27ffa974c1c6fa8faf3f96e0c27ebaff7fc  logs/stage0/service_health.txt
5ad4197aa3c2c7c2876313d77c0dbcb0b1361d6e41cf9394a6a23e19f56c5187  logs/strafer_autonomy.log
98e5c457f9680b95eaaea3941b2cf03cd6b476b47a32856fb584228898dd3c2b  logs/strafer_inference.log
bda61e0739caed31f85a472dcc9274a8b0bb4b7640f3223c748f7bfce5f78a99  logs/strafer_navigation.log
5b14a37e5c02b3ad9fda85a7906aa5d6a3a55e7d63bb43af180566afe94379a7  logs/strafer_sim_perception.log
30ffcee5b6693aefc5e85d6c449248ea514a1ab8481f7ecba2021bc169c89bae  logs/strafer_slam.log
7c55bba2197956852a43280e81353bd394727e783f0d30aa4708c71c0b6bcba4  logs/up_t0.txt
99d93c8ddc8ca43c51447392cb984a4115498048a6546eab494e25a8b5d17158  session/autonomy-urls-direct.yml
9185bb01a9c791de1730a7345bcbb17900ad2bff0040b8b4821b48cd57887910  session/dgx_up.sh
b55170eeafca319c3b25d70182008819fb05aea6fe03e81e0e7b8a8592d5dc76  session/launch_bridge.sh
cde02a6fd87c2ae1a30f5ab6603bf96b68692140b089935beeb33e5ca1904d89  session/nx_up.sh
d69aaa3c6c9b53b41e7dcd85da02781070e1ee61e111ccd281d3360ff5d6c7aa  session/stage0.sh
63ed47a69994f922f3627bc5a372a0bcdf05f26e4c072576c3e92dabc3594ea6  session/video_one.sh
c35b3022fc673efccb272ecc203a0806ed60a937707c5d1dc5c965f8f812394e  session/view.json
80f4df02901d7d136a540310bf7d4c6fdf9a069802a49a20edc499de3d70b4c9  stills/G1_1_submit.png
14f66c7113eed2b5ac0a0b1e6e19cd7915fb6abf89f350690459d76c1dc0403b  stills/G1_2_midpath.png
a8e74d91b4290a269ce14d44f53e93cf6f3cb2b9531037c22ee013a70c89d1b3  stills/G1_3_result.png
5166662992b05e56a742ac544d509ec3c56f4b2b7fcfc30dedc8c2786349d427  stills/L2_1_submit.png
5ab0b07a6c6e1c2cc2ebcac096650bfc3112c4177a809677da65f0e90a6b86c9  stills/L2_2_midpath.png
847bb87fcb8a5f46a2a1d6f10371d814f6f225dbcbab9143cfe0cfccf132425c  stills/L2_3_result.png
374fa07c44b61d8650951916a5049d58008c73256faccae1a270d40c24918dad  tools/bridge_fixed_view.py
824df89a6b8fd068a7602031cdc900ad476168cad67c1431171195b8c2ac1a8d  tools/fp_recorder.py
c66ab17f281bb751a08ac055770d3667ec030f308c366ef1587543062cb50551  tools/nudge.py
7a742d6feed5f367bd28febff8642586f582aac796b0fcacf1cb7000140ebb64  video/G1_first_person.mp4
cb07b5a439d2e2c5af6eae3debf598834958e877758d61e51f45b0cc5109f32c  video/G1_third_person.mp4.part-00
14d6f248cf6c15553ce7ea499cd807eb7ffca6ec5ebcbb69f9a79d2dbf32c9c2  video/G1_third_person.mp4.part-01
b9e33cffa84039ce22a1e99f97a35fbac6e1a065c1d428ff3c1f981c38540334  video/G1_third_person.mp4.part-02
43f91452b03f07c56b9fe3075a326c656b6b5af6db767f22ed3405903112fa03  video/G1_third_person_8x.mp4
9e15497d628ca51edb3e49c938ce3433aeba71f825eae864a0f3aafa447b9583  video/L2_first_person.mp4
bcb668c620e2cf2116b6e017e01d9fa9cadcb228c603f6cb46c9a2e7de726e93  video/L2_third_person.mp4.part-00
5f6d7e705f6d9fcdf481be203723a8653505eb7dd18f059db6b5abb52cceccc5  video/L2_third_person.mp4.part-01
b3da1756f2d7771c82b8dafeb65129217ece9899cf40af0969d93d2088ed19f5  video/L2_third_person.mp4.part-02
331cdefe529166f023849dcf21c45fa1fa45b66b6440275773c4ef35339b8adb  video/L2_third_person_8x.mp4
```
