# The command debug markers reach the robot's cameras — 2026-09-21

The command terms drew their debug markers — goal sphere and cone, subgoal sphere and cone,
and the planned path as dots — as USD point instancers under `/Visuals`. That is scene
geometry, and the renderer draws scene geometry into every camera in the stage. The D555
policy camera renders the markers in RGB and in `distance_to_image_plane` depth. Isaac Lab
flags each marker prototype `primvars:invisibleToSecondaryRays`, and that hides them from
neither output.

This record gives the evidence, what it reached, what it did and did not explain, and the fix:
- the command terms now create no scene geometry;
- `debug_vis` is off in the shared command cfgs;
- recorded video draws the command, and the robot's footprint, as a 2-D overlay;
- the play and train scripts no longer request a Kit visualizer, which also makes `--headless`
  record video again.

- **Proven:** requesting debug visualisation on `main` changes the D555 depth at the marker's
  position (§1). After the fix it leaves the depth bit-identical, pinned by a Kit test and two
  contract tests, each shown to fail on `main`.
- **Exposure:** every depth policy trained on the previous Isaac Lab had its markers drawn at
  its true subgoal every step. v3 did not: its markers never left the world origin, outside
  every room (§3).
- **Not established as a cause of v2's deploy failures.** v2's single-tick command does react
  to a marker in clean depth, at three of five poses. Under noise, though, that reaction is not
  specific to the marker, and in closed loop v2 performs the same without markers (§4).

## 1. The D555 renders the markers

**Toggle test** (`visibility/d555_marker_visibility.py`):
- The v3 exported policy drives the play env until the markers are ahead, then the robot is
  stopped.
- The markers are toggled through the command term's own `set_debug_vis` on consecutive
  stationary env steps: on, off, on, off.
- **RGB:** the D555 image shows the cyan subgoal sphere, the heading cone and the path dots.
  Hiding them removes them, and this reproduces across both toggles
  (`visibility/d555_rgb_{vis,hid,vis2,hid2}.png`).
- **Depth:** the on−off difference shows every marker as a solid blob, larger than the image's
  0.13 m saturation (`visibility/d555_depth_absdiff_vis_vs_hid.png`). The rest of that frame is
  not quiet, and the same test says so: these are consecutive env steps, and two same-state
  steps differ over almost the whole image too (230 400 and 230 389 of 230 400 pixels, by up to
  2.62 m). The blobs are read against the in-place measurement below, where re-rendering one
  state changes nothing at all.

**Deterministic measurement on `main`'s command code** (`measure/`). This uses the Kit
test's method (§5) against `615fc14`'s command terms: one environment, both command families
placed 1 m ahead of the camera, one physics state re-rendered in place.

| added to the stage | raw depth pixels changed > 0.1 m (of 230 400) | max \|Δ\| | policy cells changed > 0.1 m (of 3 600) |
|---|---|---|---|
| nothing (the same state re-rendered) | 0 | 0 | 0 |
| a real sphere, r 0.12, at the same point | 5 129 | 0.588 m | 79 |
| `SubgoalCommand` debug visualisation | 6 785 | 0.640 m | 104 |
| `GoalCommandProcRoom` debug visualisation | 10 875 | 0.665 m | 172 |

Each marker set moves more of the depth than a real object of the subgoal sphere's size.
Releasing it, like removing the sphere, restores the depth bit-exactly.

**How it reaches the policy.** At five poses along a rollout (`beacon_ab/`, poses 1–5), the
robot was stopped and two on/off pairs of policy observations taken, with corruption off. The
marker changes 400–561 of the 3 600 policy depth cells by more than 0.1 m; the floor between
two same-state stationary steps is 48–214. Pose 0 is left out: the robot was still settling,
and its floor (625–821 cells) exceeds its on/off change (185).

The subgoal sphere sits about 0.5 m from the lens, so its front face is inside the D555's
0.4 m blind zone. It therefore reaches the policy mostly as the 0.2 m near-field fill:
354–446 of about 500 footprint cells at poses 2, 4 and 5, where no marker-off frame has any
(`check1/`).

## 2. Which envs drew them, and where

| | markers created | positioned |
|---|---|---|
| every registered navigation env (training, Play, capture, bridge) | yes: `goal_command.debug_vis=True` in the four shared command cfgs, also at v2's head `69014c6` | see below |
| teleop and coverage capture | no: both turned `debug_vis` off | — |

- **When they were positioned, on this stack.** On Isaac Lab v3.0.0-beta2.patch1 the
  markers moved only from `SimulationContext.update_visualizers()`, which returns early
  unless a visualizer is registered. The play script always registered one; the train script
  did under `--video`; a bridge does under `--viz kit`.
- **Where they sat otherwise.** Without a visualizer the markers stayed at the world origin.
  That is the room centre at one environment, and a grid corner 7.07 m from the nearest room
  centre at 96 environments.
- **On the previous Isaac Lab (`IsaacLab-retired` `ae41e2a`).** `CommandTerm.set_debug_vis`
  subscribed to Kit's post-update event, and every policy-camera read pumps `app.update()`.
  So the markers were positioned at every control step in every env, with no visualizer
  needed.
- **The bridge composes the goal objective, not the subgoal one.** Its markers were a goal
  sphere (r 0.15, green/yellow/red by distance) and a blue cone. They sat at a goal the sim
  picks for itself — drawn once at reset and held for the session in the ProcRoom bridge —
  and unrelated to the subgoal the Jetson sends; the sim's only ROS subscription is
  `/cmd_vel`. The bridge publishes `d555_camera_perception`, which renders the same stage. So
  a bridge session carried either a decoy at the room centre (this stack, no visualizer) or
  one at the sim's own goal (the previous stack, or `--viz kit`). It never carried the
  training cue.

## 3. Exposure by artifact

| artifact | markers during training | reached its D555 |
|---|---|---|
| v0, the warmstart legs, v1, v2, v2.1, v2.1a, cprime (previous Isaac Lab, 96 envs, `--video`) | positioned at each env's true subgoal and path, every control step; in 32 of 33 frames sampled from the 13 runs on that stack that recorded video, the sphere and path dots are drawn along the robot's path (`exposure/training_video_sample/`) | inferred from the identical marker code, not measured on that stack |
| v3 (this stack, 96 envs, no `--video`, no visualizer) | frozen at the world origin, outside every room | the cluster is outside every walled room and uncorrelated with any subgoal. A 2-D model of the four nearest rooms (`exposure/origin_line_of_sight.*`) gives a clear line to it, through a doorway or wall slit and within the 6 m depth range, from 0.9–3.8 % of camera positions, never nearer than 2.8 m; the camera's heading is not modelled |
| NOCAM policies | — | no image input |

- **The drift-trained legs had a reason to use the marker.** For v2.1, v2.1a and cprime the
  numeric subgoal was drifted by the referent-drift DR, while the markers stayed at the true
  subgoal: a privileged, drift-free cue. v2 had no drift, so for v2 the marker only repeated
  the numeric subgoal.
- **The previous stack's depth statistics of v2's training env do not decide it.** The
  2026-08-14 `depth_obs_stats` match the current stack's to float noise (G4 of
  `isaac-lab-upgrade-stage3-2026-08-23`), but they are aggregates over 8 environments, and at
  8 environments the world origin is one environment's room centre, so the current-stack arm
  is not a marker-free reference either.
- **The bridge capture of 2026-08-22 shows no marker-shaped blob.** 48 frames sampled across
  its 1 799 were checked (`exposure/bridge_capture_mosaic.png`).

## 4. Whether v2 uses the marker

**Clean depth: v2 reacts at three of the five poses, and the reaction replicates.** The on/off
observations (`beacon_ab/`) are scored one tick from a zero hidden state, each against its
own **observed** subgoal (`check1/check1_tables.md`, clean rows).
- At poses 2–4, with the marker shown v2 heads 6–28° from the subgoal; with it hidden,
  39–116° off. Both on/off pairs show that pattern at all three poses. Only at pose 2 do they
  agree in magnitude (116° and 116°); at pose 3 the hidden heading is 53° then 39°, and at
  pose 4 105° then 51°. At poses 3 and 4 the commanded speed also drops by 37–70 % within each
  pair.
- At pose 1 v2 heads 138–179° off either way. At pose 5 it heads closer to the subgoal with
  the marker hidden (−3.5°, −5.7°) than shown (+17.1°, +18.3°).

**Check 1 — under noise, the reaction is not specific to the marker** (`check1/`, 20 paired
seeds per arm):

| arm | pattern (marker on → toward the subgoal, off → away) |
|---|---|
| v2's own training noise (σ_d 0.16, pre-#219 far-clamp fills) | 0 of 6 poses; marker off, v2 heads within 7.7° of the subgoal at poses 1–5 |
| shipped robust band | 1 of 6 poses, in one of its two replicate pairs |
| same footprint moved away from the subgoal | reproduces the marker-on command at poses 1–5 (16–20 of 20 seeds), and never heads toward the moved blob (0 of 1 500 classifications) |

- **What v2 responds to is a near blob, wherever it sits.** Clean depth is where v2 falls into
  its featureless-field default, and any near structure pulls it out.
- **What the control cannot separate.** Every displaced blob landed on the side opposite the
  observed subgoal: all six subgoals sit left (+2.7° to +34.5°), all six blobs right (−11.3° to
  −34.2°). So it shows the response is not specific to the marker, not that the blob's position
  is irrelevant. The survive, keyed and class thresholds are this analysis's own, fixed after
  the clean results were known and before the noise runs (`check1/check1_summary.md`).
- **v3 has no marker effect** once the scalar prefix is matched (|gap| ≤ 4.1° under the band).
  An earlier pose-4 "swerve" came from the robot settling between toggles, which changed the
  IMU dims, not from the marker.

**Closed loop.** In gate G7 (`isaac-lab-upgrade-stage3-2026-08-23`), v2 ran in its own
training env with corruption on: path_complete was 0.8575 ± 0.0435 on the previous stack,
where the markers were positioned, and 0.8650 ± 0.0289 on this one, where they were absent from
every room.

The attribution of the rig-gate failure to depth **content** stands. The markers are not
established as its cause.

## 5. The fix

**No scene geometry.** `GoalCommand` and `SubgoalCommand` (and so `GoalCommandProcRoom` and
`CaptureSubgoalCommand`) no longer implement `_set_debug_vis_impl` / `_debug_vis_callback`.
`set_debug_vis(True)` returns `False` and registers nothing. The five `*_visualizer_cfg`
fields stay, as the overlay's styles.

**`debug_vis` off in the four shared command cfgs.**
- The attribution walker (`goldens/`) pools the whole golden movement to one field:
  `altered commands.goal_command.debug_vis True -> False ×22`, with nothing added or removed.
- That was the expected set. It was computed before the flip, and all 25 goldens — the 22
  contract hashes, the depth-observation golden and both layout goldens — were first reproduced
  from the tree without Kit.
- The observation golden and both layout goldens are unmoved.

**Tests.**
- **Kit-free:** `test_no_camera_bearing_variant_enables_command_debug_vis` sweeps every
  composed variant with a camera; each of four single-cfg mutations fails it.
  `test_command_terms_create_no_debug_geometry` fails on `main`'s command terms, naming
  `GoalCommand`, `GoalCommandProcRoom` and `SubgoalCommand`.
- **Kit:** `test_sim/sensors/test_command_markers.py` puts both command families 1 m ahead of
  the camera and requests debug visualisation. It requires the raw depth and the policy grid to
  stay bit-identical and no `/Visuals/Command` prim to exist.
  - It compares an in-place re-render of one physics state, because consecutive stationary
    steps are not bit-identical even without markers.
  - Its positive control is a real r 0.12 sphere at the same point: it must change the depth
    and the grid, and removing it must restore both exactly.
  - An earlier in-place probe (`inplace_probe/`) saw no refresh even with the whole scene
    hidden, and is inconclusive. It ran with a visualizer registered and `update(dt=0)`. The
    test runs with neither, and its positive control is what shows its own comparison is live.
  - It passes on this tree and fails on `main` (`gates/`).

**Overlay.** `strafer_lab.tools.command_overlay` draws the goal (coloured by distance), the
rolling subgoal and the path. It projects each point through the recording camera's pose and
intrinsics, read from the stage every frame. It wraps `RecordVideo` in `play`, `train` and
`test_strafer_env` (`video/`).

**The robot is outlined too**, in magenta: the chassis footprint at the articulation's root
pose, with a line from its centre to its front.
- **Why.** Enriched episodes carry a ceiling with probability 0.7. It is culled for the overhead
  camera but still shades the room below it, and that floor renders black. Over the 80 s
  recording, enclosed frames crush 13–32 % of their pixels below 20 of 255 (median 21 %), against
  at most 2.2 % in an open room. The chassis sits 9 of 255 over a floor at 0.
- **Brightening the frame does not make it legible.** Medians over 134 enclosed frames
  (`video_robot/shadow_lift_comparison.txt`, `shadow_lift_frame*.png`). A 1.8 gamma leaves the
  floor at 0 and lifts the chassis to 36; CLAHE on L gives 24 against 6; a shadow-only lift
  raises both, to 141 against 145 — the floor ends up brighter than the robot. A light in the
  scene would change the D555's RGB, though not its depth, which is geometry-only; the
  enrichment brief measured one for this on the training path and did not ship it, because it
  cut the crushed share only 31.2 % → 21.8 % and added a specular hot spot. The 1–8-environment
  perception scenes do carry one.
- **Check.** In an 80 s play recording of v3, the outline lands on the rendered chassis in open
  rooms, and its front line follows the robot as it turns. A 2-iteration training run at 4
  environments draws it too, which is the overlay's multi-environment path (`video_robot/`).

**`--headless`.** The play script, and the train script under `--video`, requested a Kit
visualizer only so that the markers would be positioned. The deprecated `--headless` flag sets
the launcher's disable-all switch while that request still reaches the settings
(`AppLauncher._resolve_headless_settings`), so `SimulationContext` resolved no visualizer and
raised
`Explicitly requested visualizer(s) ['kit'] could not be configured`. Nothing was missing from
the install; the `isaaclab_visualizers … extension.toml` warning is unrelated.
- `HEADLESS=1` in place of the flag ran the same script with markers positioned
  (`play_markers/`).
- With the request removed, `play … --headless --video` and `train … --headless --video` both
  record (`video/`).

**Teleop's target marker does reach the cameras, so it is now opt-in.** Teleop drew a
bright-green point through Isaac Sim's debug-draw interface at the mission target, on by
default, while it recorded the perception camera — on the claim that debug-draw stays outside
the render products. It does not (`teleop_marker/`):

| product | with the marker drawn |
|---|---|
| policy camera RGB, perception camera RGB | about 2 150 pixels carry the marker's green, in two discs; none before the draw, none after `clear_points` |
| policy depth, its 3 600-cell grid, perception depth | unchanged, every pixel |

- The recorded frames are the perception camera's RGB, so the marker was in the dataset.
- `--no-target-marker` becomes `--target-marker`: the marker is off unless a session asks for
  it, and its docstring now cites the test rather than asserting the claim.
- The Kit test pins both halves. RGB is not reproducible across re-renders — two renders of the
  same state differ over about a sixth of the frame, and clearing the marker does not return the
  image to what it was — so the RGB half is measured by the marker's own colour, and the depth
  half by equality, with a sphere at the same point as the control.

## 6. v3's training contract

v3 was trained with `debug_vis=True` and its markers frozen at the world origin. The
composition contract now differs from v3's training contract by exactly
`commands.goal_command.debug_vis` — `faf86756…` → `c98d18ba…` for
`RLDepthSubgoalEnriched_Robust` — and by nothing in the observation semantics.

## 7. Angle conventions

The off-goal angles in the attribution, parity and v3 records are measured against the
**mission goal's** bearing: −8.1° at the capture pose. The subgoal the policy observes sits at
−62.5° there. Against the observed subgoal, the rig-class command is about 25–29° off and the
"toward" class about 67° off. The causal results do not change; what "toward" means does.
§4 of this record measures against the observed subgoal.

## 8. Open

> **Amendment (2026-09-23): both checks below are answered** in
> [`debug-marker-checks-2026-09-23`](../debug-marker-checks-2026-09-23/README.md). The pre-flip
> stack rendered the markers into the D555's RGB but not its depth, so for the pre-flip artifacts
> §3's "reached its D555" reads RGB yes, depth no (measured on the 80×45 camera of v1 onward). The
> depth policies' observation contract carries no RGB field in any variant
> (`strafer_shared/policy_interface.py`), so a marker was never in a pre-flip depth policy's input
> and cannot have caused the v0–cprime results through it. §4's closed-loop comparison therefore
> set two depth streams without markers against each other: its previous-stack arm had them in RGB
> only. With the markers positioned, v3's steering is biased 3.7° to the left and no outcome metric
> moves beyond its standard error.

- **Whether the previous stack rendered markers into the D555** (check 2). This is one boot on
  the retired pair, run after this change merges. It decides how every pre-flip depth
  artifact's history is read.
- **Whether v3 keys on the frozen cluster** (check 4). This is v3 in closed loop, 16 envs ×
  100 episodes with corruption on, markers positioned against off. Pre-registered: within G7's
  run-to-run spread (±0.04 path_complete, ±0.5° steering offset).
- **Markers in the livestream viewport** are filed as `livestream-command-markers`. Teleop's
  debug-draw marker is no longer open: §5 measures it, and it is opt-in. What that brief still
  carries for it is a drawing mechanism that stays out of the cameras, which debug-draw is not.

## What is not claimed

- That the markers caused v2's deploy failures.
- That the previous stack's D555 rendered them.
- Anything about the real D555 (no real-sensor frame is involved here).

## Gates

The two failure rows run this change's tests against `main`'s code. The pure, contract and
`command_markers` suites are at `f278a4f`, the head. The full Kit suite is at `a08f26f`: the
commits after it change the overlay, teleop's marker and the tests that cover them, and no
other suite exercises any of those. The recordings in `video_robot/` and the mutation arms are
at `11ea0cf`; since then the overlay names environment 0 rather than reading the viewer's
index, which is the value those recordings ran with. `video/` holds the earlier pair, recorded
before the outline. Kit verdicts are read from the JUnit XML (`gates/`).

| gate | result |
|---|---|
| pure-python suite | 1 367 passed, 1 skipped |
| composition contracts | 135 passed |
| full Kit suite (`run_tests.py all`, 15 suites) | 502 of 502 passed; `obs_dump` needed 2 boot relaunches |
| `command_markers` on `main`'s command terms | fails: `SubgoalCommand: debug vis reached the D555 depth` |
| contract tests against five single mutations | each fails: `debug_vis=True` in each of the four shared cfgs, and `main`'s command terms |
| golden attribution | `altered commands.goal_command.debug_vis True -> False ×22`, nothing else |
| `debug_vis` survey over every registered navigation env | 29 of 29 on `main`, 0 of 29 after |
| `play … --headless --video`, `train … --headless --video` | both record, with the overlay and the robot outline drawn |
| `command_markers` Kit suite at the head | 2 of 2: the command terms reach nothing, teleop's marker reaches both cameras' RGB and neither depth |
| robot outline against five single mutations (rotation sign, line reversed, filled, length and width swapped, no behind-camera guard) | each fails the overlay unit tests (`video_robot/robot_outline_mutation_proof.log`) |

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `debug-marker-leak-2026-09-21/record-files/` |
| deposit commit | `886349f8364d10b4ac0f7facd645443188599d30` |

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/debug-marker-leak-2026-09-21/record-files/. \
      docs/measurements/debug-marker-leak-2026-09-21/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/debug-marker-leak-2026-09-21/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
1b94c4087eb533fee3496f117c34f835b89f54a7751339f102dd5d8d619fcf7f  beacon_ab/beacon_ab_capture.log
83172a666d4eac2259ed1898154a9e2507547d1534d601bcd21f724019353320  beacon_ab/beacon_ab_capture.py
d4dd464839531118210c39143b8ac5ce7aa62ffb79ca043c6cb33059f205b522  beacon_ab/beacon_ab_capture_cmd.sh
1de1b6242c921582b9b8ab7dcb60ad9af43bb7feffe9c2ef5c94d4b64978f573  beacon_ab/beacon_ab_obs.json
5870057dd734138e8eea5aa3786856ccd88b26b87e9936fbe032627678ea82db  beacon_ab/beacon_ab_score.py
3c69881835e358d93f8b79f10b735d366b2b2bb10fceabb9b6d0e85511a225c9  beacon_ab/beacon_ab_scores.json
5e6f471cb99478312d51f9bc0af98fd2601ef99a40d9b073e76a38e9fd4d9f80  beacon_ab/pose0_d555_rgb_off.png
c89776a584dba22adb21cb627b92c92685ce4f126864e4e9849a18f788e9fe79  beacon_ab/pose0_d555_rgb_on.png
92845627f1b133df15040396e187a5180523525f8c56ddfb91ad3f9a8bb3a831  beacon_ab/pose1_d555_rgb_off.png
d40eaffeca7b26c81fe30baebc50a7f58c488690415e95c9e2386d3a3bd374a5  beacon_ab/pose1_d555_rgb_on.png
f921f1588971304c1d5dbe6b8b2653f81a96edfa931857d48d9372d79f4dd0be  beacon_ab/pose2_d555_rgb_off.png
99c51708d9fd3a5aedb7ba83897f707d8b4c94a0eec527150c37b89bdf921d00  beacon_ab/pose2_d555_rgb_on.png
7c0e536347149d38d33a75dc5ab822e10646cd9eaf5eff96c7df58acd792b240  beacon_ab/pose3_d555_rgb_off.png
b02e564074de7bdfe23d1c73f5f9d984f47badad6f29230366ed8d8951136be4  beacon_ab/pose3_d555_rgb_on.png
7628eb2dece8a37c9ae8734a0be812b40b2c0e631cd4ecbaf8cc4fed7fc424b3  beacon_ab/pose4_d555_rgb_off.png
9dc568a45c3d4c13261d2c0b9eed07e41b361f377040e12ee24e5300a9632159  beacon_ab/pose4_d555_rgb_on.png
e10cb165c252bff8bbab2170d511117a1056517297c5267099e82570ae50b88f  beacon_ab/pose5_d555_rgb_off.png
f761a1b1b4b6b4ad4c5e7fb2968fd5ef4cbeb4e01b62ee9b04314c982d01b240  beacon_ab/pose5_d555_rgb_on.png
2f81840615d84e28719bfcf400d57e4590f429218672c50dee69c2b908ee9bb9  beacon_ab/watchdog_beacon_ab.log
2f81840615d84e28719bfcf400d57e4590f429218672c50dee69c2b908ee9bb9  beacon_ab/watchdog_beacon_ab.log.attempt1
cbc747d1d0dd239101148da5082f67c3c3c8d51d3653b95911bbca6a0d15d173  check1/check1.py
a7e097bb93821581eaa379096852784d468a79b34b2049ac617eff53c7997dd3  check1/check1_figure.py
7f36255a59793c3aaa876869d0e8a91e35e9d1f9459d40925371eebef3831377  check1/check1_footprints.png
81e41d505d306273f62265973ce29023d4f747da5e848613019401950fc1e89b  check1/check1_prefix_probe.json
aec444f7beb77671c0f868f0a2e0f668657dd0d55e71abd04080c176e45e2740  check1/check1_prefix_probe.py
f79625edaf749fc957bbc0d63cb129fb427f8752dcd006df65023a577f69a885  check1/check1_results.json
c878cb2a2f3cbf8fce55c47a4d5b7e912ad990e0272a53814db4afcabc883c27  check1/check1_run.log
de33ee1641252b874bf178d91e9c82fc5f5a009790cc28db72f2cec6394de32c  check1/check1_summary.md
c37caaad57f3a81c4fae67f5e5cc3c92a7a327f2c955d79afb694318a911aaa4  check1/check1_tables.md
c3a2894bb1fa6ce553f2622f9835d5f7207d5e61165adc76b403ea53de028ca3  check1/pose0_displaced_depth.npy
3b58562e28ad23a0551f47cd679ead2fc21afded2fd2fac57a048fa509a3456c  check1/pose1_displaced_depth.npy
28524282f34ef35defccc8dc14d38efa553b8f824b82f63371e4473f83e47c1d  check1/pose2_displaced_depth.npy
19451a0f53fe8de8a428914439ea73053e2828907b87ddbf124af2eb4b1fce44  check1/pose3_displaced_depth.npy
b179a1abc2a9ab44d652eaa872a5ff4cb7308a43ac3d97a27129090282c3207a  check1/pose4_displaced_depth.npy
1321a01520ec0ee18445134233f74349190e35943a55047bd2092d2eb2cd8834  check1/pose5_displaced_depth.npy
2cb486ad86d279ca06b6207b54707a7f6c492be0c39080f62f5a368d703ed927  check1/run_check1.sh
0bb27361bb522c73c6ce59f8a69380c6d226421a59b552495abea09f8ae7be68  exposure/bridge_capture_mosaic.png
c452d63243d80d1a2b50d01a977246f139d14e2203efa13cfb78115284354558  exposure/bridge_capture_mosaic.py
7522c5f7c60735d4404ef84b5dcff33129d21608d61edcd364a52a51fe666471  exposure/debug_vis_survey.py
b8f442acb4f6458309e6d7d7f87ffb8d823be4c349853080e0f0c7574909200c  exposure/debug_vis_survey_after_fix.txt
31cc6ff845330207a14e8666fb526f698dbf72c0cca181d60f15e72e627fa913  exposure/debug_vis_survey_on_main.txt
ee9ed8d9e42aad90295209574034a3cbd44e5e9b2025e2ab84be4c985a91a9e3  exposure/origin_line_of_sight.py
d075a2855150eff4260fdbab4eb58c241526eb6c993bca224de6100c27718c1c  exposure/origin_line_of_sight.txt
54a384e6f20f821c27f771f6ed66ef5bc4727c2ee0e3e884cf8ca8d1dc52ab27  exposure/training_video_sample.py
83aa11a5c001ceb711b4330179eec11f57021da91ed30e24facb5f3f032113d1  exposure/training_video_sample/sample_1.png
a7d22fcf78b105ba703eab850d962f568e4c30d7019acc8bec3a71962d5fcbd3  exposure/training_video_sample/sample_2.png
52f4f37c85c55c4c28931e550b92f4f1c6f3017713f0f33b0b12284013c9204c  exposure/training_video_sample/sample_3.png
7c26fb4fa360085f9e03d7dc07633c2bc98c1c05b3d348c5c333c75eb17b4501  exposure/training_video_sample/sources.json
82d31ebbc7311d1094c17c7b9ebddc94142f52d0af6c0ba8abf1f72e604331f2  exposure/v2_training_run_20260726_221955_step9000_t3s.png
8667f3de5d47dbc3e113467b72e98bc82138bb9bbb8fca66ca53f87098438443  gates/command_markers_fixed_PASS.xml
2a273ab2af6b0d767039971351338127d7148d114a137e3c05b4cf9b52a77c90  gates/command_markers_head_PASS.xml
9376efe7c7448e5fc2fd2a4a9c019e8dd6d32d48ca2baa000b8668fdd7765572  gates/command_markers_on_main_FAIL.xml
03a3daefc2f0d49afe589a3e5dca3dddfe8fa3f0645e575cb65a12b000221e4c  gates/contract_mutation_proof.log
6bf6d839860e9fac0c35c8dd12b98d4c4ab03203435a2a54622c90b68edcc8df  gates/contract_mutation_proof.sh
f7b16359ba4ac3ad2caca274d2eba995a793fd685175cb8bf78185fbb00f74b6  gates/gate_contracts.log
2b17f3cfc9ccb5e2431f5234bd7376feeeeb5248b8bf2b0f9aaae98e54712825  gates/gate_contracts.xml
64f7b9d3ae121a66df14c53f0da88e1ffd31e6b9341e9424ce530ea626f67e96  gates/gate_pure.log
e4ee15d8ea7a1c9338590cb7ed372cd84d9cb4b602d044a4e03ded55cc009844  gates/gate_pure.xml
94c0c57465b624c3b4df3ad1b8d31eca19c7a4c51995db52b1feaf8451a85463  gates/kit_xml/test_results_actions.xml
de55ce90fd0c1b2baef84cdcdbf9b1941f037c9bc706e8a25bd303dff65788da  gates/kit_xml/test_results_camera_jitter.xml
22d5c0f2eb8db92c3c80ca75eb131f18f2e73c075d0a89fef1c7aee0243fac08  gates/kit_xml/test_results_command_markers.xml
470e464c89d199b6abace1d3c8e9959a9de83814f9766c15afddbd19b45d7e4b  gates/kit_xml/test_results_commands.xml
7ffbdb53e7c250643f1a4e5150369277ad2afc5a3a83070d22f5b6497f88ac23  gates/kit_xml/test_results_curriculums.xml
b3843625fa8bda5770dd9aa7bb65619489eccb4f8f55554fae219b95daaeecd3  gates/kit_xml/test_results_depth_noise_test_frame_drops.xml
704574d3e6082b341eee973d0b41d4a3b12bb5c7f3256511030f6d40e8739782  gates/kit_xml/test_results_depth_noise_test_gaussian.xml
a61c716b0a8bb9715bab6776dd49284ff3dea0f5cc70ad34124c8f3f632a8f27  gates/kit_xml/test_results_depth_noise_test_holes.xml
9e334a772596f4c1763e5e4ba87a41265cd47ed2d6f424d0ec90e346cd6b8eac  gates/kit_xml/test_results_env.xml
07604bffa676b82da231d3846f793e095629377b1b073bf1394ca619acb84e0a  gates/kit_xml/test_results_events.xml
442f7db4fafdb2e7051e87e390cfc3e5ea1bededbbfa41e3ed3a13a2fa9305d4  gates/kit_xml/test_results_imu_test_imu.xml
335ac0ce8abafa5b0166438f0d0ddad1b62db803bc9027271ea11228475dcb52  gates/kit_xml/test_results_imu_test_imu_collision.xml
20a824ea7cb3b827909c29be9e1bc26df1d40c6b3e19699fe6a9040701c217ab  gates/kit_xml/test_results_noise_models.xml
97a98478e753c77081150ac6b2665f9b2d77581c583eb37f247b000599062b65  gates/kit_xml/test_results_obs_dump.xml
306931ff66e2c0501e85d7ba62673bb90760b7055758b3e00b71918cd80302dd  gates/kit_xml/test_results_observations.xml
d2389fa3bf59ecdf15adf8c88679ac4881408b40d398d860503008f527cbd078  gates/kit_xml/test_results_rewards_test_collision_rewards.xml
52761c833ffbc8c1403f5cc45bc13f00970fa4aad21346d14edc64ea519ffc94  gates/kit_xml/test_results_rewards_test_rewards.xml
0612f8f0e8da2bef16605c3576dd6e4b79bf497f1b7f64c87c651a34acfd1279  gates/kit_xml/test_results_sensors.xml
4682c32c3941b8ef35b104e5eaf53c47aa11a790387e7383fbab2636430215fc  gates/kit_xml/test_results_terminations.xml
4c272ebca392d37581c4a04a119adc9680a190d045f494173640ba791cee47a6  gates/run_tests_all.log
b21b23949e8fa25eb47f4a9263d1e5dfe166dad5445b52ee0ff84eb085929d69  goldens/after.log
4f7fbc0b91e6e6bd35f6506f297c1dbeb75fc8da55291a5dd73e8dc58db114dd  goldens/after/hashes.json
e6e230c283d8b3838370a79a9fff34979517b807707d608b0b8dab1503bae208  goldens/after/preimages/contract-RLDepthEnriched_Real.json
c58a9e527859c84dbcc12f2ae69724871c9592a1c79fd24df8b7dfa7eebfbaff  goldens/after/preimages/contract-RLDepthEnriched_Real_PLAY.json
6095cb6cbc1cc503e448ecb743f0606df5837b67365eef4b9163346e4edc3335  goldens/after/preimages/contract-RLDepthEnriched_Robust.json
2676629ab0ca3c3de3fde1dadc50e623f6bd2b194d6980a9bd49b61056b9e858  goldens/after/preimages/contract-RLDepthEnriched_Robust_PLAY.json
1f558159d280e5ab55d0a7fa58c5cea84340567b8116c85e454bdb3e8a56a562  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real.json
23604bb3b6ac57b24fc3240edc9ba8987d0be60670eb5d4ca4d21d451b0c2adc  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
3f1bdc4f70e2a69714312a5c93fcbb783a7137b53e3118ff983e46fb1c1d273f  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust.json
3e8267799f8db617b4f3abc4a52bd4c676ba45b288bd6197703aafa33dd1b264  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
f98578cac577ccc85b786c0c0cb7fcf13f69a81626c5da9eaf997187b289dbf7  goldens/after/preimages/contract-RLDepthSubgoal_Real.json
0c0ab4029332169f2aee32eb97d691491999b4100137128db04b5f8dd91a170b  goldens/after/preimages/contract-RLDepthSubgoal_Real_PLAY.json
7756f87dcccc81e2989f6c507eda432803c76dbd7f3d43edcb504422b28fb196  goldens/after/preimages/contract-RLDepthSubgoal_Robust.json
112e52e179ddbf50a84194e2082ee4eea3e4e258c8c9ef10ab2f1e5d1041766f  goldens/after/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
9d84b482133fe9d23f401d67875df7a29347bdafd02f5a4664fe92b877daa5fe  goldens/after/preimages/contract-RLDepth_Real.json
7e4ad64585634743c205a416660d586c92a2d2a6ebf532e5237520aa8808684c  goldens/after/preimages/contract-RLDepth_Real_PLAY.json
210a648e8887af8100fe566e9d2e2dd919ca11baa3e78e4cc30e634983ad3195  goldens/after/preimages/contract-RLDepth_Robust.json
6771a761f6285566890fc4bae9085f16bf07779fd3ff03c3e07ec6446e745939  goldens/after/preimages/contract-RLDepth_Robust_PLAY.json
43c960379b0a2de40e683d0faa4a1bb9c31ae3418326be87acb03f10630fa667  goldens/after/preimages/contract-RLNoCam.json
76e815d3eab5bd17e1aa21d5103f22e133aadfb2049dd3c40af640c2d627ce3f  goldens/after/preimages/contract-RLNoCamSubgoal_Real.json
978df5c4d4cb3fa32e2759c5dcb40077ca161b8b5cf4e4df6f81b091e7f4b533  goldens/after/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
6ab41331e42545376d500390235c479a0ccbfb0213819582c07d07cfd83992dd  goldens/after/preimages/contract-RLNoCamSubgoal_Robust.json
b16df297516da1ca5bc14c2efa02b6809b803372aba65a2eea6cfeef7eb0f0bc  goldens/after/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
b79700c3a7c92179843055b8dde6f57bc440abfc58ad5d13c84a2d6298ddea16  goldens/after/preimages/contract-RLNoCam_PLAY.json
88b7c0e4a1ac221feb5db9f245b3e8a20e04400d90a32ab706bf0f191a762bdc  goldens/after/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/after/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/after/preimages/layout-nocam.json
edabd78ac9b3856dadb60482614648edfed43cd6464d1e67c0198d3f0bd67eb0  goldens/before.log
9f01a6271554bdf18a62c32d4eeb0ffbf0a6f34a8813527d15049f5b65ceaf2a  goldens/before/hashes.json
022a41e4807c2876400735c71219a6b69ae4eabc4988472e9e6d41502c9c7e31  goldens/before/preimages/contract-RLDepthEnriched_Real.json
57029c366aa3400627e06ad530168d12ba8376bc14650147109408b5a3895220  goldens/before/preimages/contract-RLDepthEnriched_Real_PLAY.json
1a8fb607245b2c3b41e0cc11c6b3a8d7f288d7f7d1b2c6a1cc8e146f288b62e0  goldens/before/preimages/contract-RLDepthEnriched_Robust.json
fa852dac0e54b6db1dc4184c22ddf25771feead4efc91ddfeacc31b6fff9fdf2  goldens/before/preimages/contract-RLDepthEnriched_Robust_PLAY.json
09516c10984f3ad2d9c7993074634240ae030e540d16e2b67f7e4b6b5b35abab  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real.json
75ce3c17354028d6fba7b383383ebcdd401941c5f02e67433374fa6f04906d18  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
86fe46d070f0e86adc487284246720b83cb42feabb563196bae37bf0b3a8852c  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust.json
4e29289981d1126f4da670e7ba046db84e1559ebc6a32a5643bc641460cf1858  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
f8dbf1f4f88a74e7c3840b27eaa97803c2513ef978398b26a49e650ce0ad5504  goldens/before/preimages/contract-RLDepthSubgoal_Real.json
1480bd411e52d895697af1dacf567a85f9b3296a6d2a653ae8a40043478da6ab  goldens/before/preimages/contract-RLDepthSubgoal_Real_PLAY.json
cc18d913bd3d1294f7d27327c5b1e6cbdfde76ea767889184ea2e68f966f54e0  goldens/before/preimages/contract-RLDepthSubgoal_Robust.json
f6d9d59a8ff13ec395f3eba921f5f5223bd0971e3a6e51c9cd1dec9057022e0f  goldens/before/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
e9452ec81d0272ed78b515c403bf2b2340f77052d2419219ac4251dbddf79ab7  goldens/before/preimages/contract-RLDepth_Real.json
51eced1aa04aa5bd5b2f4139d12a65368b50e2d1ae6ea3b22a0c6d7be93fd3a8  goldens/before/preimages/contract-RLDepth_Real_PLAY.json
44d40b0db19b942daa597cdf952002dfe54549d016f31df0bfb06323f3edc0aa  goldens/before/preimages/contract-RLDepth_Robust.json
4a9c1ba4ac54104d66032bee65e4c9a9107ec0cf9f3126bdc09d2ec0e233c5d4  goldens/before/preimages/contract-RLDepth_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/before/preimages/contract-RLNoCam.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/before/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/before/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/before/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/before/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/before/preimages/contract-RLNoCam_PLAY.json
88b7c0e4a1ac221feb5db9f245b3e8a20e04400d90a32ab706bf0f191a762bdc  goldens/before/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/before/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/before/preimages/layout-nocam.json
3b49094679c88771c1088e33a6cd3891da8570d7d589fa8d61db2b95f2fb050b  goldens/golden_attribution.py
78a2350010aaf715d2416e7b9fd8f072bb5a54cfea8108bda8d3818c6abeb710  inplace_probe/depth_marker_leak.py
8dc52ece9d0e1ae582ac855116d85fe7e7fe293a1c845236932e86591458218d  inplace_probe/depth_marker_leak_cmd.sh
d907bee0966a22c8ce7ec79013c847360e93ca92adc74b4d05219418c4cccd71  inplace_probe/depth_marker_leak_run1.log
1c74bfa73f6ef3b74002847eb846140e8bcfe5b192566c967583b63f2d192d89  inplace_probe/depth_marker_leak_run2.log
41870227a7fa34049e23bab751b21bd56fa45c6dcb30d697e4375bf3cde2cbab  inplace_probe/depth_marker_leak_run3.log
6accd3b49a5bc45e05a54f02e81197a67af1868cd08f05423ceccd00da72cdc4  inplace_probe/leak_run1/depth_marker_leak.json
10baf1eed455fb66e4d9ada71d68d8e615cc59dd7f1f56de9e2ebcb52a449a5f  inplace_probe/leak_run2/depth_marker_leak.json
820a21942460b8101343f36650e94e6f20d11a78e943c3aa8ace79de0c09d4fc  inplace_probe/leak_run3/depth_marker_leak.json
384c770425947b4dcbaa244129f7f3470f7e78c544426f3b0e9954cc1bdc65e7  inplace_probe/watchdog_depth_marker_leak_run1.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  inplace_probe/watchdog_depth_marker_leak_run1.log.attempt1
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  inplace_probe/watchdog_depth_marker_leak_run1.log.attempt2
384c770425947b4dcbaa244129f7f3470f7e78c544426f3b0e9954cc1bdc65e7  inplace_probe/watchdog_depth_marker_leak_run1.log.attempt3
7da3b348a4372096bca1a1790f1c87b847645b7f08ffd514f8a7b50c1318a7c3  inplace_probe/watchdog_depth_marker_leak_run2.log
7da3b348a4372096bca1a1790f1c87b847645b7f08ffd514f8a7b50c1318a7c3  inplace_probe/watchdog_depth_marker_leak_run2.log.attempt1
a900b162cdcd5cc43c8765cc7506c484150866e4b5cea35a71bd8fb44f6b1cea  inplace_probe/watchdog_depth_marker_leak_run3.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  inplace_probe/watchdog_depth_marker_leak_run3.log.attempt1
a900b162cdcd5cc43c8765cc7506c484150866e4b5cea35a71bd8fb44f6b1cea  inplace_probe/watchdog_depth_marker_leak_run3.log.attempt2
0875b7bc5266d1ef9938e70a11afacb89d69a5d536eaa8d7ccb7e87ccef7a579  measure/marker_depth_on_main.json
e25c1edadde49fec4cd1383202acb2d791b7546d731a3bb82859471e881ef305  measure/marker_depth_on_main.log
0ad732c877e5b5b98e0d7a36c8a45e161b30b31b7fd2b35dedd6be07531c2c2d  measure/marker_depth_on_main.py
a3a9a0749aafa7381843048da3963a7c240d15f63150777f972d83a23205eac0  measure/marker_depth_on_main_cmd.sh
1a88f6c38b0d4045c2dd52c0a75766ab555da8becdaf5650505bb77589e614a8  measure/watchdog_marker_depth_on_main.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  measure/watchdog_marker_depth_on_main.log.attempt1
1a88f6c38b0d4045c2dd52c0a75766ab555da8becdaf5650505bb77589e614a8  measure/watchdog_marker_depth_on_main.log.attempt2
ffba1b203b74d412230edb40cc072cdae3dd5fd88ac9ec584786079f924aeb9b  play_markers/play_headless_env.log
8323211c5c1b339215b68a0ea1e1404d58a45205d9f3dfce39ec803157db7d81  play_markers/play_headless_env_cmd.sh
1339cb52ee1790b8fa99d6c7134863effba33c816b70fb9d24ba69afe357e985  play_markers/play_headless_flag.log
a96ab83e3de2c903e10cf55eafb6ba92ea27888aba12d7b9198d2f8e4c8bd6f8  play_markers/play_headless_flag_cmd.sh
7e455f8fccad3d315f2492f27ba2925a0cab74d2bdbb367deb6a286b9304f1f8  play_markers/play_headless_flag_unwatched.log
089e810ee493d7ee5bdc7f74890078f4f34ce0f4d220e66b9e27406397a1b948  play_markers/play_videos/play_20260921_105717/rl-video-step-0.mp4
1df3ace4f78224ab0a21f93a9bfdab82af1d20cc84b9be5b810eb21b71ecf8bd  play_markers/watchdog_play_headless_env.log
1df3ace4f78224ab0a21f93a9bfdab82af1d20cc84b9be5b810eb21b71ecf8bd  play_markers/watchdog_play_headless_env.log.attempt1
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  play_markers/watchdog_play_headless_flag.log
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  play_markers/watchdog_play_headless_flag.log.attempt1
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  play_markers/watchdog_play_headless_flag.log.attempt2
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  play_markers/watchdog_play_headless_flag.log.attempt3
a206a4f023c708c82873da003716c43efb60297fa0b3487797a43e7788f9d707  teleop_marker/base_perception_rgb.png
501b98abc5f9f812a58b3998b08474907890cad85ee349b6524f9c16611a2be5  teleop_marker/base_policy_rgb.png
2f073261f08c67a9c16b9389e4875d6ad208a9e83887bc9b29ba7cf1e2bc31de  teleop_marker/cleared_perception_rgb.png
d6d76becc6e2974d1b7d3ac4c74bafba75a3ddeb61504604a82eca055734794a  teleop_marker/cleared_policy_rgb.png
50b3582505c0915a4e2f07652c4062836f02e3bd9f762ca3565ef3865be178af  teleop_marker/drawn_perception_rgb.png
ae4eefa2dc186c2a3cad7e710d9bcf842cb6cac4f38ea4ae3359287d81086e0c  teleop_marker/drawn_policy_rgb.png
f25d4b5eb895d8d762ecfa75bc24d86a78e7c9887bb22cce583ebe49e2b66c39  teleop_marker/sphere_perception_rgb.png
284345db8e4a6267e1be63e30c59a81c19c36d242b446c88c6c1cd5a6ecbd63a  teleop_marker/sphere_policy_rgb.png
7997295d627dfd4dc2f654bffd390f5b0f8a43db62e7d1d44d59bd9c7e235ca6  teleop_marker/teleop_marker_crop.png
6930b040e7979326167efa4ed763b8ef3da14ff1e095c566a578f64c6054996a  teleop_marker/teleop_marker_leak.json
323cbbf9c7cda655cf8afa5cebb78b5c65d120c2582a46483f8d515d3cdd628e  teleop_marker/teleop_marker_leak.log
6523277c1e43c97897ed95a4bc2b0cc110aa672f312f93ef7057746278aadc1d  teleop_marker/teleop_marker_leak.py
dcb3b45ac83862b641b2f005ae569e4a8bff69a9c76ab4d7f33e5e774119a61f  teleop_marker/teleop_marker_leak_cmd.sh
69a1c6d97a8dd435c73f1a2bacf8ec9f7f6b1f798382f08edb0c217ebec350af  teleop_marker/watchdog_teleop_marker_leak.log
69a1c6d97a8dd435c73f1a2bacf8ec9f7f6b1f798382f08edb0c217ebec350af  teleop_marker/watchdog_teleop_marker_leak.log.attempt1
337e758e70a9f1e9a57cad7c9ed33a98846571e936aa8f5adcd85131839762bd  video/play_headless_fixed.log
5798224cd0e669d7ad2dceae490b166df121f97911734d1631dfb5ccc749a958  video/play_videos/play_20260922_082725/rl-video-step-0.mp4
15f43ff856f562b57e271857b57702abd9154339bbf84abd99e22fb27091be39  video/train_headless_fixed.log
24e23f3b6601839d787aae64442ad74e93b7d7c416bdbff38cbaf10bd4b14044  video/train_runs/run_20260922_083035/events.out.tfevents.1790083837.gx10-d1d8.2027185.0
15263599b4a09390436d0ef3259771e672a927fcde13dafcbdad821e8110cfbd  video/train_runs/run_20260922_083035/videos/rl-video-step-0.mp4
1147ad865d5edfcd6145960dc53d3f13446cc02c9c3526703afd65e01edf8a84  video/watchdog_play_headless_fixed.log
1147ad865d5edfcd6145960dc53d3f13446cc02c9c3526703afd65e01edf8a84  video/watchdog_play_headless_fixed.log.attempt1
f62d4e02b846c194e6e39ad45f44c7faced38fb2b77f2f1ee5f9d805f2dcc6ae  video/watchdog_train_headless_fixed.log
f62d4e02b846c194e6e39ad45f44c7faced38fb2b77f2f1ee5f9d805f2dcc6ae  video/watchdog_train_headless_fixed.log.attempt1
d47dafd00b3fc562a36b7c53b15e016330d67e780c0a1b0077111410db4d7834  video_robot/play_robot_overlay.log
44e8e895eaec76938fd235efe355bce49165f643407722f7e2dd21ad3ffd6907  video_robot/play_robot_overlay_cmd.sh
178d35af7babb22a23753fb009069b61664b65a0713dcaa21e8a77d261ae2f5a  video_robot/play_videos/play_20260922_131809/rl-video-step-0.mp4
57b409ee4136ff6afcbb19552b098c1d2b644daaa44e5483b813f3fdb39dd4ff  video_robot/robot_outline_mutation_proof.log
e4e7b3f2081244149b98ac560746a0192f60bf662f8e4677ff23f407d95ceeb2  video_robot/robot_outline_mutation_proof.sh
3eabe00764f1ad440c4cf9dac3c8275c06d5b99c58ed7fcfb80c7c5db2c1e9c3  video_robot/robot_overlay_frames.py
6b8cc5207486eff0bd53549c270efc8fc404dfbcdadb29a4ca92562d1abfc492  video_robot/robot_overlay_frames.txt
85f63337c328aebadb3c850ef322fce4d3f9849851bf7a9126d3535c4cf7f8a9  video_robot/robot_overlay_sheet.png
05f285f247b10218dae742d5c662a89255a787e9af14d0cecd3272f79f4867e9  video_robot/robot_zoom.png
021c1023154de320d7639fabcf3ddb4b915a8a63a34cb678f8c66dd4f6b84766  video_robot/shadow_lift_comparison.py
67e7532f5c1940afdd0838c7ffdaa3fb9c360db7bcc93c027f01cb7d08211319  video_robot/shadow_lift_comparison.txt
4c6aa307830573cbd8a70cedc66c726692655a1e65513c69034977268b9a2c57  video_robot/shadow_lift_frame150.png
0b60378534df35e7a71e409efd77691b0935989e458b24e143e473b3cf40845c  video_robot/shadow_lift_frame290.png
e7d7e7f530efb5ffb873fa37584f4ad82006b8d7f4b1e085eac66f5f2e2c0e05  video_robot/shadow_lift_frame45.png
c26e8f20003cb971fe93bcf63f2ca99af9ec6d0eb08e62d258ead1d801c82dce  video_robot/shadow_lift_frame90.png
f1dc1e1abf58c6c3790784eed75b0e9f6af9a2dfc66914778b4c28f386c373b0  video_robot/train_robot_overlay.log
16d76f97a44f8ff64b4d7fd4a5f6cfd96efea1a0f6d76374e1d7194d6bc7b398  video_robot/train_robot_overlay_cmd.sh
c93b3494f0825d9db62d6ff4be99009e729f50d5316de6b265b65212526b3542  video_robot/train_runs/run_20260922_132554/events.out.tfevents.1790101555.gx10-d1d8.2080248.0
059414bec491d2e02fbccb7dabd70df0a0e2d293a821f1917b49b89d5ca8299c  video_robot/train_runs/run_20260922_132554/videos/rl-video-step-0.mp4
0a2c2b69c6a6bc2f0eb259becf10e28fdf47659f3a5abcb63b5ec6dcb731def5  video_robot/watchdog_play_robot_overlay.log
0a2c2b69c6a6bc2f0eb259becf10e28fdf47659f3a5abcb63b5ec6dcb731def5  video_robot/watchdog_play_robot_overlay.log.attempt1
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  video_robot/watchdog_play_robot_overlay.log.attempt2
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  video_robot/watchdog_play_robot_overlay.log.attempt3
8d48bf6ab92ea45e7de1206ab650f107350dfeec949982529a80a3fda80c4d02  video_robot/watchdog_train_robot_overlay.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  video_robot/watchdog_train_robot_overlay.log.attempt1
8d48bf6ab92ea45e7de1206ab650f107350dfeec949982529a80a3fda80c4d02  video_robot/watchdog_train_robot_overlay.log.attempt2
cd0e73abb1ad8d77e0576d3e1634e19a126b7f1cecebb1e90579112cac6aa7d3  visibility/d555_depth_absdiff_vis_vs_hid.png
854d43cbd37d656cdb2351e1d82df8c7ac8c4588472dd494174b0725b2892390  visibility/d555_marker_visibility.json
d2f4b711ed9cbf140d3aac56b62d94e200e9f38cb08c23582c03847dc48e97c6  visibility/d555_marker_visibility.log
9f5fd9a6705409489107fb8eb4e47715a97162878357ba0b986025255715850d  visibility/d555_marker_visibility.py
c6294d0bbc9d2b1965587ca16fd3c9b7b4d5288055a34df55c93ba2933223260  visibility/d555_marker_visibility_cmd.sh
0d428c1afc91e3e5b9fc39fdfdb23f2d674e888347d1113111364b0bc2ff4790  visibility/d555_rgb_absdiff_vis_vs_hid.png
9ca61622e3bbc43b71ee62f94b70ba8ccc3b6290fbf062f2986106c2609ad856  visibility/d555_rgb_hid.png
5c5581fb6eb7c60efcd21f54064cc767b648f4ab37b3636047ffc204134f7d39  visibility/d555_rgb_hid2.png
b692a037b78c8e81ecb39ac230ae52fb9736c6d54f88812c05aea86751f16ca2  visibility/d555_rgb_vis.png
12b4b6ad6159de766bad260a95051b2bb345193f6c61b6d1a78c1857330e0a01  visibility/d555_rgb_vis2.png
b4c1b6c1617ebab58bc1c5a64d9ff637a2908038297959cbab4e067f58e4f455  visibility/overhead_hid.png
b2d897f826a2d1f2351e0e9b84f176c533d977a17b003fa8c8e4c4867977e78c  visibility/overhead_hid2.png
7efdbbfeb59ec43fc9584c5027bfb5a906d7980f68ffa20f617083617c6cf574  visibility/overhead_vis.png
5fd2332762099d92bc59b806459a099327bedca25e6b3ae50c478d55906c5996  visibility/overhead_vis2.png
99a1f90bd32bca46f68896a536ceea9c7037cec061f266ff3d9870e8abed6e17  visibility/watchdog_d555_marker_visibility.log
99a1f90bd32bca46f68896a536ceea9c7037cec061f266ff3d9870e8abed6e17  visibility/watchdog_d555_marker_visibility.log.attempt1
```
