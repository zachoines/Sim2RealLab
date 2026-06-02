# Training-scene render is broken after the env-cfg-composition cutover (blank + freeze headed; black video headless)

**Status:** Shipped 2026-06-02 in `207f2b8` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/75

**Type:** investigation + fix (render / env-cfg composition)
**Owner:** DGX agent
**Priority:** P2 — raised from P3 after a new observation (below): the
break is **not** headless-only. Running **headed** `--video` opens the
viewport to a **blank screen and then freezes** — so the rendered scene
is not coming up at all under the composition, in *either* mode. That is
more than a cosmetic video-recording gap: a headed render that hangs
blocks the operator's normal "watch a training run" workflow and
suggests the composed scene's render/materialization is genuinely
broken, not just unlit in the headless path. Training itself (depth
policy camera + `Episode_Termination/*` scalars) is still unaffected,
which is why this didn't surface until a headed `--video` smoke run.
**Estimate:** S–M — a bisect across the composition epic's render-path
changes + the fix for whichever step breaks the composed-scene render.
**Branch:** `task/headless-training-video-black`

## Story

As a **DGX operator running `train_strafer_navigation.py --video`
(headed or headless)** I want **the training scene to render as it did
before the env-cfg-composition cutover** so that **I can watch a headed
run without the viewport coming up blank and freezing, and eyeball
rollout behavior (flips, stalls, wall-hugging) in the recorded clips
instead of staring at black frames**.

## Symptom

`--headless --video` training clips are **pure black** — every pixel 0
across every frame (verified by decoding frames and checking
`array.mean()`, not by eyeballing a thumbnail). The MP4s are valid and
h264-decodable; the codec is not the issue (installing/forcing h264 made
no difference). The robot's depth-policy camera (what the policy trains
on) is unaffected; only the operator-facing RGB viewport recording is
black.

**Headed is worse, not better (new — the load-bearing observation).**
The same run **headed** (drop `--headless`, keep `--video`, which
auto-injects `--viz kit`) opens the editor viewport to a **blank screen
and then freezes**. So this is not "the headless render path doesn't
light the viewport" — the **composed scene does not render at all**, in
either mode. The earlier headless-only framing was too narrow. Repro
that exhibits the freeze:

```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 64 \
  --video --video_length 200 --video_interval 500 --max_iterations 50 \
  --log_dir logs/rsl_rl/headed_repro_$(date +%s)
```

**Diagnostic caveat for whoever picks this up:** the *first* clip's
*first frame* (`rl-video-step-0.mp4`, frame 0) is legitimately black on
a good run too — it is captured during env spawn/reset before lighting
settles. Always sample a **mid-run frame** (e.g. `rl-video-step-100`
frame ~50) when deciding render-vs-black, or you will misread a working
run as broken (this cost a full investigation cycle).

## What is already ruled out (verified)

The regression was bisected by re-running the identical
`--headless --video --num_envs 64` smoke at successive commits and
pixel-checking a mid-run frame:

- **NOT the codec.** Frames decode fine; pixels are zero.
- **NOT a missing scene light in config.** The composed ProcRoom env
  cfg at HEAD resolves a normal `DomeLight` at `/World/DomeLight`
  (intensity 2000), `render_interval=4`, a default `RenderCfg`, and a
  populated `PhysxCfg` — nothing blacked out.
- **NOT the robot USD / its light rig.** The robot asset's baked
  `/OmniKit_Viewport_LightRig` (DomeLight + DistantLight) is byte-for-
  byte identical between the last-good asset and the current one.
- **NOT a physics / solver / restitution change.** A commit *before*
  those changes (still post-composition) is also black; a commit
  *before* composition is fine.
- **NOT an environmental / driver / shader-cache change.** Re-running an
  old **pre-composition** commit on the *current* machine + driver +
  shader cache renders correctly, so the host environment is not the
  cause — it is in-repo code.

## Bisect result (the window)

| Commit | Composition? | Mid-frame render | Notes |
|---|---|---|---|
| `19962d2` (pre-epic, ~May 25) | no | **renders** (mean ≈ 75) | old env id `Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0` |
| `17efec7` (post-epic) | yes | **black** (mean = 0) | composed env id `Isaac-Strafer-Nav-RLDepth-Real-v0` |
| HEAD | yes | **black** | — |

So the regression was introduced somewhere in the **env-cfg-composition
epic** — PRs **#69 (`57c19d6`)** and **#70 (`7d497e0`)** — which is the
only span between the good and bad commits that restructures the scene /
env composition. This is the **env-cfg-composition lane**, not
teleop-perf (the teleop-perf branch merely sits on top of it and
surfaced the symptom while validating a training smoke).

## Config-level checks that came back CLEAN (so the break is at render/build time)

A headless construction of the composed ProcRoom RL variant
(`StraferNavCfg_RLDepth_Real`, built without booting Kit) shows the
**config materializes completely and correctly** — which *narrows* the
bug to scene-build / render time, it does not exonerate the composition:

- **Scene members all present:** `robot` (ArticulationCfg), `terrain`
  (TerrainImporterCfg), `dome_light` (AssetBaseCfg at `/World/DomeLight`,
  intensity 2000), `room_primitives` (RigidObjectCollectionCfg),
  `d555_camera`, `contact_sensor`, `d555_imu`. `num_envs=64`,
  `env_spacing=10`, `replicate_physics=False`.
- **Events intact:** `generate_room` (`generate_proc_room`, reset),
  `reset_robot_proc_room`, the DR terms — the room actually gets built.
- **Room-building is unchanged from the good commit:** the pre-composition
  `StraferSceneCfg_ProcRoom` used the *same* `room_primitives =
  RigidObjectCollectionCfg` with the same 44-object palette + the same
  `generate_proc_room` event. So the proc-room generation path is not the
  regression.
- **The shared physics change is exonerated:** the blank/freeze was
  already present at `17efec7`, which predates the physics commits
  (`c1c37cc`/`6620b76`); `_DEFAULT_NAV_SIM_DT` / `_RENDER_INTERVAL` /
  `_DECIMATION` are byte-identical to the good commit, and the added
  `cfg.sim.physics = PhysxCfg(enable_stabilization=True)` is physics-only.

So: the composed env *config* is right, the *room* is built, the *physics*
is fine — yet the headed viewport comes up blank and freezes. The break is
in how the composition **materializes/renders** the scene at runtime, not
in any cfg field this static dump can see.

## Likely mechanism (hypothesis, not yet confirmed)

The composition cutover materializes the scene through
`composed_env_cfg.py` / `_ComposedStraferNavEnvCfg.__post_init__` rather
than the old hand-written per-cell scene classes. Since the config is
verified complete (above) and the freeze happens at first render, the
probable causes shift toward **runtime materialization / render order**:
(a) env-origin / prim-path nesting changed so the viewport camera (or the
`RecordVideo` world-frame fallback) frames an empty/unlit region; (b) the
`DomeLight` is authored but ends up at a stage path or under an env-clone
nesting the RTX viewport does not light; (c) a `sim`/render or
`InteractiveSceneCfg` setting (clone-in-fabric, replicate_physics
interaction, render product) that the old classes set is dropped or
reordered, stalling the first render. The **freeze** specifically points
at a render that never completes (a render-product/Fabric stall), not
merely a dark frame — so capture *where* it hangs (during
`gym.make`/scene clone, first `env.reset`, or first `RecordVideo`
render).

## Next step to localize

Two cheap, parallel angles — do whichever is faster on the box:

**1. Bisect #69 vs #70.** Re-run the smoke at `57c19d6` (#69) and check a
mid-run frame. If `57c19d6` renders, the break is in #70 (`7d497e0`); if
already black, it is in #69. Then diff that PR's scene-materialization
path (lights / viewer / env-origin / prim paths / clone settings) against
the pre-epic classes.

**2. Live USD-stage diff (settles the hypothesis directly).** Boot a
composed ProcRoom env headed and dump the live stage's light prims +
their *world* transforms and the active viewport camera path, then do the
same on the pre-epic commit. The composed run should reveal either the
`DomeLight` at an unexpected stage path / under an env-clone nesting, or
the viewport camera pointed at an unlit region — whichever it is, that is
the fix site. Since the headed run **freezes**, also note *where* it hangs
(scene clone during `gym.make`, first `env.reset`, or first `RecordVideo`
render) — the hang point localizes the materialization step.

Repro (headed exhibits the freeze; headless gives the black clips):

```bash
source env_setup.sh
# headed — watch for blank viewport + freeze, note where it hangs:
$ISAACLAB -p Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 64 \
  --video --video_length 100 --video_interval 100 --max_iterations 3 \
  --log_dir logs/rsl_rl/bisect_<commit>
# headless variant for the pixel check (add --headless); inspect a
# MID-run frame, not step-0 frame-0:
#   logs/rsl_rl/bisect_<commit>/*/videos/rl-video-step-100.mp4
```

## Resolution

**Root cause — RGB render product stripped from the depth-policy camera.**
#69 (`57c19d6`) introduced `_prune_scene_cameras`, which regenerates the
scene's policy camera from `sensors.policy_data_types()`. For the depth-only
RL stack that resolves to `("distance_to_image_plane",)`, so the camera is
rebuilt **without `rgb`** — defeating the reason the hand-written
`StraferSceneCfg_ProcRoom` (reused unchanged by the composition) deliberately
authored its `d555_camera` with `("rgb", "distance_to_image_plane")`: an rgb
render product is what makes the RTX renderer bring up its colour pipeline, and
without one the operator viewport / `--video` recording produce black frames
(headed, the first render stalls). The depth policy obs is unaffected, so
training kept working. The strip was deliberate — the pruner's docstring called
it a no-op for RL scenes and the contract test framed dropping rgb as a
"render-cost win" — and camera channels were intentionally excluded from the
hashed contract, so the goldens stayed green and CI never caught it. This
matches the bisect exactly: pre-epic the policy camera always rendered rgb+depth
(renders, mean ≈ 75); post-#69 it is depth-only (black, mean 0).

**Fix.** `_prune_scene_cameras` now always unions `rgb` into the policy
camera's render data types — the policy camera doubles as the scene's RGB
render product for viewport/video colour init, independent of which channels
the observation consumes. The observation layer still selects its image terms
separately, so the policy-facing contract is untouched. The fix lives in the
shared pruner, so it is source-agnostic (plane / ProcRoom / Infinigen). The
misleading "no-op for RL" / "render-cost win" comments were corrected, and a
regression test (`test_depth_policy_camera_still_renders_rgb_for_viewport`)
now asserts the depth-policy RL camera keeps its rgb channel.

**Verification.** `test/env/test_composition_contract.py` 21/21 (6 frozen
goldens + depth-obs golden green; 2 new regression cases). Cfg-level (no Kit):
rgb+depth present on the plane scaffold, both ProcRoom RL variants, and the
Infinigen Bridge variant; `policy_data_types()` still depth-only and the
`RLDepth_Real` obs still exposes only `depth_image`. Headless `--video` smoke
on `Isaac-Strafer-Nav-RLDepth-Real-v0` (`--num_envs 16 --max_iterations 3`):
`rl-video-step-100.mp4` every sampled frame mean ≈ 78; `rl-video-step-0.mp4`
frame 0 ≈ 0 (the documented spawn/reset frame) but frames 25/50/75/99 ≈ 78.

## Acceptance

- [x] Localize the regression to #69 or #70 (or a specific commit
      within), and identify the scene-materialization change that drops
      the headless viewport illumination. → **#69 (`57c19d6`)**, which
      introduced `_prune_scene_cameras`; it regenerates the depth-policy
      camera with only the observed channel, stripping the `rgb` channel
      the reused `StraferSceneCfg_ProcRoom` carried for RTX colour init.
- [x] `--headless --video` training clips show the lit robot/scene again
      (mid-run frame `array.mean()` > 0), verified on a composed env id.
      → `Isaac-Strafer-Nav-RLDepth-Real-v0`, `--num_envs 16`: mid-clip
      frames mean ≈ **78** (was 0); frame 0 of the first clip still ~0
      per the documented spawn/reset caveat.
- [x] No change to the policy-facing obs contract (the depth-policy
      camera is unaffected; this is a viewport-render fix only) — the
      `test_composition_contract` golden hashes stay green. → 21/21 pass,
      all 6 frozen goldens green (camera render channels are excluded from
      the hashed contract); `policy_data_types()` and the obs terms are
      untouched.
- [x] If the fix is a scene-cfg change, confirm it holds across the
      plane / ProcRoom / Infinigen sources the composition serves. → the
      fix is in the shared `_prune_scene_cameras`, so it is source-agnostic;
      cfg-level checks confirm rgb+depth on the plane scaffold, both
      ProcRoom RL variants, and the Infinigen Bridge variant.

## Out of scope

- **The `--headless --video` path being structurally degraded vs headed
  recording.** Pre-composition it rendered fine headless, so the bar is
  "restore the prior headless behavior," not "redesign headless video."
- **RGB perception / teleop capture rendering** (separate camera + path;
  not observed broken).
- **Any physics / solver tuning** — exonerated by the bisect.

## Triggered by

The `teleop-perf-architecture` training-stability smoke run
(`--headless --video --num_envs 64`) produced black clips; bisection
while diagnosing isolated the cause to the env-cfg-composition epic and
exonerated the teleop-perf physics changes. Surfaced as a finding rather
than fixed in-line because it belongs to the env-cfg-composition lane.
