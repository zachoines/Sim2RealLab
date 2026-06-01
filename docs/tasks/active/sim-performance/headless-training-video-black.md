# Headless training videos render black after the env-cfg-composition cutover

**Type:** investigation + fix (render / env-cfg composition)
**Owner:** DGX agent
**Priority:** P3 — operator-facing diagnostics only; does NOT affect
training itself (the policy consumes the depth-policy camera, not the
recorded RGB viewport) or the `Episode_Termination/*` TensorBoard
scalars used to judge stability. Cosmetic-but-real: the periodic
training videos are the operator's only visual check that the robot is
behaving during long headless runs.
**Estimate:** S–M — a bisect across the composition epic's render-path
changes + the fix for whichever step drops the headless viewport
illumination.
**Branch:** `task/headless-training-video-black`

## Story

As a **DGX operator running `train_strafer_navigation.py --headless
--video`** I want **the periodic MP4 clips to show the robot in the lit
scene as they did before the env-cfg-composition cutover** so that **I
can eyeball rollout behavior (flips, stalls, wall-hugging) during a long
headless training run instead of staring at black frames**.

## Symptom

`--headless --video` training clips are **pure black** — every pixel 0
across every frame (verified by decoding frames and checking
`array.mean()`, not by eyeballing a thumbnail). The MP4s are valid and
h264-decodable; the codec is not the issue (installing/forcing h264 made
no difference). The robot's depth-policy camera (what the policy trains
on) is unaffected; only the operator-facing RGB viewport recording is
black.

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

## Likely mechanism (hypothesis, not yet confirmed)

The composition cutover materializes the scene (lights, ground, robot,
viewer) through `composed_env_cfg.py` /
`_ComposedStraferNavEnvCfg.__post_init__` rather than the old
hand-written per-cell scene classes. The headless `--video` path renders
through Isaac Lab's `RecordVideo` wrapper on the env viewport (no live
Kit viewport under `--headless`). The most probable cause is that the
composed scene either (a) no longer attaches the `DomeLight` prim at a
path the headless RTX viewport sees, (b) changed env-origin / prim
nesting so the world-frame fallback camera frames an unlit region, or
(c) drops a viewport/render setting the old classes set. The fact that
the *config object* still lists a DomeLight (above) points at a
materialization/prim-path issue at scene build time rather than a
missing cfg field — confirm by dumping the live USD stage's lights +
their world transforms under a composed env vs the old one.

## Next step to localize (one bisect)

Split #69 vs #70: re-run the identical smoke at `57c19d6` (#69) and
pixel-check a mid-run frame. If `57c19d6` renders, the break is in #70
(`7d497e0`); if it is already black, it is in #69. Then diff that PR's
scene-materialization path (lights / viewer / env-origin / prim paths)
against the pre-epic scene classes.

Repro (identical to the smoke that surfaced it):

```bash
source env_setup.sh
$ISAACLAB -p Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 64 --headless \
  --video --video_length 100 --video_interval 100 --max_iterations 3 \
  --log_dir logs/rsl_rl/bisect_<commit>
# inspect a MID-run frame, not step-0 frame-0:
#   logs/rsl_rl/bisect_<commit>/*/videos/rl-video-step-100.mp4
```

## Acceptance

- [ ] Localize the regression to #69 or #70 (or a specific commit
      within), and identify the scene-materialization change that drops
      the headless viewport illumination.
- [ ] `--headless --video` training clips show the lit robot/scene again
      (mid-run frame `array.mean()` > 0), verified on a composed env id.
- [ ] No change to the policy-facing obs contract (the depth-policy
      camera is unaffected; this is a viewport-render fix only) — the
      `test_composition_contract` golden hashes stay green.
- [ ] If the fix is a scene-cfg change, confirm it holds across the
      plane / ProcRoom / Infinigen sources the composition serves.

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
