# The RGB render darkens ~28 % across the Isaac Sim 6.0 bump

**Type:** investigation → recalibration (render lane)
**Owner:** DGX / render lane
**Priority:** P2 — **not flip-blocking**; the policy lane is unaffected, because the depth
observation was measured unmoved (band profiles correlate at r = 1.000000). What is
affected is every lane calibrated against RGB luminance, and those calibrations are wrong
by a known factor now that the flip has landed.
**Estimate:** S to recalibrate, M if the settings diff finds a cause worth chasing; the
root cause is upstream and is not ours to fix.
**Branch:** n/a — this brief records a measured condition and its disposition.

## Story

As the **capture and perception lanes**, we need **the post-bump RGB luminance change
characterized and the exposure baselines recalibrated**, so that **exposure checks keep
meaning what they meant instead of failing, or passing, for a reason that has nothing to
do with the scene.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)

## What was measured

Measured 2026-08-23 in the pre-bump validation record's render leg. One 200-frame overhead
training clip per stack, identical command, `measure_perception_exposure.py` read for its
numbers rather than its verdict.

| clip | mean_luma | clip_frac | crush_frac |
|---|---|---|---|
| 2026-08-14 baseline | 86.6 | 0.0 | 0.0056 |
| **old pin, same session** | **86.6** | **0.0** | **0.0056** |
| new pin, run 1 | 68.6 | 0.0 | 0.1218 |
| new pin, run 2 | 68.3 | 0.0 | 0.1279 |

The gate's bounds were ±10 luma and ±0.01 crush. Both are exceeded — by **1.8×** and
**12×**. `clip_frac` does not move.

## Confounded — the two stacks were filmed from different camera poses

**The comparison above is not valid and the shift is not yet attributable to the
renderer.** The capture scripts anchor the recording camera over env 0 by writing
`camera_position` / `camera_target` on the recorder's capture config. Isaac Lab renamed
those fields to `eye` / `lookat` at v3.0.0-beta2 (upstream `2644c1eb0`, #5474). The
scripts kept writing the old names, which on the new stack set attributes nothing reads:
the anchor silently did nothing and the recorder filmed from its own default pose. The
"anchored" line in the logs is the script's own print of the pose it *intended*: it
reports the computed world pose whether or not the write reached a field the recorder
reads, and it is emitted even when no capture object was found at all.

So the old-pin clips are framed on env 0 and the new-pin clips are framed wherever the
recorder's default put them. A luma difference between clips of different framings of a
procedurally furnished room is expected and says nothing about photometry. The 18.0 luma
and 0.116 crush deltas measure pose, renderer, or both, in unknown proportion.

What survives: the old-pin/baseline agreement (86.6 / 0.0056 reproduced exactly) is a
same-pose comparison and still shows the old stack is stable. The two new-pin runs
agreeing with each other (68.6/68.3, 0.1218/0.1279) shows the new stack is internally
stable. Neither speaks to the gap between them.

**Re-measured with matched poses, 2026-09-12 — the shift is confirmed and it is bigger.**
The capture scripts write whichever field pair the installed Isaac Lab exposes, so the
anchor reaches the recorder on both stacks. To prove it reached it, a one-off readback
instrument took the camera prim's world transform off the stage after the recorder's first
render and compared it to the request; both pins landed on the anchor (retired 0.0 m
error, canonical 1.8e-15 m). At matched poses the retired pair reads **87.6 luma / 0.0005
crush** and the canonical pair **62.6 / 0.1713** — deltas of **25.0 luma** and **0.171
crush**, far outside the Δ ≤ 10 / Δ ≤ 0.01 that would have exonerated the renderer.

Removing the confound made the gap **larger**, not smaller: the unanchored new-pin clips
read 68.6/68.3 because the recorder's default pose flattered them. The rescale in
disposition 1 therefore stands, but at **0.715** (62.6 / 87.6), not 0.792. Crush is the
worse violation — 343× the retired pair's, 17× the Δ ≤ 0.01 comparison bound, and past
the tool's own 0.10 absolute bound — and both pins fail its absolute luma floor of 90.0,
so the tool's verdict cannot separate the stacks and only the numbers can.

The instrument is deposited with the record rather than kept in the capture scripts: it
answered its question once, and a live camera check is now a visual one (see the
sim-bridge cheatsheet's remote-view section).

Recorded in `docs/measurements/isaac-lab-upgrade-landing-2026-09-12/`.

**This is the stack, not the instrument.** The old-pin control reproduces the nine-day-old
baseline to every recorded digit, and each pin reproduces its own value across two clips.
Video-path integrity is intact: the recording is anchored on `env_0` and the `_capture`
patch found `/World/envs/env_0`, so it is not the silent world-frame fallback. The failure
is photometric, not structural.

**Depth is untouched.** The depth observation was measured on both Play tiers: mean moved
+0.02 % of a 2 % budget, band profiles correlate at r = 1.000000 with an unchanged peak
row. Depth is geometric; this is photometric. The two results do not conflict, and
together they say the policy's input distribution is unchanged while the RGB image is not.

## The change is a stable global scale — which is what makes recalibration mechanical

Per-frame luma over the deposited clips (200 frames; frame 0 is black on both pins and is
excluded):

```
delta (new − old):   first-10 −15.86   mid −18.47   last-10 −20.40
ratio (new / old):   mean 0.7919   sd 0.0116   min 0.7531   max 0.8746
```

A **uniform ≈0.79× scale across the whole clip**, not an early-frame artifact — which is
what disproves the one documented candidate below. The ratio's spread is 1.5 % relative,
so the factor is well determined.

A caveat on how to read that fit: regressing new on old gives a low R² (0.39 pure-scale,
0.44 affine), but the old clip's own luma spans just 76.9–88.3 (sd 1.42), so there is
almost no dynamic range for a regression to explain and R² is uninformative here. The
statistic that carries the recalibration is the ratio's tightness, not the fit quality,
and the affine term is not distinguishable from the scale over this narrow range.

**Recalibration factor: ×0.792** — **superseded.** This section's per-frame analysis is
sound as a description of the two deposited clips, but those clips differ in pose as well
as in stack, so 0.792 mixes both. The matched-pose factor is **0.715**; see the
2026-09-12 re-measurement above. What survives here is the *shape* of the change: a
uniform global scale across the clip, not an early-frame artifact.

## What has been ruled out

Inherited from the upstream attribution pass (2026-08-23) and not re-done here:

- **No documented change.** Nothing in the 6.0.0 GA, 6.0.1 GA, Kit 110.1.x or RTX 110.1
  notes describes a tonemapper, auto-exposure or colour-management change.
- **The OCIO config is byte-identical** between the two installs; the base kit RTX
  settings differ only by two multitick flags. The change lives inside the RTX
  110.0 → 110.1.2 binaries.
- **The one documented same-sign candidate is disproven by measurement.** A 6.0.x fix for
  *"first batch of render product images appear washed out"* would lower mean luma, but
  its removal would show as an **early-frame** artifact. The delta is uniform across the
  clip (above), so that is not this.
- **Nearest ecosystem relative:** isaac-sim/IsaacSim **#724** — our exact stack, camera
  init invalidating the RTX rendering configuration, closed without a root cause.

Re-verified here rather than taken on trust: the per-frame deltas and the 0.79 ratio
reproduce exactly from the deposited clips.

## Version context

The bump is larger than its version numbers suggest. pip `isaacsim 6.0.0.0` (the retired
pair) is the 6.0.0 **Early Developer Release** (`VERSION` = `6.0.0-rc.22`, Kit
`110.0.0+feature`);
pip `6.0.1.0` is 6.0.1 GA (Kit `110.1.2+production`). The migration therefore spans
pre-GA developer build → GA + point release, and the whole 6.0.0 GA release rode along.
Name installs by build metadata, not by pip version.

## Affected lanes

- **Capture** — RGB frames written into LeRobot datasets; any exposure gate on them.
- **Perception / VLM grounding** — the corpus and its exposure history; the `[90, 150]`
  window in `measure_perception_exposure.py` was set for perception frames.
- **Not the policy lane.** The unmoved depth observation is the evidence.
- `measure_perception_exposure.py`'s own bands are the concrete artifact to rescale.

## Disposition

1. **Diagnose before recalibrating.** A crushed-pixel fraction of 17 % is an
   image-quality regression, not a calibration offset: 17 % of the frame has gone to
   black and no rescaling of a luma band recovers what is no longer in the image. Treat
   the ×0.715 as the size of the problem, not as the fix. Both pairs are on disk and
   bootable, which makes the cheap discriminator available for the first time — dump the
   runtime carb settings from a booted Kit on each pin and diff them, tonemapper and
   auto-exposure defaults first, since those are what move mean luma and black point
   together. Recalibrate only if no setting accounts for the gap; if one does, the fix is
   the setting, not the band. Either way the new bands come from post-flip readings, not
   from arithmetic on the old ones: `[90, 150]` rescaled is `[64.4, 107.3]`, which the
   canonical pin's 62.6 still misses.
2. **Do not tune a render setting to compensate.** No exposure or renderer setting was
   touched to produce these numbers, and every *static* config surface examined is
   identical between the installs — a compensating setting would be an unexplained local
   mod of exactly the kind the candidate clone was just shown not to need. This does not
   forbid the diagnosis in item 1: reading the runtime settings to find out *why* the
   image changed is not the same as setting one to hide it.
3. **PR #147 (default-renderer A/B) re-baselines regardless**, so it inherits the new
   numbers rather than needing a separate pass.
4. **Report upstream.** No Isaac Sim issue describes this; the per-frame ratio analysis
   plus the byte-identical OCIO finding is a complete report.

## Evidence

Record `docs/measurements/isaac-lab-upgrade-stage3-2026-08-23/` (its render leg) and the
matching evidence deposit `isaac-lab-upgrade-stage3-2026-08-23/` — the three clips
(`video/g5-train-clip-OLDPIN-control.mp4`, `…-newpin-run1.mp4`, `…-newpin-run2.mp4`) and
the exposure readings under `render/`. Those are the confounded clips.

The matched-pose re-measurement that supersedes them is
`docs/measurements/isaac-lab-upgrade-landing-2026-09-12/`, with its deposit's `render/`
carrying both clips, both `camera-anchor.json` readbacks, and the exposure readings.
