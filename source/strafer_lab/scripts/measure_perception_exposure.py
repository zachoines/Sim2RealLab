"""Measure exposure quality of captured d555 PERCEPTION rgb frames.

CPU-only QA gate for the LeRobot corpus. Decodes the recorded perception
RGB video (LeRobot v3 h264, 640w x 360h) with PyAV and reports, per scene,
the numbers that decide whether the data is correctly exposed:

  - mean RGB + mean luma (Rec.601)
  - clipped fraction:  any channel >= 250/255  (blown-to-white highlights)
  - crushed fraction:  any channel <= 5/255    (lost shadow detail)
  - fully-white frame count (a confound-immune over-exposure signal)
  - per-row (top->bottom, height = 360) clipped profile, binned into 10 bands

The per-row profile attributes the blowout: a TOP-peaked profile is ceiling
bloom; a BOTTOM/MID-peaked profile is whole-scene over-exposure.

ACCEPTANCE (measured on a PRODUCTION ceiling-on, non-``--video`` capture):
  - clipped (>=250)   <= 2.0%  AND  0 fully-white frames
  - mean luma         in [90, 150]
  - crushed (<=5)     <= 10.0%
seed5 must pass clipped AND crushed simultaneously (its bimodal profile is
the binding constraint). NOTE: the on-disk ``--video`` smoke runs hide the
ceiling globally, so they are for diagnosis only — set/confirm the final
exposure on a ceiling-on capture and re-run this script on it.

Run with the Isaac Sim python (PyAV + numpy present)::

    $STRAFER_ISAACLAB_PYTHON \\
        source/strafer_lab/scripts/measure_perception_exposure.py \\
        --root data/sim_in_the_loop/<dataset>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np

REL_MP4 = "videos/observation.images.perception/chunk-000/file-000.mp4"
CLIP_HI = 250          # any channel >= this -> clipped / blown
CRUSH_LO = 5           # any channel <= this -> crushed black (acceptance floor)
N_BANDS = 10

# Acceptance bands (production ceiling-on frames).
MAX_CLIP_FRAC = 0.02
LUMA_BAND = (90.0, 150.0)
MAX_CRUSH_FRAC = 0.10
MAX_WHITE_FRAMES = 0


def measure(mp4: Path) -> dict | None:
    if not mp4.exists():
        print(f"  MISSING: {mp4}", file=sys.stderr)
        return None
    container = av.open(str(mp4))
    stream = container.streams.video[0]

    n_frames = n_pixels = clipped = crushed = white = 0
    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_luma = 0.0
    H = W = None
    row_clip = None

    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        h, w, _ = img.shape
        if H is None:
            H, W = h, w
            row_clip = np.zeros(H, dtype=np.float64)
        anyhi = (img >= CLIP_HI).any(axis=2)
        anylo = (img <= CRUSH_LO).any(axis=2)
        luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

        px = h * w
        n_frames += 1
        n_pixels += px
        sum_rgb += img.reshape(-1, 3).sum(axis=0)
        sum_luma += float(luma.sum())
        fc = int(anyhi.sum())
        clipped += fc
        crushed += int(anylo.sum())
        if fc / px >= 0.99:
            white += 1
        row_clip += anyhi.sum(axis=1)  # index = row (height)

    container.close()
    if n_frames == 0:
        return None

    row_frac = row_clip / (n_frames * W)
    edges = np.linspace(0, H, N_BANDS + 1).astype(int)
    bands = [round(float(row_frac[edges[i]:edges[i + 1]].mean()), 3)
             for i in range(N_BANDS)]

    return {
        "frames": n_frames,
        "HxW": (H, W),
        "mean_rgb": (sum_rgb / n_pixels).round(1).tolist(),
        "mean_luma": round(sum_luma / n_pixels, 1),
        "clip_frac": round(clipped / n_pixels, 4),
        "crush_frac": round(crushed / n_pixels, 4),
        "white_frames": white,
        "row_bands_top_to_bottom": bands,
    }


def verdict(res: dict) -> tuple[bool, list[str]]:
    """Return (passed, failed-check messages) against the acceptance bands."""
    fails = []
    if res["clip_frac"] > MAX_CLIP_FRAC:
        fails.append(f"clip {res['clip_frac']:.1%} > {MAX_CLIP_FRAC:.0%}")
    if res["white_frames"] > MAX_WHITE_FRAMES:
        fails.append(f"{res['white_frames']} fully-white frames > {MAX_WHITE_FRAMES}")
    if not (LUMA_BAND[0] <= res["mean_luma"] <= LUMA_BAND[1]):
        fails.append(f"mean_luma {res['mean_luma']} outside {LUMA_BAND}")
    if res["crush_frac"] > MAX_CRUSH_FRAC:
        fails.append(f"crush {res['crush_frac']:.1%} > {MAX_CRUSH_FRAC:.0%}")
    return (not fails), fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="dataset root (repeatable); defaults to the on-disk smoke runs")
    args = ap.parse_args()
    roots = args.root or [
        "data/sim_in_the_loop/video_smoke_seed5",
        "data/sim_in_the_loop/video_smoke_seed6",
        "data/sim_in_the_loop/video_smoke_seed7",
    ]
    all_pass = True
    for root in roots:
        res = measure(Path(root) / REL_MP4)
        print(f"\n=== {Path(root).name} ===")
        if res is None:
            print("  (no frames / missing)")
            all_pass = False
            continue
        for k, v in res.items():
            print(f"  {k}: {v}")
        ok, fails = verdict(res)
        all_pass = all_pass and ok
        print(f"  VERDICT: {'PASS' if ok else 'FAIL -> ' + '; '.join(fails)}")
    print(f"\n{'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
