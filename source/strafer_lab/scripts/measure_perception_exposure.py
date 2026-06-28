"""Measure exposure quality of recorded perception RGB video.

CPU-only QA gate. Decodes one or more h264 RGB MP4s with PyAV and reports,
per file, the numbers that decide whether the footage is correctly exposed:

  - mean RGB + mean luma (Rec.601)
  - clipped fraction:  any channel >= 250/255  (blown-to-white highlights)
  - crushed fraction:  any channel <= 5/255    (lost shadow detail)
  - fully-white frame count
  - per-row (top->bottom) clipped profile, binned into 10 bands (a TOP-peaked
    profile is ceiling/overhead glare; BOTTOM/MID-peaked is whole-frame
    over-exposure)

Each file is checked against the acceptance bands (overridable via flags) and
PASS/FAIL is reported. Exit code is non-zero if any file fails.

Pass MP4 paths directly, or ``--lerobot-root <dataset>`` to resolve the
perception RGB video inside a LeRobot v3 dataset. Run with a python that has
PyAV + numpy (e.g. the Isaac Sim interpreter)::

    <python> source/strafer_lab/scripts/measure_perception_exposure.py path/to/file.mp4
    <python> source/strafer_lab/scripts/measure_perception_exposure.py --lerobot-root <dataset_dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np

# Relative path to the perception RGB video inside a LeRobot v3 dataset.
LEROBOT_PERCEPTION_REL = "videos/observation.images.perception/chunk-000/file-000.mp4"
N_BANDS = 10


def measure(mp4: Path, *, clip_hi: int, crush_lo: int) -> dict | None:
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
        anyhi = (img >= clip_hi).any(axis=2)
        anylo = (img <= crush_lo).any(axis=2)
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


def verdict(res: dict, args) -> tuple[bool, list[str]]:
    """Return (passed, failed-check messages) against the acceptance bands."""
    fails = []
    if res["clip_frac"] > args.max_clip:
        fails.append(f"clip {res['clip_frac']:.1%} > {args.max_clip:.0%}")
    if res["white_frames"] > args.max_white_frames:
        fails.append(f"{res['white_frames']} fully-white frames > {args.max_white_frames}")
    if not (args.luma_min <= res["mean_luma"] <= args.luma_max):
        fails.append(f"mean_luma {res['mean_luma']} outside [{args.luma_min}, {args.luma_max}]")
    if res["crush_frac"] > args.max_crush:
        fails.append(f"crush {res['crush_frac']:.1%} > {args.max_crush:.0%}")
    return (not fails), fails


def _resolve_videos(args) -> list[Path]:
    videos = [Path(v) for v in args.video]
    videos += [Path(r) / LEROBOT_PERCEPTION_REL for r in args.lerobot_root]
    return videos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="*", help="RGB MP4 file(s) to measure")
    ap.add_argument("--lerobot-root", action="append", default=[],
                    help="LeRobot dataset dir; resolves the perception RGB video inside it (repeatable)")
    ap.add_argument("--clip-hi", type=int, default=250, help="clipped if any channel >= this")
    ap.add_argument("--crush-lo", type=int, default=5, help="crushed if any channel <= this")
    ap.add_argument("--max-clip", type=float, default=0.02, help="max clipped fraction")
    ap.add_argument("--max-crush", type=float, default=0.10, help="max crushed fraction")
    ap.add_argument("--max-white-frames", type=int, default=0, help="max fully-white frames")
    ap.add_argument("--luma-min", type=float, default=90.0, help="min mean luma")
    ap.add_argument("--luma-max", type=float, default=150.0, help="max mean luma")
    args = ap.parse_args()

    videos = _resolve_videos(args)
    if not videos:
        ap.error("pass at least one MP4 path or --lerobot-root")

    all_pass = True
    for mp4 in videos:
        res = measure(mp4, clip_hi=args.clip_hi, crush_lo=args.crush_lo)
        print(f"\n=== {mp4} ===")
        if res is None:
            print("  (no frames / missing)")
            all_pass = False
            continue
        for k, v in res.items():
            print(f"  {k}: {v}")
        ok, fails = verdict(res, args)
        all_pass = all_pass and ok
        print(f"  VERDICT: {'PASS' if ok else 'FAIL -> ' + '; '.join(fails)}")
    print(f"\n{'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
