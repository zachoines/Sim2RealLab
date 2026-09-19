"""Report the per-pixel texture of recorded depth frames.

CPU-only, numpy-only. Reads a capture and prints, for the whole frame and per
depth band, the 95th percentile of ``|d - median3x3(d)|`` and the exact-zero
share of that high-pass, with the near-field-fill class excluded.

Accepted inputs, detected by extension and content:

  - ``node_obs.jsonl`` / ``gym_obs.jsonl`` (optionally gzipped): one JSON object
    per line with an ``obs`` list; depth is the tail ``DEPTH_WIDTH*DEPTH_HEIGHT``
    values, normalized, so they are scaled by ``DEPTH_MAX`` back to metres.
  - a single JSON object with an ``obs`` list, for a one-frame capture.
  - ``.npz``: the first array, as frames in metres, shaped ``(N, 45, 80)``,
    ``(N, 3600)``, ``(45, 80)`` or ``(3600,)``.

With ``--reference`` the statistic describes the residual against that capture's
frames rather than the frames themselves, pairing them by index (one reference
frame is broadcast). A residual isolates what was added to a surface; a frame
also carries the scene's own edges, which are the larger signal::

    <python> source/strafer_lab/scripts/measure_depth_texture.py <capture>
    <python> source/strafer_lab/scripts/measure_depth_texture.py <capture> --reference <clean>
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_MAX, DEPTH_WIDTH
from strafer_shared.depth_texture import (
    DEPTH_TEXTURE_BANDS,
    texture_stats,
    texture_stats_by_band,
)

PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def load_frames(path: Path) -> np.ndarray:
    """Depth frames in metres, shaped ``(N, 45, 80)``."""
    if path.suffix == ".npz":
        with np.load(path) as bundle:
            array = np.asarray(bundle[bundle.files[0]], dtype=np.float64)
        return array.reshape(-1, DEPTH_HEIGHT, DEPTH_WIDTH)

    with _open(path) as handle:
        text = handle.read()
    # One JSON document (an object or an array of them) or one per line. A
    # pretty-printed object spans lines, so the whole-text parse comes first.
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = document if isinstance(document, list) else [document]

    rows = []
    for record in records:
        obs = record.get("obs")
        if obs is None:
            continue
        values = np.asarray(obs, dtype=np.float64)
        if values.size < PIXELS:
            raise ValueError(
                f"{path}: an observation of {values.size} values cannot hold a "
                f"{DEPTH_HEIGHT}x{DEPTH_WIDTH} depth field"
            )
        rows.append(values[-PIXELS:] * DEPTH_MAX)
    if not rows:
        raise ValueError(f"{path}: no record carried an 'obs' field")
    return np.asarray(rows).reshape(-1, DEPTH_HEIGHT, DEPTH_WIDTH)


def measure(frames: np.ndarray, references: np.ndarray | None) -> dict:
    """Per-frame whole-frame statistics, and the per-band statistics pooled."""
    whole, bands = [], []
    for index, frame in enumerate(frames):
        reference = None
        if references is not None:
            reference = references[index] if len(references) > 1 else references[0]
        whole.append(texture_stats(frame, reference=reference))
        bands.append(texture_stats_by_band(frame, reference=reference))
    return {"whole": whole, "bands": bands}


def report(result: dict) -> None:
    whole = result["whole"]
    p95 = np.asarray([row["p95_abs_highpass"] for row in whole])
    zero = np.asarray([row["exact_zero_share"] for row in whole])
    pixels = np.asarray([row["pixels"] for row in whole])
    finite = np.isfinite(p95)

    print(f"frames {len(whole)}   valid pixels/frame: median {int(np.median(pixels))} "
          f"of {PIXELS}")
    print("\nwhole frame, near-field-fill class excluded")
    print(f"  {'':10s} {'p50':>12s} {'p05':>12s} {'p95':>12s}")
    print(f"  {'hp p95 (m)':10s} {np.median(p95[finite]):12.8f} "
          f"{np.percentile(p95[finite], 5):12.8f} {np.percentile(p95[finite], 95):12.8f}")
    print(f"  {'hp ==0':10s} {np.median(zero[finite]):12.6f} "
          f"{np.percentile(zero[finite], 5):12.6f} {np.percentile(zero[finite], 95):12.6f}")

    print("\nby depth band (a pixel is binned by its own depth)")
    print(f"  {'band (m)':>10s} {'n/frame':>9s} {'hp p95 (m)':>13s} {'hp ==0':>9s}")
    for position, (low, high) in enumerate(DEPTH_TEXTURE_BANDS):
        column = [frame_bands[position] for frame_bands in result["bands"]]
        band_p95 = np.asarray([row["p95_abs_highpass"] for row in column])
        band_zero = np.asarray([row["exact_zero_share"] for row in column])
        band_n = np.asarray([row["pixels"] for row in column])
        usable = np.isfinite(band_p95)
        label = f"{low:.1f}-{high:.1f}"
        if not usable.any():
            print(f"  {label:>10s} {0:9.1f} {'(no pixels)':>13s} {'':>9s}")
            continue
        print(f"  {label:>10s} {band_n.mean():9.1f} {np.median(band_p95[usable]):13.8f} "
              f"{np.median(band_zero[usable]):9.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, help="jsonl / jsonl.gz / json / npz capture")
    parser.add_argument("--reference", type=Path, default=None,
                        help="describe the residual against this capture instead")
    args = parser.parse_args()

    frames = load_frames(args.capture)
    references = load_frames(args.reference) if args.reference else None
    if references is not None and len(references) not in (1, len(frames)):
        print(f"reference has {len(references)} frames, capture has {len(frames)}; "
              "pass one reference frame or one per capture frame", file=sys.stderr)
        return 2

    result = measure(frames, references)
    print(f"=== {args.capture}"
          + (f"  vs  {args.reference}" if args.reference else "") + " ===")
    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
