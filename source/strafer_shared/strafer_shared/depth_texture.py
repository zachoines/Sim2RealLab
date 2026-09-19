"""Per-pixel texture of a policy-resolution depth frame, shared by both lanes.

The statistic is the 95th percentile of ``|d - median3x3(d)|`` and the exact-zero
share of that high-pass. It separates a per-pixel i.i.d. field, which a 3x3 median
removes almost entirely, from a surface-consistent one, which survives it.

The near-field fill is one constant written over a third of a typical frame. It
carries no texture, so it is excluded: left in, it pulls every figure toward zero
by its own share of the frame.

``reference`` describes a residual (frame minus reference) instead of the frame.
The exclusion and the band binning then read the reference, which is the surface
the pixel is on. Without one, both read the frame.

Units are the caller's: metres for a raw frame, normalised for an observation
vector. ``fill`` and ``bands`` must be given in the same units as ``frame``.
"""

from __future__ import annotations

import numpy as np

from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_NEARFIELD_FILL,
    DEPTH_WIDTH,
)

# Depth bands, in metres, matching the sensor's own error table. A pixel is
# binned by its own depth, so the quadratic depth dependence of stereo error
# does not average across the frame.
DEPTH_TEXTURE_BANDS = (
    (0.4, 1.0),
    (1.0, 1.5),
    (1.5, 2.5),
    (2.5, 3.5),
    (3.5, 5.5),
)

_PERCENTILE = 95.0


def _as_frame(values: np.ndarray) -> np.ndarray:
    """A flat observation slice or an image, as a float64 ``(H, W)`` image."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (DEPTH_HEIGHT, DEPTH_WIDTH):
        return array
    if array.shape == (DEPTH_HEIGHT * DEPTH_WIDTH,):
        return array.reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
    raise ValueError(
        f"expected a {DEPTH_HEIGHT}x{DEPTH_WIDTH} depth frame or its "
        f"{DEPTH_HEIGHT * DEPTH_WIDTH}-value flattening; got shape {array.shape}"
    )


def median3x3(frame: np.ndarray) -> np.ndarray:
    """3x3 median of ``frame``, border replicated.

    Replication keeps every pixel in the statistic. Dropping the border instead
    would exclude the frame's outermost ring and move the percentile, so the
    two rules are not interchangeable.
    """
    image = _as_frame(frame)
    padded = np.pad(image, 1, mode="edge")
    windows = np.stack(
        [padded[y:y + image.shape[0], x:x + image.shape[1]]
         for y in range(3) for x in range(3)]
    )
    return np.median(windows, axis=0)


def highpass(frame: np.ndarray) -> np.ndarray:
    """``frame - median3x3(frame)``: what a 3x3 median would remove."""
    image = _as_frame(frame)
    return image - median3x3(image)


def nearfield_mask(
    frame: np.ndarray,
    *,
    fill: float = DEPTH_NEARFIELD_FILL,
    atol: float = 1e-4,
) -> np.ndarray:
    """Pixels holding the near-field fill, which both lanes write as one constant."""
    return np.isclose(_as_frame(frame), fill, atol=atol, rtol=0.0)


def texture_stats(
    frame: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    fill: float = DEPTH_NEARFIELD_FILL,
    atol: float = 1e-4,
) -> dict:
    """High-pass p95 and exact-zero share over the pixels that carry texture.

    Args:
        frame: The depth frame, as an image or its flattening.
        reference: Describe ``frame - reference`` instead of ``frame``.
        mask: Pixels to include. Defaults to everything but the near-field fill.
        fill: The near-field fill value, in ``frame``'s units.
        atol: Absolute tolerance on the fill comparison.

    Returns:
        ``p95_abs_highpass``, ``exact_zero_share`` and the ``pixels`` counted.
    """
    image = _as_frame(frame)
    surface = image if reference is None else _as_frame(reference)
    field = image if reference is None else image - surface

    if mask is None:
        mask = ~nearfield_mask(surface, fill=fill, atol=atol)
    selected = highpass(field)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return {"p95_abs_highpass": float("nan"),
                "exact_zero_share": float("nan"),
                "pixels": 0}
    return {
        "p95_abs_highpass": float(np.percentile(np.abs(selected), _PERCENTILE)),
        "exact_zero_share": float((selected == 0.0).mean()),
        "pixels": int(selected.size),
    }


def texture_stats_by_band(
    frame: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    bands: tuple[tuple[float, float], ...] = DEPTH_TEXTURE_BANDS,
    fill: float = DEPTH_NEARFIELD_FILL,
    atol: float = 1e-4,
) -> list[dict]:
    """``texture_stats`` per depth band, binning each pixel by its own depth.

    The depth read is the reference where there is one, so a residual's pixels
    are binned by the surface they are on rather than by the noise on them.
    """
    surface = _as_frame(frame if reference is None else reference)
    valid = ~nearfield_mask(surface, fill=fill, atol=atol)
    rows = []
    for low, high in bands:
        in_band = valid & (surface >= low) & (surface < high)
        stats = texture_stats(frame, reference=reference, mask=in_band,
                              fill=fill, atol=atol)
        rows.append({"band_m": (float(low), float(high)), **stats})
    return rows
