"""The shared per-pixel depth texture statistic.

Both lanes read the same number off a depth frame: the 95th percentile of
``|d - median3x3(d)|`` and the exact-zero share of that high-pass, over the
pixels that are not the near-field fill. It is the currency in which a training
frame's texture and a deploy frame's texture are compared, so the three
decisions inside it — which percentile, which pixels, which border rule — are
pinned here rather than left to whichever caller measured last.

Each is pinned against an independent nested-loop reference, and each test also
asserts the answer a plausible wrong decision would give, so a mutation cannot
pass by coincidence.

Pure numpy — no torch, no Isaac Sim, no Kit boot.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_NEARFIELD_FILL, DEPTH_WIDTH
from strafer_shared.depth_texture import (
    DEPTH_TEXTURE_BANDS,
    highpass,
    median3x3,
    nearfield_mask,
    texture_stats,
    texture_stats_by_band,
)

PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH


def _reference_median3x3(image: np.ndarray, *, replicate: bool) -> np.ndarray:
    """Nested-loop 3x3 median, written independently of the implementation.

    ``replicate`` clamps an out-of-frame neighbour to the edge pixel; otherwise
    the window simply loses it, which is the other rule a border could follow.
    """
    height, width = image.shape
    out = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            window = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if replicate:
                        ny = min(max(ny, 0), height - 1)
                        nx = min(max(nx, 0), width - 1)
                    elif not (0 <= ny < height and 0 <= nx < width):
                        continue
                    window.append(image[ny, nx])
            out[y, x] = np.median(window)
    return out


def _scene(seed: int = 5, noise: float = 0.01) -> np.ndarray:
    """A frame with structure, texture and a near-field patch, in metres.

    A depth ramp carries the surface, a step edge carries low-frequency
    structure the median keeps, and seeded per-pixel noise carries the texture
    the median removes. The near-field patch is the constant both lanes write.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:DEPTH_HEIGHT, 0:DEPTH_WIDTH]
    frame = 1.0 + 4.0 * x / DEPTH_WIDTH + 0.5 * y / DEPTH_HEIGHT
    frame[:, DEPTH_WIDTH // 2:] += 1.2
    if noise:
        frame += rng.normal(0.0, noise, frame.shape)
    frame[:8, :10] = DEPTH_NEARFIELD_FILL
    return frame


def _fine_structure_scene() -> np.ndarray:
    """A surface textured at the per-pixel scale, with no noise on it.

    A 3x3 median keeps a two-pixel stripe — the window still holds a majority
    of the pixel's own surface — and removes a one-pixel one, so single-pixel
    structure is what frame mode reads off a scene.
    """
    y, x = np.mgrid[0:DEPTH_HEIGHT, 0:DEPTH_WIDTH]
    return 2.0 + 1.5 * ((x + y) % 2)


class TestTheBorderRule:
    def test_the_border_is_replicated_not_shrunk(self):
        frame = _scene()
        assert np.array_equal(median3x3(frame),
                              _reference_median3x3(frame, replicate=True))
        # The alternative rule is a real alternative: it disagrees on the
        # border, so a median that silently shrank its window would be caught.
        shrunk = _reference_median3x3(frame, replicate=False)
        assert not np.array_equal(median3x3(frame), shrunk)

    def test_every_pixel_is_counted_including_the_border(self):
        frame = _scene()
        frame[:8, :10] = 3.0  # no near-field class, so nothing is excluded
        assert texture_stats(frame)["pixels"] == PIXELS

        # Dropping the one-pixel border is the other plausible rule. It keeps
        # 3354 of 3600 pixels and moves the percentile, so it cannot be
        # substituted without failing here.
        interior = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=bool)
        interior[1:-1, 1:-1] = True
        assert int(interior.sum()) < PIXELS
        assert texture_stats(frame, mask=interior)["p95_abs_highpass"] != pytest.approx(
            texture_stats(frame)["p95_abs_highpass"], abs=1e-12
        )


class TestThePercentile:
    def test_the_statistic_is_the_95th_percentile_of_the_absolute_highpass(self):
        frame = _scene()
        valid = ~nearfield_mask(frame)
        selected = np.abs(highpass(frame)[valid])
        stats = texture_stats(frame)
        assert stats["p95_abs_highpass"] == pytest.approx(
            float(np.percentile(selected, 95.0)), abs=1e-15)

    @pytest.mark.parametrize("wrong", [50.0, 90.0, 99.0])
    def test_a_neighbouring_percentile_is_a_different_number(self, wrong):
        frame = _scene()
        valid = ~nearfield_mask(frame)
        selected = np.abs(highpass(frame)[valid])
        assert texture_stats(frame)["p95_abs_highpass"] != pytest.approx(
            float(np.percentile(selected, wrong)), abs=1e-9)


class TestTheNearFieldExclusion:
    def test_the_fill_class_is_excluded_by_default(self):
        frame = _scene()
        filled = int(nearfield_mask(frame).sum())
        assert filled == 80  # the 8x10 patch
        assert texture_stats(frame)["pixels"] == PIXELS - filled

    def test_leaving_the_fill_in_changes_the_answer(self):
        """The fill is a constant: its high-pass is exactly zero, so counting it
        inflates the zero share and pulls the percentile down."""
        frame = _scene()
        excluded = texture_stats(frame)
        included = texture_stats(frame, mask=np.ones((DEPTH_HEIGHT, DEPTH_WIDTH), bool))
        assert included["pixels"] == PIXELS
        assert included["exact_zero_share"] > excluded["exact_zero_share"]
        assert included["p95_abs_highpass"] < excluded["p95_abs_highpass"]

    def test_the_fill_is_recognised_in_the_caller_s_units(self):
        """A normalised frame carries the fill at 0.2/6.0; the default constant
        is metres, so the units have to be declared rather than guessed."""
        frame = _scene() / 6.0
        assert int(nearfield_mask(frame).sum()) == 0
        assert int(nearfield_mask(frame, fill=DEPTH_NEARFIELD_FILL / 6.0).sum()) == 80


class TestTheResidualMode:
    def test_a_reference_makes_the_statistic_describe_what_was_added(self):
        rng = np.random.default_rng(9)
        clean = _scene()
        added = rng.normal(0.0, 0.02, clean.shape)
        keep = ~nearfield_mask(clean)
        residual = texture_stats(clean + added, reference=clean)
        direct = texture_stats(added, mask=keep)
        assert residual["p95_abs_highpass"] == pytest.approx(
            direct["p95_abs_highpass"], rel=1e-12)

    def test_the_exclusion_reads_the_reference_not_the_noisy_frame(self):
        """Noise moves a pixel off the fill value. Binning and excluding on the
        reference keeps the class the same size whatever the noise does."""
        rng = np.random.default_rng(11)
        clean = _scene()
        noisy = clean + rng.normal(0.0, 0.02, clean.shape)
        assert int(nearfield_mask(noisy).sum()) < int(nearfield_mask(clean).sum())
        assert texture_stats(noisy, reference=clean)["pixels"] == PIXELS - 80

    def test_a_frame_carries_the_scene_s_own_texture_and_a_residual_does_not(self):
        """The two modes answer different questions. Frame mode reads whatever
        the median removes, the surface's own fine structure included; residual
        mode reads only what was added to that surface."""
        rng = np.random.default_rng(13)
        clean = _fine_structure_scene()
        noisy = clean + rng.normal(0.0, 0.005, clean.shape)
        assert (texture_stats(noisy)["p95_abs_highpass"]
                > 10.0 * texture_stats(noisy, reference=clean)["p95_abs_highpass"])

    def test_frame_mode_reads_a_featureless_surface_as_featureless(self):
        """The coverage question the statistic exists for: a far, per-pixel
        featureless field reads as no texture at all, whatever its depth."""
        assert texture_stats(np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 5.5)) == {
            "p95_abs_highpass": 0.0, "exact_zero_share": 1.0, "pixels": PIXELS}


class TestTheBandBinning:
    def test_a_pixel_is_binned_by_its_own_depth(self):
        frame = _scene()
        rows = texture_stats_by_band(frame)
        assert [row["band_m"] for row in rows] == [
            (float(lo), float(hi)) for lo, hi in DEPTH_TEXTURE_BANDS]
        for row, (low, high) in zip(rows, DEPTH_TEXTURE_BANDS):
            in_band = (~nearfield_mask(frame)) & (frame >= low) & (frame < high)
            assert row["pixels"] == int(in_band.sum())

    def test_the_bands_partition_the_valid_pixels_they_cover(self):
        """No pixel is counted twice, and the fill is in no band."""
        frame = _scene()
        counted = sum(row["pixels"] for row in texture_stats_by_band(frame))
        covered = (~nearfield_mask(frame)) & (frame >= DEPTH_TEXTURE_BANDS[0][0]) & (
            frame < DEPTH_TEXTURE_BANDS[-1][1])
        assert counted == int(covered.sum())

    def test_an_empty_band_reports_no_pixels_rather_than_a_number(self):
        frame = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 5.0)
        rows = texture_stats_by_band(frame)
        assert rows[0]["pixels"] == 0
        assert np.isnan(rows[0]["p95_abs_highpass"])


class TestTheInputContract:
    def test_a_flat_observation_slice_is_accepted(self):
        frame = _scene()
        assert texture_stats(frame.reshape(-1)) == texture_stats(frame)

    @pytest.mark.parametrize("shape", [(45, 81), (3599,), (2, 45, 80)])
    def test_a_frame_that_is_not_the_policy_grid_is_refused(self, shape):
        with pytest.raises(ValueError):
            texture_stats(np.zeros(shape))

    def test_a_locally_flat_field_has_no_highpass_at_all(self):
        frame = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 4.0)
        stats = texture_stats(frame)
        assert stats["p95_abs_highpass"] == 0.0
        assert stats["exact_zero_share"] == 1.0
