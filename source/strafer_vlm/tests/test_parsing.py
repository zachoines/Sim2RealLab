"""Unit tests for JSON extraction, bbox coercion, normalisation, and coordinate conversion."""

from __future__ import annotations

import pytest

from strafer_vlm.inference.parsing import (
    GroundingTarget,
    _coerce_bbox,
    _coerce_bool,
    _coerce_confidence,
    _coerce_optional_text,
    bbox_to_pixel_coords,
    clamp_pixel_bbox,
    denormalize_bbox_1000,
    extract_first_json_object,
    infer_prediction_bbox_coordinate_mode,
    normalize_prediction_bbox_to_1000,
    parse_grounding_prediction,
    parse_grounding_target,
)


# -----------------------------------------------------------------------
# extract_first_json_object
# -----------------------------------------------------------------------


class TestExtractFirstJsonObject:
    def test_plain_json(self):
        assert extract_first_json_object('{"found": true}') == {"found": True}

    def test_fenced_json(self):
        text = '```json\n{"found": true, "bbox_2d": [1,2,3,4]}\n```'
        result = extract_first_json_object(text)
        assert result == {"found": True, "bbox_2d": [1, 2, 3, 4]}

    def test_fenced_without_lang(self):
        text = '```\n{"found": false}\n```'
        assert extract_first_json_object(text) == {"found": False}

    def test_json_surrounded_by_prose(self):
        text = 'Here is the result: {"found": true, "label": "door"} as requested.'
        result = extract_first_json_object(text)
        assert result == {"found": True, "label": "door"}

    def test_empty_string(self):
        assert extract_first_json_object("") is None

    def test_no_json(self):
        assert extract_first_json_object("no json here at all") is None

    def test_invalid_json(self):
        assert extract_first_json_object("{broken json") is None

    def test_returns_first_object(self):
        text = '{"a": 1} {"b": 2}'
        assert extract_first_json_object(text) == {"a": 1}

    def test_nested_object(self):
        text = '{"outer": {"inner": 1}}'
        result = extract_first_json_object(text)
        assert result == {"outer": {"inner": 1}}


# -----------------------------------------------------------------------
# _coerce_bool
# -----------------------------------------------------------------------


class TestCoerceBool:
    @pytest.mark.parametrize("value,expected", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("TRUE", True),
        ("FALSE", False),
        ("1", True),
        ("0", False),
    ])
    def test_valid_values(self, value, expected):
        assert _coerce_bool(value) == expected

    def test_none_returns_default(self):
        assert _coerce_bool(None) is False
        assert _coerce_bool(None, default=True) is True

    def test_unrecognised_string_returns_default(self):
        assert _coerce_bool("maybe") is False
        assert _coerce_bool("maybe", default=True) is True


# -----------------------------------------------------------------------
# _coerce_optional_text
# -----------------------------------------------------------------------


class TestCoerceOptionalText:
    def test_none(self):
        assert _coerce_optional_text(None) is None

    def test_empty_string(self):
        assert _coerce_optional_text("") is None

    def test_whitespace_only(self):
        assert _coerce_optional_text("   ") is None

    def test_normal_text(self):
        assert _coerce_optional_text("  door  ") == "door"

    def test_numeric_coerced(self):
        assert _coerce_optional_text(42) == "42"


# -----------------------------------------------------------------------
# _coerce_confidence
# -----------------------------------------------------------------------


class TestCoerceConfidence:
    def test_none(self):
        assert _coerce_confidence(None) is None

    def test_float(self):
        assert _coerce_confidence(0.85) == 0.85

    def test_int(self):
        assert _coerce_confidence(1) == 1.0

    def test_string(self):
        assert _coerce_confidence("0.75") == 0.75

    def test_clamp_high(self):
        assert _coerce_confidence(1.5) == 1.0

    def test_clamp_low(self):
        assert _coerce_confidence(-0.3) == 0.0

    def test_empty_string(self):
        assert _coerce_confidence("") is None

    def test_invalid_string(self):
        assert _coerce_confidence("abc") is None


# -----------------------------------------------------------------------
# _coerce_bbox
# -----------------------------------------------------------------------


class TestCoerceBbox:
    def test_valid_bbox(self):
        assert _coerce_bbox([100, 200, 300, 400]) == (100, 200, 300, 400)

    def test_float_coords_rounded(self):
        assert _coerce_bbox([100.4, 200.6, 300.5, 400.1]) == (100, 201, 300, 400)

    def test_clamped_to_1000(self):
        assert _coerce_bbox([-10, 50, 1100, 500]) == (0, 50, 1000, 500)

    def test_no_clamp(self):
        result = _coerce_bbox([0, 0, 1920, 1080], clamp_range=None)
        assert result == (0, 0, 1920, 1080)

    def test_degenerate_zero_width(self):
        assert _coerce_bbox([100, 100, 100, 200]) is None

    def test_degenerate_inverted(self):
        assert _coerce_bbox([300, 100, 100, 200]) is None

    def test_none(self):
        assert _coerce_bbox(None) is None

    def test_wrong_length(self):
        assert _coerce_bbox([1, 2, 3]) is None

    def test_non_numeric(self):
        assert _coerce_bbox(["a", "b", "c", "d"]) is None

    def test_tuple_input(self):
        assert _coerce_bbox((50, 60, 200, 300)) == (50, 60, 200, 300)


# -----------------------------------------------------------------------
# parse_grounding_prediction
# -----------------------------------------------------------------------


class TestParseGroundingPrediction:
    def test_valid_json(self):
        text = '{"found": true, "bbox_2d": [120, 80, 500, 400], "label": "chair"}'
        result = parse_grounding_prediction(text)
        assert result is not None
        assert result.found is True
        assert result.bbox_2d == (120, 80, 500, 400)
        assert result.label == "chair"

    def test_found_false(self):
        result = parse_grounding_prediction('{"found": false}')
        assert result is not None
        assert result.found is False
        assert result.bbox_2d is None

    def test_fenced_output(self):
        text = '```json\n{"found": true, "bbox_2d": [10, 20, 30, 40]}\n```'
        result = parse_grounding_prediction(text)
        assert result is not None
        assert result.found is True
        assert result.bbox_2d == (10, 20, 30, 40)

    def test_prose_with_json(self):
        text = "I can see the object. Here: {\"found\": true, \"bbox_2d\": [1,2,3,4]}"
        result = parse_grounding_prediction(text)
        assert result is not None
        assert result.found is True

    def test_no_json(self):
        assert parse_grounding_prediction("I cannot find it.") is None

    def test_found_inferred_from_bbox(self):
        result = parse_grounding_prediction('{"bbox_2d": [10, 20, 100, 200]}')
        assert result is not None
        assert result.found is True

    def test_no_clamp_on_prediction(self):
        # parse_grounding_prediction does NOT clamp — that's for normalization later
        result = parse_grounding_prediction('{"found": true, "bbox_2d": [0, 0, 1920, 1080]}')
        assert result is not None
        assert result.bbox_2d == (0, 0, 1920, 1080)


# -----------------------------------------------------------------------
# parse_grounding_target
# -----------------------------------------------------------------------


class TestParseGroundingTarget:
    def test_valid_dict(self):
        result = parse_grounding_target(
            {"found": True, "bbox_2d": [100, 200, 300, 400]},
            strict_bbox_when_found=True,
        )
        assert result.found is True
        assert result.bbox_2d == (100, 200, 300, 400)

    def test_valid_json_string(self):
        result = parse_grounding_target(
            '{"found": true, "bbox_2d": [10, 20, 90, 80]}',
            strict_bbox_when_found=True,
        )
        assert result.found is True

    def test_strict_missing_bbox(self):
        with pytest.raises(ValueError, match="missing/invalid bbox_2d"):
            parse_grounding_target(
                {"found": True},
                strict_bbox_when_found=True,
            )

    def test_non_strict_missing_bbox(self):
        result = parse_grounding_target(
            {"found": True},
            strict_bbox_when_found=False,
        )
        assert result.found is True
        assert result.bbox_2d is None

    def test_not_found(self):
        result = parse_grounding_target(
            {"found": False},
            strict_bbox_when_found=True,
        )
        assert result.found is False


# -----------------------------------------------------------------------
# infer_prediction_bbox_coordinate_mode
# -----------------------------------------------------------------------


class TestInferCoordinateMode:
    def test_none_bbox(self):
        assert infer_prediction_bbox_coordinate_mode(None, image_width=640, image_height=480) is None

    def test_fits_image_is_pixel(self):
        result = infer_prediction_bbox_coordinate_mode(
            (0, 0, 640, 480), image_width=640, image_height=480
        )
        assert result == "pixel"

    def test_small_bbox_is_normalized(self):
        result = infer_prediction_bbox_coordinate_mode(
            (100, 200, 500, 800), image_width=640, image_height=480
        )
        assert result == "normalized_1000"

    def test_exceeds_1000_is_pixel(self):
        result = infer_prediction_bbox_coordinate_mode(
            (0, 0, 1200, 900), image_width=1920, image_height=1080
        )
        assert result == "pixel"


# -----------------------------------------------------------------------
# normalize_prediction_bbox_to_1000
# -----------------------------------------------------------------------


class TestNormalizePredictionBbox:
    def test_pixel_to_normalized(self):
        pred = GroundingTarget(found=True, bbox_2d=(0, 0, 640, 480), bbox_coordinate_mode="pixel")
        result = normalize_prediction_bbox_to_1000(pred, image_width=640, image_height=480)
        assert result.bbox_coordinate_mode == "normalized_1000"
        assert result.bbox_2d == (0, 0, 1000, 1000)

    def test_already_normalized(self):
        pred = GroundingTarget(found=True, bbox_2d=(100, 200, 500, 800), bbox_coordinate_mode="normalized_1000")
        result = normalize_prediction_bbox_to_1000(pred, image_width=640, image_height=480)
        assert result.bbox_2d == (100, 200, 500, 800)

    def test_none_bbox_passthrough(self):
        pred = GroundingTarget(found=False, bbox_2d=None)
        result = normalize_prediction_bbox_to_1000(pred, image_width=640, image_height=480)
        assert result.bbox_2d is None

    def test_half_image_pixel_bbox(self):
        pred = GroundingTarget(found=True, bbox_2d=(0, 0, 320, 240), bbox_coordinate_mode="pixel")
        result = normalize_prediction_bbox_to_1000(pred, image_width=640, image_height=480)
        assert result.bbox_2d == (0, 0, 500, 500)


# -----------------------------------------------------------------------
# denormalize_bbox_1000
# -----------------------------------------------------------------------


class TestDenormalizeBbox:
    def test_full_image(self):
        assert denormalize_bbox_1000((0, 0, 1000, 1000), 640, 480) == (0, 0, 639, 479)

    def test_half_image(self):
        result = denormalize_bbox_1000((0, 0, 500, 500), 640, 480)
        assert result == (0, 0, 320, 240)

    def test_small_bbox(self):
        result = denormalize_bbox_1000((250, 250, 750, 750), 100, 100)
        assert result == (25, 25, 75, 75)


# -----------------------------------------------------------------------
# clamp_pixel_bbox
# -----------------------------------------------------------------------


class TestClampPixelBbox:
    def test_within_bounds(self):
        assert clamp_pixel_bbox((10, 20, 100, 200), image_width=640, image_height=480) == (10, 20, 100, 200)

    def test_clamps_to_bounds(self):
        result = clamp_pixel_bbox((-5, -5, 700, 500), image_width=640, image_height=480)
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 639
        assert result[3] <= 479


# -----------------------------------------------------------------------
# bbox_to_pixel_coords
# -----------------------------------------------------------------------


class TestBboxToPixelCoords:
    def test_pixel_mode(self):
        result = bbox_to_pixel_coords(
            (10, 20, 100, 200), image_width=640, image_height=480, coordinate_mode="pixel"
        )
        assert result == (10, 20, 100, 200)

    def test_normalized_mode(self):
        result = bbox_to_pixel_coords(
            (0, 0, 500, 500), image_width=640, image_height=480, coordinate_mode="normalized_1000"
        )
        assert result == (0, 0, 320, 240)

    def test_none_mode_defaults_to_denormalize(self):
        result = bbox_to_pixel_coords(
            (0, 0, 500, 500), image_width=640, image_height=480, coordinate_mode=None
        )
        assert result == (0, 0, 320, 240)
