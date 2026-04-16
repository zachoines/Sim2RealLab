"""Tests for strafer_lab.tools.infinigen_label_parser.

Pure Python — runs in .venv_vlm via the strafer_lab namespace stub.
Synthetic prim names mirror the actual Infinigen USD export pattern:

    /World/GlassPanelDoorFactory_430087__spawn_asset_5_
    /World/window
    /World/camera_0_0
    /World/_materials/shader_metal_001_deepcopy_001
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.infinigen_label_parser import (
    DEFAULT_LABEL_RULES,
    PLAIN_NAME_LABELS,
    ParsedLabel,
    factory_class_to_label,
    is_skippable_prim,
    parse_factory_label,
)


# ---------------------------------------------------------------------------
# parse_factory_label — the main entry point
# ---------------------------------------------------------------------------


class TestParseFactoryLabelStandardPattern:
    def test_glass_panel_door(self):
        result = parse_factory_label("GlassPanelDoorFactory_430087__spawn_asset_5_")
        assert result is not None
        assert result.label == "door"
        assert result.instance_id == 430087
        assert result.factory_class == "GlassPanelDoorFactory"
        assert result.spawn_index == 5

    def test_chair_with_spawn_index_zero(self):
        result = parse_factory_label("ChairFactory_991132__spawn_asset_0_")
        assert result is not None
        assert result.label == "chair"
        assert result.instance_id == 991132
        assert result.spawn_index == 0

    def test_kitchen_cabinet(self):
        result = parse_factory_label("KitchenCabinetFactory_204411__spawn_asset_2_")
        assert result is not None
        assert result.label == "cabinet"
        assert result.factory_class == "KitchenCabinetFactory"

    def test_office_chair_collapses_to_chair(self):
        result = parse_factory_label("OfficeChairFactory_555__spawn_asset_0_")
        assert result is not None
        assert result.label == "chair"

    def test_armchair_keeps_specific_label(self):
        """ArmChair has a more specific rule than Chair — should win."""
        result = parse_factory_label("ArmChairFactory_42__spawn_asset_0_")
        assert result is not None
        assert result.label == "armchair"


class TestParseFactoryLabelOptionalSpawnSuffix:
    def test_no_spawn_suffix(self):
        """Some Infinigen prims drop the __spawn_asset_N_ tail."""
        result = parse_factory_label("ChairFactory_991132")
        assert result is not None
        assert result.label == "chair"
        assert result.instance_id == 991132
        assert result.spawn_index == 0

    def test_trailing_underscore_optional(self):
        """The trailing underscore on __spawn_asset_N_ is optional."""
        result = parse_factory_label("ChairFactory_42__spawn_asset_0")
        assert result is not None
        assert result.spawn_index == 0


class TestParseFactoryLabelArchitecture:
    """Room-architecture prims like ``living_room_0_0_wall``.

    These are emitted by Infinigen's room solver, not the asset factory
    pipeline. The parser splits them into a (surface, room) pair so
    downstream code can both avoid walls AND build per-room targets.
    """

    def test_living_room_wall(self):
        result = parse_factory_label("living_room_0_0_wall")
        assert result is not None
        assert result.label == "wall"
        assert result.room_type == "living_room"
        assert result.instance_id == -1
        assert result.factory_class == ""

    def test_kitchen_floor(self):
        result = parse_factory_label("kitchen_0_0_floor")
        assert result is not None
        assert result.label == "floor"
        assert result.room_type == "kitchen"

    def test_bathroom_ceiling(self):
        result = parse_factory_label("bathroom_0_1_ceiling")
        assert result is not None
        assert result.label == "ceiling"
        assert result.room_type == "bathroom"

    def test_dining_room_with_underscore_in_name(self):
        result = parse_factory_label("dining_room_0_0_wall")
        assert result is not None
        assert result.room_type == "dining_room"
        assert result.label == "wall"

    def test_bedroom_with_blender_duplicate_suffix(self):
        result = parse_factory_label("bedroom_0_0_wall_001")
        assert result is not None
        assert result.label == "wall"
        assert result.room_type == "bedroom"


class TestParseFactoryLabelPlainNames:
    def test_plain_window(self):
        result = parse_factory_label("window")
        assert result is not None
        assert result.label == "window"
        assert result.instance_id == -1
        assert result.factory_class == ""

    def test_plain_door(self):
        assert parse_factory_label("door").label == "door"

    def test_plain_name_case_insensitive(self):
        assert parse_factory_label("WINDOW").label == "window"
        assert parse_factory_label("Door").label == "door"


class TestParseFactoryLabelSkippablePrims:
    @pytest.mark.parametrize(
        "name",
        [
            "_materials",
            "camera_0_0",
            "camrig_0",
            "Light_001",
            "env_light",
            "terrain",
            "shader_metal_001_deepcopy_001",
        ],
    )
    def test_returns_none_for_skippable(self, name):
        assert parse_factory_label(name) is None

    def test_returns_none_for_empty(self):
        assert parse_factory_label("") is None
        assert parse_factory_label("   ") is None

    def test_returns_none_for_unknown_unstructured_name(self):
        # Not a factory pattern, not in PLAIN_NAME_LABELS, not a skip
        # pattern — should be silently rejected.
        assert parse_factory_label("RandomScopeXform_12345") is None

    def test_returns_none_for_factory_pattern_with_garbage(self):
        # 'factory' lower-case isn't valid CamelCase Factory
        assert parse_factory_label("chairfactory_42__spawn_asset_0_") is None


# ---------------------------------------------------------------------------
# factory_class_to_label — tested directly for the fallback path
# ---------------------------------------------------------------------------


class TestFactoryClassToLabel:
    def test_known_class(self):
        assert factory_class_to_label("ChairFactory") == "chair"

    def test_kitchen_cabinet(self):
        assert factory_class_to_label("KitchenCabinetFactory") == "cabinet"

    def test_substring_match(self):
        # GlassPanelDoorFactory contains "Door"
        assert factory_class_to_label("GlassPanelDoorFactory") == "door"

    def test_more_specific_rule_wins(self):
        # OfficeChair contains "Chair", but the OfficeChair rule itself
        # collapses to "chair" — both paths arrive at the same answer
        # here. ArmChair has a distinct mapping that should win over
        # the generic "Chair" rule.
        assert factory_class_to_label("ArmChairFactory") == "armchair"

    def test_fallback_snake_case(self):
        # No rule matches a fictional "WidgetFactory" → snake_case fallback.
        assert factory_class_to_label("WidgetFactory") == "widget"

    def test_fallback_compound_snake_case(self):
        assert factory_class_to_label("MyWeirdAssetFactory") == "my_weird_asset"

    def test_empty_class_returns_empty_label(self):
        assert factory_class_to_label("") == ""


# ---------------------------------------------------------------------------
# is_skippable_prim
# ---------------------------------------------------------------------------


class TestIsSkippablePrim:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("_materials", True),
            ("camera_0_0", True),
            ("camrig_5", True),
            ("Light_main", True),
            ("env_light", True),
            ("terrain_blob", True),
            ("shader_xyz", True),
            # Material / shader graph internals exported from Blender's
            # Principled BSDF — observed in real Infinigen USD output.
            ("World", True),
            ("Principled_BSDF", True),
            ("Principled_BSDF_001", True),
            ("DIFFUSE_node", True),
            ("METAL_node", True),
            ("ROUGHNESS_node", True),
            ("NORMAL_node", True),
            ("uvmap", True),
            ("Looks", True),
            ("Materials", True),
            ("infinigen___version____1_19_1_", True),
            # Real geometry — should NOT be skipped
            ("ChairFactory_42__spawn_asset_0_", False),
            ("window", False),
            ("RandomMesh", False),
            ("", True),
        ],
    )
    def test_skip_decision(self, name, expected):
        assert is_skippable_prim(name) is expected


# ---------------------------------------------------------------------------
# Default rules sanity
# ---------------------------------------------------------------------------


class TestDefaultLabelRulesIntegrity:
    def test_armchair_appears_before_chair(self):
        """Otherwise ``ArmChairFactory`` collapses to plain 'chair'."""
        labels = [pair[0] for pair in DEFAULT_LABEL_RULES]
        assert labels.index("ArmChair") < labels.index("Chair")

    def test_office_chair_appears_before_chair(self):
        labels = [pair[0] for pair in DEFAULT_LABEL_RULES]
        assert labels.index("OfficeChair") < labels.index("Chair")

    def test_kitchen_cabinet_appears_before_cabinet(self):
        labels = [pair[0] for pair in DEFAULT_LABEL_RULES]
        assert labels.index("KitchenCabinet") < labels.index("Cabinet")

    def test_bed_frame_appears_before_bed(self):
        labels = [pair[0] for pair in DEFAULT_LABEL_RULES]
        assert labels.index("BedFrame") < labels.index("Bed")

    def test_known_indoor_categories_present(self):
        """Smoke test that the canonical indoor categories all appear."""
        labels = {pair[1] for pair in DEFAULT_LABEL_RULES}
        for required in (
            "door", "window", "chair", "table", "sofa", "bed",
            "lamp", "sink", "cabinet", "shelf", "rug", "plant",
        ):
            assert required in labels, f"missing canonical label: {required}"


class TestPlainNameLabelsSanity:
    def test_window_door_present(self):
        assert "window" in PLAIN_NAME_LABELS
        assert "door" in PLAIN_NAME_LABELS

    def test_lower_case_only(self):
        for name in PLAIN_NAME_LABELS:
            assert name == name.lower()
