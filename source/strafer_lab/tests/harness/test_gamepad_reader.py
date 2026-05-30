"""Pure-Python tests for the shared gamepad reader's static tables.

The :class:`GamepadReader` itself requires pygame + a physical joystick,
so it cannot be instantiated in CI. The module's static lookup tables
and helper functions can be — and they are what consumers (the RL demo
collector + the harness teleop driver) depend on, so we exercise them
here.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.gamepad_reader import (
    AXIS_MAPS,
    BUTTON_MAPS,
    GamepadFrame,
    apply_deadzone,
    describe_family_buttons,
    detect_family,
    legacy_tuple_from_frame,
)


class TestDetectFamily:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Xbox One Controller", "xbox"),
            ("Xbox Series X Controller", "xbox"),
            ("Microsoft X-Box 360 pad", "xbox"),
            ("PS5 Wireless Controller", "ps5"),
            ("DualSense Wireless Controller", "ps5"),
            ("Sony Computer Entertainment Wireless Controller", "ps5"),
            ("Wireless Controller", "ps5"),
            ("Pro Controller", "switch"),
            ("Nintendo Switch Pro Controller", "switch"),
            ("Some unknown gamepad", "xbox"),
            ("", "xbox"),
        ],
    )
    def test_canonical_names(self, name, expected):
        assert detect_family(name) == expected

    def test_case_insensitive(self):
        assert detect_family("XBOX ONE CONTROLLER") == "xbox"
        assert detect_family("DualSense") == "ps5"


class TestDeadzone:
    @pytest.mark.parametrize(
        "value,deadzone,expected",
        [
            (0.0, 0.1, 0.0),
            (0.05, 0.1, 0.0),
            (-0.05, 0.1, 0.0),
            (0.099, 0.1, 0.0),
            # Outside deadzone: linearly rescaled to [0, 1].
            (1.0, 0.1, 1.0),
            (-1.0, 0.1, -1.0),
            (0.55, 0.1, pytest.approx(0.5, abs=1e-6)),
            (-0.55, 0.1, pytest.approx(-0.5, abs=1e-6)),
            # Zero deadzone passes value through verbatim.
            (0.42, 0.0, pytest.approx(0.42, abs=1e-6)),
        ],
    )
    def test_apply_deadzone(self, value, deadzone, expected):
        assert apply_deadzone(value, deadzone) == expected


class TestButtonMapsAreComplete:
    @pytest.mark.parametrize("family", ["xbox", "ps5", "switch"])
    def test_all_logical_buttons_present(self, family):
        # The harness teleop driver depends on the full set; missing one
        # would silently disable that chord.
        expected_keys = {"a", "b", "x", "y", "select", "back", "start"}
        assert expected_keys.issubset(BUTTON_MAPS[family].keys())

    @pytest.mark.parametrize("family", ["xbox", "ps5", "switch"])
    def test_axes_present(self, family):
        assert {"lx", "ly", "rx", "ry"}.issubset(AXIS_MAPS[family].keys())

    @pytest.mark.parametrize("family", ["xbox", "ps5", "switch"])
    def test_indices_are_distinct(self, family):
        # Two logical buttons can share one physical index iff the family
        # has no dedicated equivalent (e.g. PS5's "back" aliases the PS
        # home button; Switch's "back" aliases Minus). Distinct
        # *non-alias* indices is what matters: y/x/select must not collide.
        non_alias_keys = ["a", "b", "x", "y", "start"]
        indices = [BUTTON_MAPS[family][k] for k in non_alias_keys]
        assert len(set(indices)) == len(indices), (
            f"family {family!r} has collisions among non-alias buttons: "
            f"{dict(zip(non_alias_keys, indices))}"
        )


class TestDescribeFamilyButtons:
    def test_returns_axes_and_buttons(self):
        info = describe_family_buttons("xbox")
        assert "axes" in info and "buttons" in info
        assert info["buttons"]["y"] == BUTTON_MAPS["xbox"]["y"]


class TestLegacyTupleFromFrame:
    def test_extracts_three_legacy_buttons(self):
        frame = GamepadFrame(
            lx=0.1, ly=-0.2, rx=0.3, ry=0.0,
            buttons={"a": True, "b": False, "x": True, "y": False,
                     "select": False, "back": False, "start": True},
            dpad_x=0, dpad_y=0,
        )
        lx, ly, rx, buttons = legacy_tuple_from_frame(frame)
        assert (lx, ly, rx) == (0.1, -0.2, 0.3)
        assert buttons == {"a": True, "b": False, "start": True}

    def test_legacy_tuple_ignores_extra_buttons(self):
        # collect_demos.py's loop reads only a/b/start; the adapter must
        # not surface x/y/etc. (those would be silently ignored, but the
        # contract is to return a 3-key dict).
        frame = GamepadFrame(
            lx=0.0, ly=0.0, rx=0.0, ry=0.0,
            buttons={"a": False, "b": False, "y": True, "start": False},
            dpad_x=0, dpad_y=0,
        )
        _, _, _, buttons = legacy_tuple_from_frame(frame)
        assert set(buttons.keys()) == {"a", "b", "start"}
