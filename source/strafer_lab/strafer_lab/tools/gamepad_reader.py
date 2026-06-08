"""Shared pygame gamepad reader for strafer teleop scripts.

Wraps ``pygame.joystick`` with the Xbox / PS5 / Switch-Pro auto-detection
and deadzone handling used by both the RL demo collector
(``collect_demos.py``) and the harness teleop driver
(``teleop_capture.py``). Centralizing the reader keeps the button table
in one place so the harness's extended buttons (``Y / X / SELECT / Back``
+ D-pad) stay in sync with the family auto-detection the demo collector
already relies on.

The pygame import is deferred to :meth:`GamepadReader.__init__` so the
module is importable from environments without pygame (the
unit tests use this to exercise the pure-Python
``button_state_to_episode_outcome`` translator and the static button-map
table without instantiating a real reader).

Stick axes are uniform across families (LX=0, LY=1, RX=2). Button
indices differ; the ``BUTTON_MAPS`` table below records what each family
actually reports, derived from pygame's SDL2 mapping plus the prior
``collect_demos.py`` audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Family-keyed input maps
# ---------------------------------------------------------------------------

# Stick axes — identical across all three families per pygame SDL2 mapping.
AXIS_MAPS: dict[str, dict[str, int]] = {
    "xbox":   {"lx": 0, "ly": 1, "rx": 2, "ry": 3},
    "ps5":    {"lx": 0, "ly": 1, "rx": 2, "ry": 3},
    "switch": {"lx": 0, "ly": 1, "rx": 2, "ry": 3},
}

# Button indices. The ``Y/X/SELECT/Back`` additions matter for the harness
# teleop episode-end mapping; ``A/B/Start`` carry over from the RL demo
# collector. Indices come from SDL2's default Game Controller mapping and
# match what pygame reports on the three controller families we ship for.
#
# Per-family notes:
# * Xbox (One / Series / 360 with xboxdrv): A=south, B=east, X=west, Y=north.
# * PS5 DualSense: Cross=south, Circle=east, Square=west, Triangle=north;
#   pygame surfaces them at the same numeric indices as their Xbox
#   counterparts (south/east/west/north). The ``select`` button is the
#   PS5 share button at index 4.
# * Switch Pro: physical A/B and X/Y are swapped from the Xbox layout
#   (Nintendo's south button is "B", east is "A"). We bind logical
#   ``a``=south, ``b``=east here so left-stick world-frame teleop and
#   "keep / discard" map to the same physical positions across families;
#   the table column names reflect logical south/east/west/north, not
#   silkscreen letters.
BUTTON_MAPS: dict[str, dict[str, int]] = {
    "xbox": {
        "a": 0,       # south
        "b": 1,       # east
        "x": 2,       # west
        "y": 3,       # north
        "select": 6,  # Back / View
        "back": 6,    # alias — Xbox calls this "Back / View"; harness brief
                       # uses "Back" for the discard button. Bound to the
                       # same physical chord as select for Xbox; PS5/Switch
                       # override below.
        "start": 7,   # Start / Menu
    },
    "ps5": {
        "a": 0,       # Cross (south)
        "b": 1,       # Circle (east)
        "x": 2,       # Square (west)
        "y": 3,       # Triangle (north)
        "select": 4,  # Share
        "back": 8,    # PS button (no dedicated Back; reuse the home button)
        "start": 9,   # Options
    },
    "switch": {
        "a": 0,       # south (silkscreen "B" on Switch Pro, logical south)
        "b": 1,       # east  (silkscreen "A" on Switch Pro, logical east)
        "x": 2,       # west  (silkscreen "Y")
        "y": 3,       # north (silkscreen "X")
        "select": 4,  # Minus
        "back": 4,    # Minus — Switch has no dedicated Back; alias to Minus.
        "start": 6,   # Plus
    },
}


def detect_family(joystick_name: str) -> str:
    """Map a pygame joystick name to one of ``{"xbox", "ps5", "switch"}``.

    Names are matched case-insensitively. The fallback is ``"xbox"`` —
    historically the most common controller on the project's machines
    and the SDL2 default-mapping reference.
    """
    name = (joystick_name or "").lower()
    if any(tag in name for tag in ("dualsense", "ps5", "sony", "wireless controller")):
        return "ps5"
    if any(tag in name for tag in ("pro controller", "switch", "nintendo")):
        return "switch"
    return "xbox"


def apply_deadzone(value: float, deadzone: float) -> float:
    """Snap stick values inside the deadzone to zero; rescale outside.

    Pure function — exposed at module level for unit tests that exercise
    the rescaling math without instantiating pygame.
    """
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GamepadFrame:
    """One tick of gamepad state.

    Stick values are deadzone-applied and clipped to ``[-1, 1]``. Buttons
    are bools keyed by logical name (``a``, ``b``, ``x``, ``y``,
    ``select``, ``back``, ``start``). The D-pad is exposed as a
    ``(dpad_x, dpad_y)`` pair in ``{-1, 0, 1}`` — pygame surfaces the
    cross as a hat axis, not as buttons.
    """

    lx: float
    ly: float
    rx: float
    ry: float
    buttons: dict[str, bool]
    dpad_x: int
    dpad_y: int


class GamepadReader:
    """Thin wrapper around pygame joystick with strafer's button table.

    Construction probes pygame for an attached controller, sniffs the
    family from the joystick name, and binds the button / axis indices
    from the family-keyed maps above. Subsequent ``read()`` calls return
    a :class:`GamepadFrame` snapshot.
    """

    def __init__(self, deadzone: float = 0.12, family_override: str | None = None) -> None:
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for gamepad input. Install with: pip install pygame",
            ) from exc

        self._pygame = pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect a controller and retry.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = float(deadzone)

        name = self.joystick.get_name()
        self.family = family_override or detect_family(name)
        if self.family not in BUTTON_MAPS:
            raise ValueError(
                f"Unknown gamepad family {self.family!r}; "
                f"choices: {sorted(BUTTON_MAPS)}",
            )

        amap = AXIS_MAPS[self.family]
        bmap = BUTTON_MAPS[self.family]
        self._axis_lx = amap["lx"]
        self._axis_ly = amap["ly"]
        self._axis_rx = amap["rx"]
        self._axis_ry = amap["ry"]
        self._button_indices = dict(bmap)

        # Block pygame's QUIT events so a stray window-close on the
        # editor viewport does not signal the process to exit.
        pygame.event.set_blocked(pygame.QUIT)

        print(
            f"[Gamepad] Connected: {name} "
            f"(family={self.family}, axes={self.joystick.get_numaxes()}, "
            f"buttons={self.joystick.get_numbuttons()}, "
            f"hats={self.joystick.get_numhats()})",
        )

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def read(self) -> GamepadFrame:
        """Sample the current gamepad state."""
        self._pygame.event.pump()
        # Re-block QUIT each tick in case pygame re-enabled it.
        self._pygame.event.set_blocked(self._pygame.QUIT)

        lx = apply_deadzone(self.joystick.get_axis(self._axis_lx), self.deadzone)
        ly = apply_deadzone(self.joystick.get_axis(self._axis_ly), self.deadzone)
        rx = apply_deadzone(self.joystick.get_axis(self._axis_rx), self.deadzone)
        ry = 0.0
        if self.joystick.get_numaxes() > self._axis_ry:
            ry = apply_deadzone(self.joystick.get_axis(self._axis_ry), self.deadzone)

        buttons = {
            name: bool(self.joystick.get_button(idx))
            for name, idx in self._button_indices.items()
            if idx < self.joystick.get_numbuttons()
        }

        dpad_x = 0
        dpad_y = 0
        if self.joystick.get_numhats() > 0:
            hx, hy = self.joystick.get_hat(0)
            dpad_x = int(hx)
            dpad_y = int(hy)

        return GamepadFrame(
            lx=lx, ly=ly, rx=rx, ry=ry,
            buttons=buttons,
            dpad_x=dpad_x, dpad_y=dpad_y,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the pygame joystick + subsystem."""
        try:
            self._pygame.joystick.quit()
            self._pygame.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Legacy adapter
# ---------------------------------------------------------------------------


def legacy_tuple_from_frame(frame: GamepadFrame) -> tuple[float, float, float, dict[str, bool]]:
    """Compatibility shim returning the ``collect_demos.py`` tuple shape.

    The RL demo collector reads ``(lx, ly, rx, buttons{a,b,start})`` and
    doesn't care about Y / X / D-pad. Wrapping the new frame this way
    lets the legacy script keep its loop body intact while the reader
    itself moves into the shared module.
    """
    legacy_buttons = {
        "a": frame.buttons.get("a", False),
        "b": frame.buttons.get("b", False),
        "start": frame.buttons.get("start", False),
    }
    return frame.lx, frame.ly, frame.rx, legacy_buttons


# ---------------------------------------------------------------------------
# Module-export check (executable as a smoke test)
# ---------------------------------------------------------------------------


def describe_family_buttons(family: str) -> dict[str, Any]:
    """Return the axis + button table for a family — debug helper.

    Used by tests to assert the table doesn't regress.
    """
    return {
        "axes": AXIS_MAPS[family],
        "buttons": BUTTON_MAPS[family],
    }
