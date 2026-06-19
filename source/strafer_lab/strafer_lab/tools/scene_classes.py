"""Shared structural-class vocabulary for scene authoring.

A scene's *structural* prims carry one of a small fixed set of classes
(floor / ceiling / wall / exterior / staircase). Two producers key off
that set and must agree on it:

- ``generate_scenes_metadata`` excludes structural prims from its
  spawn-point obstacle pass via a ``<room>_<i>_<j>_<class>`` prim-name
  regex.
- ``extract_scene_metadata`` excludes those same classes from the
  ``UsdSemantics`` detection labels it applies (structure dominates
  every frame and would evict furniture from the truncated detections
  column — see the Detections section of ``harness-architecture``).

Defining the set in one place keeps the regex and the label denylist
from drifting.
"""

from __future__ import annotations

import re

STRUCTURAL_CLASSES: frozenset[str] = frozenset(
    {"floor", "ceiling", "wall", "exterior", "staircase"}
)


def room_struct_regex() -> "re.Pattern[str]":
    """Compile the ``<room>_<i>_<j>_<structural-class>`` prim-name matcher.

    Matches Infinigen room-structure prim names whose trailing token is a
    member of :data:`STRUCTURAL_CLASSES`. Anchored at both ends so it
    excludes furniture whose name merely contains a structural word as a
    substring (``FloorLampFactory_*``).
    """
    alternation = "|".join(sorted(STRUCTURAL_CLASSES))
    return re.compile(rf"^[a-z]+(?:_[a-z]+)*(?:_\d+)+_({alternation})$")
