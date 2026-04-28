"""Parse Infinigen factory prim names into ``(label, instance_id)`` pairs.

Background
----------
Infinigen's USD export does not write ``semanticLabel`` / ``instanceId``
prim attributes. The only semantic signal that survives to a generated
``.usdc`` is the prim name itself, which Infinigen stamps with the
factory class that produced the asset. The convention is::

    /World/<FactoryClass>_<numeric_id>__spawn_asset_<N>_

For example::

    /World/GlassPanelDoorFactory_430087__spawn_asset_5_
    /World/ChairFactory_991132__spawn_asset_0_
    /World/KitchenCabinetFactory_204411__spawn_asset_2_

This module turns those names into a canonical ``(label, instance_id)``
pair the rest of the pipeline can use:

  - ``label`` is the broad object category (``"door"``, ``"chair"``,
    ``"cabinet"``) so navigation prompts ``"go to the door"`` work.
    The mapping is intentionally lossy: ``GlassPanelDoor``,
    ``LiteDoor``, ``LouverDoor`` all collapse to ``"door"``.
  - ``instance_id`` is the numeric ID Infinigen embedded in the prim
    name (``430087`` above), preserved verbatim so the same physical
    asset always gets the same ID across runs.

Pure Python — no ``bpy``, no ``pxr``. Importable from ``.venv_vlm``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedLabel:
    """Result of parsing an Infinigen-style prim name."""

    label: str            # Canonical object category, e.g. "door"
    instance_id: int      # Numeric ID Infinigen embedded in the prim name
    factory_class: str    # The original Factory class (e.g. "ChairFactory"), or "" for non-factory prims
    spawn_index: int      # The N in ``__spawn_asset_N_``, or 0 if not present
    room_type: str = ""   # For architecture prims (wall/floor/ceiling), the parent room (e.g. "kitchen")


# ---------------------------------------------------------------------------
# Canonical label mapping
# ---------------------------------------------------------------------------
#
# Ordered list of (substring, canonical_label). The first entry whose
# substring appears in the factory class wins. More specific matches
# come BEFORE more general ones — e.g. ``Countertop`` before any rule
# that might claim ``Top``. Substrings are case-insensitive.
#
# This list is intentionally hand-curated to broad indoor categories
# rather than auto-generated from Infinigen's class hierarchy. Broad
# labels are what the planner / executor / VLM grounding clients care
# about — ``"go to the chair"`` is more useful than
# ``"go to the OfficeChair"``. Add new entries here as new Infinigen
# factories enter regular use; drop ``DEFAULT_LABEL_RULES`` only if
# the canonical category truly changes.

DEFAULT_LABEL_RULES: tuple[tuple[str, str], ...] = (
    # Doors and door hardware
    ("Door", "door"),
    # Windows and skylights
    ("Window", "window"),
    ("Skylight", "skylight"),
    # Seating
    ("ArmChair", "armchair"),
    ("OfficeChair", "chair"),
    ("BarChair", "chair"),
    ("Chair", "chair"),
    ("Sofa", "sofa"),
    ("Stool", "stool"),
    # Sleeping
    ("BedFrame", "bed"),
    ("Bed", "bed"),
    # Surfaces and storage
    ("Countertop", "countertop"),
    ("KitchenCabinet", "cabinet"),
    ("Cabinet", "cabinet"),
    ("Drawer", "drawer"),
    ("Shelf", "shelf"),
    ("Bookshelf", "shelf"),
    ("Desk", "desk"),
    ("Table", "table"),
    # Kitchen / bath fixtures
    ("Sink", "sink"),
    ("Toilet", "toilet"),
    ("Bathtub", "bathtub"),
    ("Shower", "shower"),
    ("Faucet", "faucet"),
    ("Stove", "stove"),
    ("Oven", "oven"),
    ("Microwave", "microwave"),
    ("Dishwasher", "dishwasher"),
    ("Fridge", "fridge"),
    ("Refrigerator", "fridge"),
    # Lighting
    ("CeilingClassicLamp", "ceiling_lamp"),
    ("FloorLamp", "lamp"),
    ("DeskLamp", "lamp"),
    ("Lamp", "lamp"),
    # Electronics + decor
    ("Monitor", "monitor"),
    ("TV", "tv"),
    ("Rug", "rug"),
    ("Painting", "painting"),
    ("Mirror", "mirror"),
    ("Clock", "clock"),
    ("Vase", "vase"),
    # Plants
    ("SnakePlant", "plant"),
    ("SpiderPlant", "plant"),
    ("FlowerPlant", "plant"),
    ("Plant", "plant"),
    ("Cactus", "plant"),
    ("Succulent", "plant"),
    ("Fern", "plant"),
    # Architecture (rare in furnished scenes — usually rendered as
    # one big "Room" mesh, but Infinigen sometimes splits these out).
    ("Wall", "wall"),
    ("Floor", "floor"),
    ("Ceiling", "ceiling"),
    ("Stair", "stairs"),
)

# Plain (non-Factory) prim names that Infinigen sometimes emits at top
# level — ``window``, ``door``, etc. Lowercased for case-insensitive
# match. These are taken at face value: a prim literally named
# ``window`` is treated as a window.

PLAIN_NAME_LABELS: frozenset[str] = frozenset(
    {
        "window",
        "door",
        "wall",
        "floor",
        "ceiling",
        "skylight",
    }
)

# Prim names to skip outright. These are control / scaffolding prims
# Infinigen emits that have no semantic meaning for navigation.

SKIP_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Stage scaffolding
    re.compile(r"^World$"),               # the root prim itself
    re.compile(r"^_materials$"),
    # Cameras and lights
    re.compile(r"^camrig"),
    re.compile(r"^camera"),
    re.compile(r"^Light"),                # any prim whose name starts with "Light"
    re.compile(r"^env_light"),
    # Blender's Area / Spot / Point light types are exported by their
    # type name rather than "Light" — skip those too.
    re.compile(r"^Area(_\d+)?$"),
    re.compile(r"^Spot(_\d+)?$"),
    re.compile(r"^Point(_\d+)?$"),
    re.compile(r"^Sun(_\d+)?$"),
    # Geometry of the terrain backdrop (not navigation-relevant)
    re.compile(r"^terrain"),
    # Material / shader graph internals — Infinigen exports each
    # material as a Blender Principled BSDF node tree, which fans out
    # into many leaf prims with conventional names. None are objects
    # we'd ever navigate to.
    re.compile(r"^shader_"),
    re.compile(r"^Principled_BSDF"),
    re.compile(r".*_node$"),              # DIFFUSE_node, METAL_node, NORMAL_node, etc.
    re.compile(r"^uvmap$"),
    # USD instance / proxy scaffolding occasionally emitted by Infinigen
    re.compile(r"^Looks$"),
    re.compile(r"^Materials$"),
    # Infinigen's own version-tag prim, e.g. ``infinigen___version____1_19_1_``
    re.compile(r"^infinigen___version"),
)


# ---------------------------------------------------------------------------
# Regex for the factory pattern
# ---------------------------------------------------------------------------
#
# Matches the typical Infinigen-spawned prim name. Captures:
#   group("class") = factory class (e.g. ``GlassPanelDoorFactory``)
#   group("instance_id") = numeric ID (e.g. ``430087``)
#   group("spawn") = optional spawn index (the ``N`` in ``__spawn_asset_N_``)
#
# Infinigen also occasionally emits prims like ``ChairFactory_991132``
# without the ``__spawn_asset_N_`` tail, so the spawn group is optional.

_FACTORY_PATTERN = re.compile(
    r"""^
    (?P<class>[A-Z][A-Za-z0-9]*Factory)        # CamelCase ending in 'Factory'
    _
    (?P<instance_id>\d+)                        # Numeric instance ID
    (?:__spawn_asset_(?P<spawn>\d+)_?)?         # Optional spawn-index suffix
    (?:[._]\d+)?                                # Optional Blender duplicate suffix (.001, _001)
    $""",
    re.VERBOSE,
)

# Room-architecture prims emitted by Infinigen's room solver. Pattern:
#   <room_type>_<room_idx>_<sub_idx>_<surface>[.NNN | _NNN]
# Examples:
#   living_room_0_0_wall
#   kitchen_0_0_floor
#   bathroom_0_1_ceiling
#   bedroom_0_0_skylight
# The room name is greedy (it contains underscores like "living_room"),
# so we anchor on the trailing surface keyword.

_ARCHITECTURE_PATTERN = re.compile(
    r"""^
    (?P<room>[a-z_]+?)                          # Lazy room name with underscores
    _
    (?P<r_idx>\d+)
    _
    (?P<sub_idx>\d+)
    _
    (?P<surface>wall|floor|ceiling|skylight)
    (?:[._]\d+)?                                # Optional Blender duplicate suffix
    $""",
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_factory_label(prim_name: str) -> ParsedLabel | None:
    """Parse an Infinigen prim name into a :class:`ParsedLabel`.

    Returns ``None`` if the prim is one of:
      - a known control / scaffolding prim (camera, materials, light)
      - a plain unlabeled name we don't recognize
      - a malformed factory name

    Plain names listed in :data:`PLAIN_NAME_LABELS` are returned with
    ``instance_id=-1`` and an empty ``factory_class``.
    """

    name = (prim_name or "").strip()
    if not name:
        return None

    for pat in SKIP_NAME_PATTERNS:
        if pat.match(name):
            return None

    # Plain name (e.g. ``window``) — case-insensitive direct hit.
    lower = name.lower()
    if lower in PLAIN_NAME_LABELS:
        return ParsedLabel(
            label=lower,
            instance_id=-1,
            factory_class="",
            spawn_index=0,
        )

    match = _FACTORY_PATTERN.match(name)
    if match is not None:
        factory_class = match.group("class")
        instance_id = int(match.group("instance_id"))
        spawn_str = match.group("spawn")
        spawn_index = int(spawn_str) if spawn_str else 0

        label = factory_class_to_label(factory_class)
        return ParsedLabel(
            label=label,
            instance_id=instance_id,
            factory_class=factory_class,
            spawn_index=spawn_index,
        )

    arch = _ARCHITECTURE_PATTERN.match(name)
    if arch is not None:
        # Architecture prims encode the parent room in their name. We
        # surface the surface kind ("wall", "floor", ...) as the label
        # so navigation prompts remain consistent ("avoid the wall"),
        # and stash the room name in ``room_type`` so downstream code
        # can synthesize per-room targets ("go to the kitchen") from
        # the floor prim's pose.
        return ParsedLabel(
            label=arch.group("surface"),
            instance_id=-1,
            factory_class="",
            spawn_index=0,
            room_type=arch.group("room"),
        )

    return None


def factory_class_to_label(
    factory_class: str,
    rules: tuple[tuple[str, str], ...] = DEFAULT_LABEL_RULES,
) -> str:
    """Map a factory class name to a canonical label.

    Walks ``rules`` in order, returning the value of the first entry
    whose key (case-insensitive) appears as a substring of
    ``factory_class``. Falls back to a snake_case version of the class
    minus its trailing ``Factory`` if no rule matches — at least a
    readable, consistent string for unknown factories rather than an
    error.
    """

    bare = factory_class.strip()
    if bare.endswith("Factory"):
        bare = bare[: -len("Factory")]
    for substring, label in rules:
        if substring.lower() in factory_class.lower():
            return label

    # Fallback: convert the class CamelCase to snake_case and lowercase.
    if not bare:
        return ""
    snake = re.sub(r"(?<!^)([A-Z])", r"_\1", bare).lower()
    return snake


def is_skippable_prim(prim_name: str) -> bool:
    """Return True if the prim is a known control / scaffolding name.

    Useful for callers that want to filter prims BEFORE attempting a
    parse, e.g. when iterating a USD stage where the Traverse() call
    yields cameras and materials interleaved with the geometry.
    """

    name = (prim_name or "").strip()
    if not name:
        return True
    for pat in SKIP_NAME_PATTERNS:
        if pat.match(name):
            return True
    return False
