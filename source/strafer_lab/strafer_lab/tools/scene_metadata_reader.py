"""Read per-scene metadata embedded in a scene USD's root-prim customData.

Each scene's labeled ``objects[]`` / ``rooms[]`` payload travels *inside*
the scene USD, on the root prim's ``customData`` under the key
``strafer_scene_metadata`` (authored by ``extract_scene_metadata.py``).
This module is the single ``pxr`` touch-point for reading it back —
every other consumer operates on the plain dict :func:`load` returns.

There is no sidecar fallback: :func:`load` raises when the customData is
absent, so an un-regenerated scene surfaces immediately instead of
silently resolving stale metadata from a sibling file.

The payload is stored as a canonical JSON string rather than a USD
``VtDictionary`` because ``objects[]`` is a heterogeneous list of dicts,
which a ``VtDictionary`` cannot represent losslessly. usdview shows the
string in the property panel (benign).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CUSTOM_DATA_KEY = "strafer_scene_metadata"
VERSION_FIELD = "strafer_scene_metadata_version"
SCHEMA_VERSION = 1


class SceneMetadataError(Exception):
    """Raised when a scene's embedded metadata is missing or malformed."""


def root_prim(stage: Any) -> Any:
    """Return the prim that carries the scene's customData.

    The stage's default prim when one is set, else ``/World``, else the
    pseudo-root. The reader and the authoring pass both call this so they
    agree on where the payload lives.
    """
    default = stage.GetDefaultPrim()
    if default and default.IsValid():
        return default
    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        return world
    return stage.GetPseudoRoot()


def read_custom_data(stage: Any) -> dict[str, Any] | None:
    """Return the embedded metadata dict from an open stage, or ``None``."""
    raw = root_prim(stage).GetCustomDataByKey(CUSTOM_DATA_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise SceneMetadataError(
            f"customData['{CUSTOM_DATA_KEY}'] is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise SceneMetadataError(
            f"customData['{CUSTOM_DATA_KEY}'] is not a JSON object"
        )
    return data


def write_custom_data(stage: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Embed ``metadata`` on the stage's root prim and return the payload.

    Stamps ``strafer_scene_metadata_version`` if absent. Does not save the
    stage — the caller owns the ``stage.Save()``.
    """
    payload = dict(metadata)
    payload.setdefault(VERSION_FIELD, SCHEMA_VERSION)
    root_prim(stage).SetCustomDataByKey(
        CUSTOM_DATA_KEY, json.dumps(payload, sort_keys=True)
    )
    return payload


def load(scene_usd_path: Path | str) -> dict[str, Any]:
    """Open ``scene_usd_path`` and return its embedded scene metadata dict.

    Raises :class:`SceneMetadataError` when the USD is missing, will not
    open, or carries no embedded metadata (the clean-break behavior — no
    sidecar fallback).
    """
    try:
        from pxr import Usd  # type: ignore
    except ImportError as exc:  # pragma: no cover - pxr always present in env_isaaclab3
        raise SceneMetadataError(
            "pxr is required to read scene metadata from USD customData. "
            "Run under an Isaac Sim Python environment."
        ) from exc

    path = Path(scene_usd_path)
    if not path.exists():
        raise SceneMetadataError(f"scene USD not found: {path}")

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneMetadataError(f"failed to open USD stage: {path}")

    data = read_custom_data(stage)
    if data is None:
        raise SceneMetadataError(
            f"{path} carries no embedded scene metadata "
            f"(customData['{CUSTOM_DATA_KEY}'] is absent). Author it with "
            "extract_scene_metadata before capturing against this scene."
        )

    data.setdefault("rooms", [])
    data.setdefault("objects", [])
    data.setdefault("room_adjacency", [])
    return data


def metadata_hash(metadata: dict[str, Any]) -> str:
    """sha256-hexdigest of the canonical embedded metadata dict."""
    canonical = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
