"""Shared scene-layout path resolution for harness tools.

A single Infinigen scene is published as both a USDC export (used by
Isaac Sim) and a ``scene_metadata.json`` sidecar (used by the picker,
the bridge, and any downstream consumer that needs the labeled
``objects[]`` array). The two files are siblings inside the scene's
directory::

    Assets/generated/scenes/
      <scene_name>/                       (scene directory)
        scene_metadata.json               (sidecar)
        coarse/exports/<scene_name>.usdc  (the USD)
      <scene_name>.usdc                   (symlink → coarse/exports/...)

Tools that take both ``--scene-usd`` and ``--scene-metadata`` overrides
need a single helper to resolve which metadata file to read; this module
centralizes that logic so the rules don't drift between
``teleop_capture.py``, ``capture.py``, and future entry points.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_SEARCH_ROOT = Path("Assets/generated/scenes")
_METADATA_NAME = "scene_metadata.json"


def find_metadata_for_usd(usd_path: Path | str) -> Path | None:
    """Walk up from a USD file to find a sibling ``scene_metadata.json``.

    Handles both the symlink-at-scenes-root layout and the
    inside-scene-dir layout. Returns ``None`` when nothing is found
    within five parent levels (a safety cap so we never traverse to ``/``).
    """
    path = Path(usd_path).resolve()
    # If the USD is a symlink at the scenes root, the sibling directory
    # named after the symlink stem is the scene dir.
    if path.is_symlink() or path.parent.name == _DEFAULT_SEARCH_ROOT.name:
        sibling = path.with_suffix("") / _METADATA_NAME
        if sibling.is_file():
            return sibling
    # Otherwise walk up looking for the metadata file in any ancestor.
    current = path.parent
    for _ in range(5):
        candidate = current / _METADATA_NAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_scene_metadata_path(
    *,
    scene: str | None = None,
    metadata_override: Path | str | None = None,
    usd_override: Path | str | None = None,
    search_root: Path | str = _DEFAULT_SEARCH_ROOT,
) -> Path:
    """Resolve the ``scene_metadata.json`` path from CLI inputs.

    Precedence:

    1. ``metadata_override`` — used as-is when supplied.
    2. ``usd_override`` — sibling ``scene_metadata.json`` derived from
       the USD path. Use this when the operator points at a custom USD
       export without re-supplying the metadata path explicitly.
    3. ``<search_root>/<scene>/scene_metadata.json`` — the default
       layout for any scene authored by ``prep_room_usds.py``.

    Raises:
        FileNotFoundError: if the resolved path does not exist on disk.
        ValueError: if none of ``metadata_override``, ``usd_override``,
            or ``scene`` is supplied.
    """
    if metadata_override is not None:
        path = Path(metadata_override)
        if not path.is_file():
            raise FileNotFoundError(f"scene_metadata.json not found: {path}")
        return path

    if usd_override is not None:
        derived = find_metadata_for_usd(usd_override)
        if derived is None:
            raise FileNotFoundError(
                f"Could not locate a sibling scene_metadata.json next to "
                f"--scene-usd={usd_override}. Pass --scene-metadata explicitly.",
            )
        return derived

    if scene is not None:
        path = Path(search_root) / scene / _METADATA_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"scene_metadata.json not found: {path}. Pass "
                "--scene-metadata explicitly if the scene lives outside "
                f"{search_root}/.",
            )
        return path

    raise ValueError(
        "resolve_scene_metadata_path requires at least one of "
        "scene / metadata_override / usd_override",
    )
