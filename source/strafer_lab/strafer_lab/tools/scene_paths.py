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
    """Derive ``scene_metadata.json`` from a USD path — deterministic, no walk.

    Two layouts are recognized, both checked at exactly one candidate
    path each (no recursive ancestor crawl):

    1. **Symlink at the scenes root**: ``<scenes>/<X>.usdc`` →
       ``<scenes>/<X>/scene_metadata.json``. This is the canonical
       operator-facing form authored by ``prep_room_usds.py``.
    2. **Deep export inside the scene dir**: ``<scenes>/<X>/<...>/<X>.usdc``
       → ``<scenes>/<X>/scene_metadata.json``. We require an ancestor
       directory whose name matches the USD stem (i.e. the scene name)
       so we never resolve via an unrelated ``scene_metadata.json`` that
       happens to live further up the tree.

    Returns ``None`` when neither candidate exists; the caller should
    error loudly and tell the operator to pass ``--scene-metadata``
    explicitly. Bounded depth: at most four ancestors are inspected.
    """
    path = Path(usd_path).resolve()
    stem = path.stem

    # Layout 1: sibling dir named after the USD stem, at the same level
    # as the USD itself (covers the scenes-root symlink case).
    candidate = path.with_suffix("") / _METADATA_NAME
    if candidate.is_file():
        return candidate

    # Layout 2: an ancestor directory whose name == USD stem holds the
    # metadata. Cap at depth 4 — Infinigen's deepest export pattern is
    # ``<scene>/coarse/exports/<scene>.usdc``, which puts the stem dir
    # exactly 3 parents above the USD.
    for ancestor in list(path.parents)[:4]:
        if ancestor.name == stem:
            candidate = ancestor / _METADATA_NAME
            return candidate if candidate.is_file() else None
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
