"""Scene USD path derivation for harness tools.

A scene is keyed by one name. Its geometry — and, since the metadata
moved into the USD, its labeled ``objects[]`` / ``rooms[]`` payload — is
the top-level ``<scene>.usdc`` symlink at the scenes root::

    Assets/generated/scenes/
      <scene>/                            (scene directory: export tree, textures)
      <scene>.usdc                        (symlink → the real USDC; carries customData)

Tools resolve which USD to read from a ``--scene`` name or a
``--scene-usd`` override; the metadata reader
(:mod:`strafer_lab.tools.scene_metadata_reader`) then reads the embedded
``customData`` straight from that USD. Nothing crawls the filesystem for
a sibling JSON.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_SEARCH_ROOT = Path("Assets/generated/scenes")


def scene_usd_path(scene: str, *, search_root: Path | str = _DEFAULT_SEARCH_ROOT) -> Path:
    """Return the top-level ``<scene>.usdc`` symlink path for a scene name."""
    return Path(search_root) / f"{scene}.usdc"


def resolve_scene_usd_path(
    *,
    scene: str | None = None,
    usd_override: Path | str | None = None,
    search_root: Path | str = _DEFAULT_SEARCH_ROOT,
) -> Path:
    """Resolve the scene USD to read metadata from.

    Precedence:

    1. ``usd_override`` — used as-is (the operator pointed at a custom
       export; that USD must carry the embedded metadata).
    2. ``<search_root>/<scene>.usdc`` — the default layout for any scene
       authored by ``prep_room_usds.py``.

    Raises:
        FileNotFoundError: if the resolved path does not exist on disk.
        ValueError: if neither ``usd_override`` nor ``scene`` is supplied.
    """
    if usd_override is not None:
        path = Path(usd_override)
        if not path.exists():
            raise FileNotFoundError(f"scene USD not found: {path}")
        return path

    if scene is not None:
        path = scene_usd_path(scene, search_root=search_root)
        if not path.exists():
            raise FileNotFoundError(
                f"scene USD not found: {path}. Pass --scene-usd explicitly if "
                f"the scene lives outside {search_root}/."
            )
        return path

    raise ValueError(
        "resolve_scene_usd_path requires at least one of scene / usd_override",
    )
