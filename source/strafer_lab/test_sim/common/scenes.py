# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-in scene corpus for tests that construct Infinigen-source env cfgs.

``strafer_env_cfg.SCENE_USD_DIR`` is derived from the installed package, so it
resolves to ``<checkout>/Assets/generated/scenes`` — a transient, gitignored
directory that ``prep_room_usds.py`` produces. Every cfg whose
``scene_source.kind == "infinigen"`` reads it while constructing, so a test that
builds one passes or fails on whether the machine happens to have generated
scenes: green in a checkout that has them, ``FileNotFoundError`` in every
worktree and every fresh clone.

The tests using this assert on behaviour *given* a scene set, never on what the
corpus contains, so a byte-sized stand-in is the whole requirement. This authors
one in a tmp dir behind the real layout (``<scenes>/<stem>/export/<stem>.usdc``
with a top-level ``<scenes>/<stem>.usdc`` symlink) and repoints
``SCENE_USD_DIR`` at it — the shape ``tests/harness/test_lightest_scene.py``
already uses.

No occupancy sidecar and no ``floor_top_z`` are authored: both lookups no-op
when absent (``derive_infinigen_scene_spawn`` returns ``[]``,
``_get_infinigen_active_scene_floor_top_z`` returns ``None``), leaving the
spawn/floor event params at their cfg defaults. A test that needs a populated
spawn pool authors a real occupancy grid instead — see
``tests/harness/test_bridge_spawn_from_occupancy.py``.

Usage — import the fixture into a test module and autouse binds it there:

    from test_sim.common.scenes import hermetic_scene_corpus  # noqa: F401
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import pytest


def scene_source_kind(variant_cls) -> str | None:
    """Read a composed variant's scene source without constructing it.

    Construction is what reaches the corpus, so any check of the form "does this
    variant read scenes?" has to answer from the class to stay usable as a guard.
    """
    for field in dataclasses.fields(variant_cls):
        if field.name == "scene_source":
            return field.default_factory().kind
    return None


def write_stub_scene_corpus(
    root: Path, stems: Sequence[str] = ("scene_stub_000",)
) -> list[Path]:
    """Author stand-in scenes for ``stems`` under ``root``; return their symlinks.

    Mirrors the layout ``_get_scene_usd_paths`` discovers, top-level symlink
    included — the lightest-scene pick resolves that link to size its target, so
    a flat file would leave the resolution step untested.

    Composable: the manifest merges with whatever is already registered and an
    existing symlink is left in place, so a second call extends the corpus
    rather than replacing it.
    """
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "scenes_metadata.json"
    scenes = {}
    if meta_path.is_file():
        scenes = json.loads(meta_path.read_text()).get("scenes", {})

    links = []
    for stem in stems:
        inner = root / stem / "export" / f"{stem}.usdc"
        inner.parent.mkdir(parents=True, exist_ok=True)
        inner.write_text("(stub)")
        link = root / f"{stem}.usdc"
        if not link.is_symlink():
            link.symlink_to(inner)
        links.append(link)
        scenes.setdefault(stem, {})

    meta_path.write_text(json.dumps({"scenes": scenes}))
    return links


def _assert_corpus_is_the_bound_one(root: Path) -> None:
    """Fail loudly if an Infinigen-source cfg binds outside ``root``.

    The patch going inert is invisible on a machine that carries a real corpus:
    every test keeps passing, silently reading the machine's scenes instead of
    these. Asserting the bind fails everywhere at once instead — including where
    a corpus is present, which unaided reports nothing.
    """
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cfg = composed.StraferNavCfg_TeleopCapture()
    bound = Path(cfg.scene.scene_geometry.spawn.usd_path).resolve()
    assert root.resolve() in bound.parents, (
        "the stand-in scene corpus is not the one being read: "
        f"StraferNavCfg_TeleopCapture bound {bound}, outside {root}. "
        "SCENE_USD_DIR is not the seam _get_scene_usd_paths() reads, so these "
        "tests are reading whatever corpus this machine happens to hold."
    )


@contextlib.contextmanager
def stub_scene_corpus(tmp_path_factory):
    """Point ``SCENE_USD_DIR`` at a freshly authored stand-in corpus.

    Takes ``tmp_path_factory`` rather than ``tmp_path`` so a module-scoped
    fixture can hold it: nothing here mutates the corpus, so one per module is
    enough. Patches the module attribute rather than ``_get_scene_usd_paths``
    itself, which keeps the real discovery, metadata filtering and symlink
    resolution in the path under test.
    """
    from strafer_lab.tasks.navigation import strafer_env_cfg as cfg_mod

    root = tmp_path_factory.mktemp("scene_corpus") / "scenes"
    write_stub_scene_corpus(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfg_mod, "SCENE_USD_DIR", root)
        _assert_corpus_is_the_bound_one(root)
        yield root


@pytest.fixture(scope="module", autouse=True)
def hermetic_scene_corpus(tmp_path_factory):
    """Bind a stand-in scene corpus for every test in the importing module.

    Autouse and module-wide on purpose: the leak is structural — *any*
    construction of an Infinigen-source variant reads ``SCENE_USD_DIR``, and the
    variant sweeps in these modules construct every ``StraferNavCfg_*`` there
    is. Opting in per test would reopen the hole the moment a variant is added.

    A module binds it by importing the name. It cannot live in a
    ``conftest.py`` — the contract gate runs under ``--noconftest``.

    Inert for the plane / procroom sources, which never reach ``SCENE_USD_DIR``:
    the frozen contract goldens recompute byte-identically under it.
    """
    with stub_scene_corpus(tmp_path_factory) as root:
        yield root
