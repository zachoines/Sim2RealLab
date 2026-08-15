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

Usage:
    from test_sim.common.scenes import stub_scene_corpus

    @pytest.fixture(scope="module", autouse=True)
    def _hermetic_scene_corpus(tmp_path_factory):
        with stub_scene_corpus(tmp_path_factory) as root:
            yield root
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest


def write_stub_scene_corpus(root: Path, stem: str = "scene_stub_000") -> Path:
    """Author a one-scene stand-in corpus under ``root``; return its symlink.

    Mirrors the layout ``_get_scene_usd_paths`` discovers, top-level symlink
    included — the lightest-scene pick resolves that link to size its target, so
    a flat file would leave the resolution step untested.
    """
    inner = root / stem / "export" / f"{stem}.usdc"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text("(stub)")
    link = root / f"{stem}.usdc"
    link.symlink_to(inner)
    (root / "scenes_metadata.json").write_text(json.dumps({"scenes": {stem: {}}}))
    return link


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
        yield root
