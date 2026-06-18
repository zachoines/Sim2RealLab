"""Regression tests for prep_room_usds.py's env-var validator.

The validator exists because the wrapper used to fail-late: a missing
``STRAFER_ISAACLAB_PYTHON`` env var would crash 24+ hours into a run,
*after* the multi-hour coarse + export stages completed. The fail-fast
check at the top of ``generate_scenes`` prevents that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Import the wrapper module without going through `import strafer_lab.scripts...`
# (that package isn't on the path; the script lives under source/strafer_lab/scripts/).
# Register in sys.modules BEFORE exec so module-level @dataclass can find its
# own class via sys.modules[cls.__module__].
_PREP_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prep_room_usds.py"
)
_spec = importlib.util.spec_from_file_location("prep_room_usds", _PREP_PATH)
prep_room_usds = importlib.util.module_from_spec(_spec)
sys.modules["prep_room_usds"] = prep_room_usds
_spec.loader.exec_module(prep_room_usds)


@pytest.fixture
def tmp_python(tmp_path: Path) -> Path:
    """A non-empty path that looks like a real Python interpreter."""
    fake = tmp_path / "fake_python"
    fake.write_text("#!/usr/bin/env python\n")
    fake.chmod(0o755)
    return fake


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """A directory that looks like an Infinigen checkout."""
    root = tmp_path / "infinigen"
    root.mkdir()
    return root


def _set_all_env(monkeypatch, tmp_python, tmp_root) -> None:
    monkeypatch.setenv("INFINIGEN_ROOT", str(tmp_root))
    monkeypatch.setenv("STRAFER_INFINIGEN_PYTHON", str(tmp_python))
    monkeypatch.setenv("STRAFER_ISAACLAB_PYTHON", str(tmp_python))
    # ISAACLAB is a command string (isaaclab.sh -p); the validator checks the
    # first token exists, so a real path stands in for the launcher here.
    monkeypatch.setenv("ISAACLAB", str(tmp_python))


class TestValidateRequiredEnv:
    def test_passes_when_all_set(self, monkeypatch, tmp_python, tmp_root):
        _set_all_env(monkeypatch, tmp_python, tmp_root)
        # Should not raise.
        prep_room_usds.validate_required_env_for_generate()

    def test_launcher_not_required_when_metadata_disabled(
        self, monkeypatch, tmp_python, tmp_root,
    ):
        """--no-scene-metadata skips authoring, so ISAACLAB isn't needed."""
        monkeypatch.setenv("INFINIGEN_ROOT", str(tmp_root))
        monkeypatch.setenv("STRAFER_INFINIGEN_PYTHON", str(tmp_python))
        monkeypatch.setenv("STRAFER_ISAACLAB_PYTHON", str(tmp_python))
        monkeypatch.delenv("ISAACLAB", raising=False)
        prep_room_usds.validate_required_env_for_generate(author_scene_metadata=False)

    @pytest.mark.parametrize("missing_var", [
        "INFINIGEN_ROOT",
        "STRAFER_INFINIGEN_PYTHON",
        "STRAFER_ISAACLAB_PYTHON",
        "ISAACLAB",
    ])
    def test_raises_with_helpful_message_when_one_missing(
        self, monkeypatch, tmp_python, tmp_root, missing_var,
    ):
        """The validator must mention the missing var by name + tell the
        operator to source env_setup.sh — the historical bite was a
        late opaque crash."""
        _set_all_env(monkeypatch, tmp_python, tmp_root)
        monkeypatch.delenv(missing_var, raising=False)
        with pytest.raises(RuntimeError) as exc:
            prep_room_usds.validate_required_env_for_generate()
        msg = str(exc.value)
        assert missing_var in msg, f"missing var name not in error: {msg!r}"
        assert "env_setup.sh" in msg, f"setup hint not in error: {msg!r}"

    def test_isaaclab_python_must_resolve(self, monkeypatch, tmp_python, tmp_root):
        """STRAFER_ISAACLAB_PYTHON pointing at a non-existent path also fails fast.

        Catches the typo-in-.env class of error, not just the missing-env-var class.
        """
        _set_all_env(monkeypatch, tmp_python, tmp_root)
        monkeypatch.setenv("STRAFER_ISAACLAB_PYTHON", "/nonexistent/path/to/python")
        with pytest.raises(RuntimeError, match="non-existent path"):
            prep_room_usds.validate_required_env_for_generate()
