"""Guard: nothing outside the compat module may import ``isaacsim.core.utils``.

Isaac Sim 6.0.0 GA deprecated ``isaacsim.core.utils`` and the 6.0.x line moved it from
``isaacsim/exts/`` to ``isaacsim/extsDeprecated/``. Isaac Sim's own Kit apps put that
directory on the extension search path; Isaac Lab's apps at 3.0.0-beta2 do not — so
under ``isaaclab.sh -p`` the module is physically present but unimportable, and a direct
import raises ``ModuleNotFoundError`` at *runtime*, deep inside a booted Kit. An entry
point that imports it does not fail to start; it starts and then dies partway in.

That failure mode is invisible to every other test in this repo: the import sits inside
a function body, behind a Kit boot, on a path the pure suite cannot execute. A static
check is the only thing that catches a reintroduction cheaply, so this test walks the
tree's ASTs instead of running anything.

:mod:`strafer_lab.isaacsim_compat` is the single allowed importer — it *is* the fallback.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

DEPRECATED_ROOT = "isaacsim.core.utils"

# The compat module owns the fallback; every other file routes through it.
ALLOWED = {"strafer_lab/isaacsim_compat.py"}

_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _python_files() -> list[pathlib.Path]:
    """Every tracked-source Python file under ``source/strafer_lab``."""
    return sorted(
        p
        for p in _SOURCE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and "retired" not in p.parts
    )


def _deprecated_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for real import statements of the deprecated root.

    Parsed rather than grepped so that prose — the compat module's own docstring, the
    comments at the call sites explaining why they moved — cannot fail the test, and so
    that a re-introduction cannot hide behind unusual formatting.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a broken file is a different test's problem
        pytest.fail(f"{path} does not parse: {exc}")

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == DEPRECATED_ROOT or module.startswith(f"{DEPRECATED_ROOT}."):
                hits.append((node.lineno, f"from {module} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == DEPRECATED_ROOT or alias.name.startswith(f"{DEPRECATED_ROOT}."):
                    hits.append((node.lineno, f"import {alias.name}"))
    return hits


def test_only_the_compat_module_imports_the_deprecated_surface():
    offenders: list[str] = []
    for path in _python_files():
        rel = path.relative_to(_SOURCE_ROOT).as_posix()
        if rel in ALLOWED:
            continue
        offenders += [f"{rel}:{lineno}: {what}" for lineno, what in _deprecated_imports(path)]

    assert not offenders, (
        "these files import the deprecated isaacsim.core.utils surface directly, which "
        "raises ModuleNotFoundError under Isaac Lab's kit apps at Isaac Sim 6.0.x.\n"
        "Route them through strafer_lab.isaacsim_compat instead:\n  "
        + "\n  ".join(offenders)
    )


def test_the_compat_module_still_carries_the_fallback():
    """The allowlist is only justified while the compat module actually falls back.

    If the fallback is ever dropped, this test fails and the allowlist entry should go
    with it — otherwise the exemption outlives its reason and quietly re-opens the hole.
    """
    compat = _SOURCE_ROOT / "strafer_lab" / "isaacsim_compat.py"
    assert compat.is_file(), "strafer_lab/isaacsim_compat.py is missing"
    assert _deprecated_imports(compat), (
        "strafer_lab.isaacsim_compat no longer imports the deprecated surface, so its "
        "entry in this test's ALLOWED set is stale — remove it."
    )


def test_compat_module_imports_without_a_kit_runtime():
    """It must be importable from the pure suite, i.e. no Kit imports at module scope."""
    import strafer_lab.isaacsim_compat as compat

    for name in (
        "add_labels",
        "anchor_capture_camera",
        "enable_extension",
        "set_camera_view",
    ):
        assert callable(getattr(compat, name)), f"{name} is not exposed as a callable"
