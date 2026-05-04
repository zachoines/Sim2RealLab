"""Pytest config for tests that don't require Isaac Sim.

The sibling ``test/`` tree launches Isaac Sim once at session start; tests
in *this* tree exercise pure-Python helpers (e.g. the export plumbing
verifies a torch -> .pt -> torch.jit.load round-trip without a sim env)
and are intentionally isolated from that conftest.
"""

import sys
from pathlib import Path

# Make Scripts/ importable so tests can hit the export helpers directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "Scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
