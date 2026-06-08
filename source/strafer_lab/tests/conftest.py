"""Pytest config for tests that don't require Isaac Sim.

The sibling ``test/`` tree launches Isaac Sim once at session start; tests
in *this* tree exercise pure-Python helpers (e.g. the export plumbing
verifies a torch -> .pt -> torch.jit.load round-trip without a sim env)
and are intentionally isolated from that conftest.
"""

import sys
from pathlib import Path

# Make the strafer_lab scripts/ dir importable so tests can hit the export
# helpers directly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
