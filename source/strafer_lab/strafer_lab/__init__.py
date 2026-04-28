"""
Strafer Lab - Isaac Lab extension for Gobilda Strafer mecanum wheel robot.

This extension provides:
- Robot asset configuration (ArticulationCfg)
- RL training environments for navigation tasks
- ROS2 bridge integration for sim-to-real deployment
- Batch data-processing tools under :mod:`strafer_lab.tools` that have no
  Isaac Sim runtime dependency

Isaac Sim-dependent subpackages (``strafer_lab.assets`` and
``strafer_lab.tasks``) require Isaac Sim's Kit runtime to be active — they
pull in ``omni.*`` Kit extensions that only exist once
:class:`isaaclab.app.AppLauncher` has started the Omniverse process. Scripts
that call ``from strafer_lab.tasks import ...`` MUST launch Isaac Sim first.

Conversely, the :mod:`strafer_lab.tools` subpackage is pure Python and is
consumed by the DGX batch-processing scripts (scene metadata extraction,
description pipeline, dataset export, fine-tune data prep) which never
launch Isaac Sim. To keep those scripts importable from a plain Python env
that has no Kit runtime, we defer the Kit-dependent imports and swallow
the ``ModuleNotFoundError`` for ``omni.*`` / ``pxr.*`` / Isaac Sim packages
at package-import time. When the Kit runtime IS active, the imports
succeed exactly as before.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

try:
    # Register Gym environments
    from .tasks import *  # noqa: F401, F403

    # Make assets available at package level
    from .assets import STRAFER_CFG  # noqa: F401
except ModuleNotFoundError as _exc:
    # Only swallow the import error if it comes from Isaac Sim's Kit
    # runtime not being active. Any other missing dependency is a real
    # bug that should surface loudly.
    _missing = str(_exc.name or "")
    if _missing.startswith(("omni", "pxr", "isaacsim", "carb", "warp")):
        _logger.debug(
            "Skipping strafer_lab.tasks / strafer_lab.assets imports: "
            "Isaac Sim Kit runtime not active (missing module %r). "
            "Launch isaaclab.sh -p or call AppLauncher to use these subpackages.",
            _missing,
        )
    else:
        raise

__version__ = "0.1.0"
