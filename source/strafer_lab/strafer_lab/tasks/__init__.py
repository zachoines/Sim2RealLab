"""Task environments for Strafer robot."""

from isaaclab_tasks.utils import import_packages

# Blacklist packages that should not be auto-imported
_BLACKLIST_PKGS = ["utils", "mdp"]

# Auto-import all task packages (navigation, etc.)
import_packages(__name__, _BLACKLIST_PKGS)
