import isaacsim
import os
print("ISAAC_PATH:", os.environ.get("ISAAC_PATH", "NOT SET"))

# Search for infinigen_sdg_utils
import importlib
try:
    mod = importlib.import_module("infinigen_sdg_utils")
    print("infinigen_sdg_utils found at:", mod.__file__)
except ImportError:
    print("infinigen_sdg_utils: NOT FOUND as importable module")

# Check for standalone_examples
isaac_path = os.environ.get("ISAAC_PATH", "")
if isaac_path:
    se_path = os.path.join(isaac_path, "standalone_examples", "replicator", "infinigen")
    if os.path.exists(se_path):
        print("standalone_examples/replicator/infinigen/ EXISTS:", os.listdir(se_path))
    else:
        print("standalone_examples/replicator/infinigen/ NOT FOUND at:", se_path)
        # check if standalone_examples exists at all
        se_root = os.path.join(isaac_path, "standalone_examples")
        if os.path.exists(se_root):
            print("standalone_examples/ exists, contents:", os.listdir(se_root)[:10])
        else:
            print("standalone_examples/ does NOT exist under ISAAC_PATH")

# Check omni.replicator availability
try:
    import omni.replicator.core as rep
    print("omni.replicator.core: AVAILABLE")
except ImportError as e:
    print(f"omni.replicator.core: NOT AVAILABLE ({e})")
