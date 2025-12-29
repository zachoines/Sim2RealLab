# Type Stub Generator

Generate detailed type stub files (.pyi) for Python packages using runtime introspection. This is especially useful for C++ extension modules where source code isn't available for static analysis.

## Quick Start

```powershell
# Generate stubs for pxr (USD) modules
python Scripts/type_stubs/generate_stubs.py

# Generate stubs for other packages
python Scripts/type_stubs/generate_stubs.py isaaclab
python Scripts/type_stubs/generate_stubs.py omni
```

## What Are Type Stubs?

Type stub files (.pyi) provide type information to language servers like Pylance, enabling:
- ✅ **Autocomplete** - See available methods and classes as you type
- ✅ **Go to Definition** - Jump to class/method definitions
- ✅ **Signature Help** - View function parameters and return types
- ✅ **Type Checking** - Catch type errors before runtime

## When to Use This Tool

### ✅ Use for C++ Extension Modules
These modules are compiled and don't have readable Python source:
- **pxr** (Pixar USD) - `.pyd` or `.so` files from C++
- **omni.** modules - Isaac Sim's C++ extensions
- **torch._C** - PyTorch's C++ backend
- Any package where imports work but you get no autocomplete

### ❌ Don't Use for Pure Python Packages
Pure Python packages don't need stubs - Pylance reads the source directly:
- **isaaclab** - Already configured in `.vscode/settings.json`
- **numpy** - Type stubs available via `pip install numpy-stubs`
- Most standard libraries and pure Python packages

## How It Works

The script:
1. Imports the target module at runtime
2. Uses Python's `inspect` module to discover all classes and methods
3. Generates `.pyi` stub files with function signatures
4. Saves to `typings/<package>/` directory

Pylance then reads these stub files to provide IntelliSense.

## Usage Examples

### Generate pxr (USD) Stubs

```powershell
python Scripts/type_stubs/generate_stubs.py
```

**Output:**
- Generates stubs for 8 pxr modules
- ~8,300+ methods across 494 classes
- Saved to `typings/pxr/`

**Modules included:**
- `pxr.Usd` - Core USD stage/prim APIs
- `pxr.UsdGeom` - Geometry schemas
- `pxr.Gf` - Math types (Vec3, Matrix, Quaternion)
- `pxr.Sdf` - Scene description foundation
- `pxr.Vt` - Value types
- `pxr.Tf` - Tools foundation
- `pxr.UsdPhysics` - Physics schemas
- `pxr.UsdShade` - Shading/material schemas

### Generate Stubs for Other Packages

```powershell
# Try generating for isaaclab (may need omni modules loaded)
python Scripts/type_stubs/generate_stubs.py isaaclab

# Generate for any installed package
python Scripts/type_stubs/generate_stubs.py <package_name>
```

## Setup Requirements

### 1. Install the Package
The target package must be installed and importable:
```powershell
# For pxr
pip install usd-core

# For other packages, install them first
```

### 2. Configure VS Code
Already configured in `.vscode/settings.json`:
```json
{
    "python.analysis.stubPath": "typings",
    "python.analysis.extraPaths": ["typings"]
}
```

### 3. Generate Stubs
Run the script for your target package.

### 4. Reload VS Code
Press `Ctrl+Shift+P` → "Reload Window"

## Testing Stubs

Use the test file to verify stubs are working:

```python
from pxr import Usd, UsdGeom, Gf

# Test autocomplete
stage = Usd.Stage.CreateInMemory()
stage.Get  # <-- Type this and you should see autocomplete

prim = stage.DefinePrim("/Test")
prim.Get  # <-- Should show GetPath, GetAttribute, etc.

# Test vector math
vec = Gf.Vec3d(1, 2, 3)
vec.  # <-- Should show GetLength, Normalize, etc.
```

## Customizing for New Packages

To add a new package to the script, edit `generate_stubs.py`:

```python
package_configs = {
    "pxr": ['Usd', 'UsdGeom', 'Gf', ...],
    "your_package": ['module1', 'module2', 'module3'],
}
```

Or just run it with any package name - it will auto-discover submodules!

## Output Structure

```
typings/
├── pxr/
│   ├── Usd/
│   │   ├── __init__.pyi
│   │   └── Usd.pyi        # 52 classes, 1,150 methods
│   ├── UsdGeom/
│   │   ├── __init__.pyi
│   │   └── UsdGeom.pyi    # 39 classes, 3,057 methods
│   └── Gf/
│       ├── __init__.pyi
│       └── Gf.pyi         # 48 classes, 701 methods
└── <other_packages>/
```

## Generated Stub Example

```python
# pxr/Usd/Usd.pyi
from typing import Any, overload

class Stage:
    def CreateInMemory(*args, **kwargs): ...
    def Open(*args, **kwargs): ...
    def GetPrimAtPath(*args, **kwargs): ...
    def DefinePrim(*args, **kwargs): ...
    def Save(*args, **kwargs): ...
    # ... 87 total methods

class Prim(Object):
    def GetPath(*args, **kwargs): ...
    def GetAttribute(*args, **kwargs): ...
    def CreateAttribute(*args, **kwargs): ...
    def GetChildren(*args, **kwargs): ...
    # ... 166 total methods
```

## Limitations

### Signature Information
C++ extensions can't expose full type information, so:
- ❌ Parameter names: `(*args, **kwargs)` instead of `(path: str, name: str)`
- ❌ Return types: Not specified
- ❌ Type hints: Limited to `Any`

But you still get:
- ✅ All available method names
- ✅ All classes and constants
- ✅ Inheritance relationships
- ✅ Basic function structure

### Why This Happens
C++ extension modules (built with pybind11, Boost.Python, etc.) don't expose full type metadata to Python's `inspect` module. Only runtime introspection is possible.

### Better Alternatives
If available, use official type stubs:
- `pip install types-*` packages from Microsoft/typeshed
- Package-specific stub packages (like `torch-stubs`)

## Regenerating Stubs

Regenerate after package updates:
```powershell
python Scripts/type_stubs/generate_stubs.py pxr
```

The script overwrites existing stubs, so it's safe to rerun.

## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
**Solution:** Install the package first: `pip install <package>`

### "ImportError: DLL load failed" or similar
**Solution:** Some modules need special environments. For Isaac Sim modules, you may need to run within Isaac Sim's Python environment.

### No autocomplete after generating stubs
**Solution:**
1. Check that stubs were created in `typings/<package>/`
2. Verify `.vscode/settings.json` has correct paths
3. Reload VS Code window (`Ctrl+Shift+P` → "Reload Window")
4. Check Pylance output for errors

### Stubs generated but still no details
**Expected:** C++ extensions can't provide full signatures, so you'll see `(*args, **kwargs)` instead of detailed parameters. You still get method names and autocomplete!

## Current Generated Stubs

✅ **pxr** - 494 classes, 8,337 methods  
Location: `typings/pxr/`

To generate more, run the script with your target package!

## Support

For issues with:
- **Script itself** - Check this README and script comments
- **Specific packages** - Check package documentation
- **VS Code/Pylance** - Check Pylance output panel for errors
