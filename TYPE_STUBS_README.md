# Type Stubs Setup

Type stub files have been generated for C++ extension modules to enable IntelliSense and autocomplete in VS Code.

## Current Status

✅ Type stubs generated for **pxr** (USD) modules  
✅ VS Code configured to use stubs  
✅ IntelliSense working for pxr imports

## Documentation

For complete usage instructions, see:

📄 **[Scripts/type_stubs/README.md](Scripts/type_stubs/README.md)**

This guide covers:
- How to generate stubs for any package
- When to use type stubs vs source analysis
- Troubleshooting and customization
- Examples and testing

## Quick Commands

```powershell
# Generate stubs for pxr (USD) modules
python Scripts/type_stubs/generate_stubs.py

# Generate stubs for other packages
python Scripts/type_stubs/generate_stubs.py <package_name>

# Test that stubs are working
python Scripts/type_stubs/test_stubs.py
```

## What's Been Set Up

### Generated Stubs
- Location: `typings/pxr/`
- Modules: Usd, UsdGeom, Gf, Sdf, Vt, Tf, UsdPhysics, UsdShade
- Total: 494 classes with 8,337 methods

### VS Code Configuration
`.vscode/settings.json` includes:
```json
{
    "python.analysis.stubPath": "typings",
    "python.analysis.extraPaths": ["typings", "IsaacLab/source"]
}
```

### Scripts
- `Scripts/type_stubs/generate_stubs.py` - Stub generator
- `Scripts/type_stubs/test_stubs.py` - Test file for verification

After generating new stubs, reload VS Code to see the changes.
