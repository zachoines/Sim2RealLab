"""
Generate detailed type stubs for Python modules using runtime introspection.
Works best for C++ extension modules where source code isn't available.

Usage:
    python generate_stubs.py           # Generate stubs for pxr modules
    python generate_stubs.py isaaclab  # Generate stubs for isaaclab modules
    python generate_stubs.py <package> # Generate stubs for any package
"""

import inspect
import sys
from pathlib import Path
from typing import Any


def get_signature_string(obj: Any, name: str) -> str:
    """Try to get a signature string for a callable."""
    try:
        sig = inspect.signature(obj)
        return f"def {name}{sig}: ..."
    except (ValueError, TypeError):
        # Can't get signature from C++ extension
        return f"def {name}(*args, **kwargs): ..."


def generate_class_stub(cls: type, indent: str = "") -> list[str]:
    """Generate stub lines for a class."""
    lines = []

    # Class definition
    bases = cls.__bases__
    if bases and bases != (object,):
        base_names = ", ".join(
            b.__name__
            for b in bases
            if b.__name__
            not in ("pybind11_object", "Boost.Python.instance", "instance", "object")
        )
        if base_names:
            lines.append(f"{indent}class {cls.__name__}({base_names}):")
        else:
            lines.append(f"{indent}class {cls.__name__}:")
    else:
        lines.append(f"{indent}class {cls.__name__}:")

    # Get all members
    members = []
    for name in dir(cls):
        if name.startswith("_") and name not in ("__init__", "__new__"):
            continue
        try:
            attr = getattr(cls, name)
            members.append((name, attr))
        except:
            continue

    if not members:
        lines.append(f"{indent}    pass")
        return lines

    # Sort members: properties first, then methods
    properties = []
    methods = []

    for name, attr in members:
        try:
            if callable(attr):
                methods.append((name, attr))
            else:
                properties.append((name, attr))
        except:
            continue

    # Add properties
    for name, attr in properties:
        lines.append(f"{indent}    {name}: Any")

    # Add methods
    added_methods = False
    for name, attr in methods:
        try:
            lines.append(f"{indent}    {get_signature_string(attr, name)}")
            added_methods = True
        except:
            continue

    if not added_methods and not properties:
        lines.append(f"{indent}    pass")

    return lines


def generate_module_stub(module: Any, output_path: Path, module_name: str):
    """Generate a stub file for a module."""
    lines = []
    lines.append("from typing import Any, overload")
    lines.append("")
    lines.append(f"# Auto-generated stubs for {module_name}")
    lines.append("# Generated using runtime introspection")
    lines.append("")

    # Collect classes and functions
    classes = []
    functions = []
    constants = []

    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(module, name)
            if inspect.isclass(attr):
                classes.append((name, attr))
            elif inspect.isbuiltin(attr) or callable(attr):
                functions.append((name, attr))
            elif not inspect.ismodule(attr):
                constants.append((name, attr))
        except:
            continue

    # Write constants
    if constants:
        lines.append("# Constants and Enums")
        for name, _ in constants:
            lines.append(f"{name}: Any")
        lines.append("")

    # Write functions
    if functions:
        lines.append("# Module Functions")
        for name, func in functions:
            lines.append(get_signature_string(func, name))
        lines.append("")

    # Write classes
    if classes:
        lines.append("# Classes")
        for name, cls in sorted(classes):
            try:
                class_lines = generate_class_stub(cls)
                lines.extend(class_lines)
                lines.append("")
            except Exception as e:
                print(f"  Warning: Could not process class {name}: {e}")
                continue

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    class_count = len(classes)
    method_count = sum(
        len([l for l in generate_class_stub(cls) if "def " in l]) for _, cls in classes
    )
    print(
        f"  Generated {output_path.name}: {class_count} classes, {method_count} methods"
    )


def get_workspace_root() -> Path:
    """Get the workspace root directory (2 levels up from this script)."""
    return Path(__file__).parent.parent.parent


def main():
    workspace_root = get_workspace_root()

    # Determine which package to generate stubs for
    if len(sys.argv) > 1:
        package = sys.argv[1]
    else:
        package = "pxr"

    # Package configurations
    package_configs = {
        "pxr": [
            "Usd",
            "UsdGeom",
            "Gf",
            "Sdf",
            "Vt",
            "Tf",
            "UsdPhysics",
            "UsdShade",
        ],
        "isaaclab": [
            "app",
            "sim",
            "assets",
            "scene",
            "utils",
            "envs",
            "managers",
            "sensors",
            "terrains",
        ],
        "omni": ["isaac", "kit", "usd", "ui"],
    }

    # Get modules to process
    if package in package_configs:
        modules_to_process = package_configs[package]
    else:
        # Try to import the package and get its submodules
        try:
            pkg = __import__(package, fromlist=[""])
            modules_to_process = [
                name
                for name in dir(pkg)
                if not name.startswith("_") and inspect.ismodule(getattr(pkg, name))
            ][
                :10
            ]  # Limit to first 10 submodules
        except ImportError:
            print(f"[ERROR] Package '{package}' not found or not importable")
            print("\nSupported packages: pxr, isaaclab, omni")
            print("Or specify any installed package name")
            return

    typings_base = workspace_root / "typings" / package

    print(f"Generating type stubs for '{package}' package...")
    print(f"Output directory: {typings_base}\n")

    success_count = 0
    for module_name in modules_to_process:
        print(f"Processing {package}.{module_name}...")
        try:
            # Import module
            module = __import__(f"{package}.{module_name}", fromlist=[""])

            # Generate stub for main module
            output_path = typings_base / module_name / f"{module_name}.pyi"
            generate_module_stub(module, output_path, f"{package}.{module_name}")

            # Also create __init__.pyi
            init_path = typings_base / module_name / "__init__.pyi"
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                with open(init_path, "w") as f:
                    f.write(f"from .{module_name} import *\n")

            success_count += 1

        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

    print(
        f"\n[SUCCESS] Generated stubs for {success_count}/{len(modules_to_process)} modules"
    )
    print(f"Stubs saved to: {typings_base}")
    print("\nReload VS Code for changes to take effect.")


if __name__ == "__main__":
    main()
