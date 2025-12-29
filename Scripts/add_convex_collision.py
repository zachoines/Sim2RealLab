"""
Add collision shapes for prims whose paths contain given fragments (roller covers, rails, etc.).
Default approximations:
  - roller_cover                    -> convexHull
  - Side_Rail, Center_Rail          -> boundingCube (box)

For each matching Mesh prim:
  - Apply PhysxCollisionAPI
  - Create collisionAPI:approximation = "convexHull"
  - Keep existing visual mesh untouched

Usage:
  python Scripts/add_convex_collision.py --stage <in.usd> --out <out.usd>
"""

import argparse
from typing import List, Optional, Tuple

from pxr import Usd, UsdGeom, UsdPhysics, Sdf


DEFAULT_RULES: List[Tuple[str, str]] = [
    ("Side_Rail", "boundingCube"),
    ("Center_Rail", "boundingCube"),
    ("roller_cover", "convexHull"),
    ("core", "convexHull"),
    # side plates, joint_frames intentionally non-colliding
]


def add_collision(stage: Usd.Stage, prim: Usd.Prim, approx: str) -> bool:
    """Apply collision approximation to the given mesh prim."""
    if not prim or not prim.IsA(UsdGeom.Mesh):
        return False
    # Apply physics collision
    UsdPhysics.CollisionAPI.Apply(prim)
    # Apply MeshCollisionAPI if available to silence UI warnings.
    if hasattr(UsdPhysics, "MeshCollisionAPI"):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # Try both physics and physx names to match UI/loader expectations.
    prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set(approx)
    prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approx)
    prim.CreateAttribute("collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    return True


def ensure_editable(prim: Usd.Prim) -> Usd.Prim:
    """If prim is an instance or instance proxy, de-instance its root and return the editable prim."""
    stage = prim.GetStage()
    if prim.IsInstance() or prim.IsInstanceProxy():
        inst_root = prim
        while inst_root and not inst_root.IsInstance():
            inst_root = inst_root.GetParent()
        if inst_root and inst_root.IsInstance():
            inst_root.SetInstanceable(False)
            prim = stage.GetPrimAtPath(prim.GetPath())
    return prim


def main():
    parser = argparse.ArgumentParser(description="Add convex hull collision to selected meshes.")
    parser.add_argument("--stage", required=True, help="Input USD stage.")
    parser.add_argument("--out", required=True, help="Output USD stage.")
    parser.add_argument(
        "--fragments",
        nargs="+",
        default=[rule[0] for rule in DEFAULT_RULES],
        help="One or more substrings; any mesh path containing one will get a collider.",
    )
    parser.add_argument(
        "--approx",
        nargs="+",
        help=(
            "Optional list of approximations aligned with --fragments (e.g., convexHull box). "
            "If omitted, defaults to the internal mapping (Side_Rail:boundingCube, Center_Rail:boundingCube, roller_cover:convexHull, core:convexHull)."
        ),
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    fragments: List[str] = args.fragments
    approximations: List[str]
    if args.approx:
        if len(args.approx) != len(fragments):
            raise SystemExit("Length of --approx must match length of --fragments.")
        approximations = args.approx
    else:
        # Map fragments to defaults if present, else fall back to convexHull.
        default_map = {k: v for k, v in DEFAULT_RULES}
        approximations = [default_map.get(f, "convexHull") for f in fragments]

    # Build ordered rules so first-match wins.
    rules: List[Tuple[str, str]] = list(zip(fragments, approximations))

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    targets = []
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path_str = str(prim.GetPath())
        for frag, approx in rules:
            if frag in path_str:
                targets.append((path_str, approx))
                break

    count = 0
    for path, approx in targets:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        prim_edit = ensure_editable(prim)
        if prim_edit.IsA(UsdGeom.Mesh) and add_collision(stage, prim_edit, approx):
            count += 1

    stage.GetRootLayer().Export(args.out)
    joined = ", ".join([f"{frag}:{approx}" for frag, approx in rules])
    print(f"Added collision to {count} mesh prim(s) using rules [{joined}]; exported to {args.out}")


if __name__ == "__main__":
    main()
