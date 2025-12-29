"""
Dump a tree view of the robot hierarchy and basic physics flags.
Also list axle joint_frames meshes if needed.

Usage examples:
  python Scripts/inspect_roller_meshes.py --stage C:/Worspace/Assets/mechanum_robot_v1_collision.usd
  python Scripts/inspect_roller_meshes.py --stage C:/Worspace/Assets/mechanum_robot_v1_collision.usd --root /World/mechanum_robot_v18 --max-depth 6
"""

import argparse
from pxr import Usd, UsdGeom, UsdPhysics


def flag_str(prim):
    """Return compact flags for physics state."""
    flags = []
    if UsdPhysics.ArticulationRootAPI(prim):
        flags.append("AR")
    if UsdPhysics.RigidBodyAPI(prim):
        flags.append("RB")
    if prim.GetTypeName() == "PhysicsRevoluteJoint":
        flags.append("JNT")
    if prim.IsInstance():
        flags.append("INST")
    return ",".join(flags) if flags else "-"


def dump_tree(stage, root_path, max_depth):
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise SystemExit(f"Root prim not found: {root_path}")
    base_depth = str(root_path).count("/")
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange(root, predicate):
        path_str = str(prim.GetPath())
        depth = path_str.count("/") - base_depth
        if depth < 0 or depth > max_depth:
            continue
        indent = "  " * depth
        type_name = prim.GetTypeName()
        flags = flag_str(prim)
        print(f"{indent}{path_str.split('/')[-1]} [{type_name}] ({flags})")


def dump_axles(stage, fragment, limit):
    axles = [p for p in stage.Traverse() if fragment in str(p.GetPath()) and p.IsA(UsdGeom.Xform)]
    print(f"\nFound {len(axles)} axle xforms matching '{fragment}'")
    for axle in axles[:limit]:
        print(f"\nAxle: {axle.GetPath()}")
        found = 0
        stack = list(axle.GetChildren())
        while stack and found < 5:
            p = stack.pop()
            if p.IsA(UsdGeom.Mesh):
                print(f"  mesh: {p.GetPath()}")
                found += 1
            stack.extend(p.GetChildren())
        if axle.IsInstance():
            proto = axle.GetPrototype()
            if proto:
                proto_found = 0
                stack = [proto]
                while stack and proto_found < 5:
                    p = stack.pop()
                    if p.IsA(UsdGeom.Mesh):
                        print(f"  proto mesh: {p.GetPath()}")
                        proto_found += 1
                    stack.extend(p.GetChildren())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="Path to USD stage.")
    parser.add_argument("--root", default="/World", help="Root prim to start tree dump.")
    parser.add_argument("--max-depth", type=int, default=5, help="Max depth to print from root.")
    parser.add_argument("--fragment", default="node_606_XXXX_0096_roller_axle", help="Substring to match axle xforms.")
    parser.add_argument("--limit", type=int, default=5, help="Number of axles to sample for meshes.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    print(f"Tree from {args.root} (max depth {args.max_depth}):")
    dump_tree(stage, args.root, args.max_depth)
    dump_axles(stage, args.fragment, args.limit)


if __name__ == "__main__":
    main()
