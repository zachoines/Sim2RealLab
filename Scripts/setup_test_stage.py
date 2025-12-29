"""
Load the processed robot USD and spawn it on a flat ground plane with collider,
exporting a ready-to-sim test stage.

Usage:
  python Scripts/setup_test_stage.py --stage "<path to robot USD>" --out "<output USD>"
"""

import argparse

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


def ensure_ground(stage: Usd.Stage, size: float = 1000.0, height: float = 0.0) -> Usd.Prim:
    """Create a simple ground plane with collision."""
    plane = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/GroundPlane"))
    half = size * 0.5
    # Square plane quad
    points = [
        (-half, height, -half),
        (half, height, -half),
        (half, height, half),
        (-half, height, half),
    ]
    # two triangles
    face_vertex_counts = [3, 3]
    face_vertex_indices = [0, 1, 2, 0, 2, 3]
    plane.CreatePointsAttr(points)
    plane.CreateFaceVertexCountsAttr(face_vertex_counts)
    plane.CreateFaceVertexIndicesAttr(face_vertex_indices)
    plane.CreateExtentAttr([(-half, height, -half), (half, height, half)])
    # Collider
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    plane.GetPrim().CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    plane.GetPrim().CreateAttribute("collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    return plane.GetPrim()


def main():
    parser = argparse.ArgumentParser(description="Create a test stage with ground and the mecanum robot.")
    parser.add_argument("--stage", required=True, help="Input robot USD (already final processed).")
    parser.add_argument("--out", default=None, help="Output USD stage. If omitted, saves in place.")
    parser.add_argument("--root", default="/World/mechanum_robot_v18", help="Path to robot root in the stage.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    # Add physics scene if missing
    if not UsdPhysics.Scene.Get(stage, Sdf.Path("/World/physicsScene")):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
        # Up/down is along +Y/-Y in this asset, so apply gravity along -Y.
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)

    ensure_ground(stage, size=100.0, height=0.0)

    robot = stage.GetPrimAtPath(args.root)
    if not robot:
        raise SystemExit(f"Robot root not found at {args.root}")

    if args.out:
        stage.GetRootLayer().Export(args.out)
        print(f"Exported test stage to {args.out}")
    else:
        stage.GetRootLayer().Save()
        print(f"Saved test stage in place at {args.stage}")


if __name__ == "__main__":
    main()
