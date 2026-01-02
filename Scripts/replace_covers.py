"""
Replace all roller_cover prims with a new asset, aligned to each joint_frame.

Usage:
  python Scripts/replace_covers.py --stage "C:/Worspace/Assets/mechanum_robot_v1_jointframes.usd" --cover "C:/Worspace/Assets/props/RollerCover.usdz" --out "C:/Worspace/Assets/mechanum_robot_v1_jointframes.usd"

Run this after fix_roller_axes.py (so joint_frames exist) and before fix_articulations.py.
"""

import argparse
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom


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


def find_child_with_name(parent: Usd.Prim, name: str) -> Optional[Usd.Prim]:
    for child in parent.GetChildren():
        if child.GetName() == name:
            return child
    return None


def main():
    parser = argparse.ArgumentParser(description="Replace roller_cover prims with a new asset aligned to joint_frames.")
    parser.add_argument("--stage", required=True, help="Input USD stage path.")
    parser.add_argument("--cover", required=True, help="Path to the replacement cover asset (USD/USDZ).")
    parser.add_argument("--out", required=True, help="Output USD path (can overwrite input).")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    jf_paths = [p.GetPath() for p in Usd.PrimRange.Stage(stage, predicate) if p.GetName() == "joint_frame"]
    replaced = 0

    for jf_path in jf_paths:
        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
        joint_frame = stage.GetPrimAtPath(jf_path)
        if not joint_frame:
            continue
        joint_frame = ensure_editable(joint_frame)
        assembly = joint_frame.GetParent()
        if not assembly:
            continue
        assembly = ensure_editable(assembly)

        # Remove existing roller_cover under this assembly (direct child only).
        existing_cover = find_child_with_name(assembly, "roller_cover")
        if existing_cover:
            stage.RemovePrim(existing_cover.GetPath())

        # Define new cover prim and reference the asset.
        cover_path = assembly.GetPath().AppendChild("roller_cover")
        cover_prim = UsdGeom.Xform.Define(stage, cover_path).GetPrim()
        cover_prim.GetReferences().ClearReferences()
        cover_prim.GetReferences().AddReference(args.cover)

        # Place the cover so its world xform matches the joint_frame world xform.
        # Use a relative transform computed by the cache to match joint_frame's local pose.
        rel = xcache.ComputeRelativeTransform(joint_frame, assembly)
        local_xform = rel[0] if isinstance(rel, tuple) else rel
        xf_cover = UsdGeom.Xformable(cover_prim)
        xf_cover.ClearXformOpOrder()
        xf_cover.AddTransformOp().Set(local_xform)

        # If the asset authors a 10x scale on its mesh holder (e.g., MeshInstance/Body1), override to 0.1 to neutralize.
        body1 = stage.GetPrimAtPath(cover_path.AppendPath("MeshInstance/Body1"))
        if body1:
            body1 = ensure_editable(body1)
            xf_body1 = UsdGeom.Xformable(body1)
            xf_body1.ClearXformOpOrder()
            xf_body1.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.1))

        # Clear authored normals and set collision approximation to convexHull on the main mesh, if present.
        mesh_prim = None
        for child in Usd.PrimRange(cover_prim):
            if child.IsA(UsdGeom.Mesh):
                mesh_prim = child
                break
        if mesh_prim:
            mesh = UsdGeom.Mesh(mesh_prim)
            if mesh.GetNormalsAttr().HasAuthoredValueOpinion():
                mesh.GetNormalsAttr().Clear()
            mesh_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
            mesh_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")

        replaced += 1

    stage.GetRootLayer().Export(args.out)
    print(f"Replaced {replaced} roller_cover prim(s) with asset {args.cover}; exported to {args.out}")


if __name__ == "__main__":
    main()
