"""
Repack roller assemblies by lifting cover/axle under the assembly root and clearing stale joint_frames.

For each roller_assembly:
  - Find one cover (matching --cover-fragment) and one axle (matching --axle-fragment).
  - Pick the first descendant xform with meshes inside each (often Body1/Body1) and move it
    directly under the assembly, renaming to "roller_cover" / "roller_axle".
  - Remove any joint_frame children under the assembly so they can be regenerated.
  - Remove the old cover/axle containers if they become empty.

This preserves authored transforms of the moved xform (no baking). Run fix_roller_axes.py afterwards.
"""

import argparse
from typing import Optional

from pxr import Usd, UsdGeom, Sdf, Gf


def holder_of_first_mesh(root: Usd.Prim) -> Optional[Usd.Prim]:
    """Return the parent xform of the first mesh found under root."""
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for p in Usd.PrimRange(root, predicate):
        if p.IsA(UsdGeom.Mesh):
            parent = p.GetParent()
            if parent and parent.IsA(UsdGeom.Xform):
                return parent
    return None


def unique_child_path(parent: Usd.Prim, base_name: str) -> Sdf.Path:
    stage = parent.GetStage()
    base = parent.GetPath().AppendChild(base_name)
    if not stage.GetPrimAtPath(base):
        return base
    i = 1
    while True:
        cand = parent.GetPath().AppendChild(f"{base_name}_{i}")
        if not stage.GetPrimAtPath(cand):
            return cand
        i += 1


def remove_empty_chain(stage: Usd.Stage, prim: Usd.Prim):
    """Remove prim if it has no children and no properties; walk up while empty."""
    while prim and not list(prim.GetChildren()) and not list(prim.GetAuthoredProperties()):
        parent = prim.GetParent()
        stage.RemovePrim(prim.GetPath())
        prim = parent


def is_identity_transform(xf: UsdGeom.Xformable, tol: float = 1e-6) -> bool:
    """Return True if the authored local transform is effectively identity."""
    res = xf.GetLocalTransformation(Usd.TimeCode.Default())
    mat = res[0] if isinstance(res, tuple) else res
    ident = Gf.Matrix4d(1.0)
    for r in range(4):
        for c in range(4):
            if abs(mat[r][c] - ident[r][c]) > tol:
                return False
    return True


def flatten_redundant_chain(stage: Usd.Stage, root: Usd.Prim):
    """
    If root has a single child xform that is effectively empty (only xformOpOrder, no ops),
    hoist its children up and remove it. Repeat until branching or non-empty transforms are found.
    """
    layer = stage.GetRootLayer()
    def is_redundant_xf(xf: Usd.Prim) -> bool:
        if not xf.IsA(UsdGeom.Xform):
            return False
        xfable = UsdGeom.Xformable(xf)
        if xfable.GetOrderedXformOps():
            return False
        props = [p.GetName() for p in xf.GetAuthoredProperties() if p.GetName() != "xformOpOrder"]
        return len(props) == 0

    while True:
        kids = [c for c in root.GetChildren() if c.IsA(UsdGeom.Xform)]
        if len(kids) != 1:
            break
        child = kids[0]
        if not is_redundant_xf(child):
            break
        # Move all grandchildren up under root with unique names if needed.
        for gc in list(child.GetChildren()):
            new_path = unique_child_path(root, gc.GetName())
            Sdf.CopySpec(layer, gc.GetPath(), layer, new_path)
            stage.RemovePrim(gc.GetPath())
        stage.RemovePrim(child.GetPath())


def main():
    parser = argparse.ArgumentParser(description="Repack roller assemblies by lifting cover/axle under assembly.")
    parser.add_argument("--stage", required=True, help="Input USD stage.")
    parser.add_argument("--out", required=True, help="Output USD stage.")
    parser.add_argument("--axle-fragment", default="roller_axle", help="Substring for axle prims.")
    parser.add_argument("--cover-fragment", default="roller_cover", help="Substring for cover prims.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    assemblies = [p for p in Usd.PrimRange.Stage(stage, predicate) if "roller_assembly" in p.GetName()]

    moves = 0
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for assembly in assemblies:
        # Refresh/skip if prim was removed earlier.
        assembly = stage.GetPrimAtPath(assembly.GetPath())
        if not assembly or not assembly.IsValid():
            continue

        # Only flatten redundant wrapper xforms (no authored ops), to avoid dropping authored transforms.
        flatten_redundant_chain(stage, assembly)
        xform_cache.Clear()

        # Refresh children after potential collapse
        children = list(assembly.GetChildren())
        cover = next((c for c in children if args.cover_fragment in c.GetName()), None)
        axle = next((c for c in children if args.axle_fragment in c.GetName()), None)
        if cover is None or axle is None:
            # try searching deeper
            cover = None
            axle = None
            for p in Usd.PrimRange(assembly, predicate):
                if cover is None and args.cover_fragment in p.GetName() and p.IsA(UsdGeom.Xform):
                    cover = p
                if axle is None and args.axle_fragment in p.GetName() and p.IsA(UsdGeom.Xform):
                    axle = p
                if cover and axle:
                    break
        if cover is None or axle is None:
            continue

        if cover.IsInstanceProxy() or cover.IsInstance():
            inst = cover
            while inst and not inst.IsInstance():
                inst = inst.GetParent()
            if inst and inst.IsInstance():
                inst.SetInstanceable(False)
                cover = stage.GetPrimAtPath(cover.GetPath())
        if axle.IsInstanceProxy() or axle.IsInstance():
            inst = axle
            while inst and not inst.IsInstance():
                inst = inst.GetParent()
            if inst and inst.IsInstance():
                inst.SetInstanceable(False)
                axle = stage.GetPrimAtPath(axle.GetPath())

        cover_xf = holder_of_first_mesh(cover)
        axle_xf = holder_of_first_mesh(axle)
        if cover_xf is None or axle_xf is None:
            continue

        def hoist_with_accum(holder: Usd.Prim, new_name: str) -> Sdf.Path:
            """
            Hoist holder directly under the assembly while preserving its world-space pose.

            Computing in world space makes sure any intermediate scales/transforms (e.g., Body1 scale 10)
            are baked into the new local transform, matching a manual drag+drop in the UI.
            """
            holder_xf = UsdGeom.Xformable(holder)

            # Recompute caches after edits earlier in the loop.
            xform_cache.Clear()
            # Directly ask USD for the relative transform from assembly -> holder.
            # This matches what the DCC does when dragging a prim to a new parent.
            rel_mat, _ = xform_cache.ComputeRelativeTransform(holder, assembly)
            assembly_from_holder = rel_mat

            dst_path = unique_child_path(assembly, new_name)
            Sdf.CopySpec(stage.GetRootLayer(), holder.GetPath(), stage.GetRootLayer(), dst_path)
            dst_prim = stage.GetPrimAtPath(dst_path)
            xfable_dst = UsdGeom.Xformable(dst_prim)
            xfable_dst.ClearXformOpOrder()
            xfable_dst.AddTransformOp().Set(assembly_from_holder)
            return dst_path

        new_cover_path = hoist_with_accum(cover_xf, "roller_cover")
        new_axle_path = hoist_with_accum(axle_xf, "roller_axle")

        # Remove entire old branches to avoid leftover wrappers
        stage.RemovePrim(cover.GetPath())
        stage.RemovePrim(axle.GetPath())

        # remove stale joint_frames under assembly
        for child in list(assembly.GetChildren()):
            if child.GetName().startswith("joint_frame"):
                stage.RemovePrim(child.GetPath())

        # remove any now-empty roller_assembly wrappers under this assembly
        for child in list(assembly.GetChildren()):
            if "roller_assembly" in child.GetName():
                # If only identity xform ops remain and no children, drop it.
                if not list(child.GetChildren()):
                    props = [p for p in child.GetAuthoredProperties()]
                    other_props = [p for p in props if not p.GetName().startswith("xformOp")]
                    xformable = UsdGeom.Xformable(child) if child.IsA(UsdGeom.Xform) else None
                    drop = False
                    if not other_props and xformable and is_identity_transform(xformable):
                        drop = True
                    elif not props and not other_props:
                        drop = True
                    if drop:
                        stage.RemovePrim(child.GetPath())
                        remove_empty_chain(stage, assembly)

        moves += 2

    stage.GetRootLayer().Export(args.out)
    print(f"Repacked {moves} prim(s) (cover/axle) under roller assemblies; exported to {args.out}")


if __name__ == "__main__":
    main()
