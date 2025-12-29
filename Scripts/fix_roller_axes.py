"""
Re-orient roller joints by deriving axes from axle mesh geometry (PCA on points).

Defaults:
  - Stage: C:/Worspace/mechanum_robot_v1.usd
  - Orientation source: largest mesh under prims containing "roller_axle".
  - Joint placement: a child xform "joint_frame" under the roller axle mesh parent.

Usage:
  ./fix_roller_axes.py --stage <path_to_usd> --axle-fragment "node_606_XXXX_0096_roller_axle" --cover-fragment "node_606_XXXX_0096_roller_cover"
"""

import argparse
from typing import Iterable, Optional

import numpy as np
from pxr import Gf, Usd, UsdGeom


def principal_axes(points_np: np.ndarray) -> np.ndarray:
    """Return 3x3 matrix whose columns are major->minor principal axes."""
    p = points_np - points_np.mean(axis=0, keepdims=True)
    cov = p.T @ p / points_np.shape[0]
    vals, vecs = np.linalg.eigh(cov)  # columns = eigenvectors
    order = np.argsort(vals)[::-1]
    return vecs[:, order]


def find_largest_mesh(target_prim: Usd.Prim) -> Optional[UsdGeom.Mesh]:
    """Find the largest mesh under target_prim (traverses instance proxies)."""
    best_mesh: Optional[UsdGeom.Mesh] = None
    best_count = -1

    # traverse instance proxies so per-instance transforms are respected
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for p in Usd.PrimRange(target_prim, predicate):
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            pts = mesh.GetPointsAttr().Get()
            if pts and len(pts) > best_count:
                best_count = len(pts)
                best_mesh = mesh
    return best_mesh


def mesh_points_in_target_space(mesh: UsdGeom.Mesh, target_prim: Usd.Prim) -> Optional[np.ndarray]:
    """Return mesh points transformed into the target prim's local space."""
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return None

    mesh_xf = UsdGeom.Xformable(mesh.GetPrim())
    target_xf = UsdGeom.Xformable(target_prim)
    world_from_mesh = mesh_xf.ComputeLocalToWorldTransform(0.0)
    world_from_target = target_xf.ComputeLocalToWorldTransform(0.0)
    target_from_mesh = world_from_target.GetInverse() * world_from_mesh

    transformed = [target_from_mesh.Transform(Gf.Vec3d(p)) for p in pts]
    return np.array(transformed, dtype=float)


def ensure_joint_frame(parent_prim: Usd.Prim, name: str = "joint_frame") -> UsdGeom.Xform:
    """Create or fetch a child xform under parent to hold the derived joint frame."""
    stage = parent_prim.GetStage()
    base_path = parent_prim.GetPath().AppendChild(name)
    path = base_path
    # avoid collisions
    counter = 1
    while stage.GetPrimAtPath(path):
        if stage.GetPrimAtPath(path).IsA(UsdGeom.Xform):
            return UsdGeom.Xform(stage.GetPrimAtPath(path))
        path = parent_prim.GetPath().AppendChild(f"{name}_{counter}")
        counter += 1
    return UsdGeom.Xform.Define(stage, path)


def hoist_joint_frame(frame: UsdGeom.Xform, assembly: Usd.Prim, xform_cache: UsdGeom.XformCache) -> UsdGeom.Xform:
    """Move joint_frame under assembly, baking its world pose into the new local transform."""
    stage = assembly.GetStage()
    xform_cache.Clear()
    rel_mat, _ = xform_cache.ComputeRelativeTransform(frame.GetPrim(), assembly)
    dst_path = assembly.GetPath().AppendChild("joint_frame")
    suffix = 1
    while stage.GetPrimAtPath(dst_path):
        dst_path = assembly.GetPath().AppendChild(f"joint_frame_{suffix}")
        suffix += 1
    dst = UsdGeom.Xform.Define(stage, dst_path)
    dst.ClearXformOpOrder()
    dst.AddTransformOp().Set(rel_mat)
    stage.RemovePrim(frame.GetPrim().GetPath())
    return dst


def matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion components (w, x, y, z)."""
    t = np.trace(m)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return float(w), float(x), float(y), float(z)


def main():
    parser = argparse.ArgumentParser(description="Derive roller cover joint frames from axle mesh geometry.")
    parser.add_argument("--stage", default="C:/Worspace/mechanum_robot_v1.usd", help="Path to USD stage.")
    parser.add_argument("--out", default=None, help="Optional output USD path (defaults to in-place save).")
    parser.add_argument(
        "--axle-fragment",
        default="roller_axle",
        help="Substring to match in prim paths for axle parents (orientation source).",
    )
    parser.add_argument(
        "--cover-fragment",
        default="roller_cover",
        help="Substring to match roller cover prims (joint placement target).",
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def find_child_with_fragment(parent: Usd.Prim, fragment: str) -> Optional[Usd.Prim]:
        """Prefer direct children; fall back to any descendant xform containing fragment."""
        direct = next((c for c in parent.GetChildren() if fragment in c.GetName() and c.IsA(UsdGeom.Xform)), None)
        if direct:
            return direct
        for p in Usd.PrimRange(parent, predicate):
            if p == parent:
                continue
            if fragment in p.GetName() and p.IsA(UsdGeom.Xform):
                return p
        return None

    assemblies = [p for p in Usd.PrimRange.Stage(stage, predicate) if "roller_assembly" in p.GetName()]
    roller_pairs: list[tuple[Usd.Prim, Usd.Prim, Optional[Usd.Prim]]] = []
    for assembly in assemblies:
        assembly = stage.GetPrimAtPath(assembly.GetPath())  # refresh
        if not assembly or not assembly.IsValid():
            continue
        axle_prim = find_child_with_fragment(assembly, args.axle_fragment)
        cover_prim = find_child_with_fragment(assembly, args.cover_fragment)
        if axle_prim is None:
            continue
        roller_pairs.append((assembly, axle_prim, cover_prim))
    print(f"Found {len(roller_pairs)} roller axle prims")

    updated = 0
    for assembly, prim, cover_prim in roller_pairs:
        mesh = find_largest_mesh(prim)
        if mesh is None:
            print(f"[SKIP] No mesh found under {prim.GetPath()}")
            continue

        axle_mesh = mesh
        target_parent = axle_mesh.GetPrim().GetParent() if axle_mesh else prim
        if not target_parent or not target_parent.IsA(UsdGeom.Xform):
            print(f"[SKIP] No xform parent for axle {prim.GetPath()}")
            continue

        # remove any previous joint_frame children so we can refresh
        for child in list(target_parent.GetChildren()):
            if child.GetName().startswith("joint_frame"):
                stage.RemovePrim(child.GetPath())

        pts = mesh.GetPointsAttr().Get()
        if not pts:
            print(f"[SKIP] Empty points on {mesh.GetPath()}")
            continue
        mesh_xf = UsdGeom.Xformable(mesh.GetPrim())
        target_xf = UsdGeom.Xformable(target_parent)
        world_from_mesh = mesh_xf.ComputeLocalToWorldTransform(0.0)
        world_from_target = target_xf.ComputeLocalToWorldTransform(0.0)
        target_from_mesh = world_from_target.GetInverse() * world_from_mesh
        transformed = [target_from_mesh.Transform(Gf.Vec3d(p)) for p in pts]
        pts_np = np.array(transformed, dtype=float)
        if pts_np.shape[0] < 3:
            print(f"[SKIP] Too few points on {mesh.GetPath()}")
            continue

        bb_min = pts_np.min(axis=0)
        bb_max = pts_np.max(axis=0)
        center = (bb_min + bb_max) * 0.5
        extent = bb_max - bb_min

        primary_idx = int(np.argmax(extent))
        if extent[primary_idx] < 1e-6:
            axes = principal_axes(pts_np)
            z = np.array(axes[:, 0])
        else:
            basis = np.eye(3)
            z = basis[primary_idx]
        z_norm = np.linalg.norm(z)
        z = z / z_norm if z_norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(world_up, z)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        x = np.cross(world_up, z)
        x_norm = np.linalg.norm(x)
        x = x / x_norm if x_norm > 1e-8 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        if np.dot(np.cross(x, y), z) < 0:
            y = -y

        m = np.stack([x, y, z], axis=0)
        w, qx, qy, qz = matrix_to_quat(m)

        target = target_parent
        if target.IsInstanceProxy() or target.IsInstance():
            inst_root = target
            while inst_root and not inst_root.IsInstance():
                inst_root = inst_root.GetParent()
            if inst_root and inst_root.IsInstance():
                inst_root.SetInstanceable(False)
                target = stage.GetPrimAtPath(target.GetPath())

        frame = ensure_joint_frame(target, name="joint_frame")
        frame.ClearXformOpOrder()
        frame.AddTranslateOp().Set(Gf.Vec3f(float(center[0]), float(center[1]), float(center[2])))
        orient_op = frame.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        orient_op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(qx), float(qy), float(qz))))

        # Hoist joint_frame to assembly level (sibling of axle/cover) while preserving world pose.
        assembly = target.GetParent()
        if assembly and assembly.IsA(UsdGeom.Xform):
            frame = hoist_joint_frame(frame, assembly, xform_cache)

        updated += 1
        print(f"[OK] Added joint_frame under {target.GetPath()} using axle mesh {mesh.GetPath()}")

    if args.out:
        stage.GetRootLayer().Export(args.out)
        print(f"Updated {updated} roller axle xforms; exported to {args.out}")
    else:
        stage.GetRootLayer().Save()
        print(f"Updated {updated} roller axle xforms; saved {args.stage}")


if __name__ == "__main__":
    main()
