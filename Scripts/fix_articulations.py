"""
Attach articulation root and revolute joints for wheels and roller joint_frames.

- Articulation root: /World/mechanum_robot_v18
- Revolute joints at every prim named "joint_frame" (axis = Z, aligned to joint_frame orientation)
  body0 = parent of the joint_frame, body1 = joint_frame itself
- Revolute joints at each wheel root (axis = X: motion in YZ plane)
  Wheels:
    /World/mechanum_robot_v18/BL_Wheel_1
    /World/mechanum_robot_v18/TR_Wheel_1
    /World/mechanum_robot_v18/TL_Wheel_1
    /World/mechanum_robot_v18/BR_Wheel_1

Usage:
  python Scripts/fix_articulations.py --stage <in.usd> --out <out.usd>
"""

import argparse
from typing import List, Optional, Tuple

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


WHEEL_PATHS: List[str] = [
    "/World/mechanum_robot_v18/BL_Wheel_1/BL_Wheel",
    "/World/mechanum_robot_v18/TR_Wheel_1/TR_Wheel",
    "/World/mechanum_robot_v18/TL_Wheel_1/TL_Wheel",
    "/World/mechanum_robot_v18/BR_Wheel_1/BR_Wheel",
]

ARTICULATION_ROOT_PATH = "/World/mechanum_robot_v18"


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


def find_largest_mesh(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Return the descendant mesh with the most points."""
    best = None
    best_count = -1
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if pts and len(pts) > best_count:
                best_count = len(pts)
                best = p
    return best


def apply_rigid_body_and_collider(prim: Usd.Prim, add_collider: bool = False, reset_if_nested: bool = False) -> None:
    """Apply a RigidBodyAPI to the prim and optionally add a convex collider for mass."""
    prim = ensure_editable(prim)
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    # Ensure mass exists to avoid zero/negative mass warnings.
    if hasattr(UsdPhysics, "MassAPI"):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if not mass_api.GetMassAttr().HasAuthoredValueOpinion():
            mass_api.CreateMassAttr(1.0)
        if not mass_api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion():
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    if add_collider:
        mesh_prim = find_largest_mesh(prim)
        if mesh_prim:
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            mesh_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    # If this rigid is parented under another rigid, optionally reset the xform stack.
    if reset_if_nested:
        parent = prim.GetParent()
        if parent and UsdPhysics.RigidBodyAPI(parent):
            xform = UsdGeom.Xformable(prim)
            xform.SetResetXformStack(True)

def ensure_link(prim: Usd.Prim) -> None:
    """Apply RigidBodyAPI to the prim if not already applied."""
    if not prim:
        return
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)

def find_rigid_ancestor(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Return the nearest ancestor (excluding self) that already has RigidBodyAPI."""
    cur = prim.GetParent()
    while cur:
        if UsdPhysics.RigidBodyAPI(cur):
            return cur
        cur = cur.GetParent()
    return None


def strip_rigid_bodies(stage: Usd.Stage, name_fragment: str) -> int:
    """Remove applied RigidBodyAPI (and CollisionAPI) from prims whose name contains fragment."""
    removed = 0
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if name_fragment in prim.GetName() and UsdPhysics.RigidBodyAPI(prim):
            prim.RemoveAppliedSchema(UsdPhysics.RigidBodyAPI)
            if UsdPhysics.CollisionAPI(prim):
                prim.RemoveAppliedSchema(UsdPhysics.CollisionAPI)
            removed += 1
    return removed


def find_child_with_fragment(root: Usd.Prim, fragment: str, predicate=None) -> Optional[Usd.Prim]:
    """Return first descendant xform whose name contains fragment."""
    if predicate is None:
        predicate = Usd.PrimDefaultPredicate
    for p in Usd.PrimRange(root, predicate):
        if p == root:
            continue
        if fragment in p.GetName() and p.IsA(UsdGeom.Xform):
            return p
    return None


def move_prim_to_parent(stage: Usd.Stage, prim: Usd.Prim, new_parent: Usd.Prim, base_name: str) -> Usd.Prim:
    """Move prim under a new parent with a unique name; returns the new prim."""
    src_layer = stage.GetEditTarget().GetLayer()
    dst_layer = src_layer
    new_path = new_parent.GetPath().AppendChild(base_name)
    suffix = 1
    while stage.GetPrimAtPath(new_path):
        new_path = new_parent.GetPath().AppendChild(f"{base_name}_{suffix}")
        suffix += 1
    Sdf.CopySpec(src_layer, prim.GetPath(), dst_layer, new_path)
    # Remove original prim spec via the stage API (handles composition correctly).
    stage.RemovePrim(prim.GetPath())
    return stage.GetPrimAtPath(new_path)


def add_articulation_root(stage: Usd.Stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise SystemExit(f"Articulation root prim not found: {path}")
    if not prim.IsInstance():
        UsdPhysics.ArticulationRootAPI.Apply(prim)


def define_joint(stage: Usd.Stage, path: str) -> UsdPhysics.RevoluteJoint:
    return UsdPhysics.RevoluteJoint.Define(stage, path)

def define_fixed_joint(stage: Usd.Stage, path: str) -> UsdPhysics.FixedJoint:
    return UsdPhysics.FixedJoint.Define(stage, path)


def world_translation(xf: Gf.Matrix4d) -> Gf.Vec3d:
    """Extract translation by transforming origin."""
    return xf.Transform(Gf.Vec3d(0.0, 0.0, 0.0))


def compute_local_pos(world_from_local: Gf.Matrix4d, world_point: Gf.Vec3d) -> Gf.Vec3f:
    """Map a world-space point into a prim's local space."""
    local = world_from_local.GetInverse().Transform(world_point)
    return Gf.Vec3f(float(local[0]), float(local[1]), float(local[2]))


def extract_quat_local(parent_world: Gf.Matrix4d, child_world: Gf.Matrix4d) -> Gf.Quatf:
    """Return child rotation expressed in parent local space as a quaternion (robust to scale/shear)."""
    def rot_matrix4(mat: Gf.Matrix4d) -> Gf.Matrix4d:
        m3 = Gf.Matrix3d(
            mat[0][0], mat[0][1], mat[0][2],
            mat[1][0], mat[1][1], mat[1][2],
            mat[2][0], mat[2][1], mat[2][2],
        )
        m3.Orthonormalize()
        m4 = Gf.Matrix4d(1.0)
        for r in range(3):
            for c in range(3):
                m4[r][c] = m3[r][c]
        return m4

    parent_rot = rot_matrix4(parent_world)
    child_rot = rot_matrix4(child_world)
    local = parent_rot.GetInverse() * child_rot
    local.Orthonormalize()
    rot = local.ExtractRotation().GetQuat()
    return Gf.Quatf(float(rot.GetReal()), Gf.Vec3f(*[float(c) for c in rot.GetImaginary()]))


def process_joint_frames(stage: Usd.Stage, wheel_body_map: dict[str, Usd.Prim]) -> int:
    count = 0
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    jf_paths = [p.GetPath() for p in Usd.PrimRange.Stage(stage, predicate) if p.GetName() == "joint_frame"]
    for jf_path in jf_paths:
        prim = stage.GetPrimAtPath(jf_path)
        if not prim:
            continue
        prim = ensure_editable(prim)
        assembly = prim.GetParent()
        if not assembly:
            continue
        assembly = ensure_editable(assembly)
        axle = assembly.GetChild("roller_axle")
        if not axle:
            axle = find_child_with_fragment(assembly, "roller_axle", predicate)
        if not axle:
            continue
        axle = ensure_editable(axle)

        # joint_frame: make it rigid (no collider needed for mass, keep light inertia).
        apply_rigid_body_and_collider(prim, add_collider=False, reset_if_nested=False)
        # Axle: strip rigid/collision so it is passive and doesn't fall.
        if UsdPhysics.RigidBodyAPI(axle):
            axle.RemoveAppliedSchema(UsdPhysics.RigidBodyAPI)
        if UsdPhysics.CollisionAPI(axle):
            axle.RemoveAppliedSchema(UsdPhysics.CollisionAPI)

        # Find cover sibling under same assembly; make it rigid with collider for the fixed joint.
        cover = assembly.GetChild("roller_cover")
        if not cover:
            cover = find_child_with_fragment(assembly, "roller_cover", predicate)
        if cover:
            cover = ensure_editable(cover)
            apply_rigid_body_and_collider(cover, add_collider=True, reset_if_nested=False)

        # Bodies:
        body0_core = None  # used for core-fixed joint
        wheel_root = assembly
        while wheel_root and not wheel_root.GetName().endswith("_Wheel_1"):
            wheel_root = wheel_root.GetParent()
        if wheel_root:
            body0_core = wheel_body_map.get(str(wheel_root.GetPath()))
        if body0_core is None:
            body0_core = find_rigid_ancestor(axle)
        if body0_core is None:
            continue

        if not cover:
            continue

        joint_path = prim.GetPath().AppendChild("revolute_joint")
        joint = define_joint(stage, str(joint_path))
        joint.CreateBody0Rel().SetTargets([prim.GetPath()])   # joint_frame
        joint.CreateBody1Rel().SetTargets([cover.GetPath()])  # roller_cover
        joint.CreateAxisAttr().Set("Z")

        # Anchor the joint at the joint_frame location relative to both bodies.
        jf_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        cover_world = UsdGeom.Xformable(cover).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        anchor_world = world_translation(jf_world)
        joint.CreateLocalPos0Attr().Set(compute_local_pos(jf_world, anchor_world))     # joint_frame
        joint.CreateLocalPos1Attr().Set(compute_local_pos(cover_world, anchor_world))  # cover
        joint.CreateLocalRot0Attr().Set(extract_quat_local(jf_world, jf_world))
        joint.CreateLocalRot1Attr().Set(extract_quat_local(cover_world, jf_world))

        # Fixed joint: cover follows joint_frame so it stays attached.
        fixed_path = prim.GetPath().AppendChild("joint_frame_core_fixed")
        suffix = 1
        while stage.GetPrimAtPath(fixed_path):
            fixed_path = prim.GetPath().AppendChild(f"joint_frame_core_fixed_{suffix}")
            suffix += 1
        fixed = define_fixed_joint(stage, str(fixed_path))
        fixed.CreateBody0Rel().SetTargets([body0_core.GetPath()])  # core
        fixed.CreateBody1Rel().SetTargets([prim.GetPath()])        # joint_frame
        body0_world = UsdGeom.Xformable(body0_core).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        fixed.CreateLocalPos0Attr().Set(compute_local_pos(body0_world, anchor_world))
        fixed.CreateLocalPos1Attr().Set(compute_local_pos(jf_world, anchor_world))
        fixed.CreateLocalRot0Attr().Set(extract_quat_local(body0_world, jf_world))
        fixed.CreateLocalRot1Attr().Set(extract_quat_local(jf_world, jf_world))

        count += 1
    return count


def process_wheels(stage: Usd.Stage, root_path: str, base_body: Optional[Usd.Prim]) -> tuple[int, dict[str, Usd.Prim]]:
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False, ignoreVisibility=True)
    root = ensure_editable(stage.GetPrimAtPath(root_path))
    count = 0
    wheel_body_map: dict[str, Usd.Prim] = {}
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for wheel_path in WHEEL_PATHS:
        wheel = stage.GetPrimAtPath(wheel_path)
        if not wheel:
            continue
        wheel = ensure_editable(wheel)
        parent = ensure_editable(wheel.GetParent())
        # Pick core as preferred wheel body; make it rigid with collider. Side plates remain non-colliding.
        frame_rigids: list[Usd.Prim] = []
        core = find_child_with_fragment(wheel, "core", predicate)
        if core:
            core = ensure_editable(core)
            apply_rigid_body_and_collider(core, add_collider=True, reset_if_nested=False)
            frame_rigids.append(core)
        body = core if core else wheel
        # Remove RB/collider from wheel if present to avoid nested RB; apply to body instead.
        if wheel != body and UsdPhysics.RigidBodyAPI(wheel):
            wheel.RemoveAppliedSchema(UsdPhysics.RigidBodyAPI)
        if wheel != body and UsdPhysics.CollisionAPI(wheel):
            wheel.RemoveAppliedSchema(UsdPhysics.CollisionAPI)
        if body not in frame_rigids:
            apply_rigid_body_and_collider(body, add_collider=True, reset_if_nested=False)
        wheel_root = wheel.GetParent() if wheel.GetParent() else wheel
        wheel_body_map[str(wheel_root.GetPath())] = body

        aligned = bbox_cache.ComputeWorldBound(body).ComputeAlignedBox()
        world_min = aligned.GetMin()
        world_max = aligned.GetMax()
        world_center = (world_min + world_max) * 0.5

        root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0.0)
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0.0)
        local_center_root = compute_local_pos(root_world, world_center)
        local_center_body = compute_local_pos(body_world, world_center)

        joint_path = parent.GetPath().AppendChild("wheel_joint")
        suffix = 1
        while stage.GetPrimAtPath(joint_path):
            joint_path = parent.GetPath().AppendChild(f"wheel_joint_{suffix}")
            suffix += 1
        joint = define_joint(stage, str(joint_path))
        joint.CreateBody0Rel().SetTargets([base_body.GetPath() if base_body else root.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr().Set("X")
        joint.CreateLocalPos0Attr().Set(local_center_root)
        joint.CreateLocalPos1Attr().Set(local_center_body)
        count += 1
    return count, wheel_body_map


def main():
    parser = argparse.ArgumentParser(description="Add articulation root and revolute joints to the robot.")
    parser.add_argument("--stage", required=True, help="Input USD stage.")
    parser.add_argument("--out", required=True, help="Output USD stage.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    # Remove legacy RB/Colliders from side plates and joint_frames to avoid nested RBs.
    stripped_side = strip_rigid_bodies(stage, "slant_side_plate")
    stripped_frames = strip_rigid_bodies(stage, "joint_frame")
    if stripped_side or stripped_frames:
        print(f"Removed RigidBodyAPI from {stripped_side} slant_side_plate, {stripped_frames} joint_frame prim(s)")

    add_articulation_root(stage, ARTICULATION_ROOT_PATH)

    # Choose a base body (chassis) for wheel joints: prefer frame_body if present, else center rail, else side rail, else root.
    root_prim = stage.GetPrimAtPath(ARTICULATION_ROOT_PATH)
    base_body = stage.GetPrimAtPath(f"{ARTICULATION_ROOT_PATH}/frame_body")
    if not base_body or not base_body.IsValid():
        base_body = find_child_with_fragment(root_prim, "Center_Rail", Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    if not base_body:
        base_body = find_child_with_fragment(root_prim, "Side_Rail", Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    if base_body:
        base_body = ensure_editable(base_body)
        apply_rigid_body_and_collider(base_body, add_collider=True, reset_if_nested=False)
    else:
        base_body = ensure_editable(root_prim)
        apply_rigid_body_and_collider(base_body, add_collider=False, reset_if_nested=False)

    n_wheels, wheel_body_map = process_wheels(stage, ARTICULATION_ROOT_PATH, base_body)
    n_frames = process_joint_frames(stage, wheel_body_map)

    stage.GetRootLayer().Export(args.out)
    print(f"Applied articulation root at {ARTICULATION_ROOT_PATH}")
    print(f"Added {n_frames} revolute joints for joint_frames (axis=Z)")
    print(f"Added {n_wheels} wheel joints (axis=X)")
    print(f"Exported to {args.out}")


if __name__ == "__main__":
    main()
