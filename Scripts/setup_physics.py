"""
Apply articulation root, rigid bodies, colliders, and joints for the Strafer chassis.

This script sets up the physics for a Gobilda Strafer mecanum wheel robot:
- Articulation root on /World/strafer/body_link (identity transform in ROS frame)
- Fixed joints between body_link and frame rails
- Revolute joints between frame rails and wheel cores (with velocity drives)
- Fixed joints from wheel cores to roller axles
- Revolute joints between roller axles and roller covers

The body_link prim is created at the robot root with identity orientation to ensure
the articulation's base frame matches the robot's visual orientation (Z-up, +X forward).

Run after collapse_redundant_xforms.py and export to a new USD so the source stays intact.

Example:
  python Scripts/setup_physics.py --stage Assets/3209-0001-0006-v6/3209-0001-0006-collapsed.usd \\
    --output-usd Assets/3209-0001-0006-v6/3209-0001-0006-physics.usd \\
    --log ./setup_physics_log.txt
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None  # Fallback when PhysX schema bindings are unavailable
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


# ============================================================================
# Configuration
# ============================================================================

ROOT_PATH = "/World/strafer"
BODY_LINK_PATH = f"{ROOT_PATH}/body_link"  # Will be created with identity transform
RAILS_PATH = f"{ROOT_PATH}/frame/rails"
WHEELS_PATH = f"{ROOT_PATH}/mecanum_wheels"
FRAME_PATH = f"{ROOT_PATH}/frame"

# Wheel-to-rail mapping (ROS frame, X forward, Y left):
# wheel_1 = Front-Left, wheel_3 = Rear-Left -> left_side_rail
# wheel_2 = Front-Right, wheel_4 = Rear-Right -> right_side_rail
WHEEL_TO_RAIL = {
    "wheel_1": "left_side_rail",
    "wheel_2": "right_side_rail",
    "wheel_3": "left_side_rail",
    "wheel_4": "right_side_rail",
}

# ============================================================================
# Exclusion / Attachment Configuration
# ============================================================================

# 1) Roller assembly siblings - patterns to EXCLUDE (no physics at all)
#    These are inside each roller_assembly and are purely visual
ROLLER_ASSEMBLY_EXCLUDE_PATTERNS = [
    "roller_e_clip",      # e_clip parts
    "roller_shim",        # shim parts
    "roller_core",        # roller_core (not wheel core)
    "node_600_0307_0003", # bearing balls
]

# 2) Wheel siblings - patterns to EXCLUDE (no physics)
WHEEL_EXCLUDE_PATTERNS = [
    "node_606_XXXX_0096_text",  # text labels
    "node_812_0004_0007",       # screws/hardware
    "node_800_0004_0020",       # screws/hardware
]

# 2b) Wheel siblings - patterns to ATTACH to wheel core (fixed joint)
WHEEL_ATTACH_TO_CORE_PATTERNS = [
    "left_slant_side_plate",
    "right_slant_side_plate",
]

# 3) Frame siblings - full paths to EXCLUDE
FRAME_EXCLUDE_PATHS = [
    f"{FRAME_PATH}/mounts",
]

# 3b) Frame rails - additional rails to attach to middle_center_rail
FRAME_ATTACH_TO_MIDDLE_RAIL = [
    "back_center_rail",
]

# 4) Root siblings - full paths to EXCLUDE entirely (no physics)
ROOT_EXCLUDE_PATHS = [
    f"{ROOT_PATH}/gear_motors",
    f"{ROOT_PATH}/axles",
    f"{ROOT_PATH}/hardware",
    f"{ROOT_PATH}/miter_gears",
]


def should_exclude_by_pattern(name: str, patterns: List[str]) -> bool:
    """Check if a prim name matches any exclusion pattern."""
    for pattern in patterns:
        if pattern in name:
            return True
    return False


def should_attach_by_pattern(name: str, patterns: List[str]) -> bool:
    """Check if a prim name matches any attachment pattern."""
    for pattern in patterns:
        if pattern in name:
            return True
    return False


def delete_excluded_prims(stage: Usd.Stage, log: List[str]) -> int:
    """
    Delete all prims matching exclusion patterns for maximum simulation efficiency.
    Returns the count of deleted prims.
    """
    deleted_count = 0
    paths_to_delete: List[str] = []
    
    # 1) Collect ROOT_EXCLUDE_PATHS (full paths)
    for path in ROOT_EXCLUDE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            paths_to_delete.append(path)
    
    # 2) Collect FRAME_EXCLUDE_PATHS (full paths)
    for path in FRAME_EXCLUDE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            paths_to_delete.append(path)
    
    # 3) Collect wheel siblings matching WHEEL_EXCLUDE_PATTERNS
    for wheel_name in WHEEL_TO_RAIL.keys():
        wheel_path = f"{WHEELS_PATH}/{wheel_name}"
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if wheel_prim and wheel_prim.IsValid():
            for child in wheel_prim.GetChildren():
                if should_exclude_by_pattern(child.GetName(), WHEEL_EXCLUDE_PATTERNS):
                    paths_to_delete.append(str(child.GetPath()))
    
    # 4) Collect roller assembly siblings matching ROLLER_ASSEMBLY_EXCLUDE_PATTERNS
    for wheel_name in WHEEL_TO_RAIL.keys():
        wheel_path = f"{WHEELS_PATH}/{wheel_name}"
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if wheel_prim and wheel_prim.IsValid():
            for child in wheel_prim.GetChildren():
                if "roller_assembly" in child.GetName():
                    for roller_child in child.GetChildren():
                        if should_exclude_by_pattern(roller_child.GetName(), ROLLER_ASSEMBLY_EXCLUDE_PATTERNS):
                            paths_to_delete.append(str(roller_child.GetPath()))
    
    # Delete all collected paths (in reverse order to delete children before parents)
    paths_to_delete.sort(key=lambda p: p.count('/'), reverse=True)
    for path in paths_to_delete:
        if stage.RemovePrim(path):
            log.append(f"[DEL] Removed: {path}")
            deleted_count += 1
        else:
            log.append(f"[WARN] Failed to remove: {path}")
    
    return deleted_count


# ============================================================================
# Utility Functions
# ============================================================================


def make_editable_prim(stage: Usd.Stage, prim_path: str) -> Optional[Usd.Prim]:
    """Return a prim that is safe to author (clearing instancing if needed)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    if prim.IsInstanceProxy():
        # Walk up to find instance root and de-instance it
        inst_root = prim
        while inst_root and inst_root.IsInstanceProxy():
            inst_root = inst_root.GetParent()
        if inst_root and inst_root.IsValid() and inst_root.IsInstance():
            inst_root.SetInstanceable(False)
        # Re-fetch the prim after de-instancing
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or prim.IsInstanceProxy():
            return None
    return prim


def find_leaf_mesh(stage: Usd.Stage, root_path: str) -> Optional[str]:
    """
    Find the deepest mesh prim under the given root path.
    This traverses through Body1/BodyXXX hierarchy to find the actual mesh.
    Example: node_606_XXXX_0096_core_1/Body1/Body113 (mesh)
    
    Handles instanced prims by de-instancing them first.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    
    # De-instance the root if needed
    if root.IsInstance():
        root.SetInstanceable(False)
        root = stage.GetPrimAtPath(root_path)
    
    # First pass: de-instance any instanced children
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    
    # Re-fetch root after de-instancing
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    
    best_mesh = None
    best_depth = -1
    
    # Second pass: find meshes
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            depth = str(prim.GetPath()).count("/")
            if depth > best_depth:
                best_depth = depth
                best_mesh = str(prim.GetPath())
    
    return best_mesh


def find_child_containing(parent: Usd.Prim, fragment: str, exclude_fragment: Optional[str] = None) -> Optional[Usd.Prim]:
    """
    Find a direct child whose name contains the given fragment.
    Optionally exclude children containing exclude_fragment.
    """
    for child in parent.GetChildren():
        name = child.GetName()
        if fragment in name:
            if exclude_fragment and exclude_fragment in name:
                continue
            return child
    return None


# ============================================================================
# Physics Application Functions
# ============================================================================


def apply_collision_to_mesh(stage: Usd.Stage, mesh_path: str, approximation: str = "convexDecomposition") -> bool:
    """
    Apply collision API directly to a mesh prim with specified approximation type.
    This ensures the collision is on the actual mesh geometry.
    """
    prim = make_editable_prim(stage, mesh_path)
    if not prim:
        return False
    
    # Verify it's a mesh
    if not prim.IsA(UsdGeom.Mesh):
        return False
    
    # Apply CollisionAPI
    UsdPhysics.CollisionAPI.Apply(prim)
    
    # Set the approximation using the MeshCollisionAPI for mesh-specific settings
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set(approximation)
    
    return True


def apply_rigid_body(stage: Usd.Stage, prim_path: str, mass: float = 1.0) -> bool:
    """Apply rigid body API with mass to a prim."""
    prim = make_editable_prim(stage, prim_path)
    if not prim:
        return False
    
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    
    return True


def apply_articulation_root(stage: Usd.Stage, prim_path: str) -> bool:
    """Apply articulation root API."""
    prim = make_editable_prim(stage, prim_path)
    if not prim:
        return False
    
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    return True


def create_fixed_joint(
    stage: Usd.Stage, 
    joint_path: str, 
    body0_path: str, 
    body1_path: str
) -> bool:
    """
    Create a fixed joint between two bodies.
    body0 = parent body (the one the joint is relative to)
    body1 = child body (the one being constrained)
    
    Uses world transforms to compute correct local poses so bodies don't shift.
    """
    # Ensure bodies are editable
    b0 = make_editable_prim(stage, body0_path)
    b1 = make_editable_prim(stage, body1_path)
    if not b0 or not b1:
        return False
    
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    if not joint:
        return False
    
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    
    # Compute relative transform so bodies stay in place
    xf_cache = UsdGeom.XformCache()
    body0_world = xf_cache.GetLocalToWorldTransform(b0)
    body1_world = xf_cache.GetLocalToWorldTransform(b1)
    
    # Relative pose of body1 in body0's frame
    rel_pose = body1_world * body0_world.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    
    pos0 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot0 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    
    joint.CreateLocalPos0Attr().Set(pos0)
    joint.CreateLocalRot0Attr().Set(rot0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    
    return True


def create_revolute_joint(
    stage: Usd.Stage, 
    joint_path: str, 
    body0_path: str, 
    body1_path: str, 
    axis: str = "Z",
    add_drive: bool = False,
    drive_type: str = "velocity",
    drive_damping: float = 1000.0,
    drive_stiffness: float = 0.0,
    drive_max_force: float = 1000.0,
) -> bool:
    """
    Create a revolute joint between two bodies.
    body0 = parent body (stationary reference, e.g., rail)
    body1 = child body (rotating body, e.g., wheel core)
    
    Args:
        stage: USD stage
        joint_path: Path for the joint prim
        body0_path: Parent body path
        body1_path: Child body path
        axis: Rotation axis ("X", "Y", or "Z")
        add_drive: If True, add a DriveAPI for motor control
        drive_type: "velocity" or "position"
        drive_damping: Damping coefficient for velocity control
        drive_stiffness: Stiffness for position control (0 for pure velocity)
        drive_max_force: Maximum force/torque the drive can apply
    
    Uses world transforms to compute correct local poses so bodies don't shift.
    """
    # Ensure bodies are editable
    b0 = make_editable_prim(stage, body0_path)
    b1 = make_editable_prim(stage, body1_path)
    if not b0 or not b1:
        return False
    
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    if not joint:
        return False
    
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateAxisAttr(axis)
    
    # Compute relative transform so bodies stay in place
    xf_cache = UsdGeom.XformCache()
    body0_world = xf_cache.GetLocalToWorldTransform(b0)
    body1_world = xf_cache.GetLocalToWorldTransform(b1)
    
    # Relative pose of body1 in body0's frame
    rel_pose = body1_world * body0_world.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    
    pos0 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot0 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    
    joint.CreateLocalPos0Attr().Set(pos0)
    joint.CreateLocalRot0Attr().Set(rot0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    
    # Add Drive API for motor control if requested
    if add_drive:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim:
            # Apply DriveAPI for the angular axis
            # For revolute joints, we drive the "angular" DOF
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            
            # Set drive parameters
            # For velocity control: stiffness=0, damping>0
            # For position control: stiffness>0, damping>0
            drive_api.CreateTypeAttr("force")  # "force" or "acceleration"
            drive_api.CreateDampingAttr(drive_damping)
            drive_api.CreateStiffnessAttr(drive_stiffness)
            drive_api.CreateMaxForceAttr(drive_max_force)
            
            # Set initial target velocity to 0
            if drive_type == "velocity":
                drive_api.CreateTargetVelocityAttr(0.0)
            else:
                drive_api.CreateTargetPositionAttr(0.0)
    
    return True


# ============================================================================
# Frame Setup
# ============================================================================


def create_body_link(stage: Usd.Stage, log: List[str], mass: float = 1.0) -> Optional[str]:
    """
    Create or retrieve the body_link prim at the robot root with identity transform.
    
    The body_link serves as the articulation root in ROS convention:
    - Z-up (robot upright)
    - +X forward (robot front)
    - +Y left (robot left side)
    
    Returns the path to the body_link prim.
    """
    root_prim = stage.GetPrimAtPath(ROOT_PATH)
    if not root_prim or not root_prim.IsValid():
        log.append(f"[ERROR] Root prim not found: {ROOT_PATH}")
        return None
    
    # Check if body_link already exists
    body_link_prim = stage.GetPrimAtPath(BODY_LINK_PATH)
    if body_link_prim and body_link_prim.IsValid():
        log.append(f"[INFO] body_link already exists: {BODY_LINK_PATH}")
    else:
        # Create body_link as an Xform with identity transform
        body_link = UsdGeom.Xform.Define(stage, BODY_LINK_PATH)
        if not body_link:
            log.append(f"[ERROR] Failed to create body_link at {BODY_LINK_PATH}")
            return None
        
        # Set identity transform explicitly
        body_link.ClearXformOpOrder()
        body_link.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        body_link.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))  # Identity quaternion
        
        log.append(f"[OK] Created body_link with identity transform: {BODY_LINK_PATH}")
        body_link_prim = stage.GetPrimAtPath(BODY_LINK_PATH)
    
    # Apply articulation root API to body_link
    if apply_articulation_root(stage, BODY_LINK_PATH):
        log.append(f"[OK] ArticulationRootAPI on {BODY_LINK_PATH}")
    
    # Apply rigid body with mass
    if apply_rigid_body(stage, BODY_LINK_PATH, mass):
        log.append(f"[OK] RigidBodyAPI (mass={mass}) on {BODY_LINK_PATH}")
    
    return BODY_LINK_PATH


def setup_frame(stage: Usd.Stage, log: List[str], mass: float = 1.0) -> Optional[str]:
    """
    Set up articulation root on body_link and connect all frame rails to it.
    Returns the path to the body_link for wheel connections.
    """
    middle_rail = f"{RAILS_PATH}/middle_center_rail"
    left_rail = f"{RAILS_PATH}/left_side_rail"
    right_rail = f"{RAILS_PATH}/right_side_rail"
    
    # 1. Create body_link with articulation root
    body_link_path = create_body_link(stage, log, mass)
    if not body_link_path:
        log.append(f"[ERROR] Failed to create body_link")
        return None
    
    # Find the leaf mesh bodies for each rail
    middle_mesh = find_leaf_mesh(stage, middle_rail)
    left_mesh = find_leaf_mesh(stage, left_rail)
    right_mesh = find_leaf_mesh(stage, right_rail)
    
    if not middle_mesh:
        log.append(f"[ERROR] Cannot find mesh under {middle_rail}")
        return None
    
    log.append(f"[INFO] Middle rail mesh: {middle_mesh}")
    log.append(f"[INFO] Left rail mesh: {left_mesh}")
    log.append(f"[INFO] Right rail mesh: {right_mesh}")
    
    # 2. Set up middle_center_rail (connect to body_link)
    middle_prim = make_editable_prim(stage, middle_rail)
    if middle_prim:
        if apply_rigid_body(stage, middle_rail, mass):
            log.append(f"[OK] Rigid body (mass={mass}) on {middle_rail}")
        if apply_collision_to_mesh(stage, middle_mesh, "boundingCube"):
            log.append(f"[OK] Collision (boundingCube) on mesh {middle_mesh}")
        
        # Fixed joint: body_link -> middle_center_rail
        joint_path = f"{RAILS_PATH}/frame_rail_middle"
        if create_fixed_joint(stage, joint_path, body_link_path, middle_mesh):
            log.append(f"[OK] Fixed joint {joint_path} (name: frame_rail_middle)")
            log.append(f"     Body0: {body_link_path}")
            log.append(f"     Body1: {middle_mesh}")
        else:
            log.append(f"[WARN] Failed to create fixed joint for middle rail")
    else:
        log.append(f"[WARN] Missing prim: {middle_rail}")
        return None
    
    # 3. Set up left_side_rail (connect to body_link)
    if left_mesh:
        left_prim = make_editable_prim(stage, left_rail)
        if left_prim:
            if apply_rigid_body(stage, left_rail, mass):
                log.append(f"[OK] Rigid body (mass={mass}) on {left_rail}")
            if apply_collision_to_mesh(stage, left_mesh, "boundingCube"):
                log.append(f"[OK] Collision (boundingCube) on mesh {left_mesh}")
            
            # Fixed joint: body_link -> left_side_rail
            joint_path = f"{RAILS_PATH}/frame_rail_left"
            if create_fixed_joint(stage, joint_path, body_link_path, left_mesh):
                log.append(f"[OK] Fixed joint {joint_path} (name: frame_rail_left)")
                log.append(f"     Body0: {body_link_path}")
                log.append(f"     Body1: {left_mesh}")
            else:
                log.append(f"[WARN] Failed to create fixed joint for left rail")
    else:
        log.append(f"[WARN] Missing mesh under: {left_rail}")
    
    # 4. Set up right_side_rail (connect to body_link)
    if right_mesh:
        right_prim = make_editable_prim(stage, right_rail)
        if right_prim:
            if apply_rigid_body(stage, right_rail, mass):
                log.append(f"[OK] Rigid body (mass={mass}) on {right_rail}")
            if apply_collision_to_mesh(stage, right_mesh, "boundingCube"):
                log.append(f"[OK] Collision (boundingCube) on mesh {right_mesh}")
            
            # Fixed joint: body_link -> right_side_rail
            joint_path = f"{RAILS_PATH}/frame_rail_right"
            if create_fixed_joint(stage, joint_path, body_link_path, right_mesh):
                log.append(f"[OK] Fixed joint {joint_path} (name: frame_rail_right)")
                log.append(f"     Body0: {body_link_path}")
                log.append(f"     Body1: {right_mesh}")
            else:
                log.append(f"[WARN] Failed to create fixed joint for right rail")
    else:
        log.append(f"[WARN] Missing mesh under: {right_rail}")
    
    # 5. Attach additional rails from FRAME_ATTACH_TO_MIDDLE_RAIL (connect to body_link)
    for rail_name in FRAME_ATTACH_TO_MIDDLE_RAIL:
        extra_rail = f"{RAILS_PATH}/{rail_name}"
        extra_mesh = find_leaf_mesh(stage, extra_rail)
        if extra_mesh:
            extra_prim = make_editable_prim(stage, extra_rail)
            if extra_prim:
                if apply_rigid_body(stage, extra_rail, mass):
                    log.append(f"[OK] Rigid body (mass={mass}) on {extra_rail}")
                if apply_collision_to_mesh(stage, extra_mesh, "boundingCube"):
                    log.append(f"[OK] Collision (boundingCube) on mesh {extra_mesh}")
                
                joint_name = f"frame_rail_{rail_name.replace('_rail', '')}"
                joint_path = f"{RAILS_PATH}/{joint_name}"
                if create_fixed_joint(stage, joint_path, body_link_path, extra_mesh):
                    log.append(f"[OK] Fixed joint {joint_path} (name: {joint_name})")
                    log.append(f"     Body0: {body_link_path}")
                    log.append(f"     Body1: {extra_mesh}")
                else:
                    log.append(f"[WARN] Failed to create fixed joint for {rail_name}")
        else:
            log.append(f"[WARN] Missing mesh under: {extra_rail}")
    
    return body_link_path


# ============================================================================
# Wheel Setup
# ============================================================================


def find_wheel_core_mesh(stage: Usd.Stage, wheel_prim: Usd.Prim, log: List[str]) -> Optional[str]:
    """
    Find the wheel core's leaf mesh body path.
    Looks for prim containing "core" but not "roller_core".
    Returns the deepest mesh path like: .../node_606_XXXX_0096_core_1/Body1/Body113
    
    Core naming patterns:
    - node_606_XXXX_0096_core_1
    - node_606_XXXX_0096_core__1__1
    - node_606_XXXX_0096_core__1__2
    - node_606_XXXX_0096_core_2
    """
    # First, de-instance the wheel if needed
    wheel_path = str(wheel_prim.GetPath())
    if wheel_prim.IsInstance():
        wheel_prim.SetInstanceable(False)
        wheel_prim = stage.GetPrimAtPath(wheel_path)
    
    # De-instance all children first
    for child in wheel_prim.GetChildren():
        if child.IsInstance():
            child.SetInstanceable(False)
    
    # Re-fetch wheel prim
    wheel_prim = stage.GetPrimAtPath(wheel_path)
    
    for child in wheel_prim.GetChildren():
        name = child.GetName()
        # Match "core" but exclude "roller_core"
        # Names like: node_606_XXXX_0096_core_1, node_606_XXXX_0096_core__1__1
        if "_core_" in name or "_core__" in name:
            if "roller" in name.lower():
                continue
            core_path = str(child.GetPath())
            log.append(f"[DEBUG] Found core prim: {core_path}")
            
            # De-instance the core prim and its children
            core_prim = stage.GetPrimAtPath(core_path)
            if core_prim and core_prim.IsInstance():
                core_prim.SetInstanceable(False)
            
            # De-instance Body1 if it exists and is instanced
            body1_path = f"{core_path}/Body1"
            body1_prim = stage.GetPrimAtPath(body1_path)
            if body1_prim and body1_prim.IsValid() and body1_prim.IsInstance():
                body1_prim.SetInstanceable(False)
                log.append(f"[DEBUG] De-instanced Body1: {body1_path}")
            
            # Find the leaf mesh
            mesh_path = find_leaf_mesh(stage, core_path)
            if mesh_path:
                log.append(f"[DEBUG] Found core mesh: {mesh_path}")
                return mesh_path
            else:
                log.append(f"[WARN] Could not find mesh under core: {core_path}")
    
    # Debug: list all children
    log.append(f"[DEBUG] Children of {wheel_path}:")
    for child in wheel_prim.GetChildren():
        log.append(f"[DEBUG]   - {child.GetName()}")
    
    return None


def find_roller_mesh_bodies(stage: Usd.Stage, assembly_prim: Usd.Prim) -> Tuple[Optional[str], Optional[str]]:
    """
    Find roller axle and cover leaf mesh body paths within a roller assembly.
    Returns (axle_mesh_path, cover_mesh_path).
    
    Example paths:
    - axle:  .../node_606_XXXX_0096_roller_axle_2/Body1/Body116
    - cover: .../node_606_XXXX_0096_roller_cover_2/Body1/Body115
    """
    axle_mesh = None
    cover_mesh = None
    
    assembly_path = str(assembly_prim.GetPath())
    
    # De-instance the assembly first
    if assembly_prim.IsInstance():
        assembly_prim.SetInstanceable(False)
        assembly_prim = stage.GetPrimAtPath(assembly_path)
    
    # De-instance all children of assembly
    for child in assembly_prim.GetChildren():
        if child.IsInstance():
            child.SetInstanceable(False)
    
    # Re-fetch assembly
    assembly_prim = stage.GetPrimAtPath(assembly_path)
    
    for child in assembly_prim.GetChildren():
        name = child.GetName()
        child_path = str(child.GetPath())
        
        # De-instance child if needed
        child_prim = stage.GetPrimAtPath(child_path)
        if child_prim and child_prim.IsInstance():
            child_prim.SetInstanceable(False)
        
        # De-instance Body1 under this child
        body1_path = f"{child_path}/Body1"
        body1_prim = stage.GetPrimAtPath(body1_path)
        if body1_prim and body1_prim.IsValid() and body1_prim.IsInstance():
            body1_prim.SetInstanceable(False)
        
        if "roller_axle" in name:
            axle_mesh = find_leaf_mesh(stage, child_path)
                
        elif "roller_cover" in name:
            cover_mesh = find_leaf_mesh(stage, child_path)
    
    return (axle_mesh, cover_mesh)


def setup_wheel(
    stage: Usd.Stage,
    wheel_name: str,
    rail_name: str,
    log: List[str],
    mass: float = 1.0,
) -> None:
    """
    Set up physics for a single wheel and all its roller assemblies.
    
    Structure:
    - Revolute joint: rail_mesh -> wheel_core_mesh (wheel spins on rail)
    - For each roller assembly:
      - Fixed joint: wheel_core_mesh -> roller_axle_mesh (axle fixed to wheel)
      - Revolute joint: roller_axle_mesh -> roller_cover_mesh (cover spins on axle)
    
    All joints reference the leaf mesh prims (e.g., Body1/Body113).
    
    Respects exclusion patterns:
    - WHEEL_EXCLUDE_PATTERNS: siblings of wheel core to skip
    - WHEEL_ATTACH_TO_CORE_PATTERNS: siblings to attach to wheel core
    - ROLLER_ASSEMBLY_EXCLUDE_PATTERNS: siblings inside roller assemblies to skip
    """
    wheel_path = f"{WHEELS_PATH}/{wheel_name}"
    rail_path = f"{RAILS_PATH}/{rail_name}"
    
    wheel_prim = stage.GetPrimAtPath(wheel_path)
    if not wheel_prim or not wheel_prim.IsValid():
        log.append(f"[WARN] Missing wheel: {wheel_path}")
        return
    
    # De-instance if needed
    if wheel_prim.IsInstance():
        wheel_prim.SetInstanceable(False)
        wheel_prim = stage.GetPrimAtPath(wheel_path)
    
    # Find rail leaf mesh
    rail_mesh = find_leaf_mesh(stage, rail_path)
    if not rail_mesh:
        log.append(f"[WARN] Missing rail mesh under: {rail_path}")
        return
    
    log.append(f"\n=== Setting up {wheel_name} (connected to {rail_name}) ===")
    log.append(f"[INFO] Rail mesh: {rail_mesh}")
    
    # 1. Find wheel core leaf mesh
    core_mesh = find_wheel_core_mesh(stage, wheel_prim, log)
    if not core_mesh:
        log.append(f"[WARN] Missing wheel core mesh for {wheel_path}")
        return
    
    log.append(f"[INFO] Core mesh: {core_mesh}")
    
    # Get the parent Xform for applying rigid body (go up from mesh to the core Xform)
    # Path: .../node_606_XXXX_0096_core_1/Body1/Body113 -> .../node_606_XXXX_0096_core_1
    core_mesh_path = Sdf.Path(core_mesh)
    core_xform_path = str(core_mesh_path.GetParentPath().GetParentPath())
    
    # Apply rigid body to wheel core Xform
    if apply_rigid_body(stage, core_xform_path, mass):
        log.append(f"[OK] Rigid body (mass={mass}) on {core_xform_path}")
    
    # Apply collision to the mesh
    if apply_collision_to_mesh(stage, core_mesh, "convexDecomposition"):
        log.append(f"[OK] Collision (convexDecomposition) on mesh {core_mesh}")
    
    # 2. Create revolute joint: rail_mesh -> wheel_core_mesh
    # body0 = rail (parent, stationary), body1 = wheel core (child, rotates)
    # Extract wheel number from wheel_name (e.g., "wheel_1" -> "1")
    wheel_num = wheel_name.split("_")[1]
    wheel_joint_name = f"{wheel_name}_drive"
    wheel_joint_path = f"{core_xform_path}/{wheel_joint_name}"
    # Add velocity drive for motor control (stiffness=0 for velocity mode)
    if create_revolute_joint(
        stage, wheel_joint_path, rail_mesh, core_mesh, "Z",
        add_drive=True,
        drive_type="velocity",
        drive_damping=1000.0,    # Damping for velocity control
        drive_stiffness=0.0,     # 0 = pure velocity control
        drive_max_force=10000.0  # Max torque
    ):
        log.append(f"[OK] Revolute joint {wheel_joint_path} (name: {wheel_joint_name}) with velocity drive")
        log.append(f"     Body0 (rail): {rail_mesh}")
        log.append(f"     Body1 (core): {core_mesh}")
    else:
        log.append(f"[WARN] Failed to create revolute joint for wheel core")
    
    # 3. Process wheel siblings - attach side plates to core
    attach_count = 0
    exclude_count = 0
    for child in wheel_prim.GetChildren():
        child_name = child.GetName()
        child_path = str(child.GetPath())
        
        # Skip roller assemblies (handled separately) and core (already done)
        if "roller_assembly" in child_name:
            continue
        if "_core_" in child_name or "_core__" in child_name:
            if "roller" not in child_name.lower():
                continue
        
        # Check if should be excluded
        if should_exclude_by_pattern(child_name, WHEEL_EXCLUDE_PATTERNS):
            log.append(f"[SKIP] Excluded wheel sibling: {child_name}")
            exclude_count += 1
            continue
        
        # Check if should be attached to core
        if should_attach_by_pattern(child_name, WHEEL_ATTACH_TO_CORE_PATTERNS):
            child_mesh = find_leaf_mesh(stage, child_path)
            if child_mesh:
                # Apply rigid body to the child Xform
                if apply_rigid_body(stage, child_path, mass * 0.1):
                    log.append(f"[OK] Rigid body on side plate: {child_name}")
                
                # Create fixed joint to core
                plate_type = "left" if "left_slant" in child_name else "right"
                plate_joint_name = f"{wheel_name}_plate_{plate_type}_{attach_count}"
                joint_path = f"{child_path}/{plate_joint_name}"
                if create_fixed_joint(stage, joint_path, core_mesh, child_mesh):
                    log.append(f"[OK] Fixed joint: {child_name} -> core (name: {plate_joint_name})")
                    attach_count += 1
                else:
                    log.append(f"[WARN] Failed to attach {child_name} to core")
            else:
                log.append(f"[WARN] No mesh under {child_path}")
    
    if attach_count > 0 or exclude_count > 0:
        log.append(f"[INFO] Wheel siblings: {attach_count} attached, {exclude_count} excluded")
    
    # 4. Process all roller assemblies
    roller_index = 0  # 0-9 for each wheel
    roller_exclude_count = 0
    for child in wheel_prim.GetChildren():
        if "roller_assembly" not in child.GetName():
            continue
        
        assembly_path = str(child.GetPath())
        
        # Find axle and cover meshes
        axle_mesh, cover_mesh = find_roller_mesh_bodies(stage, child)
        
        if not axle_mesh:
            log.append(f"[WARN] Missing axle mesh in {assembly_path}")
            continue
        
        if not cover_mesh:
            log.append(f"[WARN] Missing cover mesh in {assembly_path}")
            continue
        
        # Get parent Xforms for rigid body application
        axle_mesh_path = Sdf.Path(axle_mesh)
        cover_mesh_path = Sdf.Path(cover_mesh)
        axle_xform_path = str(axle_mesh_path.GetParentPath().GetParentPath())
        cover_xform_path = str(cover_mesh_path.GetParentPath().GetParentPath())
        
        # Apply rigid body to axle Xform (no collision needed for axle pins)
        if apply_rigid_body(stage, axle_xform_path, mass):
            log.append(f"[OK] Rigid body on axle: {axle_xform_path}")
        
        # Apply rigid body and collision to cover Xform/mesh
        if apply_rigid_body(stage, cover_xform_path, mass):
            log.append(f"[OK] Rigid body on cover: {cover_xform_path}")
        if apply_collision_to_mesh(stage, cover_mesh, "convexDecomposition"):
            log.append(f"[OK] Collision (convexDecomposition) on cover mesh: {cover_mesh}")
        
        # 4a. Create fixed joint: wheel_core_mesh -> roller_axle_mesh
        # body0 = core (parent), body1 = axle (child, fixed to core)
        axle_joint_name = f"{wheel_name}_roller_{roller_index}_axle"
        fixed_joint_path = f"{axle_xform_path}/{axle_joint_name}"
        if create_fixed_joint(stage, fixed_joint_path, core_mesh, axle_mesh):
            log.append(f"[OK] Fixed joint: core -> axle (name: {axle_joint_name})")
            log.append(f"     Body0 (core): {core_mesh}")
            log.append(f"     Body1 (axle): {axle_mesh}")
        else:
            log.append(f"[WARN] Failed to create fixed joint for axle")
        
        # 4b. Create revolute joint: roller_axle_mesh -> roller_cover_mesh
        # body0 = cover (child frame), body1 = axle (provides pivot point)
        # The joint is positioned at body1's origin (axle center)
        roller_joint_name = f"{wheel_name}_roller_{roller_index}"
        revolute_joint_path = f"{axle_xform_path}/{roller_joint_name}"
        if create_revolute_joint(stage, revolute_joint_path, cover_mesh, axle_mesh, "Z"):
            log.append(f"[OK] Revolute joint: axle -> cover (name: {roller_joint_name})")
            log.append(f"     Body0 (cover): {cover_mesh}")
            log.append(f"     Body1 (axle): {axle_mesh}")
        else:
            log.append(f"[WARN] Failed to create revolute joint for roller")
        
        # 4c. Count excluded siblings in roller assembly (for logging)
        assembly_prim = stage.GetPrimAtPath(assembly_path)
        if assembly_prim:
            for roller_child in assembly_prim.GetChildren():
                roller_child_name = roller_child.GetName()
                if should_exclude_by_pattern(roller_child_name, ROLLER_ASSEMBLY_EXCLUDE_PATTERNS):
                    roller_exclude_count += 1
        
        roller_index += 1
    
    log.append(f"[INFO] Processed {roller_index} roller assemblies for {wheel_name}")
    if roller_exclude_count > 0:
        log.append(f"[INFO] Excluded {roller_exclude_count} roller assembly siblings (e_clips, shims, etc.)")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Set up articulation, rigid bodies, colliders, and joints on the Strafer chassis."
    )
    parser.add_argument("--stage", required=True, help="Input USD to read.")
    parser.add_argument("--output-usd", required=True, help="Path to write modified USD.")
    parser.add_argument("--log", help="Optional path to write a summary log.")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass to assign to bodies.")
    parser.add_argument(
        "--delete-excluded",
        action="store_true",
        help="Delete excluded prims from stage for maximum efficiency (irreversible)."
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    log: List[str] = []
    log.append("=" * 60)
    log.append("Strafer Chassis Physics Setup")
    log.append("=" * 60)
    
    # Setup frame (articulation root and rail connections)
    log.append("\n--- Setting up frame rails ---")
    middle_body = setup_frame(stage, log, mass=args.mass)
    
    if not middle_body:
        log.append("[ERROR] Failed to set up frame, aborting.")
    else:
        # Handle excluded paths
        if args.delete_excluded:
            log.append("\n--- Deleting excluded prims ---")
            deleted = delete_excluded_prims(stage, log)
            log.append(f"[INFO] Deleted {deleted} excluded prims")
        else:
            log.append("\n--- Excluded paths (no physics, use --delete-excluded to remove) ---")
            for path in FRAME_EXCLUDE_PATHS:
                log.append(f"[SKIP] Frame: {path}")
            for path in ROOT_EXCLUDE_PATHS:
                log.append(f"[SKIP] Root: {path}")
        
        # Setup each wheel
        for wheel_name, rail_name in WHEEL_TO_RAIL.items():
            setup_wheel(stage, wheel_name, rail_name, log, mass=args.mass)
    
    # Export
    out_usd = Path(args.output_usd)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    log.append(f"\n{'=' * 60}")
    log.append(f"Exported modified stage to: {out_usd.resolve()}")
    log.append("=" * 60)

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log), encoding="utf-8")
        print(f"Wrote log to {log_path.resolve()}")
    else:
        print("\n".join(log))


if __name__ == "__main__":
    main()
